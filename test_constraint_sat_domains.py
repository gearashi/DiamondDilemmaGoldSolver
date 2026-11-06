"""Full-context SAT deductions checked against independent finite arrangements."""
import itertools,random,time,unittest
from constraint_sat_assist import SatAssist
from constraint_pruning import Pruner
from validator import reverse_mask,validate_arrangement
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,brute,simple_tiles,double_face
from test_constraint_sat_assist import Fake,GoldFixtureOracle,HAS_SAT,gold_fixture


def fixture():
    _,mesh=tetrahedron_tiles()
    pairings=[[[[1,10],[1,2]],[[2,3],[0,2]],[[2,9],[0,10]]],
              [[[0,10],[1,10]],[[2,9],[0,2]],[[1,2],[2,3]]],
              [[[0,10],[0,2]],[[2,9],[1,2]],[[2,3],[1,10]]],
              [[[0,2],[2,3]],[[1,10],[1,2]],[[0,10],[2,9]]]]
    tiles=[{'id':i,'segments':segments} for i,segments in enumerate(pairings)]
    return SearchProblem(tiles,mesh),[3972,4045,4061,879]


def ordinary_fixpoint(problem,domains):
    domains=list(domains);pruner=Pruner(problem)
    while all(domains):
        before=domains.copy()
        for a,sa,b,sb in problem.edges:
            for left,side,right,other in ((a,sa,b,sb),(b,sb,a,sa)):
                allowed={reverse_mask(int(problem.masks[c,other])) for c in range(3*problem.n) if domains[right]>>c&1}
                domains[left]&=sum(1<<c for c in range(3*problem.n) if int(problem.masks[c,side]) in allowed)
        if not all(domains):break
        pruner.prune(domains)
        if domains==before:break
    return domains


def admitted(rows,domains):
    return [row for row in rows if all(mask>>code&1 for mask,code in zip(domains,row))]


class DomainFilteringTests(unittest.TestCase):
    def helper(self,problem,*,native=False,factory=None,initial=None,target='gold'):
        if initial is None:initial=[sum(1<<code for code in row) for row in problem.domains]
        if not native and factory is None:factory=lambda:GoldFixtureOracle(problem,initial)
        helper=SatAssist(problem,initial,target=target,**({} if native else {'factory':factory}))
        self.addCleanup(helper.close);return helper

    def assert_preserves(self,before,after,solutions):
        self.assertTrue(all(not b&~a for a,b in zip(before,after)))
        self.assertEqual(admitted(solutions,before),admitted(solutions,after))

    def test_full_domain_restrictions_prove_new_candidates_without_decision_facts(self):
        problem,domains=fixture();edge=brute(problem,'edges');gold=brute(problem,'gold')
        self.assertEqual(ordinary_fixpoint(problem,domains),domains)
        helper=self.helper(problem);self.assertTrue(helper.prepare(seconds=2))
        self.assertIsNone(helper.query([],seconds=.2));self.assertEqual(helper.last_outcome,'sat')
        before=domains.copy();filtered=helper.filter_domains(domains,seconds=1,max_candidates=48)
        self.assertEqual(domains,before);self.assertEqual(filtered,(3460,1737,1177,301))
        self.assertEqual(sum(a.bit_count()-b.bit_count() for a,b in zip(domains,filtered)),12)
        self.assert_preserves(domains,filtered,edge);self.assert_preserves(domains,filtered,gold)
        self.assertTrue(admitted(gold,domains))
        self.assertEqual(helper.drain_candidates(),[]);self.assertFalse(helper.cores)
        # A widened sibling must not inherit the previous branch's removals.
        widened=helper.filter_domains(helper.initial,seconds=1,max_candidates=48)
        self.assert_preserves(helper.initial,widened,edge)
        self.assertTrue(any(not all(filtered[c]>>v&1 for c,v in enumerate(row)) for row in edge))

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_actual_glucose_default_budget_and_complete_probe_filter_preserve_solutions(self):
        problem,domains=fixture();edge=brute(problem,'edges');gold=brute(problem,'gold')
        helper=self.helper(problem,native=True);self.assertTrue(helper.prepare(seconds=2))
        short=helper.filter_domains(domains,seconds=1)
        self.assertNotEqual(short,tuple(domains));self.assertLessEqual(helper.stats['sat_assist_domain_calls'],16)
        self.assert_preserves(domains,short,edge);self.assert_preserves(domains,short,gold)
        full=helper.filter_domains(domains,seconds=1,max_candidates=48)
        self.assertEqual(full,(3460,1737,1177,301))
        self.assert_preserves(domains,full,edge);self.assert_preserves(domains,full,gold)
        self.assertGreater(helper.stats['sat_assist_domain_propagations'],0)
        self.assertFalse(helper.cores);self.assertEqual(helper.drain_candidates(),[])

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_actual_unit_consequences_force_placements_without_candidate_probes(self):
        problem=SearchProblem(simple_tiles(),double_face());helper=self.helper(problem,native=True)
        self.assertTrue(helper.prepare(seconds=2));domains=[1,helper.initial[1]]
        filtered=helper.filter_domains(domains,seconds=1,max_candidates=0)
        self.assertEqual(filtered,(1,8));self.assertEqual(helper.stats['sat_assist_domain_calls'],0)
        self.assertEqual(helper.stats['sat_assist_domain_forced'],1)
        self.assert_preserves(domains,filtered,brute(problem,'gold'))

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_random_narrowed_and_widened_contexts_preserve_all_edge_and_gold_solutions(self):
        problem,unused=fixture();edge=brute(problem,'edges');gold=brute(problem,'gold');rng=random.Random(2202)
        helper=self.helper(problem,native=True);self.assertTrue(helper.prepare(seconds=2))
        for _ in range(40):
            witness=rng.choice(gold)
            domains=[(1<<witness[c])|sum(1<<code for code in problem.domains[c] if rng.randrange(3)) for c in range(problem.n)]
            filtered=helper.filter_domains(domains,seconds=.3,max_candidates=48)
            self.assert_preserves(domains,filtered,edge);self.assert_preserves(domains,filtered,gold)
        self.assertEqual(helper.drain_candidates(),[])

    def test_gold_clause_strengthening_is_seen_without_scoped_proof_leakage(self):
        problem,store,edge,gold=gold_fixture();helper=self.helper(problem)
        self.assertTrue(helper.prepare(seconds=2));domains=[1,(1<<3)|(1<<4)]
        before=helper.filter_domains(domains,seconds=1);self.assertEqual(before,tuple(domains))
        helper.add_gold_witnesses(store)
        after=helper.filter_domains(domains,seconds=1)
        self.assertEqual(after,(1,1<<4));self.assert_preserves(domains,after,gold)
        self.assertFalse(helper.cores);self.assertEqual(helper.drain_candidates(),[])
        edge_helper=self.helper(problem,target='edges');self.assertTrue(edge_helper.prepare(seconds=2))
        self.assertEqual(edge_helper.filter_domains(domains,seconds=1),tuple(domains))

    def test_model_not_ready_absent_and_zero_budget_leave_inputs_alone(self):
        problem,domains=fixture()
        helper=self.helper(problem,factory=Fake);self.assertEqual(helper.filter_domains(domains),tuple(domains))
        self.assertTrue(helper.prepare(seconds=2))
        self.assertEqual(helper.filter_domains(domains,seconds=0),tuple(domains));self.assertFalse(helper.solver.called)
        def absent():raise ImportError('optional dependency absent')
        unavailable=self.helper(problem,factory=absent)
        self.assertEqual(unavailable.filter_domains(domains),tuple(domains));self.assertEqual(unavailable.status,'unavailable')

    def test_unknown_and_late_false_answers_never_remove_candidates(self):
        problem,domains=fixture()
        class LateFalse(Fake):
            def solve_limited(self,**kwargs):self.called=True;self.ended.wait(.5);return False
        for factory in (Fake,LateFalse):
            helper=self.helper(problem,factory=factory);self.assertTrue(helper.prepare(seconds=2))
            self.assertEqual(helper.filter_domains(domains,seconds=.1),tuple(domains))
            self.assertEqual(helper.stats['sat_assist_domain_prunes'],0)
            self.assertEqual(helper.stats['sat_assist_domain_unsat'],0)
            self.assertGreater(helper.stats['sat_assist_domain_unknown'],0)

    def test_stop_discards_private_result_at_every_checked_boundary(self):
        problem,original=fixture()
        for stop_at in range(1,45):
            helper=self.helper(problem);self.assertTrue(helper.prepare(seconds=2));domains=original.copy();checks=[0]
            def stop():
                checks[0]+=1
                if checks[0]==stop_at:raise InterruptedError('requested Stop')
            try:helper.filter_domains(domains,stop,seconds=1,max_candidates=48)
            except InterruptedError:pass
            self.assertEqual(domains,original)
            self.assertEqual(helper.drain_candidates(),[]);self.assertFalse(helper.cores)
            helper.close()

    def test_unavailable_unit_propagation_falls_back_to_proved_candidate_checks(self):
        problem,domains=fixture()
        class NoPropagation(GoldFixtureOracle):
            def propagate(self,**kwargs):raise NotImplementedError('backend lacks this API')
        helper=self.helper(problem,factory=lambda:NoPropagation(problem));self.assertTrue(helper.prepare(seconds=2))
        self.assertEqual(helper.filter_domains(domains,seconds=1,max_candidates=48),(3460,1737,1177,301))
        self.assertGreater(helper.stats['sat_assist_domain_calls'],0)

    def test_late_propagation_conflict_is_not_used(self):
        problem,domains=fixture()
        class LateConflict(Fake):
            def propagate(self,**kwargs):self.ended.wait(.5);return False,[]
        helper=self.helper(problem,factory=LateConflict);self.assertTrue(helper.prepare(seconds=2))
        self.assertEqual(helper.filter_domains(domains,seconds=.1,max_candidates=0),tuple(domains))
        self.assertEqual(helper.stats['sat_assist_domain_unsat'],0)
        self.assertEqual(helper.stats['sat_assist_domain_prunes'],0)

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_domain_conflict_stays_local_and_does_not_poison_a_sibling(self):
        problem,domains=fixture();helper=self.helper(problem,native=True);self.assertTrue(helper.prepare(seconds=2))
        impossible=list(helper.initial);impossible[0]=impossible[1]=1
        result=helper.filter_domains(impossible,seconds=1)
        self.assertIn(0,result);self.assertEqual(helper.drain_candidates(),[]);self.assertFalse(helper.cores)
        sibling=helper.filter_domains(helper.initial,seconds=1,max_candidates=48)
        self.assert_preserves(helper.initial,sibling,brute(problem,'gold'))

    def test_validation_and_restricted_root_scope(self):
        problem,domains=fixture();helper=self.helper(problem,initial=domains)
        for invalid in (domains[:-1],[True,*domains[1:]],[1<<99,*domains[1:]],[-1,*domains[1:]]):
            with self.assertRaises(ValueError):helper.filter_domains(invalid)
        for maximum in (True,-1,257):
            with self.assertRaises(ValueError):helper.filter_domains(domains,max_candidates=maximum)
        self.assertTrue(helper.prepare(seconds=2));result=helper.filter_domains(domains,seconds=1,max_candidates=48)
        self.assertEqual(result,(3460,1737,1177,301));self.assertEqual(helper.drain_candidates(),[])
        with self.assertRaises(ValueError):helper.filter_domains([4095]*4)


if __name__=='__main__':unittest.main()
