import copy,json,random,time,unittest
from constraint_sat_theory_round26 import direct_propagator
from constraint_sat_session import PersistentSatSession,SessionInterrupted
from constraint_work import WorkScope
from constraint_models import solve
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,multiple_loops,double_face,brute
from test_constraint_sat_domains import fixture
from test_constraint_separator import permutation_ring
from test_dependencies import HAS_SAT


def literals(problem):
    answer=[];next_id=1
    for row in problem.domains:
        answer.append({code:next_id+i for i,code in enumerate(row)});next_id+=len(row)
    return answer

def assignments(board,variables):return {lit:lit if board[cell]==code else -lit for cell,row in enumerate(variables) for code,lit in row.items()}

@unittest.skipUnless(HAS_SAT,'optional SAT callback backend')
class TheoryTests(unittest.TestCase):
    def test_consistent_and_interrupted_affine_work_is_counted(self):
        from unittest.mock import patch
        problem,_=fixture();variables=literals(problem)
        for interrupted in (False,True):
            helper=direct_propagator(problem,variables,target='edges')
            def produce(*args,stats,**kwargs):
                stats.update(affine_reason_equations=2,affine_reason_pattern_checks=7,affine_reason_xors=11)
                if interrupted:raise SessionInterrupted('stopped')
                return None
            with patch('constraint_affine_reasons_round26.produce_affine_conflict',side_effect=produce):
                if interrupted:
                    with self.assertRaises(SessionInterrupted):helper.verify_clause([])
                else:self.assertFalse(helper.verify_clause([]))
            self.assertEqual([helper.theory_stats[k] for k in ('theory_affine_equations','theory_affine_pattern_checks','theory_affine_xors')],[2,7,11])

    def test_direct_reason_is_unit_valid_and_survives_backtracking(self):
        problem,_=fixture();variables=literals(problem);helper=direct_propagator(problem,variables)
        domains=[2428,28,1730,1851];helper.on_new_level()
        for cell,row in enumerate(variables):
            for code,lit in row.items():
                if not domains[cell]>>code&1:helper.on_assignment(-lit)
        inferred=helper.propagate();self.assertIsNone(helper.error);self.assertTrue(inferred)
        solutions=brute(problem,'gold')
        for lit in inferred:
            reason=helper.provide_reason(lit)
            self.assertEqual(reason[0],lit)
            self.assertTrue(all(helper.assignments.get(abs(other))==-other for other in reason if other!=lit))
            for board in solutions:self.assertTrue(any(assignments(board,variables)[abs(v)]==v for v in reason))
            helper.on_assignment(lit);self.assertEqual(helper.provide_reason(lit),reason)
        saved=json.loads(json.dumps(helper.semantic_snapshot()))
        restored=direct_propagator(problem,variables);self.assertEqual(len(restored.semantic_restore(saved)),len(saved['clauses']))
        bad=copy.deepcopy(saved);bad['clauses']=[[]]
        with self.assertRaises(ValueError):direct_propagator(problem,variables).semantic_restore(bad)
        helper.on_backtrack(0);self.assertFalse(helper.assignments)
        helper.propagate();self.assertIsNone(helper.error)
        self.assertGreater(helper.theory_stats['theory_affine_pattern_checks'],0)

    def test_interrupted_claim_replay_retains_the_pending_claim(self):
        problem,_=fixture();variables=literals(problem);helper=direct_propagator(problem,variables)
        # This clause is an independently valid complete cell-domain axiom.
        claim=tuple(variables[0].values());helper.pending.append(claim);helper.dirty=False
        checks=[]
        def stop():
            checks.append(1)
            if len(checks)==2:raise SessionInterrupted('stopped')
        helper.replay_check=stop;self.assertEqual(helper.propagate(),[])
        self.assertEqual(len(checks),2)
        self.assertEqual(list(helper.pending),[claim])
        helper.replay_check=lambda:None;helper.propagate()
        self.assertFalse(helper.pending);self.assertIsNone(helper.error)
        self.assertGreater(helper.theory_stats['theory_verified_clauses'],0)

    def test_actual_semantic_mutations_fail_before_reusing_theory(self):
        mutators=(lambda p:p.masks.__setitem__((0,0),int(p.masks[0,0])^1),
                  lambda p:p.tiles[0].__setitem__('segments',()),
                  lambda p:setattr(p,'edges',tuple(reversed(p.edges))),
                  lambda p:p.domains[0].pop())
        for mutate in mutators:
            problem,_=fixture();helper=direct_propagator(problem,literals(problem));mutate(problem)
            with self.assertRaisesRegex(ValueError,'semantic source'):helper.assert_semantic_source()
            self.assertEqual(helper.propagate(),[]);self.assertIsInstance(helper.error,ValueError)
            with self.assertRaisesRegex(ValueError,'semantic source'):helper.semantic_snapshot()

    def test_random_context_reasons_preserve_every_independent_gold_board(self):
        problem,_=fixture();variables=literals(problem);solutions=brute(problem,'gold');rng=random.Random(260403)
        helper=direct_propagator(problem,variables)
        for _ in range(40):
            helper.on_backtrack(0);helper.on_new_level()
            domains=[sum(1<<code for code in row if rng.random()<.6) for row in problem.domains]
            if not all(domains):continue
            for cell,row in enumerate(variables):
                for code,lit in row.items():
                    if not domains[cell]>>code&1:helper.on_assignment(-lit)
            inferred=helper.propagate();self.assertIsNone(helper.error)
            clauses=[helper.provide_reason(lit) for lit in inferred]
            while helper.has_clause():clauses.append(helper.add_clause())
            for clause in clauses:
                for board in solutions:self.assertTrue(any(assignments(board,variables)[abs(v)]==v for v in clause))
        self.assertGreater(helper.theory_stats['theory_verified_clauses'],0)

    def test_native_persistent_models_and_policy_combinations(self):
        for problem in (SearchProblem(simple_tiles(),double_face()),SearchProblem(multiple_loops(),double_face()),
                        fixture()[0],permutation_ring([(1,0,2),(0,2,1)]*2)):
            for target in ('gold','edges'):
                expected=brute(problem,target)
                for extra in ({},{'sat_guard_reuse':True},{'sat_simplification':True,'regional_compiler':True}):
                    result=solve(problem,engine='sat',target=target,seconds=8,options={'persistent_theory':True,**extra})
                    self.assertTrue(result['complete'],result['status'])
                    if expected:self.assertIn(tuple(result['codes']),expected)
                    else:self.assertEqual(result['status'],'infeasible')

    def test_warm_slices_stop_restore_and_checked_restart_clauses(self):
        problem=SearchProblem(multiple_loops(),double_face());scope=WorkScope.for_problem(problem,'gold')
        owner=PersistentSatSession(problem,scope,theory=True,guard_reuse=True)
        try:
            first=owner.advance(seconds=8)
            self.assertEqual(first['status'],'infeasible')
            native=owner.solver;prior=first['stats']['session_model_builds']
            second=owner.advance(seconds=1);self.assertIs(owner.solver,native);self.assertEqual(second['stats']['session_model_builds'],prior)
            saved=owner.snapshot();self.assertTrue(saved['theory_clauses']['clauses'])
            # Reopen the same bounded root solely to test independently checked
            # semantic restoration, not as production scheduling behavior.
            from constraint_work import frontier_snapshot,digest
            saved['frontier']=frontier_snapshot(scope,[scope.root]);saved['digest']=digest({k:v for k,v in saved.items() if k!='digest'})
        finally:owner.close()
        restored=PersistentSatSession.restore(problem,saved)
        try:
            result=restored.advance(seconds=8);self.assertEqual(result['status'],'infeasible')
            self.assertGreater(result['stats']['theory_restart_clauses'],0)
            restored.configure_theory(False);self.assertFalse(restored.theory)
        finally:restored.close()
        stopped=PersistentSatSession(problem,scope,theory=True)
        try:
            self.assertEqual(stopped.advance(seconds=8,stop=lambda:True)['status'],'stopped')
            self.assertIsNone(stopped.solver)
        finally:stopped.close()

if __name__=='__main__':unittest.main()
