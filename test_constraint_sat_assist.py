"""SAT is only a bounded UNSAT oracle, checked against direct enumeration."""
import itertools,time,threading,unittest
from constraint_sat_assist import SatAssist
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,simple_tiles,double_face,multiple_loops,brute
try:import pysat;HAS_SAT=True
except ImportError:HAS_SAT=False

def make(problem,**kwargs):
    return SatAssist(problem,[sum(1<<code for code in row) for row in problem.domains],**kwargs)

class Fake:
    def __init__(self):self.called=False;self.ended=threading.Event()
    def add_clause(self,clause):pass
    def clear_interrupt(self):self.ended.clear()
    def conf_budget(self,n):pass
    def solve_limited(self,**kwargs):self.called=True;self.ended.wait(.5);return None
    def interrupt(self):self.ended.set()
    def delete(self):pass

class AssistanceTests(unittest.TestCase):
    def test_partial_model_never_queries_or_proves_unsat(self):
        fake=Fake();helper=make(SearchProblem(simple_tiles(),double_face()),factory=lambda:fake)
        self.assertFalse(helper.prepare(seconds=0));self.assertIsNone(helper.query([]))
        self.assertFalse(fake.called);self.assertFalse(helper.ready)
        def stop():raise InterruptedError()
        with self.assertRaises(InterruptedError):helper.prepare(stop,seconds=1)
        self.assertFalse(helper.ready);helper.close()

    def test_unavailable_dependency_is_optional(self):
        def absent():raise ImportError('optional SAT unavailable')
        helper=make(SearchProblem(simple_tiles(),double_face()),factory=absent)
        self.assertEqual(helper.status,'unavailable');self.assertFalse(helper.prepare());self.assertIsNone(helper.query([]))

    def test_native_timeout_and_stop_remain_unknown(self):
        helper=make(SearchProblem(simple_tiles(),double_face()),factory=Fake)
        self.assertTrue(helper.prepare(seconds=1))
        self.assertIsNone(helper.query([],seconds=.01));self.assertFalse(helper.cache)
        checks=[0]
        def stop():
            checks[0]+=1
            if checks[0]>1:raise InterruptedError()
        with self.assertRaises(InterruptedError):helper.query([],stop,seconds=1)
        self.assertFalse(helper.cache);helper.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_assumption_cores_are_independently_unsatisfiable(self):
        problem=SearchProblem(*tetrahedron_tiles());solutions=brute(problem,'edges');helper=make(problem)
        self.assertTrue(helper.prepare(seconds=3));unsat=0
        for c in range(problem.n):
            for code in problem.domains[c]:
                decisions=[(c,code)]
                core=helper.query(decisions,seconds=.2,conflicts=10000)
                feasible=any(codes[c]==code for codes in solutions)
                if feasible:self.assertIsNone(core)
                else:
                    self.assertIsNotNone(core);self.assertLessEqual(set(core),set(decisions));unsat+=1
                    self.assertFalse(any(all(codes[i]==v for i,v in core) for codes in solutions))
                self.assertEqual(helper.query(decisions),core)
        self.assertGreater(unsat,0);self.assertGreater(helper.stats['sat_assist_cache_hits'],0)
        with self.assertRaises(ValueError):helper.query([(0,999)])
        helper.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_edge_feasibility_does_not_claim_gold(self):
        problem=SearchProblem(multiple_loops(),double_face());self.assertFalse(brute(problem,'gold'))
        helper=make(problem);self.assertTrue(helper.prepare(seconds=3))
        self.assertIsNone(helper.query([],seconds=.2));self.assertEqual(helper.stats['sat_assist_unsat'],0)
        helper.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_assistance_reuses_private_runtime_across_stopped_slices(self):
        import constraint_dfs as dfs
        problem=SearchProblem(*tetrahedron_tiles());cache={}
        first=dfs.solve(problem,seconds=0,options={'sat_assist':True},learning_state=cache)
        result=dfs.solve(problem,seconds=5,resume=first['checkpoint'],options={'sat_assist':True},learning_state=cache)
        self.assertEqual(result['status'],'solved');self.assertTrue(problem.validate(result['codes'])['valid'])
        self.assertIsInstance(cache['_runtime'].sat,SatAssist)


    def test_interrupted_false_answer_is_unknown_not_an_unsat_core(self):
        class LateFalse(Fake):
            def solve_limited(self,**kwargs):
                self.called=True;self.ended.wait(.5);return False
            def get_core(self):raise AssertionError('Interrupted result must not request a core.')
        helper=make(SearchProblem(simple_tiles(),double_face()),factory=LateFalse)
        try:
            self.assertTrue(helper.prepare(seconds=1))
            self.assertIsNone(helper.query([],seconds=.005))
            self.assertFalse(helper.cache);self.assertEqual(helper.stats['sat_assist_unsat'],0)
            self.assertEqual(helper.stats['sat_assist_unknown'],1)
        finally:helper.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_pair_assumption_core_is_needed_even_when_each_placement_is_feasible(self):
        from test_constraint_separator import ring
        problem=ring(4);solutions=brute(problem,'edges');helper=make(problem)
        try:
            self.assertTrue(helper.prepare(seconds=2))
            for placement in ((0,0),(1,0)):
                self.assertTrue(any(row[placement[0]]==placement[1] for row in solutions))
                self.assertIsNone(helper.query([placement],seconds=.2,conflicts=10000))
            core=helper.query([(0,0),(1,0)],seconds=.2,conflicts=10000)
            self.assertEqual(set(core),{(0,0),(1,0)})
            self.assertFalse(any(all(row[c]==v for c,v in core) for row in solutions))
            self.assertEqual(helper.query([(1,0),(0,0)]),core)
        finally:helper.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_proved_core_reuses_stronger_assumptions_without_native_call(self):
        from test_constraint_separator import ring
        problem=ring(4);helper=make(problem)
        try:
            self.assertTrue(helper.prepare(seconds=1))
            core=helper.query([(0,0),(1,0)],seconds=.2,conflicts=10000)
            self.assertIsNotNone(core);calls=helper.stats['sat_assist_calls']
            self.assertEqual(helper.drain_candidates(),[core])
            self.assertEqual(helper.query([(0,0),(1,0),(2,3)],seconds=.2),core)
            self.assertEqual(helper.stats['sat_assist_calls'],calls)
            self.assertEqual(helper.stats['sat_assist_subset_hits'],1)
            self.assertEqual(helper.last_outcome,'unsat');self.assertEqual(helper.drain_candidates(),[])
            self.assertFalse(any(all(board[c]==v for c,v in core) for board in brute(problem,'edges')))
        finally:helper.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_global_unsat_core_is_empty_and_partial_build_can_resume(self):
        data=simple_tiles();data[1]['segments']=[[[0,2],[1,4]]]
        problem=SearchProblem(data,double_face());self.assertFalse(brute(problem,'edges'))
        helper=make(problem);calls=[0]
        def check():
            calls[0]+=1
            if calls[0]==4:raise InterruptedError
        try:
            with self.assertRaises(InterruptedError):helper.prepare(check,seconds=1)
            self.assertFalse(helper.ready);self.assertFalse(helper.cache)
            self.assertTrue(helper.prepare(seconds=2))
            self.assertEqual(helper.query([],seconds=.2,conflicts=10000),())
            self.assertEqual(helper.query([]),())
        finally:helper.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_dfs_sat_assistance_never_promotes_multiloop_edges_to_gold(self):
        import constraint_dfs as dfs
        from search_options import BOOLEAN_OPTIONS
        problem=SearchProblem(multiple_loops(),double_face())
        options={key:False for key in BOOLEAN_OPTIONS};options.update(sat_assist=True,native='python')
        for target in ('gold','edges'):
            cache={};result=dfs.solve(problem,seconds=5,target=target,options=options,learning_state=cache)
            self.assertTrue(result['complete'])
            self.assertEqual(result['status'],'infeasible' if target=='gold' else 'edge_perfect')
            if target=='edges':self.assertFalse(problem.validate(result['codes'])['valid'])
            self.assertIsInstance(cache['_runtime'].sat,SatAssist)
            cache['_runtime'].sat.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_sat_runtime_reuse_rejects_changed_original_domains(self):
        import constraint_dfs as dfs
        problem=SearchProblem(*tetrahedron_tiles());cache={}
        result=dfs.solve(problem,seconds=5,options={'sat_assist':True,'native':'python'},learning_state=cache)
        self.assertEqual(result['status'],'solved');self.assertIsInstance(cache['_runtime'].sat,SatAssist)
        changed=SearchProblem(*tetrahedron_tiles());changed.domains[0]=changed.domains[0][1:]
        with self.assertRaises(ValueError):dfs.solve(changed,seconds=5,options={'sat_assist':True},learning_state=cache)
        cache['_runtime'].sat.close()


class GoldFixtureOracle(Fake):
    """Independent complete-board oracle; no SAT package needed for lifecycle tests.

    Base edge solutions come from the independent arrangement validator. Only
    clauses containing placement variables are interpreted here; auxiliary CNF
    is covered separately by actual Glucose tests. Gold clauses contain only
    placement variables, so their full finite effect is checked exactly.
    """
    def __init__(self,problem,initial=None):
        super().__init__();self.clauses=[];self.core=[]
        domains=problem.domains if initial is None else [[code for code in range(3*problem.n) if mask>>code&1] for mask in initial]
        self.ids={};counter=0
        for cell,row in enumerate(domains):
            for code in sorted(row):counter+=1;self.ids[cell,code]=counter
        self.placement_count=counter
        self.boards=[row for row in brute(problem,'edges') if all(code in domains[c] for c,code in enumerate(row))]
    def add_clause(self,clause):self.clauses.append(tuple(clause))
    def solve_limited(self,assumptions,**kwargs):
        self.called=True;self.core=list(assumptions)
        for row in self.boards:
            chosen={self.ids[cell,code] for cell,code in enumerate(row)}
            if not all(lit in chosen if lit>0 else -lit not in chosen for lit in assumptions):continue
            if all(any(lit in chosen if lit>0 else -lit not in chosen for lit in clause)
                   for clause in self.clauses if all(abs(lit)<=self.placement_count for lit in clause)):
                return True
        return False
    def get_core(self):return self.core


def gold_fixture():
    from constraint_knowledge import KnowledgeStore
    from test_constraint_knowledge import mixed_problem
    problem=mixed_problem();store=KnowledgeStore(problem)
    edge=brute(problem,'edges');gold=brute(problem,'gold')
    for row in edge:
        if row not in gold:store.add(list(enumerate(row)))
    return problem,store,edge,gold


class GoldAssistanceTests(unittest.TestCase):
    def helper(self,problem,initial=None,*,native=False):
        if initial is None:initial=[sum(1<<code for code in row) for row in problem.domains]
        helper=SatAssist(problem,initial,**({} if native else {'factory':lambda:GoldFixtureOracle(problem,initial)}))
        self.addCleanup(helper.close);return helper

    def verify_all_boards(self,*,native=False):
        problem,store,edge,gold=gold_fixture();helper=self.helper(problem,native=native)
        self.assertEqual((len(edge),len(gold),len(store)),(18,12,6))
        self.assertTrue(helper.prepare(seconds=2))
        for row in edge:self.assertIsNone(helper.query(list(enumerate(row)),seconds=.2,conflicts=10000))
        self.assertEqual(sum(core is None for core in helper.cache.values()),18)
        self.assertEqual(helper.add_gold_witnesses(store),6)
        self.assertEqual(helper.stats['sat_assist_gold_verified'],6)
        self.assertEqual(helper.stats['sat_assist_gold_clauses'],6)
        self.assertEqual(helper.stats['sat_assist_gold_cache_invalidations'],18)
        rejected=0
        for row in edge:
            core=helper.query(list(enumerate(row)),seconds=.2,conflicts=10000)
            if row in gold:
                self.assertIsNone(core);self.assertEqual(helper.last_outcome,'sat')
            else:
                self.assertIsNotNone(core);rejected+=1
                self.assertFalse(any(all(solution[c]==v for c,v in core) for solution in gold))
        self.assertEqual(rejected,6)
        return helper

    def test_independent_board_oracle_preserves_every_gold_and_rejects_six_multiloops(self):
        self.verify_all_boards()

    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_actual_sat_preserves_every_gold_and_rejects_six_multiloops(self):
        self.verify_all_boards(native=True)

    def test_dfs_imports_gold_knowledge_only_when_enabled(self):
        from unittest.mock import patch
        import constraint_dfs as dfs
        original=SatAssist.__init__
        def initialize(helper,problem,initial,target='gold',**kwargs):
            original(helper,problem,initial,target,factory=lambda:GoldFixtureOracle(problem,initial))
        for target,enabled in (('gold',True),('gold',False),('edges',True)):
            problem,source,edge,gold=gold_fixture()
            cache={'_knowledge':source} if target=='gold' else {}
            with self.subTest(target=target,enabled=enabled),patch.object(SatAssist,'__init__',initialize):
                result=dfs.solve(problem,target=target,seconds=5,learning_state=cache,
                    options={'sat_assist':True,'persistent_learning':enabled,'native':'python','lookahead':False})
            helper=cache['_runtime'].sat
            self.addCleanup(helper.close)
            self.assertIn(tuple(result['codes']),gold if target=='gold' else edge)
            self.assertEqual(helper.stats['sat_assist_gold_clauses'],6 if target=='gold' and enabled else 0)
            self.assertEqual(result['stats']['sat_assist_gold_clauses'],helper.stats['sat_assist_gold_clauses'])

    def test_building_queue_budget_dedup_and_proof_reuse(self):
        from unittest.mock import patch
        problem,store,_,_=gold_fixture();helper=self.helper(problem)
        self.assertEqual(helper.add_gold_witnesses(store,max_records=0),0)
        self.assertFalse(helper._gold_known)
        self.assertEqual(helper.add_gold_witnesses(store,max_records=2),2)
        self.assertEqual(len(helper._gold_pending),2);self.assertEqual(helper.stats['sat_assist_gold_clauses'],0)
        self.assertFalse(helper.ready);self.assertIsNone(helper.query([]))
        for _ in range(2):self.assertEqual(helper.add_gold_witnesses(store,max_records=2),2)
        self.assertEqual(len(helper._gold_pending),6)
        with patch('constraint_sat_assist.revalidate_witness',side_effect=AssertionError('exact retained proof repeated')):
            self.assertEqual(helper.add_gold_witnesses(store),0)
        self.assertTrue(helper.prepare(seconds=2))
        self.assertFalse(helper._gold_pending);self.assertEqual(helper.stats['sat_assist_gold_clauses'],6)
        self.assertIsNotNone(helper.query([(0,0),(1,3)],seconds=.2))

    def test_stop_during_proof_and_queued_clause_install_preserves_valid_state(self):
        problem,store,_,gold=gold_fixture();helper=self.helper(problem);checks=[0]
        def stop():
            checks[0]+=1
            if checks[0]==3:raise InterruptedError('proof finished before commit')
        with self.assertRaises(InterruptedError):helper.add_gold_witnesses(store,stop)
        self.assertFalse(helper._gold_pending);self.assertFalse(helper._gold_known)
        self.assertEqual(helper.stats['sat_assist_gold_clauses'],0)
        self.assertEqual(helper.add_gold_witnesses(store),6)
        def stop_installed():
            if helper.stats['sat_assist_gold_clauses']==1:raise InterruptedError('one committed clause')
        with self.assertRaises(InterruptedError):helper.prepare(stop_installed,seconds=2)
        self.assertFalse(helper.ready);self.assertEqual(len(helper._gold_pending),5)
        self.assertIsNone(helper.query([]))
        self.assertTrue(helper.prepare(seconds=2));self.assertEqual(helper.stats['sat_assist_gold_clauses'],6)
        for row in gold:self.assertIsNone(helper.query(list(enumerate(row)),seconds=.2))

    def test_malformed_foreign_edge_and_changed_domain_sources_are_rejected(self):
        from constraint_knowledge import KnowledgeStore
        from unittest.mock import patch
        problem,store,_,gold=gold_fixture();helper=self.helper(problem)
        foreign=KnowledgeStore(SearchProblem(multiple_loops(),double_face()))
        for source in (foreign,KnowledgeStore(problem,target='edges'),store.records()):
            with self.assertRaises(ValueError):helper.add_gold_witnesses(source)
        for records in ([[[False,0],[1,3]]],[[[0,0],[1,4]]],[[[0,0],[0,1]]],[[[0,0],[1,0]]]):
            with patch.object(store,'records',return_value=records),self.assertRaises(ValueError):
                helper.add_gold_witnesses(store)
        self.assertFalse(helper._gold_known)
        edge=SatAssist(problem,helper.initial,target='edges',factory=Fake);self.addCleanup(edge.close)
        with self.assertRaises(ValueError):edge.add_gold_witnesses(store)
        for cap in (-1,True,257):
            with self.assertRaises(ValueError):helper.add_gold_witnesses(store,max_records=cap)
        problem.domains[0].pop()
        with self.assertRaises(ValueError):helper.add_gold_witnesses(store)

    def test_restricted_initial_skips_tautologies_and_empty_core_stays_scoped(self):
        problem,store,_,gold=gold_fixture()
        initial=[1<<0,(1<<3)|(1<<4)];helper=self.helper(problem,initial)
        self.assertTrue(helper.prepare(seconds=2));self.assertEqual(helper.add_gold_witnesses(store),1)
        self.assertEqual(helper.stats['sat_assist_gold_irrelevant'],5)
        self.assertIsNotNone(helper.query([(1,3)],seconds=.2))
        self.assertIsNone(helper.query([(1,4)],seconds=.2))
        restricted=self.helper(problem,[1<<0,1<<3]);self.assertTrue(restricted.prepare(seconds=2))
        self.assertIsNone(restricted.query([],seconds=.2));restricted.add_gold_witnesses(store)
        self.assertEqual(restricted.query([],seconds=.2),())
        self.assertEqual(restricted.drain_candidates(),[()]);self.assertTrue(gold)

    def test_gold_clauses_respect_the_same_native_model_clause_cap(self):
        problem,store,_,_=gold_fixture()
        for ready in (False,True):
            helper=self.helper(problem)
            if ready:self.assertTrue(helper.prepare(seconds=2))
            else:
                helper.add_gold_witnesses(store)
                # Finish base construction, leaving queued Gold work private.
                while True:
                    try:clause=next(helper.pending)
                    except StopIteration:helper.pending=None;break
                    if clause is not None:helper.solver.add_clause(clause)
            helper.clauses=1000000
            if ready:self.assertEqual(helper.add_gold_witnesses(store),0)
            else:self.assertFalse(helper.prepare(seconds=2))
            self.assertFalse(helper.ready);self.assertEqual(helper.status,'unsupported')
            self.assertEqual(helper.stats['sat_assist_gold_clauses'],0)
            self.assertIsNone(helper.query([]))

    def test_strengthening_keeps_unsat_cache_and_memory_cap_is_unknown(self):
        problem,store,_,_=gold_fixture();helper=self.helper(problem);self.assertTrue(helper.prepare(seconds=2))
        core=helper.query([(0,0),(1,0)],seconds=.2);self.assertIsNotNone(core)
        helper.add_gold_witnesses(store,max_records=1)
        calls=helper.stats['sat_assist_calls'];self.assertEqual(helper.query([(0,0),(1,0)]),core)
        self.assertEqual(helper.stats['sat_assist_calls'],calls)
        helper._gold_limit=len(helper._gold_known)
        self.assertEqual(helper.add_gold_witnesses(store),0)
        self.assertEqual(helper.stats['sat_assist_gold_clauses'],1)
        self.assertGreater(helper.stats['sat_assist_gold_budget_skips'],0)


if __name__=='__main__':unittest.main()
