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

if __name__=='__main__':unittest.main()
