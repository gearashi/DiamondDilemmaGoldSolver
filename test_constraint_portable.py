import itertools,json,tempfile,unittest
from pathlib import Path
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,multiple_loops,double_face,brute
from constraint_dfs import _solve_core
from constraint_portable import export_frontier,dfs_roots
from constraint_work import WorkScope,frontier_snapshot,restore_frontier,CoverageLedger
from constraint_parallel import solve as parallel
from constraint_sat_session import solve as sat
from test_constraint_sat_assist import HAS_SAT

class PortableTests(unittest.TestCase):
    def test_cpu_split_roundtrip_exact_pending_products(self):
        p=SearchProblem(simple_tiles(),double_face())
        checkpoint=_solve_core(p,seconds=0)['checkpoint']
        exported=export_frontier(p,checkpoint)
        roots=dfs_roots(p,json.loads(json.dumps(exported)))
        self.assertEqual(len(roots),1);self.assertNotIn('root_symmetry',roots[0])
        self.assertEqual(export_frontier(p,roots[0]),exported)
        answer=parallel(p,seconds=10,resume=exported,workers=1)
        self.assertTrue(answer['validation']['valid'])
    def test_empty_frontier_exhausted_without_starting_worker(self):
        p=SearchProblem(simple_tiles(),double_face());empty=frontier_snapshot(WorkScope.for_problem(p),[])
        answer=parallel(p,seconds=5,resume=empty,workers=2)
        self.assertEqual(answer['status'],'infeasible');self.assertEqual(answer['stats']['parallel_workers_started'],0)
    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_cpu_sat_cpu_transport_preserves_exact_remaining_work(self):
        p=SearchProblem(multiple_loops(),double_face());scope=WorkScope.for_problem(p)
        roots=[(1<<a,1<<b) for a in range(6) for b in range(6)]
        incoming=frontier_snapshot(scope,roots);stop=[False]
        first=sat(p,seconds=5,resume=incoming,stop=lambda:stop[0],progress=lambda event:stop.__setitem__(0,True))
        self.assertEqual(first['status'],'stopped')
        remaining=export_frontier(p,first['checkpoint'])
        self.assertEqual(len(remaining['boxes']),35)
        answer=parallel(p,seconds=15,resume=remaining,workers=2)
        self.assertEqual(answer['status'],'infeasible')
    @unittest.skipUnless(HAS_SAT,'optional python-sat missing')
    def test_completed_sat_coverage_closes_cpu_queue_without_reexecution(self):
        p=SearchProblem(multiple_loops(),double_face())
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'work.sqlite');options={'shared_coverage':True,'persistent_sat':True}
            state={'_coverage_path':path}
            try:answer=sat(p,seconds=5,options=options,learning_state=state)
            finally:
                if state.get('_sat_session') is not None:state['_sat_session'].close()
            self.assertEqual(answer['status'],'infeasible')
            scope=WorkScope.for_problem(p)
            answer=parallel(p,seconds=10,resume=frontier_snapshot(scope,[scope.root]),workers=2,options=options,learning_state={'_coverage_path':path})
            self.assertEqual(answer['status'],'infeasible');self.assertEqual(answer['stats']['parallel_workers_started'],0)

if __name__=='__main__':unittest.main()
