"""Runner durability, portable ownership, and exact terminal result regressions."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from constraint_work import WorkScope,frontier_snapshot,restore_frontier,disjoint,digest
from gpu_domain_runner import solve,split_cubes,pack_native
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face,tetrahedron_tiles,multiple_loops

class DomainPartitionTests(unittest.TestCase):
    def test_already_stopped_run_never_allocates_gpu(self):
        problem=SearchProblem(*tetrahedron_tiles())
        with patch('gpu_domain_round24.DomainCUDA') as engine:
            result=solve(problem,stop=lambda:True)
        engine.assert_not_called();self.assertEqual(result['status'],'stopped')
        self.assertIsNone(result['checkpoint']['native'])
        self.assertTrue(result['checkpoint']['frontier']['boxes'])

    def test_restore_failure_closes_allocated_engine(self):
        from gpu_domain_runner import COUNTERS
        problem=SearchProblem(*tetrahedron_tiles());scope=WorkScope.for_problem(problem);closed=[]
        class Fake:
            def __init__(self,*args,**kwargs):pass
            def restore(self,payload):raise ValueError('invalid native state')
            def close(self):closed.append(True)
        native={'cubes':[[hex(x) for x in scope.root]]}
        payload={'version':1,'scope':scope.payload(),'frontier':frontier_snapshot(scope,[scope.root]),'queue':[],
                 'totals':dict.fromkeys(COUNTERS,0),'completed_lanes':0,'native':native}
        payload['digest']=digest(payload)
        with patch('gpu_domain_round24.DomainCUDA',Fake),patch('gpu_domain_runner.unpack_native',return_value={}):
            with self.assertRaisesRegex(ValueError,'invalid native state'):solve(problem,resume=payload,replicas=1)
        self.assertEqual(closed,[True])

    def test_split_has_no_overlap_or_missing_assignments(self):
        import math
        source=(0b11111,0b111,0b11)
        parts=split_cubes([source],23)
        self.assertTrue(disjoint(parts));self.assertEqual(len(parts),23)
        self.assertEqual(sum(math.prod(x.bit_count() for x in row) for row in parts),30)

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA runner tests')
class DomainRunnerTests(unittest.TestCase):
    def test_edge_only_completion_is_never_labelled_gold_solved(self):
        problem=SearchProblem(multiple_loops(),double_face())
        answer=solve(problem,seconds=5,target="edges",replicas=2)
        self.assertEqual(answer["status"],"edge_perfect")
        self.assertFalse(answer["validation"]["valid"])
        self.assertEqual(answer["validation"]["matched_edges"],len(problem.edges))

    def test_stop_native_json_resume_and_repartition(self):
        problem=SearchProblem(*tetrahedron_tiles())
        checks=iter((False,True,True))
        result=solve(problem,seconds=0,replicas=3,stop=lambda:next(checks,True))
        self.assertEqual(result['status'],'stopped');self.assertFalse(result['complete'])
        original_digest=digest(result['checkpoint'])
        payload=json.loads(json.dumps(result['checkpoint']))
        for lanes in (1,3,8):
            resumed=solve(problem,seconds=20,replicas=lanes,resume=payload)
            self.assertEqual(resumed['status'],'solved');self.assertTrue(problem.validate(resumed['codes'])['valid'])
            self.assertEqual(digest(payload),original_digest)
    def test_common_cpu_to_gpu_to_cpu_transport(self):
        from constraint_dfs import solve as cpu
        from constraint_parallel import solve as parallel
        from constraint_portable import export_frontier
        problem=SearchProblem(*tetrahedron_tiles())
        original=cpu(problem,seconds=0)
        frontier=export_frontier(problem,original['checkpoint'])
        gpu=solve(problem,replicas=2,work_frontier=frontier,stop=lambda:True)
        self.assertEqual(gpu['status'],'stopped')
        remaining=gpu['checkpoint']['frontier']
        expected=restore_frontier(frontier,WorkScope.for_problem(problem))
        actual=restore_frontier(remaining,WorkScope.for_problem(problem))
        import math
        self.assertEqual(sum(math.prod(x.bit_count() for x in row) for row in expected),sum(math.prod(x.bit_count() for x in row) for row in actual))
        answer=parallel(problem,seconds=20,workers=1,resume=remaining)
        self.assertEqual(answer['status'],'solved');self.assertTrue(problem.validate(answer['codes'])['valid'])
    def test_completed_infeasible_lanes_enter_scoped_coverage(self):
        from constraint_work import CoverageLedger
        problem=SearchProblem(multiple_loops(),double_face())
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'coverage.sqlite3'
            result=solve(problem,seconds=20,replicas=2,options={'shared_coverage':True},learning_state={'_coverage_path':str(path)})
            self.assertEqual(result['status'],'infeasible');self.assertTrue(result['complete'])
            with CoverageLedger(path,WorkScope.for_problem(problem)) as ledger:self.assertTrue(ledger.exclusions())
            again=solve(problem,seconds=20,replicas=2,options={'shared_coverage':True},learning_state={'_coverage_path':str(path)})
            self.assertEqual(again['nodes'],0);self.assertEqual(again['status'],'infeasible')
    def test_runner_detects_frontier_and_native_disagreement(self):
        problem=SearchProblem(*tetrahedron_tiles())
        checks=iter((False,True,True))
        payload=solve(problem,replicas=2,stop=lambda:next(checks,True))['checkpoint']
        payload['frontier']=frontier_snapshot(WorkScope.for_problem(problem),())
        payload['digest']=digest({k:v for k,v in payload.items() if k!='digest'})
        with self.assertRaisesRegex(ValueError,'disagree'):solve(problem,replicas=2,resume=payload)

if __name__=='__main__':unittest.main()
