"""App-controller and actual-native lifecycle for optimized recombination."""
from copy import deepcopy
import json,os,unittest
from unittest.mock import patch
import numpy as np
import gpu_sampling as sampling
from gpu_sampling_resume_round25 import SamplerSession,_SamplingInterrupted
from test_gpu_recombination_round26 import interacting_fixture
from test_gpu_sampling_round25 import FakeSampler
from gpu_incremental_sampling_round26 import canonical_labels

OPT={'optimal_crossover':True}

def semantic(value):
    value=deepcopy(value)
    def walk(node):
        if isinstance(node,dict):
            for key in ('digest','elapsed','sampler_restores','recombination_restores','recombination_model_rebuilds','recombination_source_checks','recombination_component_visits','recombination_pattern_options'):node.pop(key,None)
            for item in node.values():walk(item)
        elif isinstance(node,list):
            for item in node:walk(item)
    walk(value);return value

class OptimizerIntegrationHostTests(unittest.TestCase):
    def test_stop_retains_completed_collection_and_optimizer_work(self):
        problem,parents=interacting_fixture()
        class Parents(FakeSampler):
            top_calls=0
            def top(self):
                self.top_calls+=1
                return np.array(parents),np.array([sampling.score_board(problem,row)['objective'] for row in parents])
        kwargs=dict(replicas=8,seed=2616,backend='cuda',experiments=OPT)
        session=tail=None
        with patch.object(sampling,'_make_sampler',side_effect=Parents):
            try:
                session=SamplerSession(problem,**kwargs);session._ensure_engine()
                def check():
                    if session.crossover.stats['recombination_nodes']>=1:raise _SamplingInterrupted()
                with self.assertRaises(_SamplingInterrupted):session.collect(check=check)
                self.assertTrue(session.crossover_due);self.assertFalse(session.poisoned);self.assertEqual(session.engine.top_calls,1)
                saved=json.loads(json.dumps(session.snapshot()));checks=session.counts['archive_checks'];nodes=session.crossover.stats['recombination_nodes']
                tail=SamplerSession(problem,resume=saved,**kwargs);tail._ensure_engine();tail.collect()
                self.assertEqual(tail.engine.top_calls,0);self.assertEqual(tail.counts['archive_checks'],checks)
                self.assertGreaterEqual(tail.crossover.stats['recombination_nodes'],nodes);self.assertTrue(problem.validate(tail.solution)['valid'])
            finally:
                if session:session.close()
                if tail:tail.close()
    def test_legacy_solve_forwards_optimal(self):
        problem,_=interacting_fixture()
        with patch.object(sampling,'sample',return_value={'top_samples':[]}) as sample,patch('constraint_dfs.solve',return_value={'status':'timeout','complete':False,'stats':{},'nodes':0,'elapsed_seconds':0.}):
            sampling.solve(problem,seconds=1,repair_fraction=0,options=OPT)
        self.assertTrue(sample.call_args.kwargs['experiments']['optimal_crossover'])

class _Hardware:
    def args(self):return dict(replicas=8,seed=2616,backend=self.backend,gold_aware=True,experiments={**OPT,'incremental_sampling':True,'move_tabu':True,'replica_exchange':True,'adaptive_temperatures':True},allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
    def test_actual_prepared_parents_find_checked_gold_without_gpu_moves(self):
        problem,parents=interacting_fixture();session=SamplerSession(problem,**self.args())
        try:
            session._ensure_engine();arrays=session.engine.snapshot_arrays()
            for lane in range(8):
                row=list(parents[lane%2]);arrays[0][:,lane]=row;arrays[1][:,lane]=row
                objective=sampling.gold_score(problem,row)['gold_objective'];arrays[2][1:3,lane]=objective
                arrays[2][45:,lane]=canonical_labels(problem,row)
            session.engine.restore_arrays(arrays);session.collect()
            self.assertTrue(problem.validate(session.solution)['valid']);self.assertEqual(session.counts['move_proposals'],0)
            self.assertGreater(session.crossover.stats['recombination_nodes'],0)
        finally:session.close()
    def test_full160_combined_native_and_optimizer_progress_resume(self):
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        problem=SearchProblem(*planted_gold(2616,size=4,extra_edges=12)[:2]);kwargs=self.args()
        full=SamplerSession(problem,**kwargs);prefix=SamplerSession(problem,**kwargs);tail=None
        try:
            full.advance(30,max_dispatches=18);prefix.advance(30,max_dispatches=9);saved=json.loads(json.dumps(prefix.snapshot()));prefix.close()
            tail=SamplerSession(problem,resume=saved,**kwargs);tail.advance(30,max_dispatches=9)
            expected=full.snapshot();actual=tail.snapshot();self.assertEqual(expected['native'],actual['native'])
            self.assertEqual(semantic(expected),semantic(actual));self.assertEqual(full.launches,18)
            self.assertGreaterEqual(actual['crossover']['state']['stats']['recombination_component_visits'],expected['crossover']['state']['stats']['recombination_component_visits'])
            state=tail.engine.snapshot_arrays()
            for lane in range(8):np.testing.assert_array_equal(state[2][45:,lane],canonical_labels(problem,state[0][:,lane]))
        finally:full.close();prefix.close();tail and tail.close()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_GPU_SAMPLING')=='1','CUDA checks disabled')
class CUDARecombination(_Hardware,unittest.TestCase):backend='cuda'
@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','WebGPU checks disabled')
class WebGPURecombination(_Hardware,unittest.TestCase):backend='webgpu'
