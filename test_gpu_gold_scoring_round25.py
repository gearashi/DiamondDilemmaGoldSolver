"""Proposal36 native differential trajectories and durable score invariants."""
import itertools,json,os,unittest
from unittest.mock import patch
import numpy as np
import gpu_sampling as sampling
from gpu_sampling_resume_round25 import SamplerSession
from gpu_sampling_state_round25 import encode,restore
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,double_face,multiple_loops

LAZY={'lazy_gold_scoring':True}

def policies():
    return [{'ejection_chains':True,**{k:True for k,v in zip(('move_tabu','replica_exchange','obstruction_chains'),bits) if v}} for bits in itertools.product((False,True),repeat=3)]

def legacy_equal(test,old,new):
    np.testing.assert_array_equal(old[0],new[0]);np.testing.assert_array_equal(old[1],new[1])
    np.testing.assert_array_equal(old[2],new[2][:old[2].shape[0]])

class GoldScoringHostTests(unittest.TestCase):
    def test_flag_and_legacy_forwarding(self):
        self.assertEqual(sampling.sampling_flags(LAZY),17)
        p=SearchProblem(multiple_loops(),double_face())
        with patch.object(sampling,'sample',return_value={'top_samples':[]}) as sample,patch('constraint_dfs.solve',return_value={'status':'timeout','complete':False,'stats':{},'nodes':0,'elapsed_seconds':0.}):
            sampling.solve(p,seconds=1,repair_fraction=0,options=LAZY)
        self.assertEqual(sample.call_args.kwargs['experiments'],LAZY)
    def test_durable_policy_binds_lazy_with_no_initial_allocation(self):
        p=SearchProblem(multiple_loops(),double_face())
        result=sampling.solve(p,seconds=1,stop=lambda:True,options={'durable_hybrid':True,**LAZY})
        self.assertTrue(result['checkpoint']['sampling_policy']['experiments']['lazy_gold_scoring'])
        with self.assertRaisesRegex(ValueError,'saved replicas/seed/policy'):
            sampling.solve(p,seconds=1,stop=lambda:True,resume=result['checkpoint'],options={'durable_hybrid':True})

class _Hardware:
    def engine(self,p,policy,gold=True,r=16):
        return sampling._make_sampler(p,*sampling._inputs(p,gold_aware=gold),r,2536,self.backend,experiments=policy,
            allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
    def check_scores(self,p,engine,gold=True):
        arrays=engine.snapshot_arrays();scorer=sampling.gold_score if gold else sampling.score_board;key='gold_objective' if gold else 'objective'
        for lane in range(engine.r):
            self.assertEqual(int(arrays[2][1,lane]),scorer(p,arrays[0][:,lane])[key]);self.assertEqual(int(arrays[2][2,lane]),scorer(p,arrays[1][:,lane])[key])
        return arrays
    def test_same_trajectory_all_policy_combinations_and_temperature_boundary(self):
        p=SearchProblem(*tetrahedron_tiles())
        totals={k:0 for k in sampling.SAMPLING_COUNTERS[9:]}
        for policy in policies():
            for gold in (False,True):
                with self.subTest(policy=policy,gold=gold):
                    a=self.engine(p,policy,gold);b=self.engine(p,{**policy,**LAZY},gold)
                    try:
                        self.assertEqual(b.meta_rows,28);legacy_equal(self,a.snapshot_arrays(),b.snapshot_arrays())
                        for engine in (a,b):
                            state=engine.snapshot_arrays();state[2][3]=4094;engine.restore_arrays(state)
                        for tick in range(36):
                            steps=1 if tick<8 else 7
                            a.launch(steps);b.launch(steps)
                            if tick%8==7:a.exchange(tick//8%2);b.exchange(tick//8%2)
                            legacy_equal(self,a.snapshot_arrays(),b.snapshot_arrays())
                        self.check_scores(p,b,gold);stats=b.experiment_stats()
                        self.assertEqual(stats['gold_old_base_scans_saved'],(8+28*7)*b.r)
                        if not gold:
                            self.assertEqual(sum(stats[k] for k in sampling.SAMPLING_COUNTERS[10:14]),0)
                        else:
                            self.assertEqual(stats['gold_old_loop_traces_saved'],stats['gold_old_base_scans_saved'])
                            self.assertEqual(stats['gold_candidate_loop_traces']+stats['gold_tabu_traces_skipped']+stats['gold_metropolis_traces_skipped'],stats['gold_old_loop_traces_saved'])
                        for k in totals:totals[k]+=stats[k]
                    finally:a.close();b.close()
        self.assertGreater(totals['gold_tabu_traces_skipped'],0);self.assertGreater(totals['gold_metropolis_traces_skipped'],0);self.assertGreater(totals['gold_candidate_loop_traces'],0)
    def test_fixed_no_free_and_one_cell_cases(self):
        cases=[SearchProblem(*tetrahedron_tiles(),fixed={0:0}),SearchProblem(*tetrahedron_tiles(),fixed={c:3*c for c in range(4)})]
        from test_constraint_dfs import board
        cases.append(SearchProblem([{'id':'single','segments':[[[0,2],[1,4]]]}],board(1,[])))
        for p in cases:
            for policy in ({'ejection_chains':True},{'obstruction_chains':True,'move_tabu':True,'replica_exchange':True}):
                a=self.engine(p,policy,r=4);b=self.engine(p,{**policy,**LAZY},r=4)
                try:
                    a.launch(19);b.launch(19);legacy_equal(self,self.check_scores(p,a),self.check_scores(p,b))
                    if not b.f:self.assertEqual(b.experiment_stats()['gold_old_base_scans_saved'],0)
                finally:a.close();b.close()
    def test_full160_injection_exchange_and_trusted_native_resume(self):
        from search_fixtures import planted_gold
        data,shape,witness=planted_gold(2536,size=4,extra_edges=12);p=SearchProblem(data,shape)
        policy={'ejection_chains':True,'obstruction_chains':True,'move_tabu':True,'replica_exchange':True}
        a=self.engine(p,policy,r=4);b=self.engine(p,{**policy,**LAZY},r=4);tail=None
        try:
            for i in range(3):a.launch(17);b.launch(17)
            # Checked injection and exchange must preserve the score decomposition.
            child=witness.copy();child[0],child[17]=child[17],child[0]
            self.assertEqual(a.inject_candidate(p,child),b.inject_candidate(p,child));a.exchange(0);b.exchange(0)
            legacy_equal(self,self.check_scores(p,a),self.check_scores(p,b))
            table=sampling._inputs(p,gold_aware=True)[0];saved=json.loads(json.dumps(encode(p,table,b)))
            tail=self.engine(p,{**policy,**LAZY},r=4);restore(p,table,tail,saved)
            for i in range(3):
                a.launch(19);b.launch(19);tail.launch(19)
                a.exchange(i%2);b.exchange(i%2);tail.exchange(i%2)
                legacy_equal(self,a.snapshot_arrays(),b.snapshot_arrays())
                for x,y in zip(b.snapshot_arrays(),tail.snapshot_arrays()):np.testing.assert_array_equal(x,y)
            self.check_scores(p,b)
        finally:
            a.close();b.close()
            if tail:tail.close()
    def test_durable_controller_stop_and_resume_28_rows(self):
        from search_fixtures import planted_gold
        data,shape,_=planted_gold(2536,size=4,extra_edges=12);p=SearchProblem(data,shape)
        kw={'backend':self.backend,'replicas':4,'seed':2536,'gold_aware':True,'experiments':{**LAZY,'obstruction_chains':True,'move_tabu':True,'replica_exchange':True},
            'allow_software':os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1'}
        full=SamplerSession(p,**kw);prefix=tail=None
        try:
            full.advance(30,max_dispatches=4);expected=full.snapshot()
            prefix=SamplerSession(p,**kw);prefix.advance(30,max_dispatches=2);saved=json.loads(json.dumps(prefix.snapshot()));prefix.close()
            tail=SamplerSession(p,resume=saved,**kw);tail.advance(30,stop=lambda:True);self.assertIsNone(tail.engine)
            tail.advance(30,max_dispatches=2);actual=tail.snapshot()
            self.assertEqual(expected['native'],actual['native']);self.assertEqual(expected['experiment_stats'],actual['experiment_stats'])
            self.assertEqual(actual['native']['binding']['meta_rows'],28)
        finally:
            full.close()
            if prefix:prefix.close()
            if tail:tail.close()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','CUDA finite checks require explicit opt-in')
class CUDAGoldScoring(_Hardware,unittest.TestCase):backend='cuda'
@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','WebGPU finite checks require explicit opt-in')
class WebGPUGoldScoring(_Hardware,unittest.TestCase):backend='webgpu'
if __name__=='__main__':unittest.main()
