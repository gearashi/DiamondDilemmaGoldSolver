"""Exact incremental sampler scores/cache across both native backends."""
import itertools,os,unittest
from unittest.mock import patch
import numpy as np
import gpu_sampling as sampling
from search_problem import SearchProblem
from gpu_incremental_sampling_round26 import canonical_labels,state_size,COUNTERS,CACHE_BASE
from gpu_sampling_state_round25 import encode,restore
from gpu_sampling_resume_round25 import SamplerSession
from test_constraint_dfs import tetrahedron_tiles,double_face,multiple_loops,board

class IncrementalHostTests(unittest.TestCase):
    def test_layout_cap_and_flags(self):
        self.assertEqual(sampling.sampling_flags({'incremental_sampling':True}),65)
        size=state_size(160,131072,365,65)
        self.assertEqual(size['meta_rows'],228)
        self.assertEqual(size['snapshot_bytes'],287309824)
        self.assertLess(size['snapshot_replica_cap'],131072)
    def test_limit_rejects_before_device_selection(self):
        from search_fixtures import planted_gold
        data,shape,_=planted_gold(2615,size=4,extra_edges=12);p=SearchProblem(data,shape)
        # Dense valid inventory raises the cache payload independently of VRAM.
        from types import SimpleNamespace
        heavy=SimpleNamespace(n=160,total_segments=2560,domains=[[0]]*160,fixed={})
        from gpu_incremental_sampling_round26 import validate_capacity
        with self.assertRaisesRegex(ValueError,'snapshot bound'):validate_capacity(160,131072,2560,65)
        with patch('gpu_backends.choose_backend',side_effect=AssertionError('device selection')):
            with self.assertRaisesRegex(ValueError,'snapshot bound'):sampling.sample(heavy,seconds=1,replicas=131072,gold_aware=True,experiments={'incremental_sampling':True})

    def test_independent_labels_count_raw_closed_components(self):
        p=SearchProblem(*tetrahedron_tiles());rng=np.random.default_rng(2615)
        for _ in range(100):
            codes=3*rng.permutation(p.n)+rng.integers(0,3,p.n);packed=canonical_labels(p,codes)
            labels=[(int(packed[i//2])>>(16*(i%2)))&65535 for i in range(p.total_segments)]
            self.assertEqual(len(set(labels)-{0}),sampling.gold_score(p,codes)['closed_proper_loops'])
    def test_durable_and_legacy_forwarding(self):
        p=SearchProblem(*tetrahedron_tiles())
        with patch.object(sampling,'sample',return_value={'top_samples':[]}) as sample,patch('constraint_dfs.solve',return_value={'status':'timeout','complete':False,'stats':{},'nodes':0,'elapsed_seconds':0.}):
            sampling.solve(p,seconds=1,repair_fraction=0,options={'incremental_sampling':True})
        self.assertTrue(sample.call_args.kwargs['experiments']['incremental_sampling'])
        value=sampling.solve(p,seconds=1,stop=lambda:True,options={'durable_hybrid':True,'incremental_sampling':True})
        self.assertTrue(value['checkpoint']['sampling_policy']['experiments']['incremental_sampling'])

class _Hardware:
    def engine(self,p,policy,gold=True,r=8,initialize=True):
        return sampling._make_sampler(p,*sampling._inputs(p,gold_aware=gold),r,2615,self.backend,experiments=policy,initialize=initialize,
            allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
    def check(self,p,engine,gold=True):
        arrays=engine.snapshot_arrays();scorer=sampling.gold_score if gold else sampling.score_board;key='gold_objective' if gold else 'objective'
        for lane in range(engine.r):
            for index,row in ((0,1),(1,2)):self.assertEqual(int(arrays[2][row,lane]),scorer(p,arrays[index][:,lane])[key])
            if gold and engine.flags&64:np.testing.assert_array_equal(arrays[2][CACHE_BASE:,lane],canonical_labels(p,arrays[0][:,lane]))
        return arrays
    def equal(self,a,b):
        x=a.snapshot_arrays();y=b.snapshot_arrays()
        for old,new in zip(x,y):np.testing.assert_array_equal(old,new[:len(old)])
    def test_private_measurement_preserves_generator_and_charges_initialization(self):
        p=SearchProblem(*tetrahedron_tiles());policy={'ejection_chains':True}
        a=self.engine(p,policy);b=sampling._make_sampler(p,*sampling._inputs(p,True),8,2615,self.backend,experiments=policy,_measure_scoring=True,allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
        try:
            a.launch(13);b.launch(13);self.equal(a,b)
            stats=b.experiment_stats();self.assertEqual(stats['incremental_domain_terms'],27*p.n*8)
            self.assertEqual(stats['incremental_seam_terms'],27*len(p.edges)*8)
        finally:a.close();b.close()

    def test_all_policy_trajectories_and_component_transitions(self):
        p=SearchProblem(*tetrahedron_tiles());accepted=rejected=0
        for bits in itertools.product((False,True),repeat=4):
            policy={'ejection_chains':True,**{key:True for key,value in zip(('move_tabu','replica_exchange','obstruction_chains','lazy_gold_scoring'),bits) if value}}
            for gold in (False,True):
                with self.subTest(policy=policy,gold=gold):
                    a=self.engine(p,policy,gold);b=self.engine(p,{**policy,'incremental_sampling':True},gold)
                    try:
                        self.equal(a,b)
                        for engine in (a,b):
                            value=engine.snapshot_arrays();value[2][3]=4094;engine.restore_arrays(value)
                        for tick in range(16):
                            a.launch(1 if tick<8 else 5);b.launch(1 if tick<8 else 5)
                            if tick%4==3:a.exchange((tick//4)%2);b.exchange((tick//4)%2)
                            self.equal(a,b);self.check(p,b,gold)
                        stats=b.experiment_stats();accepted+=stats['accepted_moves'];rejected+=stats['incremental_proposals']-stats['accepted_moves']
                        self.assertEqual(stats['incremental_proposals'],48*b.r)
                    finally:a.close();b.close()
        self.assertGreater(accepted,0);self.assertGreater(rejected,0)
    def test_fixed_empty_and_multiple_loop_cases(self):
        cases=[SearchProblem(multiple_loops(),double_face()),SearchProblem(*tetrahedron_tiles(),fixed={0:0}),
               SearchProblem(*tetrahedron_tiles(),fixed={i:3*i for i in range(4)}),
               SearchProblem([{'id':'one','segments':[[[0,2],[1,4]]]}],board(1,[]))]
        for p in cases:
            a=self.engine(p,{'ejection_chains':True});b=self.engine(p,{'incremental_sampling':True})
            try:
                for _ in range(12):a.launch(3);b.launch(3);self.equal(a,b);self.check(p,b)
            finally:a.close();b.close()
    def test_native_restore_validation_injection_exchange_full160(self):
        from search_fixtures import planted_gold
        tiles,shape,witness=planted_gold(2615,size=4,extra_edges=12);p=SearchProblem(tiles,shape)
        policy={'ejection_chains':True,'obstruction_chains':True,'replica_exchange':True,'move_tabu':True,'lazy_gold_scoring':True,'adaptive_temperatures':True}
        a=self.engine(p,policy);b=self.engine(p,{**policy,'incremental_sampling':True});tail=None
        try:
            for _ in range(3):a.launch(11);b.launch(11);self.equal(a,b)
            a.set_temperatures([.2,.25,.4,.6,.8,1.,2.,3.]);b.set_temperatures([.2,.25,.4,.6,.8,1.,2.,3.])
            self.assertEqual(a.inject_candidate(p,witness),b.inject_candidate(p,witness));a.exchange(0);b.exchange(0)
            self.equal(a,b);self.check(p,b)
            value=encode(p,sampling._inputs(p,True)[0],b);tail=self.engine(p,{**policy,'incremental_sampling':True},initialize=False)
            restore(p,sampling._inputs(p,True)[0],tail,value)
            for _ in range(8):b.launch(2);tail.launch(2);self.equal(b,tail)
            self.check(p,tail)
            saved=tail.snapshot_arrays();bad=[x.copy() for x in saved];bad[2][CACHE_BASE,0]^=1
            with self.assertRaisesRegex(ValueError,'cache labels'):tail.restore_arrays(bad)
            for x,y in zip(saved,tail.snapshot_arrays()):np.testing.assert_array_equal(x,y)
            bad=[x.copy() for x in saved];bad[2][1,0]+=1
            with self.assertRaisesRegex(ValueError,'objective'):tail.restore_arrays(bad)
            self.assertGreater(b.experiment_stats()['incremental_full_fallbacks'],0)
        finally:a.close();b.close();tail and tail.close()
    def test_durable_session_stop_resume_and_counter_carry(self):
        from search_fixtures import planted_gold
        p=SearchProblem(*planted_gold(2615,size=4,extra_edges=12)[:2]);policy={'incremental_sampling':True,'move_tabu':True,'replica_exchange':True,'adaptive_temperatures':True}
        args=dict(replicas=8,seed=2615,backend=self.backend,gold_aware=True,experiments=policy,allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
        a=SamplerSession(p,**args);b=SamplerSession(p,**args);c=None
        try:
            a.advance(30,max_dispatches=18);b.advance(30,max_dispatches=9);saved=b.snapshot();b.close()
            c=SamplerSession(p,resume=saved,**args);c.advance(30,max_dispatches=9)
            from benchmark_hybrid_round25 import continuation_signature
            self.assertEqual(continuation_signature(a.snapshot()),continuation_signature(c.snapshot()))
            state=c.engine.snapshot_arrays();state[2][29]=0xfffffffe;state[2][30]=0;c.engine.restore_arrays(state);c.engine.launch(3)
            self.assertEqual(c.engine.experiment_stats()['incremental_proposals'],8*(2**32+1))
        finally:a.close();b.close();c and c.close()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_GPU_SAMPLING')=='1','CUDA finite tests disabled')
class CUDAIncremental(_Hardware,unittest.TestCase):backend='cuda'
@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','WebGPU finite tests disabled')
class WebGPUIncremental(_Hardware,unittest.TestCase):backend='webgpu'
