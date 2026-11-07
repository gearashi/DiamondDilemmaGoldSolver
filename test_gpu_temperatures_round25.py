"""Adaptive exchange native state, scheduling, and failure boundaries."""
import json,os,unittest
from unittest.mock import Mock,patch
import numpy as np
import gpu_sampling as sampling
from gpu_sampling_resume_round25 import SamplerSession
from gpu_sampling_state_round25 import encode,restore
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,multiple_loops,double_face

ADAPT={'adaptive_temperatures':True}

class TemperatureHostTests(unittest.TestCase):
    def test_flags_and_small_replica_rejection_precede_backend(self):
        self.assertEqual(sampling.sampling_flags(ADAPT),36)
        p=SearchProblem(*tetrahedron_tiles())
        with patch('gpu_backends.choose_backend',side_effect=AssertionError('no GPU')):
            with self.assertRaisesRegex(ValueError,'eight replicas'):SamplerSession(p,replicas=7,experiments=ADAPT)
            with self.assertRaisesRegex(ValueError,'eight replicas'):sampling.sample(p,seconds=1,replicas=7,experiments=ADAPT)
    def test_feedback_uint32_aggregation_and_values(self):
        rows=np.zeros((2,16),np.uint32);rows[0,0]=2**32-1;rows[0,8]=2;rows[1,0]=1
        temps=np.array([.2*1.5**i for i in range(8)],np.float32)
        got=sampling._temperature_feedback(rows,temps.view(np.uint32))
        self.assertEqual(got['attempts'][0],1);self.assertEqual(got['accepts'][0],1)
        for bad in ([1]*8,[float('nan')]*8,[True]*8,[.2]*7,[-1,*list(map(float,temps))[1:]]):
            with self.assertRaises(ValueError):sampling._temperature_values(bad)
    def test_feedback_or_device_failure_poison_and_freeze_has_no_reads(self):
        engine=Mock();engine._poisoned=False;controller=Mock();controller.frozen=False;controller.observe.return_value=[.2*1.5**i for i in range(8)]
        engine.set_temperatures.side_effect=RuntimeError('partial write')
        with self.assertRaises(RuntimeError):sampling._temperature_step(controller,engine,16)
        self.assertTrue(engine._poisoned)
        engine.reset_mock();controller.frozen=True;sampling._temperature_step(controller,engine,32);engine.temperature_feedback.assert_not_called()
    def test_legacy_live_progress_exposes_controller_counters_without_crossover(self):
        from test_gpu_sampling_round25 import FakeSampler
        clock=[0.];events=[]
        class Timed(FakeSampler):
            def launch(self,steps):super().launch(steps);clock[0]+=.1
        p=SearchProblem(multiple_loops(),double_face())
        with patch.object(sampling,'_make_sampler',side_effect=Timed),patch.object(sampling,'choose_backend',return_value='cuda'),patch.object(sampling.time,'monotonic',side_effect=lambda:clock[0]):
            sampling.sample(p,seconds=1.2,replicas=8,backend='cuda',experiments=ADAPT,progress=events.append)
        live=[e for e in events if e['event']=='sampling_progress'];self.assertTrue(live)
        self.assertEqual(live[0]['sampling_experiments']['temperature_feedback_windows'],0)
    def test_legacy_and_durable_hybrid_forward_the_policy(self):
        p=SearchProblem(multiple_loops(),double_face())
        with patch.object(sampling,'sample',return_value={'top_samples':[]}) as sample,patch('constraint_dfs.solve',return_value={'status':'timeout','complete':False,'stats':{},'nodes':0,'elapsed_seconds':0.}):
            sampling.solve(p,seconds=1,repair_fraction=0,options=ADAPT)
        self.assertEqual(sample.call_args.kwargs['experiments'],ADAPT)
        saved=sampling.solve(p,seconds=1,stop=lambda:True,options={'durable_hybrid':True,**ADAPT})['checkpoint']
        self.assertTrue(saved['sampling_policy']['experiments']['adaptive_temperatures'])

class _Hardware:
    def engine(self,p,policy,r=16):return sampling._make_sampler(p,*sampling._inputs(p,gold_aware=True),r,2537,self.backend,experiments=policy,
        allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
    def test_unchanged_initial_ladder_preserves_pre_adjustment_trajectory(self):
        p=SearchProblem(*tetrahedron_tiles())
        for policy in ({'replica_exchange':True},{'replica_exchange':True,'move_tabu':True,'obstruction_chains':True,'lazy_gold_scoring':True}):
            a=self.engine(p,policy);b=self.engine(p,{**policy,**ADAPT})
            try:
                self.assertEqual(b.meta_rows,29)
                for launch in range(1,17):
                    a.launch(7);b.launch(7)
                    if launch%8==0:a.exchange((launch//8)%2);b.exchange((launch//8)%2)
                    old,new=a.snapshot_arrays(),b.snapshot_arrays()
                    for x,y in zip(old[:2],new[:2]):np.testing.assert_array_equal(x,y)
                    np.testing.assert_array_equal(old[2],new[2][:a.meta_rows])
                self.assertEqual(a.experiment_stats(),b.experiment_stats())
                feedback=b.temperature_feedback();self.assertEqual(sum(feedback['attempts']),14)
                self.assertTrue(all(x<=y for x,y in zip(feedback['accepts'],feedback['attempts'])))
            finally:a.close();b.close()
    def test_ladder_update_only_changes_temperature_row_and_bad_restore_rejected(self):
        p=SearchProblem(*tetrahedron_tiles());engine=self.engine(p,{**ADAPT,'lazy_gold_scoring':True});target=None
        try:
            before=engine.snapshot_arrays();values=engine.temperature_feedback()['temperatures'];changed=[float(np.float32(x**1.02)) for x in values]
            engine.set_temperatures(changed);after=engine.snapshot_arrays()
            for x,y in zip(before[:2],after[:2]):np.testing.assert_array_equal(x,y)
            np.testing.assert_array_equal(before[2][:28],after[2][:28]);self.assertEqual(engine.temperature_feedback()['temperatures'],changed)
            stable=engine.snapshot_arrays()
            with self.assertRaises(ValueError):engine.set_temperatures([1.]*8)
            for x,y in zip(stable,engine.snapshot_arrays()):np.testing.assert_array_equal(x,y)
            after[2][28,8]=after[2][28,9];engine.restore_arrays(after)
            table=sampling._inputs(p,gold_aware=True)[0];bad=encode(p,table,engine);target=self.engine(p,{**ADAPT,'lazy_gold_scoring':True})
            with self.assertRaisesRegex(ValueError,'per-group'):restore(p,table,target,bad)
        finally:
            engine.close()
            if target:target.close()
    def test_partial_native_ladder_write_poison_prevents_publication(self):
        p=SearchProblem(*tetrahedron_tiles());engine=self.engine(p,ADAPT)
        try:
            values=engine.temperature_feedback()['temperatures']
            if self.backend=='cuda':
                original=engine.meta
                class WriteThenFail:
                    def __setitem__(self,key,value):original[key]=value;raise RuntimeError('injected ladder write failure')
                    def __getattr__(self,key):return getattr(original,key)
                engine.meta=WriteThenFail()
            else:
                original=engine.queue
                class WriteThenFail:
                    def __getattr__(self,key):return getattr(original,key)
                    def write_buffer(self,*args,**kw):original.write_buffer(*args,**kw);raise RuntimeError('injected ladder write failure')
                engine.queue=WriteThenFail()
            with self.assertRaisesRegex(RuntimeError,'ladder write failure'):engine.set_temperatures(values)
            self.assertTrue(engine._poisoned)
            for call in (engine.snapshot_arrays,lambda:engine.launch(1),engine.experiment_stats,engine.temperature_feedback,lambda:engine.set_temperatures(values)):
                with self.assertRaisesRegex(RuntimeError,'fresh sampler'):call()
        finally:engine.close()
    def test_full160_adaptation_stop_resume_binds_native_ladder_and_cadence(self):
        from search_fixtures import planted_gold
        data,shape,_=planted_gold(2537,size=4,extra_edges=12);p=SearchProblem(data,shape)
        kw={'backend':self.backend,'replicas':8,'seed':2537,'gold_aware':True,'experiments':{**ADAPT,'ejection_chains':True,'move_tabu':True,'lazy_gold_scoring':True},
            'allow_software':os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1'}
        full=SamplerSession(p,**kw);prefix=tail=None
        try:
            full.advance(30,max_dispatches=18);expected=full.snapshot();self.assertEqual(full.temperatures.last_dispatch,16)
            prefix=SamplerSession(p,**kw);prefix.advance(30,max_dispatches=15);saved=json.loads(json.dumps(prefix.snapshot()));prefix.close()
            tail=SamplerSession(p,resume=saved,**kw);tail.advance(30,stop=lambda:True);self.assertIsNone(tail.engine)
            tail.advance(30,max_dispatches=3);actual=tail.snapshot()
            self.assertEqual(actual['sampler_session_version'],3);self.assertEqual(expected['native'],actual['native']);self.assertEqual(expected['temperatures'],actual['temperatures'])
            self.assertEqual(full.report()['sampling_experiments']['temperature_feedback_windows'],1)
            boards,best,meta=tail.engine.snapshot_arrays()
            for lane in range(8):
                self.assertEqual(int(meta[1,lane]),sampling.gold_score(p,boards[:,lane])['gold_objective']);self.assertEqual(int(meta[2,lane]),sampling.gold_score(p,best[:,lane])['gold_objective'])
            # A self-consistent native state with a different valid ladder still
            # cannot masquerade as the saved controller's trajectory.
            altered=tail.engine.temperature_feedback()['temperatures'];altered[3]=float(np.float32((altered[2]+altered[3])/2));tail.engine.set_temperatures(altered)
            mismatch=tail.snapshot();fresh=SamplerSession(p,resume=mismatch,**kw)
            try:
                with self.assertRaisesRegex(ValueError,'ladder differs'):fresh.advance(1,max_dispatches=1)
            finally:fresh.close()
        finally:
            full.close()
            if prefix:prefix.close()
            if tail:tail.close()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','CUDA finite checks require explicit opt-in')
class CUDATemperatures(_Hardware,unittest.TestCase):backend='cuda'
@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','WebGPU finite checks require explicit opt-in')
class WebGPUTemperatures(_Hardware,unittest.TestCase):backend='webgpu'
if __name__=='__main__':unittest.main()
