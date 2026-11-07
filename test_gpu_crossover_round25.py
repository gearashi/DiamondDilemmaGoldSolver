"""Native crossover lane injection invariants; host policy checks follow separately."""
import itertools,json,os,unittest
import numpy as np
import gpu_sampling as sampling
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles
from unittest.mock import Mock,patch
from gpu_sampling_resume_round25 import SamplerSession

class CrossoverCollectionTests(unittest.TestCase):
    def setUp(self):
        self.problem=SearchProblem(*tetrahedron_tiles());self.codes=[0,4,6,9]
        self.metrics=sampling.gold_score(self.problem,self.codes);self.controller=Mock();self.engine=Mock()
        self.controller.propose.return_value={'candidate':self.codes,'metrics':self.metrics,'solution':None,'stats':{}}
        self.engine.inject_candidate.return_value={'injected':True}
    def test_checked_child_is_injected_once_and_added_to_archive(self):
        archive,solution=sampling._crossover_step(self.problem,self.controller,[],self.engine,gold_aware=True)
        self.assertIsNone(solution);self.assertEqual(archive[0]['codes'],self.codes)
        self.engine.inject_candidate.assert_called_once_with(self.problem,self.codes);self.controller.record_injection.assert_called_once()
    def test_gold_child_is_returned_without_gpu_injection(self):
        goal=[0,3,6,9];self.controller.propose.return_value={'candidate':goal,'metrics':sampling.gold_score(self.problem,goal),'solution':goal,'stats':{}}
        archive,solution=sampling._crossover_step(self.problem,self.controller,[],self.engine,gold_aware=True)
        self.assertEqual(solution,goal);self.assertEqual(archive[0]['codes'],goal);self.engine.inject_candidate.assert_not_called()
    def test_finishing_phase_keeps_hint_without_mutating_gpu(self):
        archive,solution=sampling._crossover_step(self.problem,self.controller,[],self.engine,gold_aware=True,inject=False)
        self.assertEqual(archive[0]['codes'],self.codes);self.engine.inject_candidate.assert_not_called();self.controller.record_injection.assert_not_called()
    def test_invalid_type_score_or_gold_claim_rejected_before_write(self):
        for change in ({'candidate':[float(x) for x in self.codes]},{'metrics':{**self.metrics,'gold_objective':999}},{'solution':self.codes}):
            self.controller.propose.return_value={'candidate':self.codes,'metrics':self.metrics,'solution':None,'stats':{},**change}
            with self.assertRaises((ValueError,RuntimeError)):sampling._crossover_step(self.problem,self.controller,[],self.engine,gold_aware=True)
        self.engine.inject_candidate.assert_not_called()
    def test_domain_illegal_child_is_never_injected(self):
        self.problem.domains[1]=[3]
        self.controller.propose.return_value={'candidate':self.codes,'metrics':sampling.gold_score(self.problem,self.codes),'solution':None,'stats':{}}
        with self.assertRaisesRegex(RuntimeError,'domain/score'):sampling._crossover_step(self.problem,self.controller,[],self.engine,gold_aware=True)
        self.engine.inject_candidate.assert_not_called()

def install_population(session,parents):
    session._ensure_engine();arrays=session.engine.snapshot_arrays()
    for lane in range(session.config['replicas']):
        codes=parents[lane%len(parents)];arrays[0][:,lane]=codes;arrays[1][:,lane]=codes
        metrics=sampling.gold_score(session.problem,codes) if session.config['gold_aware'] else sampling.score_board(session.problem,codes)
        arrays[2][1:3,lane]=metrics['gold_objective' if session.config['gold_aware'] else 'objective']
    session.engine.restore_arrays(arrays)

class CrossoverSessionHostTests(unittest.TestCase):
    def test_failed_collection_cannot_publish_or_continue_partial_state(self):
        from test_gpu_sampling_round25 import FakeSampler
        p=SearchProblem(*tetrahedron_tiles());session=SamplerSession(p,backend='cuda',replicas=4,experiments={'cycle_crossover':True})
        with patch.object(sampling,'_make_sampler',side_effect=FakeSampler),patch('gpu_backends.choose_backend',return_value='cuda'):
            install_population(session,[[1,4,6,9],[0,3,7,10]])
            session.engine.inject_candidate=Mock(side_effect=RuntimeError('partial injection'))
            with self.assertRaisesRegex(RuntimeError,'partial injection'):session.collect()
            for call in (session.snapshot,lambda:session.advance(1),session.finish_phase,session.report):
                with self.assertRaisesRegex(RuntimeError,'fresh session'):call()
            session.close();self.assertTrue(session.closed)
    def test_legacy_sample_gold_child_stops_before_any_gpu_move_when_collected(self):
        from test_gpu_sampling_round25 import FakeSampler
        from test_gpu_crossover_reference_round25 import complementary_fixture
        p,parents,goal=complementary_fixture()
        class Prepared(FakeSampler):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                for lane in range(self.r):self.arrays[0][:,lane]=parents[lane%2]['codes'];self.arrays[1][:,lane]=parents[lane%2]['codes']
            def inject_candidate(self,*args):raise AssertionError('Gold child need not be injected')
            def launch(self,steps):self.launches+=1 # Keep stipulated parent archive for the collection discriminator.
        # Legacy final collection actually invokes crossover even without durable_hybrid.
        ticks=iter([0.,0.,0.,2.,2.,2.,2.,2.])
        with patch.object(sampling,'_make_sampler',side_effect=Prepared),patch.object(sampling,'choose_backend',return_value='cuda'),patch.object(sampling.time,'monotonic',side_effect=lambda:next(ticks,2.)):
            result=sampling.sample(p,seconds=1,replicas=4,backend='cuda',experiments={'cycle_crossover':True})
        self.assertEqual(result['solution'],goal);self.assertEqual(result['move_proposals'],0)
        self.assertEqual(result['sampling_experiments']['cycle_gold_solutions'],1)

class _Hardware:
    def engine(self,p,flags):return sampling._make_sampler(p,*sampling._inputs(p,gold_aware=True),4,2535,self.backend,experiments=flags,
        allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
    def test_injection_preserves_other_lanes_better_best_rng_and_work_counters(self):
        p=SearchProblem(*tetrahedron_tiles());ranked={}
        for rotation in itertools.product(range(3),repeat=4):
            board=[3*i+r for i,r in enumerate(rotation)];ranked.setdefault(sampling.gold_score(p,board)['gold_objective'],board)
        levels=sorted(ranked);low=ranked[levels[0]];middle=ranked[levels[len(levels)//2]];goal=[0,3,6,9]
        self.assertLess(sampling.gold_score(p,middle)['gold_objective'],sampling.gold_score(p,goal)['gold_objective'])
        for flags in ({'cycle_crossover':True},{'cycle_crossover':True,'move_tabu':True},{'cycle_crossover':True,'obstruction_chains':True,'move_tabu':True,'replica_exchange':True}):
            engine=self.engine(p,flags)
            try:
                arrays=engine.snapshot_arrays();arrays[0][:]=np.array(low,np.uint32)[:,None];arrays[1][:]=arrays[0];arrays[1][:,0]=goal
                arrays[2][1]=sampling.gold_score(p,low)['gold_objective'];arrays[2][2]=arrays[2][1];arrays[2][2,0]=sampling.gold_score(p,goal)['gold_objective'];arrays[2][3]=np.arange(4)+23
                if engine.meta_rows>4:arrays[2][4:13]=17;arrays[2][13:]=31
                engine.restore_arrays(arrays);before=engine.snapshot_arrays();outcome=engine.inject_candidate(p,middle);after=engine.snapshot_arrays()
                self.assertTrue(outcome['injected']);self.assertEqual(outcome['lane'],0);self.assertFalse(outcome['best_updated'])
                self.assertEqual(after[0][:,0].tolist(),middle);np.testing.assert_array_equal(after[1],before[1])
                for a,b in zip(after,before):np.testing.assert_array_equal(a[:,1:],b[:,1:])
                np.testing.assert_array_equal(after[2][[0,3]],before[2][[0,3]])
                if engine.meta_rows>4:
                    self.assertFalse(after[2][4:13,0].any());np.testing.assert_array_equal(after[2][13:],before[2][13:])
                ties=engine.snapshot_arrays();ties[0][:]=np.array(middle,np.uint32)[:,None];ties[1][:]=np.array(goal,np.uint32)[:,None]
                ties[2][1]=sampling.gold_score(p,middle)['gold_objective'];ties[2][2]=sampling.gold_score(p,goal)['gold_objective'];engine.restore_arrays(ties)
                tie=engine.inject_candidate(p,middle);self.assertTrue(tie['injected']);self.assertEqual(tie['lane'],0);self.assertFalse(tie['best_updated'])
                engine.launch(7);current,best,meta=engine.snapshot_arrays()
                for lane in range(4):
                    self.assertEqual(int(meta[1,lane]),sampling.gold_score(p,current[:,lane])['gold_objective'])
                    self.assertEqual(int(meta[2,lane]),sampling.gold_score(p,best[:,lane])['gold_objective'])
            finally:engine.close()
    def test_partial_device_injection_poison_prevents_snapshot_and_dispatch(self):
        p=SearchProblem(*tetrahedron_tiles());engine=self.engine(p,{'cycle_crossover':True,'move_tabu':True})
        try:
            if self.backend=='cuda':
                original=engine.boards
                class WriteThenFail:
                    def __setitem__(self,key,value):original[key]=value;raise RuntimeError('injected device write failure')
                    def __getattr__(self,key):return getattr(original,key)
                engine.boards=WriteThenFail()
            else:
                original=engine.queue
                class WriteThenFail:
                    calls=0
                    def __getattr__(self,key):return getattr(original,key)
                    def write_buffer(self,*args,**kw):
                        original.write_buffer(*args,**kw);self.calls+=1
                        if self.calls==1:raise RuntimeError('injected device write failure')
                engine.queue=WriteThenFail()
            with self.assertRaisesRegex(RuntimeError,'device write failure'):engine.inject_candidate(p,[0,3,6,9])
            self.assertTrue(engine._poisoned)
            for call in (engine.snapshot_arrays,lambda:engine.launch(1),lambda:engine.inject_candidate(p,[0,3,6,9]),engine.experiment_stats):
                with self.assertRaisesRegex(RuntimeError,'fresh sampler'):call()
        finally:engine.close()
    def test_actual_collector_injection_checkpoint_does_not_replay(self):
        p=SearchProblem(*tetrahedron_tiles());kw={'backend':self.backend,'replicas':4,'seed':2535,'gold_aware':True,
            'experiments':{'cycle_crossover':True,'move_tabu':True,'obstruction_chains':True},'allow_software':os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1'}
        session=SamplerSession(p,**kw);restored=None
        try:
            install_population(session,[[1,4,6,9],[0,3,7,10]]);session.collect()
            self.assertFalse(session.done);self.assertEqual(session.crossover.stats['cycle_injections'],1)
            saved=json.loads(json.dumps(session.snapshot()));self.assertEqual(saved['sampler_session_version'],2)
            restored=SamplerSession(p,resume=saved,**kw);restored.advance(30,stop=lambda:True)
            self.assertIsNone(restored.engine);self.assertEqual(restored.crossover.snapshot(),session.crossover.snapshot())
            session.advance(30,max_dispatches=1);restored.advance(30,max_dispatches=1)
            actual=restored.snapshot();expected=session.snapshot()
            self.assertEqual(actual['native'],expected['native']);self.assertEqual(actual['crossover'],expected['crossover']);self.assertEqual(actual['archive'],expected['archive'])
            self.assertEqual(actual['crossover']['state']['stats']['cycle_injections'],1)
        finally:
            session.close()
            if restored:restored.close()
    def test_actual_full160_archive_child_gold_without_injection(self):
        from search_fixtures import planted_gold
        data,geometry,goal=planted_gold(2535,size=4,extra_edges=12);p=SearchProblem(data,geometry)
        a=goal.copy();a[0],a[17]=a[17],a[0];b=goal.copy();b[93],b[100]=b[100],b[93]
        self.assertFalse(p.validate(a)['valid']);self.assertFalse(p.validate(b)['valid'])
        session=SamplerSession(p,backend=self.backend,replicas=4,seed=2535,gold_aware=True,experiments={'cycle_crossover':True},allow_software=os.environ.get('DIAMOND_TEST_SOFTWARE_GPU')=='1')
        try:
            install_population(session,[a,b]);session.collect();self.assertEqual(session.solution,goal);self.assertTrue(session.done)
            self.assertEqual(session.crossover.stats['cycle_injections'],0);self.assertEqual(session.counts['move_proposals'],0)
            self.assertTrue(p.validate(session.solution)['valid'])
        finally:session.close()
    def test_invalid_or_worse_injection_does_not_mutate_native_state(self):
        p=SearchProblem(*tetrahedron_tiles());goal=[0,3,6,9];engine=self.engine(p,{'cycle_crossover':True})
        try:
            arrays=engine.snapshot_arrays();arrays[0][:]=np.array(goal,np.uint32)[:,None];arrays[1][:]=arrays[0]
            arrays[2][1:3]=sampling.gold_score(p,goal)['gold_objective'];engine.restore_arrays(arrays);before=engine.snapshot_arrays()
            self.assertFalse(engine.inject_candidate(p,[0,4,6,9])['injected'])
            with self.assertRaises(ValueError):engine.inject_candidate(p,[0,0,0,0])
            p.domains[0]=[0]
            with self.assertRaisesRegex(ValueError,'scope changed'):engine.inject_candidate(p,goal)
            for a,b in zip(engine.snapshot_arrays(),before):np.testing.assert_array_equal(a,b)
        finally:engine.close()

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA hardware')
class CUDACrossover(_Hardware,unittest.TestCase):backend='cuda'
@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','Optional native or explicit CI software WebGPU')
class WebGPUCrossover(_Hardware,unittest.TestCase):backend='webgpu'
if __name__=='__main__':unittest.main()
