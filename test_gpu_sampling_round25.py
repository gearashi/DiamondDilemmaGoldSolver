"""Durable hybrid controller regressions; native trajectory checks opt in."""
from copy import deepcopy
from types import SimpleNamespace
import json,os,unittest
from unittest.mock import patch
import numpy as np
from constraint_work import digest
from search_problem import SearchProblem
from test_constraint_dfs import double_face,multiple_loops,tetrahedron_tiles
from gpu_sampling_resume_round25 import SamplerSession
from constraint_lns_resume_round25 import RepairSession
import gpu_sampling as sampling
import gpu_backends as backends
import gpu_hybrid_resume_round25 as hybrid

EXPERIMENTS={'ejection_chains':True,'move_tabu':True,'replica_exchange':True}
def problem():return SearchProblem(multiple_loops(),double_face())
def copied(value):return json.loads(json.dumps(value))
def rehash(value):value['digest']=digest({k:v for k,v in value.items() if k!='digest'});return value

def fake_backend(name='auto'):
    """Host-only selection paired with FakeSampler; it never probes a driver."""
    if name not in ('auto','cuda','webgpu'):raise ValueError('Invalid fake backend.')
    return 'cuda' if name=='auto' else name

class FakeSampler:
    instances=[]
    def __init__(self,p,table,free,r,seed,backend,*,experiments=None,initialize=True,allow_software=False):
        self.problem=p;self.n=p.n;self.r=r;self.f=free;self.flags=sampling.sampling_flags(experiments);self.gold_total=0;self.meta_rows=18 if self.flags else 4
        self.device='finite fake';self.backend=backend;self.closed=False;self.initialized=initialize;self.launches=0
        self.arrays=[np.tile(np.arange(self.n,dtype=np.uint32)[:,None]*3,(1,r)) for _ in range(2)]+[np.zeros((self.meta_rows,r),np.uint32)]
        self.arrays[2][0]=np.arange(r,dtype=np.uint32)+1
        self.instances.append(self)
    def launch(self,steps):
        self.launches+=1;self.arrays[2][3]+=1;self.arrays[0][0]=self.arrays[0][0]//3*3+(self.arrays[0][0]+1)%3;self.arrays[1]=self.arrays[0].copy()
    def exchange(self,parity):self.arrays[2][-1]+=parity+1
    def experiment_stats(self):return {'exchange_attempts':int(self.arrays[2][-1].sum())} if self.flags else {}
    def runtime_identity(self):return {'backend':self.backend,'device':self.device,'runtime':'fake'}
    def snapshot_arrays(self):return [a.copy() for a in self.arrays]
    def restore_arrays(self,arrays):self.arrays=[a.copy() for a in arrays]
    def top(self):
        boards=self.arrays[1].T.copy();return boards,np.array([sampling.score_board(self.problem,b)['objective'] for b in boards])
    def close(self):self.closed=True

class HostSamplerTests(unittest.TestCase):
    def setUp(self):
        FakeSampler.instances=[];self.p=problem()
        self.enterContext(patch.object(sampling,'_make_sampler',side_effect=FakeSampler))
        self.backend_choice=self.enterContext(patch.object(backends,'choose_backend',side_effect=fake_backend))
    def session(self,**kw):return SamplerSession(self.p,replicas=4,backend='cuda',**kw)
    def test_complete_state_and_exchange_cadence_resume(self):
        a=self.session(experiments=EXPERIMENTS);a.advance(20,max_dispatches=18);a.finish_phase();expected=a.snapshot();a.close()
        b=self.session(experiments=EXPERIMENTS);b.advance(20,max_dispatches=9);saved=copied(b.snapshot());b.close()
        c=self.session(experiments=EXPERIMENTS,resume=saved);c.advance(20,max_dispatches=9);c.finish_phase();actual=c.snapshot();c.close()
        self.assertEqual(expected['native'],actual['native']);self.assertEqual(expected['archive'],actual['archive']);self.assertEqual(actual['launches'],18)
        self.assertEqual(actual['counts']['initial_boards'],4);self.assertEqual(actual['counts']['sampler_restores'],1);self.assertFalse(FakeSampler.instances[-1].initialized)
    def test_restored_uncollected_tail_finish_and_closed_guard(self):
        a=self.session();a.advance(20,max_dispatches=7);saved=copied(a.snapshot());a.close()
        b=self.session(resume=saved);self.assertEqual(b.last_collected,-1);b.finish_phase();self.assertEqual(b.last_collected,7);self.assertTrue(b.archive);b.close()
        for call in (b.snapshot,b.finish_phase,lambda:b.advance(1)):
            with self.assertRaisesRegex(RuntimeError,'closed'):call()
    def test_immediate_stop_allocates_nothing(self):
        a=self.session();a.advance(20,stop=lambda:True);self.assertEqual(FakeSampler.instances,[]);self.assertIsNone(a.snapshot()['native']);self.backend_choice.assert_not_called();a.close()
    def test_changed_live_input_cannot_publish_old_scope(self):
        a=self.session();self.p.domains[0]=self.p.domains[0][1:]
        with self.assertRaisesRegex(ValueError,'input/source changed'):a.snapshot()
        with self.assertRaisesRegex(ValueError,'input/source changed'):a.advance(1)
        a.close()
    def test_wrong_replica_policy_and_backend_rejected(self):
        a=self.session();a.advance(20,max_dispatches=1);saved=a.snapshot();a.close()
        for kw,reason in (({'replicas':8,'backend':'cuda'},'policy'),({'replicas':4,'backend':'webgpu'},'saved backend')):
            with self.assertRaisesRegex(ValueError,reason):SamplerSession(self.p,resume=saved,**kw)
    def test_host_fake_on_darwin_never_selects_or_constructs_native_device(self):
        with patch.object(backends,'sys',SimpleNamespace(platform='darwin')),patch.object(backends,'_automatic_backend',side_effect=AssertionError('native auto selection')) as automatic,patch.object(sampling,'_CUDA',side_effect=AssertionError('native CUDA')) as cuda,patch.object(sampling,'_WebGPU',side_effect=AssertionError('native WebGPU')) as webgpu:
            a=self.session()
            try:
                a.advance(20,max_dispatches=1);saved=copied(a.snapshot())
                self.assertEqual(saved['selected'],'cuda');self.assertEqual(saved['launches'],1)
            finally:a.close()
            self.backend_choice.assert_called_once_with('cuda')
            automatic.assert_not_called();cuda.assert_not_called();webgpu.assert_not_called()

    def test_corrupt_native_does_not_write_device(self):
        a=self.session();a.advance(20,max_dispatches=1);saved=a.snapshot();a.close()
        saved['native']['compressed']='AAAA';rehash(saved['native']);rehash(saved)
        b=self.session(resume=saved)
        with self.assertRaises(ValueError):b.advance(10,max_dispatches=1)
        self.assertTrue(FakeSampler.instances[-1].closed);b.close()

class BackendSelectionIsolationTests(unittest.TestCase):
    def test_actual_darwin_cuda_policy_rejects_before_native_factory(self):
        # The host fake must not weaken the real platform policy test.
        with patch.object(backends,'sys',SimpleNamespace(platform='darwin')),patch.object(sampling,'_make_sampler',side_effect=AssertionError('native factory')) as factory:
            session=SamplerSession(problem(),replicas=4,backend='cuda')
            try:
                with self.assertRaisesRegex(ValueError,'CUDA is unavailable on macOS'):session.advance(1,max_dispatches=1)
            finally:session.close()
            factory.assert_not_called()

class DurableRepairTests(unittest.TestCase):
    def session(self,**kw):return RepairSession(problem(),[0,3],phase_seconds=30,patch_sizes=(2,),max_attempts=4,**kw)
    def test_stop_does_not_consume_pending_selection(self):
        a=self.session();a.advance(10,stop=lambda:True);saved=copied(a.snapshot());self.assertEqual(saved['index'],0);self.assertIsNone(saved['pending']);self.assertEqual(saved['stats']['attempts'],0)
    def test_changed_live_input_cannot_publish_cached_queries(self):
        a=self.session();a.problem.domains[0]=a.problem.domains[0][1:]
        with self.assertRaisesRegex(ValueError,'input/source changed'):a.snapshot()
        with self.assertRaisesRegex(ValueError,'input/source changed'):a.report()
    def test_terminal_status_requires_actual_solution(self):
        saved=self.session().snapshot();saved.update(done=True,status='solved');rehash(saved)
        with self.assertRaisesRegex(ValueError,'certificate'):self.session(resume=saved)
    def test_wrong_scope_and_input_archive_bound(self):
        saved=self.session().snapshot();saved['config']['seed']=7;rehash(saved)
        with self.assertRaises(ValueError):self.session(resume=saved)
        with self.assertRaises(ValueError):self.session(candidate_boards=[[0,3]]*65)
    def test_pending_query_retries_exact_context_and_gold_cuts(self):
        stop=[False];observed=[]
        def fake(p,board,cells,**kwargs):
            observed.append((board.copy(),list(cells),kwargs['seed'],deepcopy(kwargs['resume_state'])))
            stop[0]=True;return None,'UNKNOWN'
        a=self.session()
        with patch('constraint_lns_resume_round25._load_cp_model',return_value=object()),patch('constraint_lns_resume_round25._solve_patch',side_effect=fake):a.advance(10,stop=lambda:stop[0])
        saved=copied(a.snapshot());self.assertIsNotNone(saved['pending']);self.assertEqual(saved['index'],0)
        b=self.session(resume=saved);stop[0]=False
        with patch('constraint_lns_resume_round25._load_cp_model',return_value=object()),patch('constraint_lns_resume_round25._solve_patch',side_effect=fake):b.advance(10,stop=lambda:stop[0])
        self.assertEqual(observed[0],observed[1]);self.assertEqual(b.stats['repair_native_retries'],1);self.assertEqual(b.stats['repair_model_rebuilds'],0)
    def test_completed_scoped_queries_cache_across_restore(self):
        calls=[]
        def fake(p,board,cells,**kw):calls.append((tuple(board),tuple(cells)));kw['diagnostics']['model_built']=1;return None,'INFEASIBLE'
        with patch('constraint_lns_resume_round25._load_cp_model',return_value=object()),patch('constraint_lns_resume_round25._solve_patch',side_effect=fake):
            a=self.session();a.advance(10,max_attempts_per_slice=1);saved=copied(a.snapshot());b=self.session(resume=saved);b.advance(10)
        self.assertEqual(len(calls),1);self.assertEqual(b.stats['attempt_cache_hits'],3);self.assertEqual(b.index,4);self.assertFalse(b.report()['complete'])
    def test_checkpoint_publication_obeys_reader_byte_bound(self):
        session=self.session()
        with patch('constraint_lns_resume_round25.MAX_SNAPSHOT_BYTES',1):
            with self.assertRaisesRegex(ValueError,'pending state was retained'):session.snapshot()
        self.assertFalse(session.done);self.assertEqual(session.index,0)
    def test_all_legacy_repair_modes_integrate_and_resume_without_global_claims(self):
        try:from ortools.sat.python import cp_model
        except ImportError:self.skipTest('Optional OR-Tools')
        experiments={name:True for name in ('core_repair','loop_trades','fragment_crossover','repair_learning')}
        a=self.session(experiments=experiments);a.advance(20,max_attempts_per_slice=1);saved=copied(a.snapshot());b=self.session(experiments=experiments,resume=saved);b.advance(20)
        self.assertEqual(b.experiment_index,4);self.assertEqual(len(b.experimental_reports),4);self.assertFalse(b.report()['complete'])
        for report in b.experimental_reports:
            self.assertFalse(report.get('native_state_restored',False));self.assertNotEqual(report.get('scope'),'original_puzzle_exhausted')
    def test_real_loop_cuts_completed_queries_reused(self):
        try:from ortools.sat.python import cp_model
        except ImportError:self.skipTest('Optional OR-Tools')
        a=self.session();a.advance(20,max_attempts_per_slice=1);saved=copied(a.snapshot());b=self.session(resume=saved);b.advance(20)
        self.assertEqual(b.index,4);self.assertEqual(b.stats['attempts'],3);self.assertEqual(b.stats['attempt_cache_hits'],1)
        self.assertGreater(b.stats['loop_cuts'],0);self.assertFalse(b.report()['complete']);self.assertFalse(b._solved(b.best))

class HybridTests(unittest.TestCase):
    def test_immediate_stop_saves_once_allocated_whole_run_budgets(self):
        with patch.object(sampling,'_make_sampler',side_effect=AssertionError('allocated')):
            result=sampling.solve(problem(),seconds=1,stop=lambda:True,options={'durable_hybrid':True},learning_state={'_hybrid_run_seconds':0})
        saved=copied(result['checkpoint']);self.assertEqual(saved['budgets'],{'sampling':30.,'repair':60.});self.assertEqual(saved['phase'],'sampling')
        resumed=sampling.solve(problem(),seconds=.1,stop=lambda:True,resume=saved,options={'durable_hybrid':False})
        self.assertEqual(resumed['checkpoint']['budgets'],saved['budgets']);self.assertTrue(resumed['durable_resume']);self.assertEqual(resumed['checkpoint']['resumes'],1)
    def test_zero_heuristic_budgets_dispatch_exact_and_wrap_progress(self):
        events=[];stack={'finite':'cursor'}
        def exact(p,**kw):kw['progress']({'event':'checkpoint','payload':stack});return {'status':'timeout','complete':False,'nodes':7,'stats':{},'elapsed_seconds':.01,'checkpoint':stack}
        with patch('constraint_dfs.solve',side_effect=exact),patch.object(sampling,'_make_sampler',side_effect=AssertionError('allocated')):
            result=sampling.solve(problem(),seconds=5,sampling_fraction=0,repair_fraction=0,options={'durable_hybrid':True},progress=events.append)
        self.assertEqual(result['checkpoint']['phase'],'exact_dfs');self.assertEqual(result['checkpoint']['dfs'],stack)
        wraps=[e for e in events if e['event']=='checkpoint'];self.assertEqual(wraps[0]['payload']['hybrid_version'],3)
    def test_legacy_checkpoint_skips_heuristics_and_exact_allows_changed_gpu_settings(self):
        stack={'finite':'cursor'}
        def exact(p,**kw):self.assertEqual(kw['resume'],stack);return {'status':'timeout','complete':False,'nodes':0,'stats':{},'elapsed_seconds':0.,'checkpoint':stack}
        with patch('constraint_dfs.solve',side_effect=exact),patch.object(hybrid,'SamplerSession',side_effect=AssertionError('repeated')):
            a=sampling.solve(problem(),seconds=1,resume={'hybrid_version':2,'dfs':stack,'sampling':None,'repair':None},options={'durable_hybrid':True})
            b=sampling.solve(problem(),seconds=1,resume=copied(a['checkpoint']),replicas=8192,backend='webgpu',options={'durable_hybrid':False,'move_tabu':True})
        self.assertEqual(b['checkpoint']['phase'],'exact_dfs')
    def test_heuristic_source_mismatch_rejects_without_gpu(self):
        a=sampling.solve(problem(),seconds=1,stop=lambda:True,options={'durable_hybrid':True});saved=a['checkpoint'];saved['sources']={};rehash(saved)
        with self.assertRaisesRegex(ValueError,'source semantics'):sampling.solve(problem(),seconds=1,resume=saved)
    def test_completed_sampling_commits_before_repair_and_stop(self):
        stopped=[False];events=[]
        class DoneSampler(SamplerSession):
            def advance(self,seconds,**kw):super().advance(seconds,max_dispatches=8,**kw);self.finish_phase()
        def progress(event):
            events.append(copied(event))
            if event.get('event')=='phase' and event.get('phase')=='repair':stopped[0]=True
        with patch.object(hybrid,'SamplerSession',DoneSampler),patch.object(sampling,'_make_sampler',side_effect=FakeSampler),patch.object(backends,'choose_backend',side_effect=fake_backend) as selected,patch.object(backends,'sys',SimpleNamespace(platform='darwin')),patch.object(sampling,'_CUDA',side_effect=AssertionError('native CUDA')) as cuda,patch.object(sampling,'_WebGPU',side_effect=AssertionError('native WebGPU')) as webgpu:
            first=sampling.solve(problem(),seconds=10,replicas=4,backend='cuda',stop=lambda:stopped[0],progress=progress,options={'durable_hybrid':True})
            selected.assert_called_once_with('cuda');cuda.assert_not_called();webgpu.assert_not_called()
        saved=copied(first['checkpoint']);self.assertEqual(saved['phase'],'repair');self.assertIsNone(saved['sampler_state'])
        commits=[e for e in events if e['event']=='checkpoint'];self.assertEqual(commits[-1]['payload']['phase'],'repair');self.assertEqual(saved['sampling']['dispatches'],8)
        with patch.object(hybrid,'SamplerSession',side_effect=AssertionError('sampling repeated')):
            second=sampling.solve(problem(),seconds=1,resume=saved,stop=lambda:True,replicas=999,backend='webgpu',options={'durable_hybrid':False})
        self.assertEqual(second['checkpoint']['sampling']['dispatches'],8)
    def test_repair_transition_commits_and_fresh_store_requeues_claims(self):
        original=sampling.solve(problem(),seconds=1,stop=lambda:True,options={'durable_hybrid':True})['checkpoint']
        original.update(phase='exact_dfs',claims_offered=True,repair={'experimental_repair':[{'exclusions':{'kind':'gold_repair_assumption_cores','fingerprint':problem().fingerprint,'domains':[list(row) for row in problem().domains],'cores':[[[0,0]]]}}]});rehash(original)
        offers=[]
        class Store:
            def __init__(self,*a,**kw):pass
            def configure_policy(self,**kw):pass
            def offer(self,row):offers.append(row);return True
        def exact(p,**kw):self.assertIn('_mask_conflicts',kw['learning_state']);return {'status':'timeout','complete':False,'nodes':0,'stats':{},'elapsed_seconds':0.,'checkpoint':{'cursor':1}}
        shared={'_mask_conflicts':Store()}
        with patch('constraint_mask_conflicts.VerifiedMaskConflictStore',Store),patch('constraint_dfs.solve',side_effect=exact):
            result=sampling.solve(problem(),seconds=5,resume=original,options={'smart_learning':True},learning_state=shared)
            self.assertEqual(offers,[((0,1),)])
            again=sampling.solve(problem(),seconds=5,resume=copied(result['checkpoint']),options={'smart_learning':True},learning_state=shared)
            self.assertEqual(offers,[((0,1),)]) # Same live verifier gets no repeated offer.
            sampling.solve(problem(),seconds=5,resume=copied(again['checkpoint']),options={'smart_learning':True},learning_state={'_mask_conflicts':Store()})
        self.assertEqual(offers,[((0,1),),((0,1),)]);self.assertEqual(result['checkpoint']['phase'],'exact_dfs')
    def test_repair_completion_checkpoint_precedes_exact_phase_stop(self):
        saved=sampling.solve(problem(),seconds=1,stop=lambda:True,options={'durable_hybrid':True})['checkpoint']
        saved.update(phase='repair',repair_inputs=[[0,3]],hints=[0,3]);saved['budgets']['repair']=0.;rehash(saved)
        stopped=[False];events=[]
        def progress(event):
            events.append(copied(event))
            if event.get('event')=='phase' and event.get('phase')=='exact_dfs':stopped[0]=True
        with patch('constraint_dfs.solve',side_effect=AssertionError('exact started afterStop')):
            result=sampling.solve(problem(),seconds=5,resume=saved,stop=lambda:stopped[0],progress=progress)
        commits=[e for e in events if e['event']=='checkpoint'];self.assertEqual(commits[-1]['payload']['phase'],'exact_dfs');self.assertIsNone(commits[-1]['payload']['repair_state'])
        self.assertEqual(result['checkpoint']['phase'],'exact_dfs')
    def test_empty_original_domain_reaches_exact_not_false_heuristic_unsat(self):
        p=problem();p.domains[0]=[]
        with patch.object(sampling,'_make_sampler',side_effect=AssertionError('GPU allocated for empty domain')),patch('gpu_backends.choose_backend',side_effect=AssertionError('GPU selected for empty domain')):
            result=sampling.solve(p,seconds=10,options={'durable_hybrid':True,'native':'python'})
        self.assertTrue(result['complete']);self.assertEqual(result['phase'],'exact_dfs');self.assertEqual(result['status'],'infeasible')
        self.assertEqual(result['gpu_sampling']['status'],'skipped')
    def test_real_exact_handoff_preserves_warm_caches_and_controller_metadata(self):
        import constraint_dfs
        from constraint_knowledge import KnowledgeStore
        p=problem();knowledge=KnowledgeStore(p,target='gold')
        marker=(object(),'retained report');coverage='unused-hybrid-coverage.sqlite'
        shared={'_hybrid_run_seconds':200,'_hybrid_offered_claims':marker,
                '_coverage_path':coverage,'_knowledge':knowledge}
        real=constraint_dfs.solve;seen=[]
        def first_slice(problem,**kw):
            inner=kw['learning_state'];seen.append(inner)
            self.assertIs(inner['_knowledge'],knowledge)
            self.assertFalse({'_hybrid_run_seconds','_hybrid_offered_claims','_coverage_path'}&set(inner))
            # Stop only after actual DFS has constructed its reusable support cache.
            kw['stop']=lambda:'_support' in inner
            return real(problem,**kw)
        options={'durable_hybrid':True,'native':'python'}
        with patch.object(sampling,'_make_sampler',side_effect=AssertionError('GPU allocated')),patch('gpu_backends.choose_backend',side_effect=AssertionError('GPU selected')):
            with patch.object(constraint_dfs,'solve',side_effect=first_slice):
                first=sampling.solve(p,seconds=5,sampling_fraction=0,repair_fraction=0,options=options,learning_state=shared)
            self.assertEqual(first['status'],'stopped');self.assertEqual(first['phase'],'exact_dfs')
            owners={key:shared[key] for key in ('_nogoods','_runtime','_support','_knowledge')}
            resumed=sampling.solve(p,seconds=5,resume=copied(first['checkpoint']),options=options,learning_state=shared)
        self.assertEqual(resumed['status'],'infeasible');self.assertTrue(resumed['complete'])
        for key,value in owners.items():self.assertIs(shared[key],value)
        self.assertIs(shared['_hybrid_offered_claims'],marker)
        self.assertEqual(shared['_hybrid_run_seconds'],200);self.assertEqual(shared['_coverage_path'],coverage)
        self.assertIsNot(seen[0],shared)

    def test_real_exact_guard_still_rejects_unknown_malformed_and_wrong_scope(self):
        from constraint_knowledge import KnowledgeStore
        for bad in ({'_unexpected_cache':object()},{'_runtime':object()},
                    {'_knowledge':KnowledgeStore(problem(),target='edges')}):
            with self.subTest(keys=list(bad)):
                shared={'_hybrid_run_seconds':200,'_coverage_path':'unused.sqlite',**bad}
                before=dict(shared)
                with patch.object(sampling,'_make_sampler',side_effect=AssertionError('GPU allocated')):
                    with self.assertRaises(ValueError):
                        sampling.solve(problem(),seconds=5,sampling_fraction=0,repair_fraction=0,
                                       options={'durable_hybrid':True,'native':'python'},learning_state=shared)
                self.assertEqual(set(shared),set(before))
                for key,value in before.items():self.assertIs(shared[key],value)

    def test_exact_exception_hands_back_real_created_cache_resources(self):
        import constraint_dfs
        real=constraint_dfs.solve;created={};shared={'_hybrid_run_seconds':200}
        def failing(problem,**kw):
            inner=kw['learning_state'];kw['stop']=lambda:'_support' in inner
            answer=real(problem,**kw);self.assertEqual(answer['status'],'stopped')
            created.update(inner)
            raise RuntimeError('failure after real cache creation')
        with patch.object(sampling,'_make_sampler',side_effect=AssertionError('GPU allocated')),patch.object(constraint_dfs,'solve',side_effect=failing):
            with self.assertRaisesRegex(RuntimeError,'after real cache'):
                sampling.solve(problem(),seconds=5,sampling_fraction=0,repair_fraction=0,
                               options={'durable_hybrid':True,'native':'python'},learning_state=shared)
        self.assertIn('_runtime',created);self.assertIn('_support',created)
        for key,value in created.items():self.assertIs(shared[key],value)
        self.assertEqual(shared['_hybrid_run_seconds'],200)

    def test_exact_handoff_mirrors_removals_even_on_exception(self):
        old=object();new=object();marker=object()
        shared={'_hybrid_run_seconds':200,'_hybrid_offered_claims':marker,
                '_coverage_path':'unused.sqlite','_knowledge':old}
        def failing(problem,**kw):
            inner=kw['learning_state'];self.assertIs(inner.pop('_knowledge'),old)
            inner['_runtime']=new
            raise RuntimeError('ownership boundary failure')
        with patch.object(sampling,'_make_sampler',side_effect=AssertionError('GPU allocated')),patch('constraint_dfs.solve',side_effect=failing):
            with self.assertRaisesRegex(RuntimeError,'ownership boundary'):
                sampling.solve(problem(),seconds=5,sampling_fraction=0,repair_fraction=0,
                               options={'durable_hybrid':True,'native':'python'},learning_state=shared)
        self.assertNotIn('_knowledge',shared);self.assertIs(shared['_runtime'],new)
        self.assertIs(shared['_hybrid_offered_claims'],marker);self.assertEqual(shared['_hybrid_run_seconds'],200)
        self.assertEqual(shared['_coverage_path'],'unused.sqlite')

    def test_saved_valid_gold_solution_returns_before_gpu(self):
        tiles,board=tetrahedron_tiles();p=SearchProblem(tiles,board)
        with patch.object(sampling,'_make_sampler',side_effect=AssertionError('allocated')):
            result=sampling.solve(p,seconds=5,seed_board=[0,3,6,9],options={'durable_hybrid':True})
        self.assertTrue(result['validation']['valid']);self.assertTrue(result['complete'])

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_GPU_SAMPLING')=='1','Optional actual CUDA/WebGPU continuation checks')
class NativeSamplerTests(unittest.TestCase):
    def compare(self,backend,experiments,gold_aware=False,fixed=False):
        p=problem()
        if fixed:p.fixed[0]=0
        kw={'replicas':128,'backend':backend,'experiments':experiments,'gold_aware':gold_aware,'seed':2533}
        a=SamplerSession(p,**kw);b=c=None
        try:
            a.advance(60,max_dispatches=18);a.finish_phase();expected=a.snapshot()
            b=SamplerSession(p,**kw);b.advance(60,max_dispatches=9);saved=copied(b.snapshot());b.close();b=None
            c=SamplerSession(p,resume=saved,**kw);c.advance(60,max_dispatches=9);c.finish_phase();actual=c.snapshot()
            self.assertEqual(expected['native'],actual['native']);self.assertEqual(expected['archive'],actual['archive']);self.assertEqual(expected['experiment_stats'],actual['experiment_stats'])
            self.assertEqual(actual['launches'],18);self.assertEqual(actual['counts']['initial_boards'],128);self.assertEqual(actual['counts']['sampler_restores'],1)
        finally:
            a.close()
            if b:b.close()
            if c:c.close()
    def test_cuda_resume_all_policies(self):
        for flags in ({},EXPERIMENTS):
            for gold in (False,True):
                with self.subTest(flags=flags,gold=gold):self.compare('cuda',flags,gold)
    def test_webgpu_resume_all_policies(self):
        for flags in ({},EXPERIMENTS):
            for gold in (False,True):
                with self.subTest(flags=flags,gold=gold):self.compare('webgpu',flags,gold)
    def test_native_hybrid_stop_and_disabled_flag_resume_preserve_dispatches(self):
        factory=sampling._make_sampler
        for backend in ('cuda','webgpu'):
            with self.subTest(backend=backend):
                counter=[0]
                def wrapped(*args,**kw):
                    engine=factory(*args,**kw);launch=engine.launch
                    def run(steps,*a,**k):launch(steps,*a,**k);counter[0]+=int(steps>0)
                    engine.launch=run;return engine
                options={'durable_hybrid':True,'gpu_gold_sampling':True,**EXPERIMENTS}
                with patch.object(sampling,'_make_sampler',side_effect=wrapped):
                    first=sampling.solve(problem(),seconds=30,replicas=128,backend=backend,seed=2533,options=options,
                        learning_state={'_hybrid_run_seconds':100},stop=lambda:counter[0]>=9)
                    counter[0]=0
                    second=sampling.solve(problem(),seconds=30,replicas=128,backend=backend,seed=2533,options={**options,'durable_hybrid':False},
                        resume=copied(first['checkpoint']),stop=lambda:counter[0]>=9)
                saved=second['checkpoint']['sampler_state'];self.assertEqual(saved['launches'],18);self.assertEqual(saved['counts']['initial_boards'],128)
                expected=SamplerSession(problem(),replicas=128,backend=backend,seed=2533,experiments=EXPERIMENTS,gold_aware=True)
                try:
                    expected.advance(60,max_dispatches=18);native=expected.snapshot()
                    self.assertEqual(saved['native'],native['native']);self.assertEqual(saved['archive'],native['archive'])
                finally:expected.close()
    def test_fixed_placement_resume(self):
        for backend in ('cuda','webgpu'):
            with self.subTest(backend=backend):self.compare(backend,EXPERIMENTS,True,True)

if __name__=='__main__':unittest.main()
