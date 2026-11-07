"""Benchmark accounting and cleanup regressions; no search workload is launched."""
import contextlib,io,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import benchmark_round25 as benchmark
from constraint_work import WorkScope,frontier_snapshot,restore_frontier
from test_dependencies import requires_sat


class BenchmarkAccountingTests(unittest.TestCase):
 def setUp(self):self.scope=WorkScope('finite','gold',(3,3),(3,3))
 def checkpoint(self,rows):
  return {'jobs':[{'checkpoint':{'stack':[{'domains':[hex(x) for x in row],'cell':-1,'choices':[],'next':0}]}} for row in rows]}
 def test_validated_committed_frontiers_have_exact_cartesian_volume(self):
  rows=((1,3),(2,1));result={'status':'timeout','complete':False,'checkpoint':self.checkpoint(rows)}
  self.assertEqual(benchmark.pending_frontier(self.scope,result,'cpu'),rows)
  self.assertEqual(benchmark.volume(rows),3)
  result['checkpoint']={'frontier':frontier_snapshot(self.scope,rows)}
  self.assertEqual(benchmark.pending_frontier(self.scope,result,'gpu'),rows)
 def test_missing_incomplete_overlap_and_escaped_scope_do_not_claim_closed_work(self):
  for result in ({'status':'timeout','complete':False},{'status':'infeasible','complete':False}):
   with self.assertRaises(ValueError):benchmark.pending_frontier(self.scope,result,'cpu')
  for rows in (((1,3),(1,3)),((4,3),)):
   with self.assertRaises(ValueError):benchmark.pending_frontier(self.scope,{'status':'timeout','checkpoint':self.checkpoint(rows)},'cpu')
  self.assertEqual(benchmark.pending_frontier(self.scope,{'status':'infeasible','complete':True},'cpu'),())
  self.assertIsNone(benchmark.pending_frontier(self.scope,{'status':'solved','complete':True},'cpu'))
 def test_one_worker_uses_parallel_original_scope_fresh_ledgers_and_fixture_provenance(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-contract-',dir=benchmark.ROOT) as directory:
   args=SimpleNamespace(case='small',seed=817,free_cells=60,engine='cpu',workers=1,seconds=.01,
       profile='reuse',replicas=128,output=Path(directory)/'case.json')
   called=[]
   def fake(problem,**kwargs):
    called.append(kwargs)
    self.assertEqual(kwargs['workers'],1)
    self.assertEqual(kwargs['resume']['scope']['root'],kwargs['resume']['scope']['original'])
    self.assertIn('_coverage_path',kwargs['learning_state']);self.assertIn('_regional_cache_dir',kwargs['learning_state'])
    # No solver runs: this fake result tests result routing/accounting only.
    return {'status':'infeasible','complete':True,'nodes':0,'stats':{}}
   with patch('constraint_parallel.solve',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
    benchmark.worker(args)
    with self.assertRaises(FileExistsError):benchmark.worker(args)
   self.assertEqual(len(called),1)
   report=json.loads(args.output.read_text())
   self.assertEqual(report['solver_engine'],'parallel-dfs');self.assertEqual(report['workers'],1)
   self.assertEqual(report['provenance']['kind'],'synthetic_planted_fixture')
   self.assertTrue(report['provenance']['planted_reference_independently_validated'])
   self.assertTrue(report['benchmark_state']['coverage_enabled']);self.assertEqual(len(report['domains_sha256']),64)


class BenchmarkProvenanceTests(unittest.TestCase):
 def test_fixture_dependency_and_native_bytes_are_frozen_with_correct_line_endings(self):
  from benchmark_provenance import source_snapshot
  with tempfile.TemporaryDirectory(prefix='benchmark-provenance-',dir=benchmark.ROOT) as directory:
   root=Path(directory);(root/'.native').mkdir();(root/'runtime').mkdir()
   (root/'test_fixture.py').write_bytes(b'fixture=1\r\n')
   (root/'kernel.c').write_bytes(b'kernel\n')
   (root/'.native/library.dll').write_bytes(b'\x00\r\n')
   (root/'runtime/live.json').write_text('{}',encoding='utf-8')
   before=source_snapshot(root)
   self.assertEqual(set(before),{'test_fixture.py','kernel.c','.native/library.dll'})
   self.assertNotIn('lf_sha256',before['.native/library.dll'])
   (root/'test_fixture.py').write_bytes(b'fixture=1\n')
   after=source_snapshot(root)
   self.assertEqual(before['test_fixture.py']['lf_sha256'],after['test_fixture.py']['lf_sha256'])
   self.assertNotEqual(before['test_fixture.py']['raw_sha256'],after['test_fixture.py']['raw_sha256'])
   (root/'.native/library.dll').write_bytes(b'\x00\n')
   self.assertNotEqual(after['.native/library.dll'],source_snapshot(root)['.native/library.dll'])


 def test_launcher_and_requirement_changes_are_detected_and_lf_canonicalized(self):
  from benchmark_provenance import source_snapshot
  with tempfile.TemporaryDirectory(prefix='benchmark-launchers-',dir=benchmark.ROOT) as directory:
   root=Path(directory);names=('Setup.ps1','OpenDashboard.cmd','Setup.sh','requirements-webgpu.txt')
   for name in names:(root/name).write_bytes(b'first\r\n')
   before=source_snapshot(root);self.assertEqual(set(before),set(names))
   for name in names:
    (root/name).write_bytes(b'first\n');after=source_snapshot(root)
    self.assertEqual(before[name]['lf_sha256'],after[name]['lf_sha256'])
    self.assertNotEqual(before[name]['raw_sha256'],after[name]['raw_sha256'])
    (root/name).write_bytes(b'changed\n');changed=source_snapshot(root)
    self.assertNotEqual(after[name]['raw_sha256'],changed[name]['raw_sha256'])
    self.assertNotEqual(after[name]['lf_sha256'],changed[name]['lf_sha256'])


class BenchmarkFreezeTests(unittest.TestCase):
 def test_frozen_run_rejects_uncommitted_sources_before_search(self):
  for repository in ({}, {'commit':'abc','dirty':True}):
   with tempfile.TemporaryDirectory(prefix='benchmark-freeze-',dir=benchmark.ROOT) as directory:
    argv=['benchmark_round25.py','--output',directory,'--suite','finite','--evidence-stage','frozen']
    with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'repository_snapshot',return_value=repository),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen') as launch:
     with self.assertRaisesRegex(RuntimeError,'clean, readable committed'):benchmark.main()
    launch.assert_not_called()
 def test_serial_worker_matrix_records_each_config_and_reverses_second_seed(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-workers-',dir=benchmark.ROOT) as directory:
   folder=Path(directory);data=folder/'data.json';data.write_text('{}',encoding='utf-8')
   argv=['benchmark_round25.py','--output',str(folder/'matrix'),'--data',str(data),'--suite','original','--profiles','baseline','--worker-counts','4','8','4','--evidence-stage','frozen']
   jobs=[]
   def fake(command,**kwargs):
    output=Path(command[command.index('--output')+1]);workers=int(command[command.index('--workers')+1]);seed=int(command[command.index('--seed')+1]);jobs.append((seed,workers))
    output.write_text(json.dumps({'status':'infeasible','complete':True,'workers':workers,'provenance':{'input_sha256':benchmark.hashlib.sha256(data.read_bytes()).hexdigest()}}),encoding='utf-8')
    return SimpleNamespace(returncode=0,poll=lambda:0,communicate=lambda **kw:('',''))
   repository={'commit':'abc','dirty':False}
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'repository_snapshot',return_value=repository),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen',side_effect=fake),patch.object(benchmark.platform,'platform',return_value='test-platform'),contextlib.redirect_stdout(io.StringIO()):benchmark.main()
   self.assertEqual(jobs,[(71,4),(71,8),(197,8),(197,4)])
   report=json.loads((folder/'matrix/report.json').read_text(encoding='utf-8'))
   self.assertEqual(report['cpu_worker_counts'],[4,8]);self.assertEqual(report['evidence_stage'],'frozen');self.assertEqual(report['completed_cases'],4)


 def test_finite_depths_dedupe_without_repeating_small_and_reverse_second_seed(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-depths-',dir=benchmark.ROOT) as directory:
   argv=['benchmark_round25.py','--output',directory,'--suite','finite','--profiles','baseline','reuse','baseline','--finite-cases','small','residual','small','--free-cell-counts','8','16','8','60','--finite-seconds','7','--seconds','.01','--evidence-stage','frozen']
   jobs=[]
   def fake(command,**kwargs):
    value=lambda key:command[command.index(key)+1]
    case=value('--case');depth=int(value('--free-cells')) if case=='residual' else None
    jobs.append((int(value('--seed')),case,depth,value('--profile'),float(value('--seconds'))))
    Path(value('--output')).write_text(json.dumps({'status':'infeasible','complete':True}),encoding='utf-8')
    return SimpleNamespace(returncode=0,poll=lambda:0,communicate=lambda **kw:('',''))
   repository={'commit':'abc','dirty':False}
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'repository_snapshot',return_value=repository),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen',side_effect=fake),patch.object(benchmark.platform,'platform',return_value='test-platform'),contextlib.redirect_stdout(io.StringIO()):benchmark.main()
   first=[(case,depth,profile,7.0) for case,depth in [('small',None),('residual',8),('residual',16),('residual',60)] for profile in ('baseline','reuse')]
   self.assertEqual(jobs,[(817,*config) for config in first]+[(291,*config) for config in reversed(first)])
   report=json.loads((Path(directory)/'report.json').read_text(encoding='utf-8'))
   self.assertEqual(report['planned_cases'],16);self.assertEqual(report['finite_cases'],['small','residual'])
   self.assertEqual(report['free_cell_counts'],[8,16,60]);self.assertEqual(report['finite_seconds'],7)
   self.assertEqual([(row['seed'],row['case'],row['free_cells'],row['profile'],row['budget_seconds']) for row in report['results']],jobs)
   self.assertEqual(sum(row['case']=='small' for row in report['results']),4)
 def test_residual_only_selector_and_default_depth_do_not_add_small_cases(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-residuals-',dir=benchmark.ROOT) as directory:
   argv=['benchmark_round25.py','--output',directory,'--suite','finite','--profiles','baseline','--finite-cases','residual','--free-cells','24','--finite-seconds','3']
   jobs=[]
   def fake(command,**kwargs):
    value=lambda key:command[command.index(key)+1]
    jobs.append((value('--case'),int(value('--free-cells')),float(value('--seconds'))))
    Path(value('--output')).write_text(json.dumps({'status':'infeasible','complete':True}),encoding='utf-8')
    return SimpleNamespace(returncode=0,poll=lambda:0,communicate=lambda **kw:('',''))
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={}),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen',side_effect=fake),patch.object(benchmark.platform,'platform',return_value='test-platform'),contextlib.redirect_stdout(io.StringIO()):benchmark.main()
   self.assertEqual(jobs,[('residual',24,3.0)]*2)
 def test_invalid_finite_budgets_and_depths_fail_before_launch(self):
  for option,values in (('--finite-seconds',['0']),('--finite-seconds',['3601']),('--finite-seconds',['nan']),('--free-cell-counts',['0']),('--free-cell-counts',['8','161'])):
   with tempfile.TemporaryDirectory(prefix='benchmark-invalid-',dir=benchmark.ROOT) as directory:
    argv=['benchmark_round25.py','--output',directory,option,*values]
    with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark.subprocess,'Popen') as launch,contextlib.redirect_stderr(io.StringIO()):
     with self.assertRaises(SystemExit):benchmark.main()
    launch.assert_not_called()


class FiniteGPUBenchmarkFreezeTests(unittest.TestCase):
 def test_dirty_gpu_evidence_refuses_before_loading_or_running_fixtures(self):
  import benchmark_gpu_round25 as gpu
  with tempfile.TemporaryDirectory(prefix='gpu-finite-freeze-',dir=benchmark.ROOT) as directory:
   argv=['benchmark_gpu_round25.py','--output',str(Path(directory)/'report.json'),'--evidence-stage','frozen']
   with patch.object(benchmark.sys,'argv',argv),patch.object(gpu,'source_snapshot',return_value={}),patch.object(gpu,'runtime_snapshot',return_value={}),patch.object(gpu,'repository_snapshot',return_value={'commit':'abc','dirty':True}),patch.object(gpu,'_load_fixtures') as load,patch.object(gpu,'run26') as run:
    with self.assertRaisesRegex(RuntimeError,'clean, readable committed'):gpu.main()
   load.assert_not_called();run.assert_not_called()
 def test_gpu_evidence_records_complete_identity_and_rejects_mid_run_changes(self):
  import benchmark_gpu_round25 as gpu
  for changed in ('none','source','commit'):
   with self.subTest(changed=changed),tempfile.TemporaryDirectory(prefix='gpu-finite-identity-',dir=benchmark.ROOT) as directory:
    output=Path(directory)/'report.json';before={'fixture':{'raw_sha256':'a'}};after={'fixture':{'raw_sha256':'b'}} if changed=='source' else before
    repo={'commit':'abc','dirty':False};repo_after={**repo,'commit':'def'} if changed=='commit' else repo
    argv=['benchmark_gpu_round25.py','--output',str(output),'--evidence-stage','frozen']
    with patch.object(benchmark.sys,'argv',argv),patch.object(gpu,'source_snapshot',side_effect=[before,after]),patch.object(gpu,'runtime_snapshot',return_value={'assertions_enabled':True}),patch.object(gpu,'repository_snapshot',side_effect=[repo,repo_after]),patch.object(gpu,'run26',return_value=[]),patch.object(gpu,'_hardware_snapshot',return_value={'test_only':True}) as hardware,contextlib.redirect_stdout(io.StringIO()):
     if changed=='none':gpu.main()
     else:
      with self.assertRaisesRegex(RuntimeError,'changed'):gpu.main()
    if changed=='none':
     report=json.loads(output.read_text(encoding='utf-8'));self.assertEqual(report['evidence_stage'],'frozen');self.assertEqual(report['repository_after'],repo)
     self.assertEqual(report['source_before'],before);self.assertEqual(report['runtime_before'],report['runtime_after'])
    else:self.assertFalse(output.exists());hardware.assert_not_called()


class LearningBenchmarkFreezeTests(unittest.TestCase):
 def test_dirty_frozen_learning_run_never_invokes_candidate(self):
  import benchmark_learning_round25 as learning
  for repo in ({},{'commit':'abc','dirty':True}):
   with tempfile.TemporaryDirectory(prefix='learning-freeze-',dir=benchmark.ROOT) as directory:
    argv=['benchmark_learning_round25.py','--candidate','1','--output',str(Path(directory)/'result.json'),'--evidence-stage','frozen']
    with patch.object(benchmark.sys,'argv',argv),patch.object(learning,'source_snapshot',return_value={}),patch.object(learning,'runtime_snapshot',return_value={}),patch.object(learning,'repository_snapshot',return_value=repo),patch.object(learning,'core_lifting') as candidate:
     with self.assertRaisesRegex(RuntimeError,'clean, readable committed'):learning.main()
    candidate.assert_not_called()
 def test_learning_evidence_records_stage_and_rejects_mid_run_commit_changes(self):
  import benchmark_learning_round25 as learning
  for changed in (False,True):
   with tempfile.TemporaryDirectory(prefix='learning-provenance-',dir=benchmark.ROOT) as directory:
    output=Path(directory)/'result.json'
    argv=['benchmark_learning_round25.py','--candidate','1','--output',str(output),'--evidence-stage','frozen']
    repo={'commit':'abc','dirty':False};after={**repo,'commit':'def'} if changed else repo
    with patch.object(benchmark.sys,'argv',argv),patch.object(learning,'source_snapshot',return_value={}),patch.object(learning,'runtime_snapshot',return_value={}),patch.object(learning,'repository_snapshot',side_effect=[repo,after]),patch.object(learning,'core_lifting',return_value={'proposal':1}),patch.object(learning,'sat_provenance',return_value={}),contextlib.redirect_stdout(io.StringIO()):
     if changed:
      with self.assertRaisesRegex(RuntimeError,'checkout changed'):learning.main()
     else:learning.main()
    if changed:self.assertFalse(output.exists())
    else:
     row=json.loads(output.read_text(encoding='utf-8'));self.assertEqual(row['evidence_stage'],'frozen');self.assertEqual(row['repository_after'],repo)


class BenchmarkCleanupTests(unittest.TestCase):
 def test_windows_tree_failure_aborts_instead_of_starting_another_case(self):
  child=SimpleNamespace(pid=424242,poll=lambda:None,wait=lambda **kw:None)
  with patch.object(benchmark.os,'name','nt'),patch.object(benchmark.subprocess,'run',return_value=SimpleNamespace(returncode=1)):
   with self.assertRaisesRegex(RuntimeError,'cleanup failed'):benchmark.terminate_tree(child)
 def test_posix_cleanup_targets_group_after_coordinator_exits(self):
  child=SimpleNamespace(pid=424242,poll=lambda:17,wait=lambda **kw:17)
  with patch.object(benchmark.os,'name','posix'),patch.object(benchmark.os,'killpg',create=True) as kill,patch.object(benchmark.signal,'SIGKILL',9,create=True):
   status=benchmark.terminate_tree(child)
  kill.assert_called_once_with(child.pid,9)
  self.assertIn('coordinator reaped',status)
 def test_nonzero_exit_writes_partial_report_and_never_launches_next_case(self):
  child=SimpleNamespace(pid=424242,returncode=17,poll=lambda:17,communicate=lambda **kw:('partial output','native failure'))
  with tempfile.TemporaryDirectory(prefix='benchmark-failed-',dir=benchmark.ROOT) as directory:
   output=Path(directory)/'matrix';argv=['benchmark_round25.py','--output',str(output),'--suite','finite','--profiles','baseline']
   # platform.platform may itself spawn a subprocess on POSIX; keep that
   # unrelated introspection outside this coordinator-process mock.
   def cleanup(owned):
    self.assertIs(owned,child)
    self.assertTrue((output/'report.json').exists(),'Partial report precedes cleanup')
    self.assertTrue((output/'00-cpu-small-baseline-817.stderr.txt').exists())
    return 'Descendant cleanup cannot be confirmed.'
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen',return_value=child) as launch,patch.object(benchmark,'terminate_tree',side_effect=cleanup),patch.object(benchmark.platform,'platform',return_value='test-platform'),contextlib.redirect_stdout(io.StringIO()):
    with self.assertRaisesRegex(RuntimeError,'remaining matrix was not started'):benchmark.main()
   launch.assert_called_once();report=json.loads((output/'report.json').read_text())
   self.assertTrue(report['aborted_after_child_failure']);self.assertEqual(report['completed_cases'],1)
   self.assertGreater(report['planned_cases'],1);self.assertEqual(report['results'][0]['process_exit_code'],17)
   self.assertEqual(report['results'][0]['status'],'process_error')
   self.assertIn('cannot be confirmed',report['results'][0]['cleanup'])

 def test_keyboard_interrupt_retires_the_owned_child(self):
  child=SimpleNamespace(pid=424242,returncode=None,poll=lambda:None)
  def interrupted(**kwargs):raise KeyboardInterrupt()
  child.communicate=interrupted
  with tempfile.TemporaryDirectory(prefix='benchmark-cancel-',dir=benchmark.ROOT) as directory:
   argv=['benchmark_round25.py','--output',str(Path(directory)/'matrix'),'--suite','finite','--profiles','baseline']
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen',return_value=child) as launch,patch.object(benchmark,'terminate_tree') as cleanup:
    with self.assertRaises(KeyboardInterrupt):benchmark.main()
   launch.assert_called_once();cleanup.assert_called_once_with(child)


 def test_malformed_exited_child_still_cleans_owned_group_and_keeps_original_error(self):
  for cleanup_fails in (False,True):
   with self.subTest(cleanup_fails=cleanup_fails),tempfile.TemporaryDirectory(prefix='benchmark-malformed-',dir=benchmark.ROOT) as directory:
    output=Path(directory)/'matrix';argv=['benchmark_round25.py','--output',str(output),'--suite','finite','--profiles','baseline']
    def fake(command,**kw):
     Path(command[command.index('--output')+1]).write_text('invalid-json',encoding='utf-8')
     return SimpleNamespace(pid=424242,returncode=0,poll=lambda:0,communicate=lambda **kw:('native output',''))
    with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen',side_effect=fake) as launch,patch.object(benchmark,'terminate_tree',side_effect=RuntimeError('cleanup failure') if cleanup_fails else None) as cleanup:
     with self.assertRaises(json.JSONDecodeError) as caught:benchmark.main()
    launch.assert_called_once();cleanup.assert_called_once()
    if cleanup_fails:self.assertIn('cleanup failure',caught.exception.__notes__[0])
    self.assertFalse((output/'report.json').exists())

class GPUBenchmarkTests(unittest.TestCase):
 def test_gpu_profiles_forward_effective_policy_after_no_search_warmup(self):
  for name,overrides in benchmark.GPU_PROFILES.items():
   with self.subTest(profile=name),tempfile.TemporaryDirectory(prefix='benchmark-gpu-profile-',dir=benchmark.ROOT) as directory:
    args=SimpleNamespace(case='small',seed=817,free_cells=8,engine='gpu',workers=1,seconds=.01,profile=name,replicas=128,output=Path(directory)/'case.json')
    events=[]
    def warm(problem,options):
     events.append('warmup')
     self.assertEqual({k:options[k] for k in overrides},overrides)
     return {'initialization_only':True,'search_steps':0,'wall_seconds':2.0}
    def fake(problem,**kwargs):
     self.assertEqual(events,['warmup']);events.append('search')
     self.assertEqual({k:kwargs['options'][k] for k in overrides},overrides)
     self.assertEqual(kwargs['replicas'],128)
     self.assertEqual(set(kwargs['learning_state']),{'_coverage_path','_gpu_checkpoint_dir'})
     self.assertEqual(kwargs['work_frontier']['scope']['root'],kwargs['work_frontier']['scope']['original'])
     return {'status':'stopped','complete':False,'nodes':0,'stats':{},'checkpoint':{'frontier':kwargs['work_frontier']}}
    with patch.object(benchmark,'warmup_domain',side_effect=warm),patch('gpu_domain_runner.solve',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):benchmark.worker(args)
    report=json.loads(args.output.read_text(encoding='utf-8'))
    self.assertEqual(events,['warmup','search']);self.assertEqual(report['gpu_warmup']['wall_seconds'],2.0)
    self.assertGreaterEqual(report['solver_wall_seconds'],0);self.assertEqual(report['profile_overrides'],overrides)
    self.assertEqual(report['benchmark_state']['coverage_enabled'],bool(overrides.get('shared_coverage')))
    self.assertEqual(report['replicas'],128);self.assertEqual(report['raw_space']['closed'],'0')
 def test_gpu_original_matrix_dedupes_and_reverses_all_configurations(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-gpu-matrix-',dir=benchmark.ROOT) as directory:
   folder=Path(directory);data=folder/'tiles.json';data.write_text('{}',encoding='utf-8')
   argv=['benchmark_round25.py','--output',str(folder/'matrix'),'--data',str(data),'--suite','gpu','--gpu-profiles','baseline','joint','baseline','--replica-counts','128','512','128','--evidence-stage','frozen']
   calls=[]
   def fake(command,**kwargs):
    value=lambda key:command[command.index(key)+1]
    calls.append((int(value('--seed')),value('--profile'),int(value('--replicas'))))
    Path(value('--output')).write_text(json.dumps({'status':'infeasible','complete':True,'provenance':{'input_sha256':benchmark.hashlib.sha256(data.read_bytes()).hexdigest()}}),encoding='utf-8')
    return SimpleNamespace(returncode=0,poll=lambda:0,communicate=lambda **kw:('',''))
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={'commit':'abc','dirty':False}),patch.object(benchmark.subprocess,'Popen',side_effect=fake),patch.object(benchmark.platform,'platform',return_value='test'),contextlib.redirect_stdout(io.StringIO()):benchmark.main()
   first=[(profile,lanes) for lanes in (128,512) for profile in ('baseline','joint')]
   self.assertEqual(calls,[(71,*config) for config in first]+[(197,*config) for config in reversed(first)])
   report=json.loads((folder/'matrix/report.json').read_text(encoding='utf-8'))
   self.assertEqual(report['gpu_replica_counts'],[128,512]);self.assertEqual(report['completed_cases'],8)
   self.assertEqual([(r['seed'],r['profile'],r['replicas']) for r in report['results']],calls)
 def test_gpu_finite_matrix_varies_depth_without_repeating_small_per_depth(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-gpu-finite-',dir=benchmark.ROOT) as directory:
   argv=['benchmark_round25.py','--output',directory,'--suite','gpu','--gpu-cases','small','residual','small','--gpu-profiles','baseline','joint','--replica-counts','128','256','128','--free-cell-counts','8','16','8','--finite-seconds','3']
   calls=[]
   def fake(command,**kwargs):
    value=lambda key:command[command.index(key)+1]
    case=value('--case');depth=int(value('--free-cells')) if case=='residual' else None
    calls.append((int(value('--seed')),case,depth,value('--profile'),int(value('--replicas')),float(value('--seconds'))))
    Path(value('--output')).write_text(json.dumps({'status':'infeasible','complete':True}),encoding='utf-8')
    return SimpleNamespace(returncode=0,poll=lambda:0,communicate=lambda **kw:('',''))
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen',side_effect=fake),patch.object(benchmark.platform,'platform',return_value='test'),contextlib.redirect_stdout(io.StringIO()):benchmark.main()
   first=[(case,depth,profile,lanes,3.0) for case,depth in [('small',None),('residual',8),('residual',16)] for lanes in (128,256) for profile in ('baseline','joint')]
   self.assertEqual(calls,[(817,*config) for config in first]+[(291,*config) for config in reversed(first)])
   self.assertEqual(sum(case=='small' for seed,case,*rest in calls),8)
   report=json.loads((Path(directory)/'report.json').read_text(encoding='utf-8'))
   self.assertEqual(report['completed_cases'],24);self.assertIsNone(report['input_sha256'])
   self.assertEqual([(r['seed'],r['case'],r['free_cells'],r['profile'],r['replicas'],r['budget_seconds']) for r in report['results']],calls)
 def test_invalid_gpu_replica_counts_are_rejected_before_launch(self):
  for extra in (['--replicas','0'],['--replicas','513'],['--replica-counts','128','513']):
   with tempfile.TemporaryDirectory(prefix='benchmark-gpu-invalid-',dir=benchmark.ROOT) as directory:
    with patch.object(benchmark.sys,'argv',['benchmark_round25.py','--output',directory,'--suite','gpu',*extra]),patch.object(benchmark.subprocess,'Popen') as launch,contextlib.redirect_stderr(io.StringIO()):
     with self.assertRaises(SystemExit):benchmark.main()
    launch.assert_not_called()


class SATBenchmarkTests(unittest.TestCase):
 def args(self,folder,engine='sat',profile='lifted'):
  return SimpleNamespace(case='small',seed=817,free_cells=8,engine=engine,workers=1,seconds=.01,
      profile=profile,replicas=128,output=Path(folder)/'case.json')
 def test_deterministic_partition_covers_same_original_scope_and_incomplete_scope_is_rejected(self):
  scope=WorkScope('finite','gold',(255,255),(255,255))
  first=benchmark.initial_frontier(scope,'sat');second=benchmark.initial_frontier(scope,'sat')
  self.assertEqual(first,second);self.assertEqual(len(first['boxes']),32)
  self.assertEqual(WorkScope.restore(first['scope']),scope)
  self.assertEqual(benchmark.volume(restore_frontier(first,scope)),64)
  result={'status':'timeout','complete':False,'checkpoint':{'frontier':first}}
  self.assertEqual(benchmark.pending_frontier(scope,result,'sat'),restore_frontier(first,scope))
  narrow=WorkScope('finite','gold',scope.original,(1,255))
  result['checkpoint']['frontier']=frontier_snapshot(narrow,[narrow.root])
  with self.assertRaisesRegex(ValueError,'changed its original owned scope'):benchmark.pending_frontier(scope,result,'sat')
  result['checkpoint']['frontier']=frontier_snapshot(scope,[])
  with self.assertRaisesRegex(ValueError,'empty remaining'):benchmark.pending_frontier(scope,result,'sat')
  for status,complete,boxes in (('timeout',True,[]),('infeasible',True,[scope.root]),('infeasible',False,[scope.root])):
   bad={'status':status,'complete':complete,'checkpoint':{'frontier':frontier_snapshot(scope,boxes)}}
   with self.assertRaises(ValueError):benchmark.pending_frontier(scope,bad,'sat')
  with patch('gpu_domain_runner.split_cubes',return_value=[(1,255)]):
   with self.assertRaisesRegex(ValueError,'cover the original root'):benchmark.initial_frontier(scope,'sat')
 def test_sat_worker_forwards_effective_policy_and_saves_hashed_checkpoint(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-sat-',dir=benchmark.ROOT) as directory:
   args=self.args(directory);calls=[];closed=[]
   def fake(problem,**kwargs):
    calls.append(kwargs);options=kwargs['options']
    self.assertTrue(options['persistent_sat']);self.assertFalse(options['native_sat_persistence'])
    self.assertTrue(options['sat_core_lifting']);self.assertTrue(options['symbolic_coverage']);self.assertTrue(options['shared_coverage'])
    self.assertEqual(len(kwargs['resume']['boxes']),32)
    self.assertEqual(kwargs['resume']['scope']['root'],kwargs['resume']['scope']['original'])
    self.assertEqual(set(kwargs['learning_state']),{'_coverage_path'})
    kwargs['learning_state']['_sat_session']=SimpleNamespace(close=lambda:closed.append(True))
    return {'status':'stopped','complete':False,'nodes':5,'stats':{'session_decisions':3,'session_core_decisions':2,'session_total_decisions':5},
        'checkpoint':{'schema':'diamond-sat-session-v1','frontier':kwargs['resume'],'stats':{}}}
   with patch('constraint_sat_session.solve',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):benchmark.worker(args)
   self.assertEqual(closed,[True]);self.assertEqual(len(calls),1)
   result=json.loads(args.output.read_text(encoding='utf-8'));artifact=result['checkpoint_artifact'];raw=Path(artifact['path']).read_bytes()
   self.assertEqual(artifact['sha256'],benchmark.hashlib.sha256(raw).hexdigest());self.assertEqual(artifact['bytes'],len(raw))
   self.assertNotIn('checkpoint',result);self.assertEqual(result['raw_space']['closed'],'0')
   self.assertEqual(result['sat_work']['session_core_decisions'],2);self.assertEqual(result['sat_work']['session_decisions'],3)
   self.assertTrue(result['benchmark_state']['coverage_enabled']);self.assertEqual(result['solver_engine'],'sat')
   self.assertIn('repeated measurements',result['solver_seed_effect'])
   initial=json.loads(Path(result['initial_frontier']['path']).read_text(encoding='utf-8'))
   self.assertEqual(result['initial_frontier']['digest'],initial['digest'])
 def test_warm_slices_keep_same_session_and_always_close_on_failure(self):
  state={};closed=[];initial={'initial':True};saved={'saved':True};calls=[]
  def fake(problem,**kwargs):
   calls.append(kwargs);self.assertIs(kwargs['learning_state'],state)
   state.setdefault('_sat_session',SimpleNamespace(close=lambda:closed.append(True)))
   return {'status':'timeout','checkpoint':saved} if len(calls)==1 else {'status':'infeasible','complete':True}
  with patch('constraint_sat_session.solve',side_effect=fake),patch.object(benchmark.time,'monotonic',side_effect=[0,30]):
   result=benchmark.run_sat(None,seconds=60,seed=71,initial=initial,options={},learning_state=state)
  self.assertEqual([call['seconds'] for call in calls],[30,30]);self.assertIs(calls[0]['resume'],initial);self.assertIs(calls[1]['resume'],saved)
  self.assertEqual(result['benchmark_sat_slices'],2);self.assertEqual(closed,[True])
  with patch('constraint_sat_session.solve',side_effect=RuntimeError('failed')):
   with self.assertRaisesRegex(RuntimeError,'failed'):benchmark.run_sat(None,seconds=1,seed=71,initial=initial,options={},learning_state=state)
  self.assertEqual(closed,[True,True])
 def test_cpu_and_gpu_final_checkpoints_are_preserved_without_running_engines(self):
  for engine,module in (('cpu','constraint_parallel'),('gpu','gpu_domain_runner')):
   with tempfile.TemporaryDirectory(prefix='benchmark-artifact-',dir=benchmark.ROOT) as directory:
    args=self.args(directory,engine,'baseline')
    def fake(problem,**kwargs):
     initial=kwargs.get('resume',kwargs.get('work_frontier'))
     saved={'frontier':initial} if engine=='gpu' else {'jobs':[{'checkpoint':{'stack':[{'domains':initial['scope']['root'],'cell':-1,'choices':[],'next':0}]}}]}
     return {'status':'stopped','complete':False,'nodes':0,'stats':{},'checkpoint':saved}
    with patch(module+'.solve',side_effect=fake),patch.object(benchmark,'warmup_domain',return_value={'initialization_only':True,'search_steps':0}),contextlib.redirect_stdout(io.StringIO()):benchmark.worker(args)
    result=json.loads(args.output.read_text(encoding='utf-8'))
    self.assertTrue(Path(result['checkpoint_artifact']['path']).is_file());self.assertNotIn('checkpoint',result)
 def test_existing_outputs_states_and_wrong_engine_profiles_fail_before_solver(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-collision-',dir=benchmark.ROOT) as directory:
   args=self.args(directory);args.output.write_text('existing input',encoding='utf-8')
   with patch('constraint_sat_session.solve') as solve:
    with self.assertRaises(FileExistsError):benchmark.worker(args)
   solve.assert_not_called();self.assertEqual(args.output.read_text(encoding='utf-8'),'existing input')
   args.output=Path(directory)/'other.json';args.output.with_suffix('.state').mkdir()
   with self.assertRaises(FileExistsError):benchmark.worker(args)
   args.output=Path(directory)/'source.py'
   with self.assertRaisesRegex(ValueError,'fresh .json'):benchmark.worker(args)
   args.output=Path(directory)/'valid.json';args.engine='gpu';args.profile='lifted'
   with self.assertRaisesRegex(ValueError,'unavailable'):benchmark.worker(args)
   self.assertFalse(args.output.exists());self.assertFalse((Path(directory)/'source.py').exists())
 def test_solved_claim_requires_independent_gold_witness(self):
  for codes in (None,[True]):
   with tempfile.TemporaryDirectory(prefix='benchmark-witness-',dir=benchmark.ROOT) as directory:
    args=self.args(directory)
    with patch('constraint_sat_session.solve',return_value={'status':'solved','complete':True,'codes':codes}),contextlib.redirect_stdout(io.StringIO()):
     with self.assertRaisesRegex(ValueError,'witness|domains'):benchmark.worker(args)
    self.assertFalse(args.output.exists())
 @requires_sat
 def test_tiny_real_sat_profiles_complete_and_preserve_independent_gold_answers(self):
  from search_problem import SearchProblem
  from search_options import normalize_options
  from test_constraint_dfs import simple_tiles,multiple_loops,double_face,brute
  for tiles in (simple_tiles(),multiple_loops()):
   problem=SearchProblem(tiles,double_face());expected=brute(problem,'gold');scope=WorkScope.for_problem(problem)
   initial=benchmark.initial_frontier(scope,'sat')
   for profile in benchmark.SAT_PROFILES.values():
    with tempfile.TemporaryDirectory(prefix='benchmark-sat-finite-',dir=benchmark.ROOT) as directory:
     state={'_coverage_path':Path(directory)/'coverage.sqlite'}
     answer=benchmark.run_sat(problem,seconds=5,seed=71,initial=initial,options=normalize_options(profile),learning_state=state)
     self.assertEqual(answer['status'],'solved' if expected else 'infeasible')
     self.assertFalse(answer['native_persistence']['enabled']);self.assertIsNone(state['_sat_session'].solver)
     if expected:self.assertIn(tuple(answer['codes']),{tuple(row) for row in expected});self.assertIsNone(benchmark.pending_frontier(scope,answer,'sat'))
     else:self.assertEqual(benchmark.pending_frontier(scope,answer,'sat'),())


class SATBenchmarkMatrixTests(unittest.TestCase):
 def test_sat_cases_profiles_dedupe_reverse_and_use_separate_budgets(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-sat-matrix-',dir=benchmark.ROOT) as directory:
   folder=Path(directory);data=folder/'input.json';data.write_text('{}',encoding='utf-8')
   argv=['benchmark_round25.py','--output',str(folder/'matrix'),'--data',str(data),'--suite','sat','--sat-profiles','baseline','lifted','baseline','--sat-cases','small','residual','original','small','--free-cell-counts','8','16','8','--finite-seconds','3','--seconds','7','--evidence-stage','frozen']
   jobs=[]
   def fake(command,**kwargs):
    value=lambda key:command[command.index(key)+1]
    case=value('--case');depth=int(value('--free-cells')) if case=='residual' else None
    jobs.append((value('--engine'),int(value('--seed')),case,depth,value('--profile'),float(value('--seconds'))))
    result={'status':'infeasible','complete':True}
    if case=='original':result['provenance']={'input_sha256':benchmark.hashlib.sha256(data.read_bytes()).hexdigest()}
    Path(value('--output')).write_text(json.dumps(result),encoding='utf-8')
    return SimpleNamespace(returncode=0,poll=lambda:0,communicate=lambda **kw:('',''))
   repo={'commit':'abc','dirty':False}
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'repository_snapshot',return_value=repo),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen',side_effect=fake),patch.object(benchmark.platform,'platform',return_value='test-platform'),contextlib.redirect_stdout(io.StringIO()):benchmark.main()
   finite=[(case,depth,profile,3.0) for case,depth in [('small',None),('residual',8),('residual',16)] for profile in ('baseline','lifted')]
   expected=[('sat',817,*row) for row in finite]+[('sat',291,*row) for row in reversed(finite)]
   expected += [('sat',seed,'original',None,profile,7.0) for seed,profiles in ((71,('baseline','lifted')),(197,('lifted','baseline'))) for profile in profiles]
   self.assertEqual(jobs,expected)
   report=json.loads((folder/'matrix/report.json').read_text(encoding='utf-8'))
   self.assertEqual(report['planned_cases'],16);self.assertEqual(report['sat_partition_boxes'],32);self.assertFalse(report['sat_profiles']['lifted']['native_sat_persistence'])
 def test_planned_case_or_summary_collision_prevents_any_child(self):
  for name in ('report.json','results.json','01-sat-small-lifted-817.state','00-sat-small-baseline-817.stdout.txt'):
   with tempfile.TemporaryDirectory(prefix='benchmark-plan-collision-',dir=benchmark.ROOT) as directory:
    folder=Path(directory);(folder/name).write_text('do not overwrite',encoding='utf-8')
    argv=['benchmark_round25.py','--output',directory,'--suite','sat','--sat-cases','small']
    with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={}),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen') as launch:
     with self.assertRaises(FileExistsError):benchmark.main()
    launch.assert_not_called();self.assertEqual((folder/name).read_text(encoding='utf-8'),'do not overwrite')
 def test_optimized_python_refuses_all_benchmark_entrypoints_before_output(self):
  for filename,extra in (('benchmark_round25.py',['--suite','finite']),('benchmark_learning_round25.py',['--candidate','1']),('benchmark_regions_round25.py',['--candidate','14']),('benchmark_gpu_round25.py',['--proposal','26'])):
   with tempfile.TemporaryDirectory(prefix='benchmark-optimize-',dir=benchmark.ROOT) as directory:
    output=Path(directory)/'result.json'
    result=subprocess.run([sys.executable,'-B','-O',str(benchmark.ROOT/filename),'--output',str(output),*extra],cwd=benchmark.ROOT,capture_output=True,text=True,timeout=20)
    self.assertNotEqual(result.returncode,0);self.assertIn('assert',result.stderr.lower());self.assertFalse(output.exists())



class HybridPipelineBenchmarkTests(unittest.TestCase):
 def checkpoint(self,problem,phase='sampling'):
  from constraint_work import digest
  payload={'hybrid_version':3,'scope':{'fingerprint':problem.fingerprint,'domains':[list(row) for row in problem.domains],
      'fixed':[[int(c),int(v)] for c,v in sorted(problem.fixed.items())],'target':'gold'},'phase':phase,'cursor':1}
  payload['digest']=digest(payload);return payload
 def args(self,folder,profile='combined'):
  return SimpleNamespace(case='small',seed=817,free_cells=8,engine='hybrid',workers=1,seconds=.01,
      profile=profile,replicas=4096,hybrid_backend='webgpu',output=Path(folder)/'case.json')
 def test_profiles_share_durable_contract_and_validate_for_hybrid_only(self):
  from search_options import normalize_options,validate_algorithm_options
  for name,profile in benchmark.HYBRID_PROFILES.items():
   self.assertIs(profile['durable_hybrid'],True);validate_algorithm_options('hybrid',normalize_options(profile))
  self.assertEqual(tuple(benchmark.DEFAULT_HYBRID_PROFILES),('baseline','legacy-sampling','repair','combined'))
  self.assertFalse(any(name in benchmark.HYBRID_PROFILES['baseline'] for name in ('ejection_chains','core_repair','hall_rematch')))
  self.assertTrue(benchmark.HYBRID_PROFILES['combined-cpu']['hall_rematch'])
 def test_worker_uses_pipeline_not_portable_coverage_and_hashes_v3_artifact(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-hybrid-',dir=benchmark.ROOT) as directory:
   args=self.args(directory);calls=[]
   def fake(problem,**kw):
    calls.append(kw);self.assertIsNone(kw['resume']);self.assertNotIn('work_frontier',kw)
    self.assertEqual(kw['learning_state']['_hybrid_run_seconds'],args.seconds)
    self.assertTrue(kw['options']['durable_hybrid']);self.assertTrue(kw['options']['core_repair'])
    self.assertEqual(kw['replicas'],4096);self.assertEqual(kw['backend'],'webgpu')
    checkpoint=self.checkpoint(problem,'repair');kw['progress']({'event':'checkpoint','payload':checkpoint})
    return {'status':'stopped','complete':False,'phase':'repair','nodes':0,'stats':{},'checkpoint':checkpoint,
      'gpu_sampling':{'initial_boards':4096,'move_proposals':262144,'dispatches':1,'sampling_experiments':{'guided_candidate_checks':17}},
      'local_repair':{'stats':{'attempts':1,'repair_native_calls':3},'experimental_repair':[]}}
   with (patch('gpu_sampling.solve',side_effect=fake),patch.object(benchmark,'initial_frontier',side_effect=AssertionError('portable root')),
        patch.object(benchmark,'pending_frontier',side_effect=AssertionError('coverage metric')),contextlib.redirect_stdout(io.StringIO())):
    benchmark.worker(args)
    with self.assertRaises(FileExistsError):benchmark.worker(args)
   report=json.loads(args.output.read_text());self.assertEqual(len(calls),1)
   for key in ('raw_space','initial_frontier','same_full_original_root','checkpoint'):self.assertNotIn(key,report)
   self.assertEqual(report['solver_engine'],'hybrid');self.assertEqual(report['replicas'],4096)
   work=report['pipeline_work'];self.assertEqual(work['gpu_sampling']['move_proposals'],262144);self.assertEqual(work['sampling_experiments']['guided_candidate_checks'],17)
   self.assertEqual(work['local_repair']['stats']['repair_native_calls'],3);self.assertEqual(work['exact_dfs']['nodes'],0)
   artifact=report['checkpoint_artifact'];raw=Path(artifact['path']).read_bytes()
   self.assertEqual(artifact['sha256'],benchmark.hashlib.sha256(raw).hexdigest());self.assertEqual(json.loads(raw)['hybrid_version'],3)
   self.assertEqual(report['checkpoint_publication']['published_count'],2)
   self.assertFalse((args.output.with_suffix('.state')/'initial-frontier.json').exists())
   self.assertIn('symmetry',json.loads(Path(report['pipeline_input']['path']).read_text())['search_contract'])
 def test_slices_keep_same_state_and_whole_budget_and_close_each_owner_once(self):
  from search_problem import SearchProblem
  from test_constraint_dfs import tetrahedron_tiles
  problem=SearchProblem(*tetrahedron_tiles());calls=[];closed=[];state={};queued=[]
  publication=SimpleNamespace(progress=lambda event:None,publish=queued.append,failed=lambda:False)
  owner=SimpleNamespace(close=lambda:closed.append(True))
  def fake(p,**kw):
   calls.append(kw);self.assertIs(kw['learning_state'],state);self.assertEqual(state['_hybrid_run_seconds'],61)
   state['_shared_sat']=owner;state['_mask_conflicts']=owner
   return {'status':'timeout' if len(calls)<3 else 'stopped','complete':False,'phase':'exact_dfs','nodes':len(calls),
      'stats':{},'checkpoint':self.checkpoint(p,'exact_dfs')}
  with patch('gpu_sampling.solve',side_effect=fake),patch.object(benchmark.time,'monotonic',side_effect=[0,30,60]):
   result=benchmark.run_hybrid(problem,seconds=61,seed=71,replicas=128,backend='cuda',options={'durable_hybrid':True},learning_state=state,publication=publication)
  self.assertEqual([row['seconds'] for row in calls],[30,30,1]);self.assertIsNone(calls[0]['resume']);self.assertEqual(len(queued),3)
  self.assertIs(calls[1]['resume'],queued[0]);self.assertEqual(closed,[True]);self.assertEqual(result['benchmark_hybrid_slices'],3)
 def test_failure_closes_helpers_and_preserves_original_error(self):
  state={};closed=[];publication=SimpleNamespace(progress=lambda e:None,publish=lambda p:None,failed=lambda:False)
  def fake(*a,**kw):
   state['_mask_conflicts']=SimpleNamespace(close=lambda:closed.append(True));raise RuntimeError('native failure')
  with patch('gpu_sampling.solve',side_effect=fake):
   with self.assertRaisesRegex(RuntimeError,'native failure'):
    benchmark.run_hybrid(None,seconds=1,seed=0,replicas=128,backend='auto',options={'durable_hybrid':True},learning_state=state,publication=publication)
  self.assertEqual(closed,[True])
 def test_incomplete_and_heuristic_infeasible_results_are_rejected(self):
  publication=SimpleNamespace(progress=lambda e:None,publish=lambda p:None,failed=lambda:False)
  from search_problem import SearchProblem
  from test_constraint_dfs import tetrahedron_tiles
  problem=SearchProblem(*tetrahedron_tiles())
  for result in ({'status':'timeout','complete':False},{'status':'infeasible','complete':False,'phase':'exact_dfs'},
                 {'status':'infeasible','complete':True,'phase':'repair'}):
   with self.subTest(result=result),patch('gpu_sampling.solve',return_value=result):
    with self.assertRaises(ValueError):benchmark.run_hybrid(problem,seconds=1,seed=0,replicas=128,backend='auto',options={'durable_hybrid':True},learning_state={},publication=publication)
  saved=self.checkpoint(problem);saved['scope']['domains'][0]=[]
  with patch('gpu_sampling.solve',return_value={'status':'stopped','complete':False,'checkpoint':saved}):
   with self.assertRaisesRegex(ValueError,'continuation'):benchmark.run_hybrid(problem,seconds=1,seed=0,replicas=128,backend='auto',options={'durable_hybrid':True},learning_state={},publication=publication)
 def test_gold_result_independently_validates_and_missing_or_invalid_witness_fails(self):
  from search_fixtures import planted_gold
  reference=planted_gold(817,size=1,extra_edges=3)[2]
  for mode in ('valid','missing','invalid'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory(prefix='benchmark-hybrid-gold-',dir=benchmark.ROOT) as directory:
    args=self.args(directory,'baseline')
    def fake(problem,**kw):
     result={'status':'solved','complete':True,'phase':'sampling','nodes':0,'stats':{}}
     if mode!='missing':result['codes']=list(reference) if mode=='valid' else [0]*problem.n
     return result
    with patch('gpu_sampling.solve',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):
     if mode=='valid':benchmark.worker(args)
     else:
      with self.assertRaises(ValueError):benchmark.worker(args)
    if mode=='valid':self.assertTrue(json.loads(args.output.read_text())['independent_validation']['valid'])
    else:self.assertFalse(args.output.exists())
 def test_matrix_is_opt_in_dedupes_bounds_and_reverses_second_seed(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-hybrid-matrix-',dir=benchmark.ROOT) as directory:
   folder=Path(directory);data=folder/'data.json';data.write_text('{}');jobs=[]
   argv=['benchmark_round25.py','--output',str(folder/'matrix'),'--data',str(data),'--suite','hybrid',
     '--hybrid-cases','small','residual','original','small','--hybrid-profiles','baseline','repair','baseline',
     '--hybrid-replica-counts','128','131072','128','--hybrid-backend','webgpu','--free-cell-counts','8','16','8',
     '--finite-seconds','3','--seconds','7','--evidence-stage','frozen']
   def fake(command,**kw):
    value=lambda key:command[command.index(key)+1]
    self.assertEqual(value('--engine'),'hybrid');self.assertEqual(value('--hybrid-backend'),'webgpu')
    case=value('--case');depth=int(value('--free-cells')) if case=='residual' else None
    jobs.append((int(value('--seed')),case,depth,value('--profile'),int(value('--replicas')),float(value('--seconds'))))
    result={'status':'infeasible','complete':True}
    if case=='original':result['provenance']={'input_sha256':benchmark.hashlib.sha256(data.read_bytes()).hexdigest()}
    Path(value('--output')).write_text(json.dumps(result));return SimpleNamespace(returncode=0,poll=lambda:0,communicate=lambda **kw:('',''))
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={'commit':'abc','dirty':False}),patch.object(benchmark.subprocess,'Popen',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):benchmark.main()
   first=[(case,depth,profile,lanes,3.) for case,depth in [('small',None),('residual',8),('residual',16)] for lanes in (128,131072) for profile in ('baseline','repair')]
   originals=[('original',None,profile,lanes,7.) for lanes in (128,131072) for profile in ('baseline','repair')]
   self.assertEqual(jobs,[(817,*row) for row in first]+[(291,*row) for row in reversed(first)]+[(71,*row) for row in originals]+[(197,*row) for row in reversed(originals)])
   report=json.loads((folder/'matrix/report.json').read_text());self.assertEqual(report['planned_cases'],32)
   self.assertEqual(report['hybrid_replica_counts'],[128,131072]);self.assertEqual(report['hybrid_backend'],'webgpu')
   self.assertTrue(all(row['replicas'] in (128,131072) for row in report['results']))
 def test_all_suite_does_not_silently_add_hybrid_and_bad_limits_fail_before_launch(self):
  with tempfile.TemporaryDirectory(prefix='benchmark-hybrid-optin-',dir=benchmark.ROOT) as directory:
   folder=Path(directory);data=folder/'data.json';data.write_text('{}');engines=[]
   def fake(command,**kw):
    engine=command[command.index('--engine')+1];engines.append(engine)
    Path(command[command.index('--output')+1]).write_text(json.dumps({'status':'infeasible','complete':True,'provenance':{'input_sha256':benchmark.hashlib.sha256(data.read_bytes()).hexdigest()}}))
    return SimpleNamespace(returncode=0,poll=lambda:0,communicate=lambda **kw:('',''))
   argv=['benchmark_round25.py','--output',str(folder/'matrix'),'--data',str(data),'--suite','all','--profiles','baseline','--finite-cases','small']
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={}),patch.object(benchmark.subprocess,'Popen',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):benchmark.main()
   self.assertTrue(engines);self.assertNotIn('hybrid',engines)
  for extra in (['--hybrid-replica-counts','0'],['--hybrid-replica-counts','131073'],['--replicas','131073'],['--finite-seconds','nan'],['--hybrid-profiles','adaptive','--hybrid-replica-counts','7']):
   with tempfile.TemporaryDirectory(prefix='benchmark-hybrid-bound-',dir=benchmark.ROOT) as directory:
    argv=['benchmark_round25.py','--output',directory,'--suite','hybrid',*extra]
    with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark.subprocess,'Popen') as launch,contextlib.redirect_stderr(io.StringIO()):
     with self.assertRaises(SystemExit):benchmark.main()
    launch.assert_not_called()

if __name__=='__main__':unittest.main()
