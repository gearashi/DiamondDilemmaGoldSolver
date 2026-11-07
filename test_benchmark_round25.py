"""Benchmark accounting and cleanup regressions; no search workload is launched."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import benchmark_round25 as benchmark
from constraint_work import WorkScope,frontier_snapshot


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
   def cleanup(owned):
    self.assertIs(owned,child)
    self.assertTrue((output/'report.json').exists(),'Partial report precedes cleanup')
    self.assertTrue((output/'00-cpu-small-baseline-817.stderr.txt').exists())
    return 'Descendant cleanup cannot be confirmed.'
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark.subprocess,'Popen',return_value=child) as launch,patch.object(benchmark,'terminate_tree',side_effect=cleanup),contextlib.redirect_stdout(io.StringIO()):
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
   with patch.object(benchmark.sys,'argv',argv),patch.object(benchmark,'source_hashes',return_value={}),patch.object(benchmark.subprocess,'Popen',return_value=child) as launch,patch.object(benchmark,'terminate_tree') as cleanup:
    with self.assertRaises(KeyboardInterrupt):benchmark.main()
   launch.assert_called_once();cleanup.assert_called_once_with(child)


if __name__=='__main__':unittest.main()
