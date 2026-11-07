"""Sampler benchmark accounting/provenance tests; no GPU work is launched."""
import contextlib,copy,io,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import benchmark_hybrid_round25 as bench

class SamplerBenchmarkTests(unittest.TestCase):
    def test_matrix_deduplicates_and_reverses_all_configs_for_second_seed(self):
        got=bench.matrix_configurations(['baseline','exchange','baseline'],['gold','gold'],[128,4096,128],[71,197,71])
        first=[(71,'baseline','gold',128),(71,'exchange','gold',128),(71,'baseline','gold',4096),(71,'exchange','gold',4096)]
        self.assertEqual(got,first+[(197,*row[1:]) for row in reversed(first)])

    def test_signature_ignores_only_time_restore_bookkeeping_and_outer_digest(self):
        original={'native':{'bytes':'abc'},'archive':[],'elapsed':1.0,'counts':{'sampler_restores':0,'move_proposals':64},'digest':'a'}
        changed=copy.deepcopy(original);changed.update(elapsed=8.0,digest='b');changed['counts']['sampler_restores']=1
        self.assertEqual(bench.continuation_signature(original),bench.continuation_signature(changed))
        changed['native']['bytes']='def'
        self.assertNotEqual(bench.continuation_signature(original),bench.continuation_signature(changed))
        self.assertEqual(original['counts']['sampler_restores'],0)

    def test_resume_pair_requires_exact_artifact_work_and_native_state(self):
        artifact={'path':'saved.json','sha256':'sha','bytes':10}
        full={'continuation_signature':'same','completed_dispatches':16}
        prefix={'snapshot_artifact':artifact,'cumulative_dispatches':8,'completed_dispatches':8,'status':'paused'}
        resumed={'continuation_signature':'same','resume_artifact':artifact,'initial_dispatches':8,'completed_dispatches':8,'report':{'sampler_restores':1}}
        self.assertEqual(bench.validate_pair(full,prefix,resumed)['completed_dispatches_replayed'],0)
        for key,value in [('continuation_signature','different'),('resume_artifact',{}),('initial_dispatches',7),('completed_dispatches',9),('report',{'sampler_restores':2})]:
            with self.subTest(key=key),self.assertRaises(ValueError):bench.validate_pair(full,prefix,{**resumed,key:value})
        solved={**prefix,'status':'solved'}
        self.assertTrue(bench.validate_pair({**full,'completed_dispatches':8},solved,{**resumed,'completed_dispatches':0,'report':{'sampler_restores':0}})['same_final_state'])

    def test_frozen_dirty_source_fails_before_gpu_or_process_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            argv=['benchmark_hybrid_round25.py','--output',directory,'--evidence-stage','frozen']
            with patch.object(bench.sys,'argv',argv),patch.object(bench,'source_snapshot',return_value={}),patch.object(bench,'runtime_snapshot',return_value={}),patch.object(bench,'repository_snapshot',return_value={'dirty':True,'commit':'abc'}),patch.object(bench,'launch') as launch:
                with self.assertRaisesRegex(RuntimeError,'clean committed'):bench.main()
            launch.assert_not_called()

    def test_optimized_python_refuses_before_gpu_work(self):
        for args,env in [(['-O'],None),([],{**bench.os.environ,'PYTHONOPTIMIZE':'1'})]:
            result=subprocess.run([sys.executable,*args,'-B',str(bench.ROOT/'benchmark_hybrid_round25.py'),'--help'],capture_output=True,text=True,env=env,timeout=30)
            self.assertNotEqual(result.returncode,0);self.assertIn('require assertions',result.stderr)

    def test_serial_fresh_process_matrix_retains_exact_resume_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);data=folder/'input.json';data.write_text('{}',encoding='utf-8');output=folder/'matrix';calls=[]
            def fake(args,path,**kw):
                calls.append(kw);prefix=not kw['finish'];resumed=kw.get('resume') is not None
                artifact={'path':str(output/'saved.json'),'sha256':'abc','bytes':3}
                return {'status':'paused' if prefix else 'sampled','continuation_signature':'prefix' if prefix else 'same',
                    'snapshot_artifact':artifact,'resume_artifact':artifact if resumed else None,
                    'initial_dispatches':2 if resumed else 0,'completed_dispatches':kw['dispatches'],
                    'cumulative_dispatches':4 if resumed or not prefix else 2,
                    'input_sha256':bench.hashlib.sha256(data.read_bytes()).hexdigest(),
                    'report':{'sampler_restores':int(resumed),'best_score':200,'move_proposals':512}}
            argv=['benchmark_hybrid_round25.py','--data',str(data),'--output',str(output),'--profiles','baseline','--objectives','gold','--replica-counts','128','--seeds','71','--dispatches','4','--resume-comparison','--evidence-stage','frozen']
            with patch.object(bench.sys,'argv',argv),patch.object(bench,'source_snapshot',return_value={}),patch.object(bench,'runtime_snapshot',return_value={}),patch.object(bench,'repository_snapshot',return_value={'dirty':False,'commit':'abc'}),patch.object(bench,'launch',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):bench.main()
            self.assertEqual([row['dispatches'] for row in calls],[4,2,2]);self.assertTrue(calls[2]['resume'])
            report=json.loads((output/'report.json').read_text(encoding='utf-8'));self.assertEqual(report['completed_cases'],1)
            self.assertTrue(report['results'][0]['comparison']['same_final_state'])

    def test_source_drift_and_failed_child_do_not_leave_accepted_report(self):
        for fail in ('drift','child'):
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as directory:
                folder=Path(directory);data=folder/'input.json';data.write_text('{}',encoding='utf-8');output=folder/'matrix'
                argv=['benchmark_hybrid_round25.py','--data',str(data),'--output',str(output),'--profiles','baseline','--objectives','gold','--replica-counts','128','--seeds','71']
                row={'input_sha256':bench.hashlib.sha256(data.read_bytes()).hexdigest(),'report':{'best_score':200,'move_proposals':512}}
                with patch.object(bench.sys,'argv',argv),patch.object(bench,'source_snapshot',side_effect=[{}, {'changed':True}]),patch.object(bench,'runtime_snapshot',return_value={}),patch.object(bench,'repository_snapshot',return_value={'dirty':False,'commit':'abc'}),patch.object(bench,'launch',side_effect=RuntimeError('child failed') if fail=='child' else None,return_value=row),contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaises(RuntimeError):bench.main()
                self.assertFalse((output/'report.json').exists())

    def test_worker_closes_session_and_rejects_incomplete_fixed_work(self):
        from test_constraint_search import pair_problem
        data,problem=pair_problem()
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);source=folder/'input.json';source.write_text(json.dumps(data),encoding='utf-8')
            args=SimpleNamespace(output=folder/'result.json',data=source,resume=None,seed=71,replicas=128,backend='cuda',
                objective='edges',profile='baseline',limit_seconds=1,dispatches=2,finish=True)
            class Fake:
                done=False;launches=0;closed=False
                def report(self):return {'dispatches':0}
                def advance(self,*args,**kwargs):pass
                def close(self):self.closed=True
            fake=Fake()
            with patch('search_problem.SearchProblem',return_value=problem),patch('gpu_sampling_resume_round25.SamplerSession',return_value=fake):
                with self.assertRaisesRegex(RuntimeError,'time bound'):bench.worker(args)
            self.assertTrue(fake.closed);self.assertFalse(args.output.exists())

    def test_child_failure_stops_and_preserves_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);args=SimpleNamespace(data=folder/'input.json',backend='cuda',limit_seconds=1)
            child=SimpleNamespace(returncode=1,poll=lambda:1,communicate=lambda **kw:('diagnostic','failure'))
            with patch.object(bench.subprocess,'Popen',return_value=child),patch.object(bench,'terminate_tree',return_value='owned cleanup') as cleanup:
                with self.assertRaisesRegex(RuntimeError,'no further cases'):bench.launch(args,folder/'case.json',profile='baseline',objective='gold',seed=71,replicas=128,dispatches=2,finish=True)
            cleanup.assert_called_once_with(child);self.assertEqual((folder/'case.stderr.txt').read_text(),'failure')

    def test_exited_child_with_malformed_json_still_attempts_owned_group_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);args=SimpleNamespace(data=folder/'input.json',backend='cuda',limit_seconds=1)
            path=folder/'case.json'
            def fake(*args,**kwargs):
                path.write_text('malformed',encoding='utf-8')
                return SimpleNamespace(returncode=0,poll=lambda:0,communicate=lambda **kw:('',''))
            with patch.object(bench.subprocess,'Popen',side_effect=fake),patch.object(bench,'terminate_tree',return_value='owned group reaped') as cleanup:
                with self.assertRaises(json.JSONDecodeError):bench.launch(args,path,profile='baseline',objective='gold',seed=71,replicas=128,dispatches=2,finish=True)
            cleanup.assert_called_once()

class AdaptivePopulationTests(unittest.TestCase):
    def test_invalid_population_stops_before_worker_or_matrix_launch(self):
        for extra in (['--profiles','adaptive','--replica-counts','7'],['--worker','--profile','adaptive','--replicas','7']):
            with self.subTest(extra=extra),tempfile.TemporaryDirectory() as directory:
                argv=['benchmark_hybrid_round25.py','--output',str(Path(directory)/'out'),*extra]
                with patch.object(bench.sys,'argv',argv),patch.object(bench,'worker') as worker,patch.object(bench,'launch') as launch,contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):bench.main()
                worker.assert_not_called();launch.assert_not_called()
                self.assertFalse((Path(directory)/'out').exists())

if __name__=='__main__':unittest.main()
