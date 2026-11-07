"""Application integration of reviewed R25 engine capabilities and persistence."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import constraint_search as runner
from constraint_shared_conflicts import VerifiedConflictStore
from constraint_work import WorkScope,frontier_snapshot,digest
from gpu_domain_runner import COUNTERS
from test_constraint_search import pair_problem

class ApplicationRound25Tests(unittest.TestCase):
    def test_gpu_clause_flag_owns_original_scope_verifiers_and_closes_them(self):
        data,problem=pair_problem();scope=WorkScope.for_problem(problem)
        for enabled in (False,True):
            with self.subTest(enabled=enabled),tempfile.TemporaryDirectory(prefix='r25-app-') as directory:
                root=Path(directory).resolve();source=root/'tiles.json';output=root/'runtime';source.write_text(json.dumps(data),encoding='utf-8')
                stores=[];executed=[]
                def create(*args,**kwargs):
                    store=VerifiedConflictStore(*args,**kwargs);stores.append(store);return store
                def fake_solver(actual,algorithm,**kwargs):
                    self.assertIs(actual,problem);self.assertEqual(algorithm,'gpu-domains');executed.append(True)
                    checkpoint_dir=Path(kwargs['learning_state']['_gpu_checkpoint_dir'])
                    self.assertEqual(checkpoint_dir,output/'searches/gpu-domains-gold/gpu-checkpoints')
                    self.assertFalse(checkpoint_dir.exists(),'Provisioning a path alone must not allocate a store.')
                    active=kwargs['learning_state'].get('_shared_conflicts')
                    self.assertEqual(active is not None,enabled)
                    if enabled:self.assertTrue(active.matches(problem,'gold'));self.assertEqual(active.records(),[])
                    payload={'version':1,'scope':scope.payload(),'queue':[[hex(x) for x in scope.root]],'native':None,
                             'totals':dict.fromkeys(COUNTERS,0),'completed_lanes':0,'active_stats':{},'frontier':frontier_snapshot(scope,[scope.root])}
                    payload['digest']=digest(payload)
                    return {'status':'stopped','complete':False,'nodes':0,'stats':{},'checkpoint':payload,
                            'shared_conflicts':active.snapshot() if active is not None else None}
                args=['--algorithm','gpu-domains','--seconds','5','--data',str(source),'--output',str(output),
                      '--search-options',json.dumps({'gpu_shared_clauses':enabled})]
                with patch.object(runner,'SearchProblem',return_value=problem),patch.object(runner,'validate_runtime_selection'),patch.object(runner,'run_solver',side_effect=fake_solver),patch('constraint_shared_conflicts.VerifiedConflictStore',side_effect=create),contextlib.redirect_stdout(io.StringIO()):runner.main(args)
                self.assertEqual(executed,[True]);self.assertEqual(len(stores),2 if enabled else 0)
                self.assertTrue(all(store._closed for store in stores));self.assertFalse((output/'run.lock').exists())
                status=json.loads((output/'status.json').read_text(encoding='utf-8'))
                self.assertEqual(status['state'],'stopped');self.assertEqual('shared_conflicts' in status,enabled)

    def test_gpu_references_publish_exact_frontier_and_resume_with_feature_disabled(self):
        from gpu_checkpoint_round25 import CheckpointStore,is_reference
        data,problem=pair_problem();scope=WorkScope.for_problem(problem)
        with tempfile.TemporaryDirectory(prefix='r25-app-reference-') as directory:
            root=Path(directory).resolve();source=root/'tiles.json';output=root/'runtime';source.write_text(json.dumps(data),encoding='utf-8')
            folder=output/'searches/gpu-domains-gold';plain={'version':1,'scope':scope.payload(),'queue':[[hex(x) for x in scope.root]],'native':None,
                'totals':dict.fromkeys(COUNTERS,0),'completed_lanes':0,'active_stats':{},'frontier':frontier_snapshot(scope,[scope.root])}
            plain['digest']=digest(plain);calls=[];references=[]
            def fake(actual,algorithm,**kwargs):
                calls.append(kwargs)
                if len(calls)==1:
                    self.assertIsNone(kwargs['resume']);self.assertTrue(kwargs['options']['gpu_incremental_checkpoint'])
                    store=CheckpointStore(kwargs['learning_state']['_gpu_checkpoint_dir'],scope.payload())
                    saved=store.write(plain,{});references.append(saved)
                    kwargs['progress']({'event':'checkpoint','payload':saved})
                else:
                    self.assertFalse(kwargs['options']['gpu_incremental_checkpoint'])
                    saved=kwargs['resume'];self.assertFalse(is_reference(saved))
                    self.assertEqual(set(saved),set(plain)|{'checkpoint_origin'})
                    self.assertEqual(saved['checkpoint_origin'],references[0])
                    self.assertEqual({k:v for k,v in saved.items() if k not in ('digest','checkpoint_origin')},{k:v for k,v in plain.items() if k!='digest'})
                    self.assertEqual(saved['digest'],digest({k:v for k,v in saved.items() if k!='digest'}))
                return {'status':'stopped','complete':False,'nodes':0,'stats':{},'checkpoint':saved,
                        'gpu_checkpoint_status':'incremental' if len(calls)==1 else 'off','gpu_checkpoint_note':'test publication' if len(calls)==1 else ''}
            base=['--algorithm','gpu-domains','--seconds','5','--data',str(source),'--output',str(output)]
            with patch.object(runner,'SearchProblem',return_value=problem),patch.object(runner,'validate_runtime_selection'),patch.object(runner,'run_solver',side_effect=fake),patch('gpu_checkpoint_codec_round25.decode_native',side_effect=AssertionError('App must not decode native arrays')),contextlib.redirect_stdout(io.StringIO()):
                runner.main(base+['--search-options',json.dumps({'gpu_incremental_checkpoint':True})])
                envelope=json.loads((folder/'checkpoint.json').read_text(encoding='utf-8'))
                self.assertTrue(is_reference(envelope['payload']))
                status=json.loads((output/'status.json').read_text(encoding='utf-8'))
                self.assertEqual(status['gpu_checkpoint_status'],'incremental');self.assertEqual(status['gpu_checkpoint_note'],'test publication')
                self.assertEqual(json.loads((folder/'remaining-work.json').read_text(encoding='utf-8')),plain['frontier'])
                runner.main(base+['--resume','--search-options',json.dumps({'gpu_incremental_checkpoint':False})])
            self.assertEqual(len(calls),2);self.assertFalse((output/'run.lock').exists())
            self.assertEqual(json.loads((folder/'checkpoint.json').read_text(encoding='utf-8'))['payload'],calls[1]['resume'])
            self.assertEqual(json.loads((output/'status.json').read_text(encoding='utf-8'))['gpu_checkpoint_status'],'off')

    def test_corrupt_gpu_generation_cannot_replace_saved_pending_work(self):
        from gpu_checkpoint_round25 import CheckpointStore
        data,problem=pair_problem();scope=WorkScope.for_problem(problem)
        with tempfile.TemporaryDirectory(prefix='r25-app-corrupt-') as directory:
            root=Path(directory).resolve();source=root/'tiles.json';output=root/'runtime';source.write_text(json.dumps(data),encoding='utf-8')
            folder=output/'searches/gpu-domains-gold';refs=[]
            def fake(actual,algorithm,**kwargs):
                plain={'version':1,'scope':scope.payload(),'queue':[[hex(x) for x in scope.root]],'native':None,
                    'totals':dict.fromkeys(COUNTERS,0),'completed_lanes':0,'active_stats':{},'frontier':frontier_snapshot(scope,[scope.root])}
                plain['digest']=digest(plain)
                saved=CheckpointStore(kwargs['learning_state']['_gpu_checkpoint_dir'],scope.payload()).write(plain,{})
                refs.append(saved);return {'status':'stopped','complete':False,'nodes':0,'stats':{},'checkpoint':saved}
            args=['--algorithm','gpu-domains','--seconds','5','--data',str(source),'--output',str(output),'--search-options',json.dumps({'gpu_incremental_checkpoint':True})]
            with patch.object(runner,'SearchProblem',return_value=problem),patch.object(runner,'validate_runtime_selection'),patch.object(runner,'run_solver',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):runner.main(args)
            saved_bytes=(folder/'checkpoint.json').read_bytes();frontier_bytes=(folder/'remaining-work.json').read_bytes()
            (folder/'gpu-checkpoints/manifests'/str(refs[0]['generation']+'.json')).write_text('{}',encoding='utf-8')
            with patch.object(runner,'SearchProblem',return_value=problem),patch.object(runner,'validate_runtime_selection'),patch.object(runner,'run_solver') as solve,contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(SystemExit,'checksum'):runner.main(args+['--resume'])
            solve.assert_not_called();self.assertFalse((output/'run.lock').exists())
            self.assertEqual((folder/'checkpoint.json').read_bytes(),saved_bytes)
            self.assertEqual((folder/'remaining-work.json').read_bytes(),frontier_bytes)

if __name__=='__main__':unittest.main()
