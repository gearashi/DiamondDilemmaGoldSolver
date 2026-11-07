"""Checkpoint evidence includes immutable dependencies without starting a GPU."""
import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import benchmark_round25 as benchmark
from benchmark_checkpoint_artifacts import retained_store_snapshot,reopen_gpu_checkpoint
from gpu_checkpoint_round25 import CheckpointStore,checkpoint_frontier
from gpu_checkpoint_codec_round25 import encode_native
from test_gpu_checkpoint_round25 import fixture,payload
from constraint_work import WorkScope,frontier_snapshot,digest


class CheckpointArtifactTests(unittest.TestCase):
    def make(self,root):
        engine,scope,snapshot=fixture();spec,blobs,_=encode_native(engine,snapshot)
        directory=root/'store';store=CheckpointStore(directory,scope.payload())
        ref=store.write(payload(scope,spec),blobs)
        artifact=benchmark.write_artifact(root/'checkpoint.json',ref)
        return scope,directory,ref,artifact
    def test_saved_reference_and_all_retained_dependencies_are_hashed_without_decoding(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-receipt-',dir=benchmark.ROOT) as folder:
            scope,directory,ref,artifact=self.make(Path(folder))
            with patch('gpu_checkpoint_codec_round25.zlib.decompressobj',side_effect=AssertionError('No GPU arrays should be decoded')):
                result=reopen_gpu_checkpoint(artifact,directory,scope)
            self.assertTrue(result['verified']);self.assertTrue(result['referenced_store'])
            self.assertFalse(result['native_arrays_decoded']);self.assertFalse(result['search_resumed'])
            report=retained_store_snapshot(directory)
            self.assertEqual(report['file_count'],len(report['files']))
            self.assertEqual(report['logical_file_bytes'],sum(row['bytes'] for row in report['files'].values()))
            for name,row in report['files'].items():
                raw=(directory/name).read_bytes();self.assertEqual(row['bytes'],len(raw))
                self.assertEqual(row['sha256'],benchmark.hashlib.sha256(raw).hexdigest())
            self.assertIn('identity.json',report['files']);self.assertIn('HEAD.json',report['files'])
            self.assertTrue(any(name.startswith('chunks/') for name in report['files']))
    def test_missing_chunk_never_becomes_valid_saved_evidence(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-missing-',dir=benchmark.ROOT) as folder:
            scope,directory,ref,artifact=self.make(Path(folder))
            next((directory/'chunks').glob('*.bin')).unlink()
            self.assertEqual(checkpoint_frontier(ref,directory),frontier_snapshot(scope,[scope.root]))
            with self.assertRaisesRegex(ValueError,'Missing'):reopen_gpu_checkpoint(artifact,directory,scope)
            with self.assertRaisesRegex(ValueError,'exact local store'):
                benchmark.pending_frontier(scope,{'status':'stopped','checkpoint':ref},'gpu')
    def test_artifact_corruption_scope_and_store_bounds_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-guards-',dir=benchmark.ROOT) as folder:
            scope,directory,ref,artifact=self.make(Path(folder))
            other=WorkScope('different',scope.target,scope.original,scope.root)
            with self.assertRaisesRegex(ValueError,'scope differs'):reopen_gpu_checkpoint(artifact,directory,other)
            with patch('gpu_checkpoint_round25.DEFAULT_STORE_BYTES',1):
                with self.assertRaisesRegex(ValueError,'declared bound'):retained_store_snapshot(directory)
            with Path(artifact['path']).open('ab') as stream:stream.write(b' ')
            with self.assertRaisesRegex(ValueError,'differs from its receipt'):reopen_gpu_checkpoint(artifact,directory,scope)
            self.assertIsNone(retained_store_snapshot(Path(folder)/'absent'))
            with self.assertRaisesRegex(ValueError,'regular local directory'):retained_store_snapshot(Path(artifact['path']))
    def test_benchmark_publishes_and_reopens_current_reference_with_dependency_receipt(self):
        with tempfile.TemporaryDirectory(prefix='checkpoint-worker-',dir=benchmark.ROOT) as folder:
            args=SimpleNamespace(case='small',seed=817,free_cells=8,engine='gpu',workers=1,seconds=.01,profile='reuse',replicas=128,output=Path(folder)/'case.json')
            references=[]
            def fake(problem,**kwargs):
                scope=WorkScope.for_problem(problem,'gold');store=CheckpointStore(kwargs['learning_state']['_gpu_checkpoint_dir'],scope.payload())
                value={'version':1,'scope':scope.payload(),'native':None,'frontier':kwargs['work_frontier'],'totals':{'decisions':0}}
                value['digest']=digest(value);first=store.write(value,{})
                kwargs['progress']({'event':'checkpoint','payload':first})
                second=store.write(value,{});references.extend([first,second])
                return {'status':'stopped','complete':False,'nodes':0,'stats':{},'checkpoint':second}
            with patch.object(benchmark,'warmup_domain',return_value={'initialization_only':True,'search_steps':0}),patch('gpu_domain_runner.solve',side_effect=fake),contextlib.redirect_stdout(io.StringIO()):benchmark.worker(args)
            report=json.loads(args.output.read_text(encoding='utf-8'))
            self.assertEqual(report['raw_space']['closed'],'0');self.assertTrue(report['checkpoint_reopen']['referenced_store'])
            publication=report['checkpoint_publication'];self.assertEqual(publication['published_count'],2)
            self.assertFalse(publication['writer_alive']);self.assertTrue(publication['closed'])
            self.assertEqual(json.loads(Path(publication['path']).read_text(encoding='utf-8')),references[-1])
            dependencies=json.loads(Path(report['checkpoint_dependencies_artifact']['path']).read_text(encoding='utf-8'))
            for ref in references:self.assertIn('manifests/'+ref['generation']+'.json',dependencies['files'])


if __name__=='__main__':unittest.main()
