"""Checkpoint publication contracts with barriers and isolated CPU-only I/O."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import benchmark_checkpoint_publication as publication
from io_utils import atomic_json as actual_atomic_json

ROOT=Path(__file__).resolve().parent


class CheckpointPublicationTests(unittest.TestCase):
    def folder(self):
        return tempfile.TemporaryDirectory(prefix='checkpoint-publication-test-',dir=ROOT)

    def test_fifo_event_shapes_identity_and_actual_serialized_byte_counts(self):
        snapshots=[{'value':1},{'value':2,'text':'café'},{'value':3}]
        seen=[];sizes=[]
        def recording(path,payload):
            seen.append(payload)
            actual_atomic_json(path,payload)
            sizes.append(path.stat().st_size)
        with self.folder() as folder,patch.object(publication,'atomic_json',side_effect=recording):
            path=Path(folder)/'published.json'
            with publication.CheckpointPublisher(path,queue_capacity=2) as writer:
                self.assertFalse(writer.progress({'event':'progress','nodes':10}))
                self.assertFalse(writer.publish(None))
                writer.progress({'event':'checkpoint','payload':snapshots[0],'checkpoint':snapshots[1]})
                writer.progress({'event':'progress','checkpoint':snapshots[1]})
                writer.publish(snapshots[2]);writer.publish(snapshots[2])
            report=writer.report()
            self.assertEqual(len(seen),4)
            for actual,expected in zip(seen,[*snapshots,snapshots[-1]]):self.assertIs(actual,expected)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),snapshots[-1])
            self.assertEqual(report['submitted_count'],4);self.assertEqual(report['published_count'],4)
            self.assertEqual(report['published_bytes'],sum(sizes));self.assertEqual(report['final_file_bytes'],sizes[-1])
            self.assertEqual(report['discarded_count'],0);self.assertEqual(report['failed_write_count'],0)
            self.assertLessEqual(report['max_queue_depth'],2);self.assertGreaterEqual(report['write_seconds'],0)
            self.assertTrue(report['closed']);self.assertFalse(report['writer_alive']);self.assertFalse(writer.failed())
            self.assertFalse(path.with_name(path.name+'.writer.lock').exists())
            self.assertIn('not physical SSD writes',report['scope'])

    def test_queue_capacity_backpressures_without_copy_or_coalescing(self):
        entered=threading.Event();release=threading.Event();waiting=threading.Event();finished=threading.Event()
        seen=[];errors=[]
        def blocked(path,payload):
            seen.append(payload)
            if len(seen)==1:
                entered.set()
                if not release.wait(5):raise RuntimeError('test barrier was not released')
            actual_atomic_json(path,payload)
        with self.folder() as folder,patch.object(publication,'atomic_json',side_effect=blocked):
            writer=publication.CheckpointPublisher(Path(folder)/'state.json',queue_capacity=1)
            producer=None
            try:
                with writer:
                    writer.publish({'value':1});self.assertTrue(entered.wait(5));writer.publish({'value':2})
                    original_wait=writer._condition.wait
                    def observed_wait(timeout=None):
                        if threading.current_thread() is producer:waiting.set()
                        return original_wait(timeout)
                    def produce():
                        try:writer.publish({'value':3})
                        except BaseException as error:errors.append(error)
                        finally:finished.set()
                    producer=threading.Thread(target=produce)
                    with patch.object(writer._condition,'wait',side_effect=observed_wait):
                        producer.start();self.assertTrue(waiting.wait(5))
                        self.assertFalse(finished.is_set());self.assertEqual(writer.report()['queued'],1)
                        release.set();producer.join(5)
                        self.assertFalse(producer.is_alive())
            finally:
                release.set()
                if producer is not None:producer.join(5)
            self.assertFalse(errors);self.assertEqual([row['value'] for row in seen],[1,2,3])
            self.assertEqual(writer.report()['max_queue_depth'],1);self.assertFalse(writer.report()['writer_alive'])

    def test_writer_failure_wakes_blocked_producer_and_invalidates_context(self):
        entered=threading.Event();release=threading.Event();waiting=threading.Event();errors=[]
        def failed_write(path,payload):
            entered.set()
            if not release.wait(5):raise RuntimeError('test barrier was not released')
            raise OSError('disk failed')
        with self.folder() as folder,patch.object(publication,'atomic_json',side_effect=failed_write):
            writer=publication.CheckpointPublisher(Path(folder)/'state.json',queue_capacity=1);producer=None
            try:
                with self.assertRaisesRegex(publication.CheckpointPublicationError,'disk failed'):
                    with writer:
                        writer.publish({'value':1});self.assertTrue(entered.wait(5));writer.publish({'value':2})
                        original_wait=writer._condition.wait
                        def observed_wait(timeout=None):
                            if threading.current_thread() is producer:waiting.set()
                            return original_wait(timeout)
                        def produce():
                            try:writer.publish({'value':3})
                            except BaseException as error:errors.append(error)
                        producer=threading.Thread(target=produce)
                        with patch.object(writer._condition,'wait',side_effect=observed_wait):
                            producer.start();self.assertTrue(waiting.wait(5));release.set();producer.join(5)
                            self.assertFalse(producer.is_alive());self.assertTrue(writer.failed())
                        with self.assertRaises(publication.CheckpointPublicationError):writer.progress({'event':'progress'})
            finally:
                release.set()
                if producer is not None:producer.join(5)
            self.assertEqual(len(errors),1);self.assertIsInstance(errors[0],publication.CheckpointPublicationError)
            report=writer.report();self.assertEqual(report['submitted_count'],2)
            self.assertEqual(report['published_count'],0);self.assertEqual(report['discarded_count'],1)
            self.assertEqual(report['failed_write_count'],1);self.assertFalse(report['writer_alive'])
            self.assertFalse(writer._lock_path.exists())

    def test_close_drains_accepted_queue_and_rejects_late_admission(self):
        entered=threading.Event();release=threading.Event();closed=threading.Event();errors=[]
        def blocked(path,payload):
            if payload['value']==1:
                entered.set()
                if not release.wait(5):raise RuntimeError('test barrier was not released')
            actual_atomic_json(path,payload)
        with self.folder() as folder,patch.object(publication,'atomic_json',side_effect=blocked):
            writer=publication.CheckpointPublisher(Path(folder)/'state.json',queue_capacity=1);writer.__enter__()
            closer=None
            try:
                writer.publish({'value':1});self.assertTrue(entered.wait(5));writer.publish({'value':2})
                def close():
                    try:writer.close()
                    except BaseException as error:errors.append(error)
                    finally:closed.set()
                closer=threading.Thread(target=close);closer.start()
                with writer._condition:
                    self.assertTrue(writer._condition.wait_for(lambda:writer._closing,timeout=5))
                self.assertFalse(closed.is_set())
                with self.assertRaisesRegex(RuntimeError,'not accepting'):writer.publish({'value':3})
                release.set();closer.join(5);self.assertFalse(closer.is_alive())
                self.assertEqual(json.loads(writer.path.read_text(encoding='utf-8')),{'value':2})
                self.assertEqual(writer.report()['published_count'],2);self.assertFalse(writer.report()['writer_alive'])
                writer.close()  # Successful closure is idempotent.
                with self.assertRaises(RuntimeError):writer.progress({'event':'checkpoint','payload':{'late':True}})
            finally:
                release.set()
                if closer is not None:closer.join(5)
                if not writer.report()['closed']:writer.close()
            self.assertFalse(errors)

    def test_body_exception_stays_primary_and_writer_failure_is_reported(self):
        with self.folder() as folder,patch.object(publication,'atomic_json',side_effect=OSError('disk failed')):
            writer=publication.CheckpointPublisher(Path(folder)/'state.json')
            with self.assertRaisesRegex(ValueError,'body failed') as caught:
                with writer:
                    writer.publish({'value':1})
                    raise ValueError('body failed')
            self.assertTrue(any('disk failed' in note for note in caught.exception.__notes__))
            self.assertFalse(writer.report()['writer_alive']);self.assertTrue(writer.failed())

    def test_cleanup_error_preserves_body_and_writer_failures_after_join(self):
        with self.folder() as folder:
            writer=publication.CheckpointPublisher(Path(folder)/'state.json')
            release=writer._release_lock
            with patch.object(publication,'atomic_json',side_effect=OSError('disk failed')),patch.object(writer,'_release_lock',side_effect=PermissionError('lock unlink failed')):
                with self.assertRaisesRegex(ValueError,'body failed') as caught:
                    with writer:
                        writer.publish({'value':1})
                        raise ValueError('body failed')
            report=writer.report();self.assertFalse(report['writer_alive']);self.assertTrue(writer.failed())
            self.assertEqual(report['write_error'],'disk failed');self.assertEqual(report['cleanup_error'],'lock unlink failed')
            notes=' '.join(caught.exception.__notes__)
            self.assertIn('disk failed',notes);self.assertIn('lock unlink failed',notes)
            release()  # Only this test's owned sidecar remains after mocked denial.

    def test_cleanup_error_invalidates_an_otherwise_successful_publication(self):
        with self.folder() as folder:
            writer=publication.CheckpointPublisher(Path(folder)/'state.json')
            release=writer._release_lock
            with patch.object(writer,'_release_lock',side_effect=PermissionError('lock unlink failed')):
                with self.assertRaisesRegex(publication.CheckpointPublicationError,'cleanup: lock unlink failed') as caught:
                    with writer:writer.publish({'value':1})
            self.assertIsInstance(caught.exception.__cause__,PermissionError)
            report=writer.report();self.assertFalse(report['writer_alive']);self.assertTrue(report['failed'])
            self.assertEqual(report['published_count'],1);self.assertIsNone(report['write_error'])
            self.assertEqual(report['cleanup_error'],'lock unlink failed');self.assertTrue(writer.path.is_file())
            release()

    def test_serialization_failure_leaves_no_unreported_success_or_orphan(self):
        with self.folder() as folder:
            writer=publication.CheckpointPublisher(Path(folder)/'state.json')
            with self.assertRaises(publication.CheckpointPublicationError):
                with writer:writer.publish({'not_json':object()})
            self.assertTrue(writer.failed());self.assertEqual(writer.report()['published_count'],0)
            self.assertFalse(writer.report()['writer_alive']);self.assertFalse(writer.path.exists())
            self.assertFalse(list(Path(folder).iterdir()))

    def test_fresh_target_and_exclusive_ownership_preserve_existing_files(self):
        with self.folder() as folder:
            root=Path(folder);existing=root/'input.json';existing.write_text('preserve',encoding='utf-8')
            with self.assertRaises(FileExistsError):
                with publication.CheckpointPublisher(existing):self.fail('Existing input was accepted')
            self.assertEqual(existing.read_text(encoding='utf-8'),'preserve')
            with self.assertRaises(ValueError):publication.CheckpointPublisher(root/'source.py')
            with self.assertRaises(ValueError):
                with publication.CheckpointPublisher(root/'missing'/'state.json'):self.fail('Missing parent accepted')
            path=root/'state.json'
            with publication.CheckpointPublisher(path) as writer:
                with self.assertRaises(FileExistsError):
                    with publication.CheckpointPublisher(path):self.fail('Concurrent target accepted')
                self.assertTrue(writer._lock_path.exists())
                writer.publish({'value':1})
            self.assertFalse(writer._lock_path.exists())
            with self.assertRaises(RuntimeError):writer.__enter__()

    def test_bounds_and_lifecycle_fail_without_starting_writers(self):
        with self.folder() as folder:
            path=Path(folder)/'state.json'
            for capacity in (0,17,True,1.5):
                with self.assertRaises(ValueError):publication.CheckpointPublisher(path,queue_capacity=capacity)
            writer=publication.CheckpointPublisher(path)
            with self.assertRaises(RuntimeError):writer.publish({'value':1})
            with self.assertRaises(RuntimeError):writer.close()
            with writer:
                with self.assertRaises(TypeError):writer.publish([1])
                with self.assertRaises(TypeError):writer.progress(None)
            self.assertFalse(path.exists());self.assertFalse(writer.report()['writer_alive'])


if __name__=='__main__':unittest.main()
