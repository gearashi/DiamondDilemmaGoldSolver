"""Atomic JSON regressions, including the real Windows reader-sharing failure.

These tests keep temporary files under the standalone solver directory. They
never change permissions, Defender settings, or the actual runtime snapshots.
"""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest

ROOT=Path(__file__).resolve().parent


@contextmanager
def scratch():
    with tempfile.TemporaryDirectory(prefix='atomic-io-test-',dir=ROOT) as name:
        directory=Path(name).resolve()
        if directory.parent!=ROOT.resolve():
            raise RuntimeError('Temporary test directory escaped the solver folder')
        yield directory


@contextmanager
def windows_reader_without_delete_sharing(path):
    """Read handle that recreates CRT-style interference with atomic rename."""
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    create=kernel.CreateFileW
    create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,
                     ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype=wintypes.HANDLE
    close=kernel.CloseHandle
    close.argtypes=[wintypes.HANDLE];close.restype=wintypes.BOOL
    handle=create(str(path),0x80000000,0x00000001|0x00000002,None,3,0x80,None)
    if handle==ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    released=False
    def release():
        nonlocal released
        if not released:
            if not close(handle):
                raise ctypes.WinError(ctypes.get_last_error())
            released=True
    try:
        yield release
    finally:
        release()


class AtomicJSONTests(unittest.TestCase):
    def atomic_json(self,path,value):
        from io_utils import atomic_json
        return atomic_json(path,value)

    def test_roundtrip_and_no_temporary_files(self):
        with scratch() as directory:
            path=directory/'state.json'
            self.atomic_json(path,{'version':1,'text':'gold \u2194 loop'})
            self.atomic_json(path,{'version':2,'values':list(range(50))})
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),{'version':2,'values':list(range(50))})
            self.assertEqual({p.name for p in directory.iterdir()},{'state.json'})

    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_real_reader_lock_reproduces_replace_failure_then_retries(self):
        with scratch() as directory:
            target=directory/'state.json';probe=directory/'probe.json'
            old={'version':1};new={'version':2,'payload':'x'*5000}
            target.write_text(json.dumps(old),encoding='utf-8')
            probe.write_text(json.dumps(new),encoding='utf-8')
            with windows_reader_without_delete_sharing(target) as release:
                # Establish that this lock actually blocks replacement on this OS.
                with self.assertRaises(PermissionError) as blocked:
                    os.replace(probe,target)
                self.assertIn(blocked.exception.winerror,(5,32))
                self.assertEqual(json.loads(target.read_text()),old)
                probe.unlink()
                # Independent reader releases its normal read handle shortly later.
                timer=threading.Timer(.15,release)
                timer.start();started=time.monotonic()
                try:
                    self.atomic_json(target,new)
                    elapsed=time.monotonic()-started
                finally:
                    timer.join(timeout=5)
                self.assertGreaterEqual(elapsed,.10,'Writer bypassed atomic replacement instead of waiting for the reader')
            self.assertEqual(json.loads(target.read_text()),new)
            self.assertEqual({p.name for p in directory.iterdir()},{'state.json'})

    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_shared_reader_completes_old_snapshot_during_atomic_update(self):
        from io_utils import read_json_shared
        from unittest.mock import patch
        with scratch() as directory:
            target=directory/'state.json'
            target.write_text('{"version":1}',encoding='utf-8')
            opened=threading.Event();release=threading.Event();results=[];failures=[]
            original_load=json.load
            def delayed_load(stream,*args,**kwargs):
                opened.set()
                if not release.wait(timeout=5):
                    raise TimeoutError('Test did not release the held reader')
                return original_load(stream,*args,**kwargs)
            def reader():
                try:results.append(read_json_shared(target))
                except Exception as error:failures.append(repr(error))
            with patch('io_utils.json.load',side_effect=delayed_load):
                thread=threading.Thread(target=reader,daemon=True);thread.start()
                try:
                    self.assertTrue(opened.wait(timeout=2))
                    # Windows may still reject replacing an open destination even
                    # with FILE_SHARE_DELETE. The bounded writer retry remains
                    # required while this reader finishes its original snapshot.
                    timer=threading.Timer(.15,release.set)
                    timer.start()
                    self.atomic_json(target,{'version':2})
                    self.assertEqual(json.loads(target.read_text()),{'version':2})
                finally:
                    release.set();thread.join(timeout=5)
                    if 'timer' in locals():timer.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(failures,[])
            self.assertEqual(results,[{'version':1}])

    @unittest.skipUnless(os.name=='nt','Windows sharing semantics')
    def test_retry_timeout_preserves_original_and_replacement(self):
        from io_utils import replace_with_retry
        with scratch() as directory:
            target=directory/'state.json';replacement=directory/'replacement.json'
            target.write_text('{"version":1}',encoding='utf-8')
            replacement.write_text('{"version":2}',encoding='utf-8')
            with windows_reader_without_delete_sharing(target):
                with self.assertRaises(PermissionError):
                    replace_with_retry(replacement,target,timeout=.03)
                self.assertEqual(json.loads(target.read_text()),{'version':1})
                self.assertEqual(json.loads(replacement.read_text()),{'version':2})

    def test_concurrent_readers_never_observe_partial_json(self):
        with scratch() as directory:
            target=directory/'state.json'
            self.atomic_json(target,{'version':0,'payload':'x'*12000})
            stop=threading.Event();ready=threading.Event();failures=[];read_count=[0]
            def reader():
                ready.set()
                while not stop.is_set():
                    try:
                        state=json.loads(target.read_text(encoding='utf-8'))
                        if not 0<=state['version']<=20 or state['payload']!='x'*12000:
                            failures.append('Incomplete or invalid snapshot')
                        read_count[0]+=1
                    except PermissionError:
                        # A transient read denial is retryable; a partial JSON body is not.
                        pass
                    except Exception as error:
                        failures.append(repr(error))
                    time.sleep(.001)
            thread=threading.Thread(target=reader,daemon=True);thread.start()
            self.assertTrue(ready.wait(timeout=2))
            try:
                for version in range(1,21):
                    self.atomic_json(target,{'version':version,'payload':'x'*12000})
            finally:
                stop.set();thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            self.assertGreater(read_count[0],0)
            self.assertEqual(failures,[])
            self.assertEqual(json.loads(target.read_text())['version'],20)
            self.assertEqual({p.name for p in directory.iterdir()},{'state.json'})


if __name__=='__main__':
    unittest.main(verbosity=2)
