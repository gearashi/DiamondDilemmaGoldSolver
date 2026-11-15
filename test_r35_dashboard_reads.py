"""Verify lightweight dashboard identity and immutable snapshot reuse."""
from http.client import HTTPConnection
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import dashboard_server as dashboard
from dashboard_snapshots import read_display_file
from io_utils import atomic_json, read_json_shared
import platform_launcher
import runtime_capabilities


class DashboardReadTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.enterContext(patch.object(dashboard, 'RUNTIME', self.root))
        self.enterContext(patch.object(dashboard, 'CHILD', None))

    def test_identity_needs_no_state_even_when_its_response_exceeds_launcher_limit(self):
        atomic_json(self.root/'history.json', [{'score':207, 'tested':i} for i in range(42000)])
        server = dashboard.DashboardHTTPServer(('127.0.0.1',0), dashboard.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_port
            conn = HTTPConnection('127.0.0.1',port,timeout=5)
            try:
                conn.request('GET','/api/state')
                response=conn.getresponse()
                self.assertEqual(response.status,200)
                self.assertGreater(len(response.read()),1024*1024)
            finally:
                conn.close()
            with patch.object(dashboard,'state_payload',side_effect=AssertionError('Read full state')):
                identity=platform_launcher.read_dashboard(port)
                self.assertEqual(identity,dashboard.installation_identity())
                self.assertEqual(set(identity),{'service','installation_root'})
                conn=HTTPConnection('127.0.0.1',port,timeout=5)
                try:
                    conn.request('GET','/api/identity',headers={'Host':'unrelated.example'})
                    self.assertEqual(conn.getresponse().status,403)
                finally:
                    conn.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_snapshot_replacement_deletion_corruption_and_recovery(self):
        path=self.root/'best.json'
        first={'score':207}
        atomic_json(path,first)
        previous=path.stat()
        reader=Mock(wraps=read_json_shared)
        with patch.object(dashboard,'read_json_shared',reader):
            self.assertEqual(dashboard.read_json(path.name),first)
            self.assertEqual(dashboard.read_json(path.name),first)
            self.assertEqual(reader.call_count,1)
            tmp=self.root/'next'
            atomic_json(tmp,{'score':205})
            os.utime(tmp,ns=(previous.st_atime_ns,previous.st_mtime_ns))
            os.replace(tmp,path)
            self.assertEqual(dashboard.read_json(path.name),{'score':205})
            self.assertEqual(reader.call_count,2)
            path.unlink()
            self.assertIsNone(dashboard.read_json(path.name))
            path.write_text('{')
            self.assertIsNone(dashboard.read_json(path.name))
            atomic_json(path,{'score':202})
            self.assertEqual(dashboard.read_json(path.name),{'score':202})

    def test_replacement_during_decode_does_not_publish_under_wrong_version(self):
        path=self.root/'live.json'
        atomic_json(path,{'step':1})
        def decode(target):
            result=read_json_shared(target)
            if result['step']==1:
                atomic_json(target,{'step':2})
            return result
        self.assertEqual(read_display_file(path,decode),{'step':1})
        self.assertEqual(read_display_file(path,decode),{'step':2})

    def test_concurrent_polls_share_each_published_generation(self):
        from concurrent.futures import ThreadPoolExecutor
        path = self.root/'status.json'
        read = Mock(wraps=read_json_shared)
        with ThreadPoolExecutor(max_workers=6) as pool:
            for generation in range(6):
                atomic_json(path,{'generation':generation})
                jobs = [pool.submit(read_display_file,path,read) for _ in range(6)]
                self.assertEqual([job.result() for job in jobs],[{'generation':generation}]*6)
        self.assertEqual(read.call_count,6)

    def test_sat_package_metadata_is_read_only_on_supported_profiles(self):
        for environment, arch, version, installed, eligible in (
                ('win32','AMD64',(3,12),True,False),
                ('darwin','arm64',(3,15),True,False),
                ('linux','aarch64',(3,12),True,False),
                ('darwin','arm64',(3,12),False,False),
                ('linux','x86_64',(3,12),True,True),
                ('darwin','arm64',(3,12),True,True)):
            with self.subTest(environment=environment,arch=arch,version=version), \
                    patch.object(runtime_capabilities.sys,'platform',environment), \
                    patch.object(runtime_capabilities.sys,'version_info',version), \
                    patch.object(runtime_capabilities.platform,'machine',return_value=arch), \
                    patch.object(runtime_capabilities.platform,'python_implementation',return_value='CPython'), \
                    patch.object(runtime_capabilities,'_installed',return_value=installed), \
                    patch.object(runtime_capabilities,'package_version',return_value='1.9.dev15') as metadata:
                caps=runtime_capabilities.runtime_capabilities(display=True)
                self.assertEqual(caps['options']['native_sat_persistence']['available'],eligible)
                self.assertEqual(metadata.call_count,int(eligible))

    def test_native_display_caches_digests_but_strict_validation_checks_current_bytes(self):
        root=self.root
        folder=root/'.native'
        folder.mkdir()
        native=folder/'demo.bin'
        source=root/'constraint_native_search.c'
        source.write_bytes(b'source')
        native.write_bytes(b'library')
        manifest=dict(abi=1,platform=sys.platform,machine=platform.machine(),library=native.name,
                      source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      library_sha256=hashlib.sha256(native.read_bytes()).hexdigest())
        atomic_json(folder/'search-build.json',manifest)
        with patch.object(runtime_capabilities,'_digest',wraps=runtime_capabilities._digest) as digest, \
                patch.object(runtime_capabilities,'_build_metadata',wraps=runtime_capabilities._build_metadata) as metadata:
            for _ in range(3):
                self.assertTrue(runtime_capabilities._native_explanation_build(root,display=True)['available'])
            self.assertEqual(digest.call_count,2)
            self.assertTrue(runtime_capabilities._native_explanation_build(root)['available'])
            self.assertEqual(digest.call_count,4)
            replacement=folder/'update'
            replacement.write_bytes(b'changed')
            os.replace(replacement,native)
            self.assertFalse(runtime_capabilities._native_explanation_build(root,display=True)['available'])
            self.assertEqual(digest.call_count,5)
            self.assertFalse(runtime_capabilities._native_explanation_build(root)['available'])
            self.assertEqual(digest.call_count,7)
            self.assertEqual(metadata.call_count,3)
            # Failed parses never enter the cache. A replace repairs the display.
            (folder/'search-build.json').write_text('{')
            self.assertFalse(runtime_capabilities._native_explanation_build(root,display=True)['available'])
            atomic_json(folder/'search-build.json',{**manifest,'library_sha256':hashlib.sha256(b'changed').hexdigest()})
            self.assertTrue(runtime_capabilities._native_explanation_build(root,display=True)['available'])
            self.assertEqual(metadata.call_count,5)
