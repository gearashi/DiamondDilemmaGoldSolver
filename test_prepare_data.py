"""Offline tests using synthetic data; no source diagrams or GPU are required."""
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import prepare_data as prep


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.offline = self.root / 'offline'
        self.output = self.root / 'output'
        self.scripts = self.root / 'scripts'
        for path in (self.offline, self.output, self.scripts):
            path.mkdir()
        self.sources = []
        for name in prep.SOURCE_NAMES:
            payload = ('synthetic-test-' + name).encode()
            (self.offline / name).write_bytes(payload)
            self.sources.append({'name': name, 'url': prep.SOURCE_URLS[name],
                                 'sha256': hashlib.sha256(payload).hexdigest()})
        hashes = {s['name']: s['sha256'] for s in self.sources}
        tiles = []
        for number in range(1, 161):
            group, prefix = ('silver', 'S') if number <= 32 else ('red', 'R') if number <= 80 else ('blue', 'B')
            segments = [[[0, 1], [1, 2]], [[0, 3], [2, 4]]]
            if number <= 45:
                segments.append([[1, 5], [2, 6]])
            tiles.append({'id': f'{prefix}{number:02d}', 'number': number, 'group': group, 'segments': segments})
        self.tiles = {'source_sha256': hashes, 'tiles': tiles}
        self.manifest = {'sources': self.sources, 'expected': {'tile_count': 160, 'segment_count': 365,
                         'endpoint_count': 730, 'topology_sha256': prep.topology_digest(self.tiles)}}
        self.audit = {'source_sha256': hashes, 'tile_count': 160, 'gold_segment_count': 365,
                      'gold_endpoint_count': 730, 'unexplained_strong_gold_pixels': 0,
                      'minimum_pairing_margin': 1.0}
        self.endpoints = {'source_sha256': hashes, 'tiles': [
            {'id': t['id'], 'endpoints': [p for s in t['segments'] for p in s]}
            for t in tiles]}
        for name in ('audit_endpoints.py', 'extract_tiles.py'):
            (self.scripts / name).write_text('# synthetic subprocess fixture\n')
        self.patch_manifest = patch.object(prep, 'load_manifest', return_value=self.manifest)
        self.patch_root = patch.object(prep, 'ROOT', self.scripts)
        self.patch_manifest.start()
        self.patch_root.start()
        self.addCleanup(self.patch_manifest.stop)
        self.addCleanup(self.patch_root.stop)

    def artifacts(self, directory):
        for name, content in [('tiles.json', self.tiles), ('endpoint_audit.json', self.endpoints),
                              ('extraction_audit.json', self.audit)]:
            (directory / name).write_text(json.dumps(content), encoding='utf-8')
        for name in prep.GENERATED:
            if name.endswith('.png'):
                (directory / name).write_bytes(b'\x89PNG\r\n\x1a\nsynthetic-overlay')

    def extract(self, args, *, check, cwd):
        self.assertTrue(check)
        self.assertEqual(args[0], sys.executable)
        self.assertEqual(cwd, self.scripts)
        self.artifacts(Path(args[3]))
        return subprocess.CompletedProcess(args, 0)

    def prepare(self):
        with patch.object(prep.subprocess, 'run', side_effect=self.extract) as run, \
             patch.object(prep, 'download_source', side_effect=AssertionError('network forbidden')):
            result = prep.prepare(self.output, self.offline)
        return result, run

    def test_offline_preparation_and_idempotent_verification(self):
        result, run = self.prepare()
        self.assertEqual(result, self.output)
        self.assertEqual(run.call_count, 2)
        before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.output.iterdir()}
        _, run = self.prepare()
        self.assertEqual(run.call_count, 0)
        self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in self.output.iterdir()})
        self.assertEqual(json.loads((self.output / 'preparation.json').read_text())['source_mode'], 'offline')

    def test_existing_bad_hash_is_preserved(self):
        target = self.output / prep.SOURCE_NAMES[0]
        target.write_bytes(b'user-owned-conflicting-file')
        with self.assertRaisesRegex(prep.PreparationError, 'SHA-256 mismatch'):
            self.prepare()
        self.assertEqual(target.read_bytes(), b'user-owned-conflicting-file')
        self.assertEqual(len(list(self.output.iterdir())), 1)

    def test_bad_offline_source_is_not_installed(self):
        name = prep.SOURCE_NAMES[0]
        (self.offline / name).write_bytes(b'wrong')
        with self.assertRaisesRegex(prep.PreparationError, 'SHA-256 mismatch'):
            self.prepare()
        self.assertFalse((self.output / name).exists())
        self.assertFalse(any(self.output.glob('.prepare-*')))

    def test_missing_offline_source_never_downloads(self):
        (self.offline / prep.SOURCE_NAMES[0]).unlink()
        with self.assertRaisesRegex(prep.PreparationError, 'Missing, empty, or oversized'):
            self.prepare()

    def test_missing_artifact_rebuilds_without_touching_existing_files(self):
        self.prepare()
        target = self.output / 'tiles.json'
        before = (target.stat().st_mtime_ns, target.read_bytes())
        (self.output / 'extraction_blue.png').unlink()
        _, run = self.prepare()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(before, (target.stat().st_mtime_ns, target.read_bytes()))
        self.assertTrue((self.output / 'extraction_blue.png').is_file())

    def test_wrong_topology_is_rejected_without_subprocess(self):
        self.prepare()
        wrong = copy.deepcopy(self.tiles)
        wrong['tiles'][0]['segments'][0][0] = [0, 11]
        (self.output / 'tiles.json').write_text(json.dumps(wrong))
        with self.assertRaisesRegex(prep.PreparationError, 'topology differs'):
            self.prepare()

    def test_wrong_endpoint_audit_is_rejected(self):
        self.prepare()
        wrong = copy.deepcopy(self.endpoints)
        wrong['tiles'][0]['endpoints'][0] = [0, 11]
        (self.output / 'endpoint_audit.json').write_text(json.dumps(wrong))
        with self.assertRaisesRegex(prep.PreparationError, 'Endpoint audit'):
            self.prepare()

    def test_path_traversal_and_directory_target_rejected(self):
        for name in ('../outside.gif', '..\\outside.gif', '/tmp/outside', '.', '..'):
            with self.assertRaises(prep.PreparationError):
                prep.protected_child(self.output, name)
        (self.output / prep.SOURCE_NAMES[0]).mkdir()
        with self.assertRaisesRegex(prep.PreparationError, 'regular file'):
            self.prepare()

    def test_atomic_install_does_not_clobber(self):
        staged, target = self.root / 'staged', self.output / 'target'
        staged.write_bytes(b'new')
        target.write_bytes(b'existing')
        with self.assertRaisesRegex(prep.PreparationError, 'Conflicting existing'):
            prep.install_missing(staged, target)
        self.assertEqual(target.read_bytes(), b'existing')

    def test_subprocess_failure_installs_no_generated_artifacts(self):
        with patch.object(prep.subprocess, 'run', side_effect=subprocess.CalledProcessError(2, ['fixture'])):
            with self.assertRaises(subprocess.CalledProcessError):
                prep.prepare(self.output, self.offline)
        self.assertFalse(any((self.output / name).exists() for name in prep.GENERATED))
        self.assertFalse(any(self.output.glob('.prepare-*')))

    def test_missing_program_is_clear(self):
        (self.scripts / 'audit_endpoints.py').unlink()
        with self.assertRaisesRegex(prep.PreparationError, 'Missing extraction program'):
            self.prepare()

    def test_missing_generated_file_rejected(self):
        self.artifacts(self.output)
        (self.output / 'tiles.json').unlink()
        with self.assertRaisesRegex(prep.PreparationError, 'Missing generated artifact'):
            prep.validate_artifacts(self.output, self.manifest)

    def test_download_rejects_unlisted_url_before_network(self):
        source = dict(self.sources[0], url='https://example.com/unlisted')
        with patch.object(prep.urllib.request, 'build_opener', side_effect=AssertionError('network forbidden')):
            with self.assertRaisesRegex(prep.PreparationError, 'allowlist'):
                prep.download_source(source, self.root / 'download')

    def test_download_hash_and_size_limits_using_fake_response(self):
        class Response(io.BytesIO):
            headers = {}
        payload = (self.offline / self.sources[0]['name']).read_bytes()
        destination = self.root / 'download'
        with patch.object(prep.urllib.request, 'build_opener') as factory:
            factory.return_value.open.return_value = Response(payload)
            prep.download_source(self.sources[0], destination)
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(factory.return_value.open.call_args.kwargs['timeout'], prep.SOCKET_TIMEOUT)
        destination.unlink()
        with patch.object(prep.urllib.request, 'build_opener') as factory:
            factory.return_value.open.return_value = Response(b'wrong')
            with self.assertRaisesRegex(prep.PreparationError, 'SHA-256 mismatch'):
                prep.download_source(self.sources[0], destination)
        destination.unlink()
        with patch.object(prep.urllib.request, 'build_opener') as factory, patch.object(prep, 'MAX_SOURCE_BYTES', 4):
            factory.return_value.open.return_value = Response(b'12345')
            with self.assertRaisesRegex(prep.PreparationError, 'size limit'):
                prep.download_source(self.sources[0], destination)

    def test_download_deadline_and_redirect_rejected(self):
        class Response(io.BytesIO):
            headers = {}
        with patch.object(prep.urllib.request, 'build_opener') as factory, \
             patch.object(prep.time, 'monotonic', side_effect=[0, prep.DOWNLOAD_DEADLINE + 1]):
            factory.return_value.open.return_value = Response(b'x')
            with self.assertRaisesRegex(prep.PreparationError, 'time limit'):
                prep.download_source(self.sources[0], self.root / 'download')
        with self.assertRaisesRegex(prep.PreparationError, 'redirect'):
            prep._NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.com')


if __name__ == '__main__':
    unittest.main()
