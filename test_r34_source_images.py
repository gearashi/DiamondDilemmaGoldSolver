"""Source archives retain exact captured bytes and independent writable files."""
from concurrent.futures import ThreadPoolExecutor
import errno
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import source_image_storage as storage
from exact_source_evidence import snapshot_search_sources


class SourceImageTests(TestCase):
    def test_copy_fallback_and_changed_source_do_not_reuse_links(self):
        with TemporaryDirectory() as root:
            root=Path(root)
            src=root/'module.py'
            src.write_bytes(b'captured')
            evidence=root/'run'
            evidence.mkdir()
            calls=[]
            def unsupported(src,stage):
                calls.append(src)
                stage.write_bytes(b'partial')
                raise OSError(errno.ENOTSUP,'no block cloning')
            with mock.patch.object(storage,'_native_clone',return_value=unsupported):
                with storage.source_image_writer(evidence) as write:
                    for _ in range(2):
                        self.assertFalse(write(src,b'captured'))
            self.assertEqual(len(calls),1)
            for implementation in (lambda src,stage:stage.write_bytes(b'new source bytes'),
                                   lambda src,stage:os.link(src,stage)):
                with mock.patch.object(storage,'_native_clone',return_value=implementation):
                    with storage.source_image_writer(evidence) as write:
                        self.assertFalse(write(src,b'captured'))
                (evidence/src.name).write_bytes(b'changed run')
                self.assertEqual(src.read_bytes(),b'captured')

    def test_interruption_keeps_previous_and_concurrent_publications_are_whole(self):
        with TemporaryDirectory() as root:
            root=Path(root)
            evidence=root/'run'
            evidence.mkdir()
            source=root/'module.py'
            source.write_bytes(b'installed')
            out=evidence/source.name
            out.write_bytes(b'previous')
            def interrupted(source,stage):
                stage.write_bytes(b'partial')
                raise KeyboardInterrupt
            with mock.patch.object(storage,'_native_clone',return_value=interrupted):
                with self.assertRaises(KeyboardInterrupt),storage.source_image_writer(evidence) as write:
                    write(source,b'never published')
            self.assertEqual(list(evidence.iterdir()),[out])
            self.assertEqual(out.read_bytes(),b'previous')
            rows=[bytes([n])*16384 for n in range(8)]
            def publish(data):
                with storage.source_image_writer(evidence) as write:
                    write(source,data)
            with ThreadPoolExecutor(4) as executor:
                list(executor.map(publish,rows))
            self.assertIn(out.read_bytes(),rows)
            self.assertEqual(source.read_bytes(),b'installed')
            self.assertEqual(list(evidence.iterdir()),[out])

    def test_interrupted_new_raw_image_has_no_receipt_and_can_be_replaced(self):
        from hashlib import sha256
        with TemporaryDirectory() as root:
            root=Path(root)
            dest=root/'run'
            dest.mkdir()
            source=root/'module.py'
            data=b'import math\n'
            source.write_bytes(data)
            # A killed first write can leave an incomplete file in an unsealed
            # run. It cannot match the receipt and a retry must fully replace it.
            target=dest/source.name
            target.write_bytes(data[:3])
            self.assertNotEqual(sha256(target.read_bytes()).digest(),sha256(data).digest())
            with mock.patch.object(storage,'_native_clone',return_value=None):
                manifest=snapshot_search_sources(root,dest,[source.name],include_families=False)
            self.assertEqual(manifest[source.name],sha256(data).hexdigest())
            self.assertEqual(target.read_bytes(),data)

    def test_raw_archive_api_and_previous_run_independence(self):
        with TemporaryDirectory() as root:
            root=Path(root)
            source=root/'module.py'
            body=b'import math\n'
            source.write_bytes(body)
            first=root/'run1'
            second=root/'run2'
            first.mkdir()
            second.mkdir()
            for dest in (first,second):
                hashes=snapshot_search_sources(root,dest,[source.name],include_families=False)
                self.assertEqual(set(hashes),{source.name})
            source.write_bytes(b'install changed')
            (first/source.name).write_bytes(b'first run changed')
            self.assertEqual((second/source.name).read_bytes(),body)
            self.assertEqual(source.read_bytes(),b'install changed')

    def test_absent_darwin_symbol_and_directory_collision(self):
        with mock.patch.object(storage.sys,'platform','darwin'):
            for failure in (AttributeError('clonefile'),OSError('library unavailable')):
                with mock.patch.object(storage,'_darwin_api',side_effect=failure):
                    self.assertIsNone(storage._native_clone())
        with TemporaryDirectory() as root:
            root=Path(root)
            dest=root/'run'
            dest.mkdir()
            source=root/'module.py'
            source.write_bytes(b'installed')
            collision=dest/source.name
            collision.mkdir()
            sentinel=collision/'user-file'
            sentinel.write_bytes(b'unchanged')
            with mock.patch.object(storage,'_native_clone',return_value=None):
                with self.assertRaises(OSError),storage.source_image_writer(dest) as write:
                    write(source,b'new image')
            self.assertEqual(sentinel.read_bytes(),b'unchanged')

    def test_native_clones_have_private_data_and_distinct_file_ids(self):
        with TemporaryDirectory() as root:
            root=Path(root)
            source=root/'shader.wgsl'
            body=b'shader\n'*8192
            source.write_bytes(body)
            copies=[]
            for i in range(2):
                dest=root/str(i)
                dest.mkdir()
                with storage.source_image_writer(dest) as write:
                    cloned=write(source,body)
                if not cloned:
                    if os.getenv('DIAMOND_REQUIRE_SOURCE_CLONE')=='1':
                        self.fail('Native same-volume source cloning is required for this test run.')
                    self.skipTest('Local filesystem has no tested block-clone API.')
                copies.append(dest/source.name)
            self.assertEqual(len({p.stat().st_ino for p in [source,*copies]}),3)
            source.write_bytes(b'installed source changed')
            copies[0].write_bytes(b'run zero changed')
            self.assertEqual(copies[1].read_bytes(),body)
