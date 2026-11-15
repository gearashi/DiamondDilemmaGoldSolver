"""Classic signed checkpoint storage and crash boundary through the production writer."""
import errno
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

import systematic_evidence as proof
import systematic_search as runner


class ClassicStorageTests(unittest.TestCase):
    def test_fast_archive_preserves_layout_receipt_and_previous_checkpoint_on_failure(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'systematic/checkpoint.npz'
            origin='20261115T000000000000Z'
            (root/'runs'/origin).mkdir(parents=True)
            fields=dict(exact_source_origin=np.array(origin),exact_source_binding=np.array('bound'),
                        exact_generation=np.int64(1),boards=np.zeros((160,2048),np.int16),
                        paused=np.empty((160,0),np.uint16),message=np.array('physical'),
                        fortran=np.asfortranarray(np.arange(48).reshape(6,8)))
            report=runner.atomic_npz(path,fields)
            self.assertTrue(report['compressed'])
            self.assertLess(report['file_bytes'],20000)
            proof.check_receipt(root,origin,1,path,'bound')
            with zipfile.ZipFile(path) as z:
                self.assertTrue(all(entry.compress_type==zipfile.ZIP_DEFLATED for entry in z.infolist()))
            with np.load(path,allow_pickle=False) as z:
                self.assertEqual(set(z.files),set(fields))
                for name,value in fields.items():
                    np.testing.assert_array_equal(z[name],value)
                    self.assertEqual(z[name].dtype,np.asanyarray(value).dtype)
            old=hashlib.sha256(path.read_bytes()).digest()
            dirty={**fields,'exact_generation':np.int64(2)}
            err=OSError(errno.ENOSPC,'injected disk full')
            for target in ('np.lib.format.write_array','os.fsync','atomic_json','replace_with_retry'):
                with self.subTest(failure=target), patch('systematic_search.'+target,side_effect=err):
                    with self.assertRaisesRegex(OSError,'disk full'):
                        runner.atomic_npz(path,dirty)
                self.assertEqual(hashlib.sha256(path.read_bytes()).digest(),old)
                proof.check_receipt(root,origin,1,path,'bound')
                self.assertEqual(list(path.parent.glob('*.tmp')),[])
            runner.atomic_npz(path,dirty)
            proof.check_receipt(root,origin,2,path,'bound')
            with path.open('ab') as stream:
                stream.write(b'x')
            with self.assertRaisesRegex(ValueError,'receipt'):
                proof.check_receipt(root,origin,2,path,'bound')

    def test_ordinary_frontier_is_stored_with_legacy_layout(self):
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'frontier.npz'
            fields={'prefixes':np.arange(32,dtype=np.int16),'metadata':np.array('frontier')}
            stats=runner.atomic_npz(path,fields)
            self.assertFalse(stats['compressed'])
            with zipfile.ZipFile(path) as z:
                self.assertTrue(all(entry.compress_type==zipfile.ZIP_STORED for entry in z.infolist()))
            with np.load(path,allow_pickle=False) as z:
                for name,value in fields.items():
                    np.testing.assert_array_equal(z[name],value)
