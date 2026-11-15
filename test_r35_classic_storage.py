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

    def test_detached_suffix_portability_and_failed_final_receipt_preserve_exact_progress(self):
        from classic_parked_storage import ParkedCheckpointWriter, restore_parked, REFERENCE
        from types import SimpleNamespace
        with TemporaryDirectory() as tmp, patch('classic_parked_storage.MIN_PAUSED',2), patch('classic_parked_storage.MIN_BANK_BYTES',0):
            root=Path(tmp)
            path=root/'systematic/checkpoint.npz'
            origin='20261115T000000000000Z'
            (root/'runs'/origin).mkdir(parents=True)
            owner=SimpleNamespace(ids=np.arange(12,dtype=np.int64))
            def packet(gen,offset=0,bank=owner):
                n=len(bank.ids)
                fields=dict(exact_source_origin=np.array(origin),exact_source_binding=np.array('scope'),
                    exact_terminal=np.array(''),exact_frontier_sha256=np.array('front'),exact_data_sha256=np.array('tiles'),
                    exact_generation=np.int64(gen),boards=np.zeros((160,3),np.int16),
                    exact_paused_job_ids=bank.ids[offset:],
                    exact_paused_prefix_codes=np.arange(n*160,dtype=np.int16).reshape(n,160)[offset:],
                    exact_paused_boards=np.arange(n*160,dtype=np.int16).reshape(160,n)[:,offset:])
                fields['exact_paused_shifts']=np.random.default_rng(35068).integers(0,65535,(160,n),dtype=np.uint16)[:,offset:]
                return fields,bank,offset,True
            def compare(gen,offset,external):
                expected=packet(gen,offset)[0]
                with np.load(path,allow_pickle=False) as head:
                    self.assertEqual(REFERENCE in head.files,external)
                    proof.check_receipt(root,origin,gen,path,'scope',parked=external)
                    with restore_parked(path,head) as restored:
                        for k,v in expected.items():
                            np.testing.assert_array_equal(restored[k],v)
            ledger=ParkedCheckpointWriter(path,runner.atomic_npz)
            self.assertFalse(ledger.write(packet(3))['parked_bank_external'])
            self.assertTrue(ledger.write(packet(4,2))['parked_bank_external'])
            compare(4,2,True)
            sidecar,=path.parent.glob('parked-bank-*.npz')
            self.assertTrue(ledger.write(packet(5,6))['parked_bank_external'])
            compare(5,6,True)
            readonly=ParkedCheckpointWriter(path,runner.atomic_npz)
            self.assertIsNone(readonly.base)
            old_bytes=path.read_bytes()
            with patch('systematic_search.replace_with_retry',side_effect=OSError(errno.ENOSPC,'full')):
                with self.assertRaises(OSError):
                    ledger.portable()
            self.assertEqual(path.read_bytes(),old_bytes)
            compare(5,6,True)
            before=sidecar.read_bytes()
            with sidecar.open('ab') as f:
                f.write(b'damage')
            with self.assertRaisesRegex(ValueError,'digest'):
                readonly.portable()
            self.assertEqual(path.read_bytes(),old_bytes)
            sidecar.write_bytes(before)
            readonly.portable()
            self.assertFalse(sidecar.exists())
            compare(5,6,False)
            newcomer=SimpleNamespace(ids=np.arange(12,dtype=np.int64))
            self.assertFalse(readonly.write(packet(6,3,newcomer))['parked_bank_external'])
            self.assertTrue(readonly.write(packet(7,5,newcomer))['parked_bank_external'])
            compare(7,5,True)
            # A near-empty bank goes straight back to a self-contained checkpoint.
            self.assertFalse(readonly.write(packet(8,11,newcomer))['parked_bank_external'])
            compare(8,11,False)
            self.assertFalse(list(path.parent.glob('parked-bank-*.npz')))

    def test_hard_link_unavailable_falls_back_to_standalone_and_rejects_invalid_pointer(self):
        from classic_parked_storage import ParkedCheckpointWriter, restore_parked, REFERENCE, COUNT, OFFSET
        from types import SimpleNamespace
        with TemporaryDirectory() as tmp, patch('classic_parked_storage.MIN_PAUSED',1), patch('classic_parked_storage.MIN_BANK_BYTES',0):
            root=Path(tmp)
            path=root/'systematic/checkpoint.npz'
            origin='20261115T000000000000Z'
            (root/'runs'/origin).mkdir(parents=True)
            owner=SimpleNamespace(ids=np.array([1],np.int64))
            fields=dict(exact_generation=np.int64(1),exact_terminal=np.array(''),
                exact_source_origin=np.array(origin),exact_source_binding=np.array('scope'),
                exact_frontier_sha256=np.array('front'),exact_data_sha256=np.array('tiles'),exact_paused_job_ids=owner.ids)
            fields['exact_paused_prefix_codes']=np.random.default_rng(7).integers(0,32767,(1,4096),dtype=np.int16)
            writer=ParkedCheckpointWriter(path,runner.atomic_npz)
            writer.write((fields,owner,0,True))
            fields={**fields,'exact_generation':np.int64(2)}
            with patch('classic_parked_storage.os.link',side_effect=OSError('unsupported')) as link:
                writer.write((fields,owner,0,True))
                writer.write(({**fields,'exact_generation':np.int64(3)},owner,0,True))
            self.assertEqual(link.call_count,1)
            with np.load(path,allow_pickle=False) as f:
                self.assertNotIn(REFERENCE,f.files)
            del fields['exact_paused_job_ids'],fields['exact_paused_prefix_codes']
            for pointer,count,offset in (('../outside',1,0),('e'*64,0,0),('e'*64,1,-1),('e'*64,1,1.5)):
                with self.subTest(pointer=pointer,count=count,offset=offset):
                    np.savez(path,**fields,**{REFERENCE:np.array(pointer),COUNT:np.array(count),OFFSET:np.array(offset)})
                    with np.load(path,allow_pickle=False) as saved, self.assertRaises(ValueError):
                        with restore_parked(path,saved):
                            self.fail('invalid parked checkpoint accepted')
