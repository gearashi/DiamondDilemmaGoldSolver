"""Independent physical versus frame-verified checkpoint board transport."""
import os
import json
from pathlib import Path
import tempfile
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

from classic_checkpoint_ranges import CheckpointRanges
from r35_board_codec import PhysicalBoardCodec,encode_layout,native_unpack,unpack,verified_decoder
from systematic_jobs import JobLedger
from test_r35_checkpoint_ranges import NativeControls
from test_gpu_reasoning import varied_gold_fixture


def check_built_decoder(directory):
    fn=PhysicalBoardCodec(SimpleNamespace(n=1),native_directory=directory).decoder
    if fn is unpack:raise AssertionError('A matching native decoder did not load.')
    layout,n=encode_layout(1,np.array([0]),np.array([3]))
    packet=np.zeros(11+n,np.uint32)
    packet[1]=3
    packet[11]=7|(479<<16)
    np.testing.assert_array_equal(fn(layout,packet)[0],unpack(layout,packet)[0])


class PacketTests(unittest.TestCase):
    def test_missing_malformed_and_wrong_library_take_numpy_path(self):
        with tempfile.TemporaryDirectory() as folder:
            gpu=SimpleNamespace(n=2)
            self.assertIs(PhysicalBoardCodec(gpu,native_directory=folder).decoder,unpack)
            manifest=Path(folder)/'physical-board-build.json'
            for raw in (b'{',b'[]',b'{"library":"../escape.dll","abi":1}'):
                manifest.write_bytes(raw)
                with patch('ctypes.CDLL',side_effect=AssertionError('Unverified binary was opened')):
                    self.assertIsNone(verified_decoder(folder))

    def test_validated_native_source_binary_platform_and_explicit_reference(self):
        from build_native import find_compiler,_build_one
        try:compiler=find_compiler()
        except RuntimeError as exc:self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as folder:
            built=_build_one(compiler=compiler,output_dir=folder,stem='r35_board_codec_decode',
                             manifest_name='physical-board-build.json')
            gpu=SimpleNamespace(n=1)
            # Windows keeps a loaded DLL mapped. The child owns the positive
            # decode and releases it before the temporary directory is removed.
            subprocess.run([sys.executable,'-B','-c',
                            'from test_r35_board_projections import check_built_decoder; '
                            'import sys; check_built_decoder(sys.argv[1])',folder],check=True)
            self.assertIs(PhysicalBoardCodec(gpu,'numpy',native_directory=folder).decoder,unpack)
            base=built['manifest']
            for field,value in (('source_sha256','0'*64),('library_sha256','f'*64),('abi',2),
                                ('platform','wrong'),('machine','wrong'),('library','../bad.dll')):
                Path(folder,'physical-board-build.json').write_text(json.dumps({**base,field:value}))
                self.assertIsNone(verified_decoder(folder))

    def test_all_physical_cell_bits_int16_patterns_and_raw_used_words(self):
        k=7
        caps=np.array([0,1,2,3,32,91,160],np.int16)
        ids=np.array([6,0,1,2,3,4,5])
        layout,words=encode_layout(k,ids,caps)
        packet=np.zeros(1+10*k+words,np.uint32)
        masks=packet[1:1+5*k].reshape(5,k)
        raw_used=np.arange(5*k,dtype=np.uint32).reshape(5,k)^np.uint32(0xf0a53c27)
        packet[1+5*k:1+10*k]=raw_used.ravel()
        expected=np.full((160,k),-1,np.int16)
        rng=np.random.default_rng(9873)
        special=np.array([-32768,-32000,-2,0,479,480,32767],np.int16)
        for i,lane in enumerate(ids):
            cells=np.sort(rng.choice(160,int(caps[lane]),replace=False))
            for rank,cell in enumerate(cells):
                code=special[(rank+i)%len(special)]
                expected[cell,i]=code
                masks[cell//32,i]|=np.uint32(1)<<np.uint32(cell%32)
                at=1+10*k+int(layout[2+k+i])+rank//2
                packet[at]|=np.uint32(code.view(np.uint16))<<np.uint32(16*(rank%2))
        decoders=[unpack]
        if path:=os.environ.get('DIAMOND_PHYSICAL_DECODER'):
            import ctypes
            decoders.append(native_unpack(ctypes.CDLL(path)))
        for decode in decoders:
            board,used=decode(layout,packet)
            np.testing.assert_array_equal(board,expected)
            np.testing.assert_array_equal(used,raw_used)
            self.assertIsNone(decode(layout[:-1],packet))
            self.assertIsNone(decode(layout,packet[:-1]))
            self.assertIsNone(decode(layout,packet.view(np.int32)))
            broken=layout.copy()
            broken[2+k]=511
            self.assertIsNone(decode(broken,packet))
        self.assertEqual(board.dtype,np.int16)
        self.assertEqual(used.dtype,np.uint32)
        self.assertIsNone(unpack(layout,packet[:-1]))
        packet[0]=1
        self.assertIsNone(unpack(layout,packet))
        self.assertIsNone(encode_layout(1,np.array([0]),np.array([-1],np.int16)))

    def test_stale_overreservation_and_damaged_packet_are_separate(self):
        layout,words=encode_layout(2,np.arange(2),np.array([5,0],np.int16))
        packet=np.zeros(1+20+words,np.uint32)
        packet[1]=0b101
        packet[21]=np.uint32(479|(47<<16))
        board,_=unpack(layout,packet)
        self.assertEqual(board[[0,2],0].tolist(),[479,47])
        packet[2]=1
        self.assertIsNone(unpack(layout,packet))


class ProjectionNative:
    pool=NativeControls.pool

    def setup_lanes(self,dynamic=False):
        data,board=varied_gold_fixture()
        ids=np.array([0,5,9,17,36,62,65],np.int32)
        opts={'gpu_features':{'forced':True,'dynamic_mrv':True},'gold_data':data} if dynamic else {}
        gpu=self.pool(data,board,80,**opts)
        gpu.load_prefixes([[3*i] for i in range(len(ids))],lanes=ids)
        return gpu,ids

    def arrays(self,gpu):
        return gpu.boards.get(),gpu.used.get()

    def assert_columns(self,expected,projected,ids):
        self.assertIsNotNone(projected)
        for src,got in zip(expected,projected):
            self.assertEqual(src.dtype,got.dtype)
            np.testing.assert_array_equal(src[:,ids],got)

    def test_sparse_static_and_dynamic_match_independent_physical_copy(self):
        for dynamic in (False,True):
            gpu,ids=self.setup_lanes(dynamic)
            codec=PhysicalBoardCodec(gpu)
            for count in (0,3,17):
                if count:gpu.step(count)
                fields=gpu.snapshot_checkpoint()
                arrays=self.arrays(gpu)
                for subset in (ids,ids[1:],ids[::2]):
                    with self.subTest(dynamic=dynamic,depth=count,subset=len(subset)):
                        self.assert_columns(arrays,codec.capture(fields,subset),subset)
            # A later CPU table mutation cannot change a physical result.
            value=gpu.host_pool.copy()
            try:
                gpu.host_pool[:]=np.roll(value,3)
                self.assert_columns(self.arrays(gpu),codec.capture(fields,ids),ids)
            finally:gpu.host_pool[:]=value

    def test_stale_cpu_metadata_never_changes_a_physical_copy(self):
        gpu,ids=self.setup_lanes(True)
        gpu.step(9)
        fields=gpu.snapshot_checkpoint()
        physical=self.arrays(gpu)
        codec=PhysicalBoardCodec(gpu)
        # Stale valid-looking dynamic slots, empty frames, contradictory floors,
        # changed CPU depth and even garbage order must not synthesize a board.
        stale=dict(fields)
        for field in ('firsts','lengths','cursors','shifts','order','floors'):
            stale[field]=np.zeros_like(fields[field])
        stale['depths']=np.full(gpu.n,160,np.int16)
        self.assert_columns(physical,codec.capture(stale,ids[::-1]),ids[::-1])
        stale['depths']=np.zeros(gpu.n,np.int16)
        self.assertIsNone(codec.capture(stale,ids))
        self.assert_columns(physical,codec.capture(fields,ids),ids)

    def test_damaged_native_payload_is_preserved_and_rejected_on_restore(self):
        gpu,ids=self.setup_lanes()
        fields=gpu.snapshot_checkpoint()
        codec=PhysicalBoardCodec(gpu)
        used=fields['used'].copy()
        used[4,ids[0]]^=np.uint32(1<<20)
        gpu.used[:,:]=gpu.cp.asarray(used)
        self.assert_columns(self.arrays(gpu),codec.capture(fields,ids),ids)
        with self.assertRaises(ValueError):gpu.read_checkpoint(gpu.snapshot_checkpoint())
        raw=fields['boards'].copy()
        cell=int(np.flatnonzero(raw[:,ids[0]]<0)[0])
        raw[cell,ids[0]]=-32768
        raw[int(gpu.order[0]),ids[1]]=32767
        gpu.boards[:,:]=gpu.cp.asarray(raw)
        self.assertIsNone(codec.capture(fields,ids))
        loose=dict(fields,depths=np.full(gpu.n,160,np.int16))
        self.assert_columns(self.arrays(gpu),codec.capture(loose,ids),ids)
        # The public checkpoint on an under-reserved lane must keep native bytes.
        ledger=JobLedger.empty(0,gpu.n)
        reader=CheckpointRanges(gpu,ledger,board_mode='physical')
        reader.capture()
        previous=reader.board_fallbacks
        captured=reader.capture()
        np.testing.assert_array_equal(captured['boards'],raw)
        np.testing.assert_array_equal(captured['used'],used)
        self.assertIsInstance(reader.board_projection,PhysicalBoardCodec)
        self.assertEqual(reader.board_fallbacks,previous+1)


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','requires CUDA device')
class CudaProjectionTests(ProjectionNative,unittest.TestCase):
    web=False


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','requires WebGPU device')
class WebProjectionTests(ProjectionNative,unittest.TestCase):
    web=True
