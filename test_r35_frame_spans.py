"""Bound native frame intervals, column descriptors and lossless snapshot replay."""
import ctypes
import os
import unittest
from types import SimpleNamespace

import numpy as np

from r35_frame_spans import FIELDS, FULL, FrameSpans, blocks, extents, prepare, rectangle_wire_bytes


def mark(a,b,c,d):
    return np.uint32(a|(b<<8)|(c<<16)|(d<<24))


class TypeCheckedMemory(FrameSpans):
    """Independent interpreter of the transfer descriptor and little-endian wire packet."""
    def __init__(self,physical,**kwargs):
        super().__init__(SimpleNamespace(n=physical[0].shape[1]),**kwargs)
        self.physical=physical
        self.flags=set()

    def _read(self,desc,words):
        n,k,lc,mc,wc,mode=map(int,desc[:6]); self.flags.add(mode)
        packet=np.zeros(words,np.uint32); consumed=0
        for i in range(k):
            if mode&8:
                lane=lc+i
                if mode&16:
                    bits,table=map(int,desc[6:8]); per=32//bits
                    code=int(desc[mc+i//per])>>(bits*(i%per))&((1<<bits)-1)
                    packed=int(desc[table+code])
                else:
                    packed=int(desc[mc+i])
                if i%32==0: self_assert(consumed==desc[wc+i//32])
                at=consumed
            else:
                lane=int(desc[lc+i]) if mode&1 else lc+i
                packed=int(desc[mc+i]) if mode&2 else mc
                at=int(desc[wc+i]) if mode&4 else wc*i
            a,b,c,d=(packed>>s&255 for s in (0,8,16,24))
            if c==0: continue
            pieces=[self.physical[j][a-1:b,lane] for j in (0,1,3)] if a else []
            pieces.append(self.physical[2][c-1:d,lane])
            data=np.concatenate(pieces).astype(np.uint32)
            if len(data)%2: data=np.r_[data,np.uint32(0)]
            count=len(data)//2
            self_assert(at==consumed and at+count<=len(packet))
            packet[at:at+count]=data[::2]|(data[1::2]<<16)
            consumed+=count
        self_assert(consumed==words)
        return packet


def self_assert(condition):
    if not condition: raise AssertionError('Invalid frame descriptor or offset.')


class FrameSpanTests(unittest.TestCase):
    def fixture(self,lanes,marks,reset,limit=None,merge=None):
        n=346
        rng=np.random.default_rng(35068)
        prior=tuple(rng.integers(0,65536,(160,n),np.uint16) for _ in FIELDS)
        physical=tuple(arr.copy() for arr in prior)
        expected=tuple(arr.copy() for arr in prior)
        for lane,raw,replaced in zip(lanes,marks,reset):
            bits=int(FULL) if replaced else int(raw)
            a,b,c,d=(bits>>s&255 for s in (0,8,16,24))
            for field,start,stop in ((0,a,b),(1,a,b),(3,a,b),(2,c,d)):
                if start:
                    data=rng.integers(0,65536,stop+1-start,np.uint16)
                    expected[field][start-1:stop,lane]=data
                    physical[field][start-1:stop,lane]=data
        for arr in prior: arr.setflags(write=False)
        reader=TypeCheckedMemory(physical,**({'merge':merge} if merge else {}))
        answer=reader.capture(prior,lanes,marks,reset,limit=limit)
        for saved,actual in zip(expected,answer): np.testing.assert_array_equal(saved,actual)
        return reader,prior,answer

    def test_mixed_spans_resets_sparse_chunks_and_maximum_frame(self):
        ids=np.array([0,3,4,8,31,32,33,77,85,159,161,211,345])
        marks=np.array([0,mark(1,1,1,2),mark(4,9,3,10),mark(0,0,160,160),
                        mark(34,50,32,60),0,mark(4,9,3,10),mark(159,160,155,160),
                        mark(2,2,1,3),mark(1,1,1,1),0,mark(7,7,7,7),mark(0,0,9,13)],np.uint32)
        reset=np.zeros(len(ids),bool)
        reset[0]=True
        reader,_,_=self.fixture(ids,marks,reset,limit=1280)
        self.assertGreater(reader.maps,1)

    def test_dense_holes_group_offsets_uniform_and_only_cursors(self):
        ids=np.r_[0:94,96:110]
        marks=np.array([mark(3,3+(r%9),2,4+(r%11)) if r%9<=r%11+1
                        else mark(2,2,2,2) for r in ids],np.uint32)
        reader,_,_=self.fixture(ids,marks,np.zeros(len(ids),bool))
        self.assertIn(8,reader.flags)
        ids=np.r_[0:150,152:291]
        marks=np.array([mark(3,3+r%5,2,10) for r in ids],np.uint32)
        reader,_,_=self.fixture(ids,marks,np.zeros(len(ids),bool))
        self.assertIn(24,reader.flags)
        ids=np.arange(65)
        marks=np.full(65,mark(0,0,160,160),np.uint32)
        reader,prior,actual=self.fixture(ids,marks,np.zeros(65,bool))
        self.assertIn(0,reader.flags)
        for f in (0,1,3): self.assertIs(prior[f],actual[f])
        self.assertIsNot(prior[2],actual[2])

    def test_no_write_and_invalid_bounds(self):
        reader,prior,actual=self.fixture(np.array([],int),np.array([],np.uint32),np.array([],bool))
        self.assertIs(prior,actual)
        self.assertEqual((reader.maps,reader.read_bytes,reader.upload_bytes),(0,0,0))
        for bad in (mark(0,1,1,1),mark(1,0,1,1),mark(1,2,0,0),mark(4,3,1,7),
                    mark(1,7,3,8),mark(0,0,1,161),mark(0,0,9,0)):
            with self.assertRaises(ValueError): extents(np.array([bad],np.uint32))
            result=prepare(5,np.array([2]),np.array([bad],np.uint32),np.array([True]))
            self.assertEqual(int(result[1][0]),int(FULL))
        for ids in ([-1],[345,344],[2,2],[346]):
            with self.assertRaises(ValueError):
                prepare(346,np.array(ids),np.zeros(len(ids),np.uint32),np.zeros(len(ids),bool))
        with self.assertRaises(ValueError):
            self.fixture(np.array([3]),np.array([FULL],np.uint32),np.array([False]),limit=1279)

    def test_byte_gate_refuses_before_device_work_and_retries_after_error(self):
        n=6
        old=tuple(np.zeros((160,n),np.uint16) for _ in FIELDS)
        current=tuple(np.full((160,n),21000+i,np.uint16) for i in range(4))
        gpu=SimpleNamespace(n=n)
        ids=np.array([0,1,2])
        marks=np.full(3,mark(1,1,1,1),np.uint32)
        resets=np.zeros(3,bool)
        codec=TypeCheckedMemory(current)
        price=rectangle_wire_bytes(gpu,[(0,1,ids)])
        self.assertEqual(price,24)
        self.assertIsNone(codec.capture(old,ids,marks,resets,max_wire_bytes=price))
        self.assertEqual((codec.maps,codec.read_bytes,codec.upload_bytes),(0,0,0))
        with self.assertRaises(ValueError):
            codec.capture(old,ids,np.array([FULL,0,mark(1,0,2,3)],np.uint32),resets)
        self.assertTrue(all(not f.any() for f in old))
        recovered=codec.capture(old,ids,marks,resets,max_wire_bytes=49)
        self.assertIsNotNone(recovered)
        self.assertEqual(codec.read_bytes+codec.upload_bytes,48)
        self.assertTrue(all(not f.any() for f in old))

    def test_maximum_pool_resets_split_at_requested_binding_limit(self):
        n=131072
        ids=np.arange(n,dtype=np.uint32)
        ids,marks,widths=prepare(n,ids,np.zeros(n,np.uint32),np.ones(n,bool))
        cap=128*1024**2
        plans=list(blocks(n,ids,marks,widths,cap))
        self.assertEqual(len(plans),2)
        self.assertEqual([sl.start for sl,_,_ in plans],[0,plans[0][0].stop])
        self.assertEqual(plans[-1][0].stop,n)
        self.assertEqual(sum(4*w for _,_,w in plans),160*n*8)
        for _,desc,words in plans:
            self.assertLessEqual(max(desc.nbytes,words*4),cap)
            self.assertEqual(desc.nbytes,24)

    def test_checked_shared_library_loader(self):
        import tempfile
        from pathlib import Path
        from build_native import _build_one, find_compiler
        try:
            compiler=find_compiler()
        except RuntimeError:
            self.skipTest('A C compiler is required for optional checked decoder.')
        with tempfile.TemporaryDirectory() as tmp:
            gpu=SimpleNamespace(n=3)
            self.assertFalse(FrameSpans(gpu,native_directory=tmp).native)
            _build_one(compiler=compiler,output_dir=tmp,stem='r35_board_codec_decode',
                             manifest_name='physical-board-build.json')
            import subprocess
            import sys
            subprocess.run([sys.executable,'-B','-c',
                            'from r35_frame_spans import FrameSpans; from types import SimpleNamespace; '
                            'import sys; assert FrameSpans(SimpleNamespace(n=3),native_directory=sys.argv[1]).native',
                            tmp],check=True)
            Path(tmp,'physical-board-build.json').write_text('{}')
            self.assertFalse(FrameSpans(gpu,native_directory=tmp).native)

    @unittest.skipUnless(os.environ.get('DIAMOND_SPAN_DECODER'),'requires independent benchmark C build')
    def test_native_decoder_reference_and_negative_packets(self):
        from r35_frame_span_decode import native_merge
        merge=native_merge(ctypes.CDLL(os.environ['DIAMOND_SPAN_DECODER']))
        ids=np.arange(310)
        marks=np.array([mark(2,2+(i%19),1,2+(i%19)) for i in ids],np.uint32)
        reset=(ids%11)==0
        self.fixture(ids,marks,reset,limit=2120,merge=merge)
        fields=tuple(np.zeros((160,1),np.uint16) for _ in FIELDS)
        for raw,words in ((FULL,319),(FULL,321),(mark(0,2,1,3),5)):
            with self.assertRaises(ValueError):
                merge(fields,np.array([0],np.uint32),np.array([raw],np.uint32),None,np.zeros(words,np.uint32))

class PhysicalSpanChecks:
    def test_physical_words_match_two_intervals_across_odd_chunks(self):
        from test_gpu_gold import gold_fixture
        from test_r35_checkpoint_ranges import NativeControls
        n=513
        gpu=NativeControls.pool(self,*gold_fixture(),n)
        ids=np.r_[1:211,213:512].astype(np.int32)
        marks=np.array([mark(1+i%3,12+i%3,1,14+i%3) if i%5 else mark(0,0,158,160)
                        for i in ids],np.uint32)
        reset=np.zeros(len(ids),bool);reset[-1]=True
        expected=[]
        old=[]
        rng=np.random.default_rng(11829)
        for field,name in enumerate(FIELDS):
            data=rng.integers(0,65536,(160,n),np.uint16)
            before=rng.integers(0,65536,(160,n),np.uint16)
            stored=before.copy()
            for j,lane in enumerate(ids):
                m=int(FULL if reset[j] else marks[j]); off=16 if field==2 else 0
                a,b=(m>>off)&255,(m>>(off+8))&255
                if a: stored[a-1:b,lane]=data[a-1:b,lane]
            getattr(gpu,name)[:,:]=gpu.cp.asarray(data)
            expected.append(stored);before.flags.writeable=False;old.append(before)
        reader=FrameSpans(gpu)
        for limit in (None,2120):
            answer=reader.capture(tuple(old),ids,marks,reset,limit=limit)
            for a,b in zip(answer,expected): np.testing.assert_array_equal(a,b)
            if limit: self.assertGreater(reader.maps,1)


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','requires CUDA')
class CudaSpanTests(PhysicalSpanChecks,unittest.TestCase):
    web=False


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','requires WebGPU')
class WebSpanTests(PhysicalSpanChecks,unittest.TestCase):
    web=True


if __name__=='__main__':
    unittest.main()
