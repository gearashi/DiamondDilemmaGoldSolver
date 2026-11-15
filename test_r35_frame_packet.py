"""Paired physical frame packets, device limits, untouched tails and maximum pool width."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from classic_checkpoint_ranges import FRAME_FIELDS, _packed_frames
from classic_frame_packet import FramePacket, _blocks, _DESCRIPTOR_BYTES, _successors
from test_gpu_gold import gold_fixture
from test_r35_checkpoint_ranges import NativeControls


def ordered(windows):
    return [(row,int(lane)) for low,high,lanes in windows for row in range(low,high) for lane in lanes]


class BlockTests(unittest.TestCase):
    def test_every_split_preserves_exact_cell_sequence_and_granted_storage(self):
        ids=np.arange(67,dtype=np.int32)
        windows=[(0,3,ids),(4,11,ids[1::4]),(125,158,ids[11:14]),(159,160,ids[-1:])]
        for cap in (1,2,3,16,17,24,66,67,68,133,134,177,2000):
            fake=SimpleNamespace(backend='webgpu',gpu_device=SimpleNamespace(limits={
                'max-storage-buffer-binding-size':_DESCRIPTOR_BYTES+4*cap,
                'max-buffer-size':max(8*cap,_DESCRIPTOR_BYTES+4*cap)}))
            actual=FramePacket(fake)._capacity()
            self.assertLessEqual(actual,cap)
            blocks=list(_blocks(windows,actual))
            self.assertEqual([cell for b in blocks for cell in ordered(b)],ordered(windows))
            for block in blocks:
                desc,width,count=_packed_frames(67,block)
                self.assertLessEqual(8*count,min(fake.gpu_device.limits.values()))
                self.assertLessEqual(desc.nbytes,min(fake.gpu_device.limits.values()))
                self.assertGreater(width,0)
        self.assertEqual(list(_blocks([],1)),[])

    def test_every_odd_paired_field_boundary_has_the_correct_physical_successor(self):
        rng=np.random.default_rng(718)
        n=71
        source=np.arange(n,dtype=np.int32)
        for _ in range(48):
            rows=rng.choice(160,9,replace=False)
            windows=[]
            for row in rows:
                howmany=int(rng.integers(1,15))
                ids=np.sort(rng.choice(source,howmany,replace=False))
                windows.append((int(row),int(row)+1,ids))
            physical=[r*n+c for r,c in ordered(windows)]
            for block in _blocks(windows,41):
                layout,_,count=_packed_frames(n,block)
                after=_successors(layout,n)[-160:]
                local=[r*n+c for r,c in ordered(block)]
                self.assertTrue(set(local)<=set(physical))
                rebuilt={}
                next_cell=local[1:]+local[:1]
                offset=0
                for start,stop,ids in block:
                    for r in range(start,stop):
                        self.assertEqual(int(after[r]),next_cell[offset+len(ids)-1])
                        for field in range(4):
                            for c in range((field*count+offset)%2,len(ids),2):
                                position=offset+c
                                right=int(after[r]) if c+1==len(ids) else r*n+int(ids[c+1])
                                rebuilt[(field*count+position)//2]=((field,local[position]),(field+int(position+1==count),right))
                        offset+=len(ids)
                flat=[(f,cell) for f in range(4) for cell in local]
                self.assertEqual([item for k in range(2*count) for item in rebuilt[k]],flat)

    def test_empty_and_impossible_grants(self):
        fake=SimpleNamespace(n=1,backend='cuda')
        codec=FramePacket(fake)
        self.assertEqual([len(v) for v in codec.capture([])],[0]*4)
        self.assertEqual((codec.maps,codec.read_bytes),(0,0))
        with self.assertRaises(ValueError):list(_blocks([(0,1,np.arange(1))],0))


class NativeFrameChecks:
    pool=NativeControls.pool

    def test_nonconsecutive_windows_and_immutable_frame_tails_match_physical_copy(self):
        n=513
        gpu=self.pool(*gold_fixture(),n)
        ids=np.arange(n,dtype=np.int32)
        windows=[(152,160,ids[5::9]),(0,1,ids[-1:]),(14,77,ids[2:138]),(1,9,ids[::3])]
        codec=FramePacket(gpu)
        oracle=[]
        for i,name in enumerate(FRAME_FIELDS):
            raw=(np.arange(n*160,dtype=np.uint32).reshape(160,n)*(37+2*i)+65535-23*i).astype(np.uint16)
            raw[-1,-1-i]=65535
            getattr(gpu,name)[:,:]=gpu.cp.asarray(raw)
            physical=getattr(gpu,name).get()
            oracle.append(np.concatenate([physical[a:b,cs].ravel() for a,b,cs in windows]))
        for forced in (None,2047):
            if forced is None:actual=codec.capture(windows)
            else:
                with patch.object(codec,'_capacity',return_value=forced):actual=codec.capture(windows)
            for want,got in zip(oracle,actual,strict=True):
                self.assertTrue(got.flags.c_contiguous)
                self.assertEqual(got.dtype,np.uint16)
                np.testing.assert_array_equal(got,want)
            if forced is None:self.assertEqual(codec.maps,1)
            else:self.assertGreater(codec.maps,1)
            self.assertEqual(codec.read_bytes,8*len(oracle[0]))
        first=actual
        gpu.cursors[:,:]=gpu.cp.asarray(np.zeros((160,n),np.uint16))
        after=codec.capture(windows)
        np.testing.assert_array_equal(first[2],oracle[2])
        self.assertFalse(after[2].any())

    def test_complete_maximum_pool_and_held_out_tail(self):
        n=131072
        base=self.pool(*gold_fixture(),2)
        source_bytes=n*160*4
        if self.web:
            adapter=base.adapter
            if min(adapter.limits['max-buffer-size'],adapter.limits['max-storage-buffer-binding-size'])<source_bytes:
                self.skipTest('Adapter cannot represent all 131072 physical frame columns.')
            device=adapter.request_device_sync(required_limits={'max-buffer-size':source_bytes,'max-storage-buffer-binding-size':source_bytes})
            self.addCleanup(device.destroy)
            usage=base.wgpu.BufferUsage
            buffers=[None]
            gpu=SimpleNamespace(n=n,backend='webgpu',gpu_device=device,wgpu=base.wgpu,queue=device.queue,buffers=buffers,readback_bytes=0)
        else:
            gpu=SimpleNamespace(n=n,backend='cuda',cp=base.cp)
        for i,name in enumerate(FRAME_FIELDS):
            if self.web:
                buf=device.create_buffer(size=source_bytes,usage=usage.STORAGE|usage.COPY_SRC|usage.COPY_DST)
                self.addCleanup(buf.destroy)
                gpu.buffers.append(buf)
            else:setattr(gpu,name,gpu.cp.empty((160,n),gpu.cp.uint16))
            lanes=np.arange(n,dtype=np.uint32)
            for row in range(0,160,8):
                offset=np.arange(row,row+8,dtype=np.uint32)[:,None]*np.uint32(7919)
                raw=(lanes[None,:]*(19+i*6)+offset+i*717+0xfffa)|np.uint32(0x80000000)
                if self.web:gpu.queue.write_buffer(buf,row*n*4,raw)
                else:getattr(gpu,name)[row:row+8]=gpu.cp.asarray(raw.astype(np.uint16))
        ids=np.arange(n,dtype=np.int32)
        codec=FramePacket(gpu)
        def read(index,name):
            if self.web:
                return np.frombuffer(gpu.queue.read_buffer(gpu.buffers[index+1]),np.uint32).reshape(160,n).astype(np.uint16)
            return getattr(gpu,name).get()
        for windows in ([(159,160,ids[17::113])],[(0,160,ids)]):
            actual=codec.capture(windows)
            for i,(name,flat) in enumerate(zip(FRAME_FIELDS,actual,strict=True)):
                physical=read(i,name)
                want=np.concatenate([physical[a:b,cols].ravel() for a,b,cols in windows])
                np.testing.assert_array_equal(flat,want)
                del physical,want
            self.assertEqual(codec.read_bytes,sum((b-a)*len(cs)*8 for a,b,cs in windows))
            if self.web and len(actual[0])==160*n:self.assertEqual(codec.maps,2)
            else:self.assertEqual(codec.maps,1)


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','requires a CUDA device')
class CudaPacketTests(NativeFrameChecks,unittest.TestCase):
    web=False


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','requires a WebGPU device')
class WebPacketTests(NativeFrameChecks,unittest.TestCase):
    web=True


if __name__=='__main__':
    unittest.main()
