"""Check complete physical frame recovery against arbitrary restored u16 arrays."""
import os
from types import SimpleNamespace
import unittest

import numpy as np

from classic_frame_defaults import DEFAULTS,FIELDS,FrameDefaults,_pages,_prices,_restore
from classic_frame_packet import FramePacket
from test_gpu_gold import gold_fixture
from test_r35_checkpoint_ranges import NativeControls


def rows(n):
    return tuple(np.full((160,n),value,np.uint16) for value in DEFAULTS)


def summary(fields):
    dirty=np.logical_or.reduce([a!=d for a,d in zip(fields,DEFAULTS)])
    n=dirty.shape[1]
    meta=np.zeros((160,7),np.uint32)
    meta[:,3:]=65535
    meta[:,1]=n
    for row,line in enumerate(dirty):
        ids=np.flatnonzero(line)
        if len(ids):
            meta[row,:3]=len(ids),ids[0],ids[-1]+1
            for i,field in enumerate(fields):
                vals=field[row,ids]
                meta[row,3+i]=int(vals.min())|(int(vals.max())<<16)
    return dirty,meta


def control(n):
    a,b,c,d=fields=rows(n)
    a[0,0]=65535
    d[-1,-1]=65535
    b[11,::2]=65534
    c[32,1::511]=0
    a[83,1::127]=0xf045
    d[84,::32]=0x8000
    a[85,:]=np.arange(n,dtype=np.uint16)
    b[85,:]=0xa001
    d[85,-17:]=0xedfd
    return fields


def packet(fields,dirty,row,desc):
    data=[]
    for j,p in enumerate(desc[3:].reshape(-1,10),row):
        mode,lo,hi,start,directory,count,base1,base2,widths,total=map(int,p)
        if not mode:continue
        assert len(data)==start
        ids=np.flatnonzero(dirty[j]).tolist()
        if mode==1:ids=range(lo,hi)
        elif mode==2:
            masks=[0]*((hi-1)//32-lo//32+1)
            for lane in ids:masks[lane//32-lo//32]|=1<<(lane%32)
            data.extend(masks)
        elif mode==3:data.extend(ids)
        stream=0
        at=0
        for i in ids:
            for f,field in enumerate(fields):
                bias=((base1 if f<2 else base2)>>(16*(f%2)))&65535
                width=(widths>>(5*f))&31
                value=int(field[j,i])-bias
                assert 0<=value<(1<<width)
                stream|=value<<at
                at+=width
        assert at==total*len(ids)
        data.extend((stream>>k)&0xffffffff for k in range(0,at,32))
    return np.array(data,np.uint32)


class PriceControls(unittest.TestCase):
    def test_all_physical_directory_forms_and_batch_boundaries(self):
        n=2051
        fields=control(n)
        dirty,meta=summary(fields)
        plans,sizes=_prices(n,meta)
        self.assertEqual(set(plans[:,0]),{0,1,2,3})
        for limit in (int(sizes.max()),int(sizes.sum())):
            out=rows(n)
            for row,desc,size in _pages(n,plans,sizes,limit):
                data=packet(fields,dirty,row,desc)
                self.assertEqual(len(data),size)
                _restore(out,row,desc,data)
            for want,got in zip(fields,out):np.testing.assert_array_equal(want,got)

    def test_lossless_range_packing_for_all_u16_widths_and_odd_bit_offsets(self):
        n=257
        fields=rows(n)
        rnd=np.random.default_rng(5331)
        for r in range(160):
            ids=np.arange(r%7,n,1+r%9,dtype=np.int32)
            for i,field in enumerate(fields):
                width=(r*(2*i+1)+i)%17
                limit=(1<<width)-1
                bias=7*i if width<16 else 0
                vals=rnd.integers(0,limit+1,len(ids),dtype=np.uint16)
                vals[:2]=[0,limit]
                field[r,ids]=vals+bias
        dirty,meta=summary(fields)
        plans,sizes=_prices(n,meta)
        widths={(int(word)>>(5*f))&31 for word in plans[:,8] for f in range(4)}
        self.assertEqual(widths,set(range(17)))
        offsets={(rank*int(width))%32 for width in plans[:,9] for rank in range(32)}
        self.assertEqual(offsets,set(range(32)))
        out=rows(n)
        for row,desc,size in _pages(n,plans,sizes,int(sizes.max())):
            body=packet(fields,dirty,row,desc)
            self.assertEqual(len(body),size)
            _restore(out,row,desc,body)
        for got,want in zip(out,fields):np.testing.assert_array_equal(got,want)

    def test_stale_or_damaged_payload_cannot_be_used_as_a_full_frame(self):
        n=2051
        fields=control(n)
        dirty,meta=summary(fields)
        plans,sizes=_prices(n,meta)
        row,desc,_=next(_pages(n,plans,sizes,sizes.sum()))
        packed=packet(fields,dirty,row,desc)
        for j in (11,32):
            corrupted=packed.copy()
            at=int(plans[j,0]);self.assertNotEqual(at,0)
            start=int(desc[3+10*j+3])
            corrupted[start]^=1
            with self.assertRaises(ValueError):
                _restore(rows(n),row,desc,corrupted)
        for bad in ((0,0,0),(9,5,6),(1,n,n)):
            changed=meta.copy();changed[0,:3]=bad
            with self.assertRaises(ValueError):_prices(n,changed)

    def test_small_pool_uses_packed_fallback_without_a_device_request(self):
        native=FrameDefaults(SimpleNamespace(n=1))
        self.assertIsNone(native.capture())
        self.assertEqual((native.maps,native.read_bytes),(0,0))


class NativeChecks:
    pool=NativeControls.pool

    def test_cold_arbitrary_restored_tails_dense_fallback_and_reinstalled_lanes(self):
        n=2051
        gpu=self.pool(*gold_fixture(),n)
        native=FrameDefaults(gpu)
        def check(mode):
            reference=tuple(getattr(gpu,f).get().ravel() for f in FIELDS)
            result=native.capture()
            if mode=='packed':
                self.assertIsNone(result)
                result=FramePacket(gpu).capture([(0,160,np.arange(n))])
            else:
                self.assertEqual(mode=='empty',native.maps==1)
            for a,b in zip(reference,result):
                self.assertEqual(b.dtype,np.uint16)
                self.assertTrue(b.flags.c_contiguous)
                np.testing.assert_array_equal(a,b)
            return result
        cold=check('empty')
        self.assertEqual(native.read_bytes,20)
        for key,field in zip(FIELDS,control(n)):
            getattr(gpu,key)[:,:]=gpu.cp.asarray(field)
        restored=check('sparse')
        self.assertTrue(all(native.distribution[m]>0 for m in ('span','bits','indices')))
        gpu.load_prefixes([[0],[3],[6]],lanes=np.array([0,128,2049]))
        check('sparse')
        # Arbitrary high-bit tails, even on completely inactive lanes, remain physical data.
        for i,key in enumerate(FIELDS):
            field=np.full((160,n),0xf123+i,np.uint16)
            getattr(gpu,key)[:,:]=gpu.cp.asarray(field)
        check('sparse')
        self.assertEqual(native.maps,2)
        rng=np.random.default_rng(35068)
        for key in FIELDS:
            field=rng.integers(0,65536,size=(160,n),dtype=np.uint16)
            getattr(gpu,key)[:,:]=gpu.cp.asarray(field)
        check('packed')
        for k,field in enumerate(cold):self.assertTrue(np.all(field==DEFAULTS[k]))
        for k,want in enumerate(control(n)):np.testing.assert_array_equal(restored[k],want.ravel())


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','requires a CUDA device')
class CudaDefaultTests(NativeChecks,unittest.TestCase):
    web=False


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','requires a WebGPU device')
class WebDefaultTests(NativeChecks,unittest.TestCase):
    web=True


if __name__=='__main__':unittest.main()
