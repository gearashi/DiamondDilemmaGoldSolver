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



def check_native(directory):
    from classic_frame_defaults import verified_restore
    decode=verified_restore(directory)
    assert decode is not None
    n=257
    fields=control(n)
    rnd=np.random.default_rng(350680)
    for r in range(20,60):
        ids=np.arange(1,n-1,1+r%7)
        for i,field in enumerate(fields):
            width=(r+i*5)%17
            limit=(1<<width)-1
            values=rnd.integers(0,limit+1,len(ids),dtype=np.uint16)
            values[:2]=[0,limit]
            field[r,ids]=values
    dirty,meta=summary(fields)
    plans,sizes=_prices(n,meta)
    widths={(int(word)>>(5*f))&31 for word in plans[:,8] for f in range(4)}
    assert widths==set(range(17))
    out=rows(n)
    for start,desc,size in _pages(n,plans,sizes,int(sizes.max())):
        raw=packet(fields,dirty,start,desc)
        decode(out,start,desc,raw)
    for a,b in zip(out,fields):np.testing.assert_array_equal(a,b)
    # Single row with one 1-bit field per lane. Padding, bias, counts,
    # raw word count and descriptor offsets each have an independent adverse control.
    n=7
    small=rows(n)
    small[0][0,:]=[1,2,2,1,2,1,2]
    dirty,meta=summary(small)
    plans,sizes=_prices(n,meta)
    start,desc,size=next(_pages(n,plans,sizes,int(sizes.sum())))
    raw=packet(small,dirty,start,desc)
    for f in (decode,_restore):
        for key,changed in ((3,7),(4,2),(5,n+1),(6,1),(7,12),(8,6),
                             (9,65535),(11,18),(12,17),(13+8,17)):
            d=desc.copy();d[key]=changed
            try:f(rows(n),start,d,raw)
            except ValueError:pass
            else:raise AssertionError(('Accepted bad descriptor',key))
        for bad in (np.r_[raw,np.uint32(2)],raw[:-1],raw|np.uint32(1<<31)):
            try:f(rows(n),start,desc,bad)
            except ValueError:pass
            else:raise AssertionError('Accepted bad packed word count or nonzero padding.')
    small=rows(2051)
    small[1][6,::2]=65432
    small[0][7,1::509]=49200
    dirty,meta=summary(small)
    plans,sizes=_prices(2051,meta)
    start,desc,size=next(_pages(2051,plans,sizes,int(sizes.sum())))
    raw=packet(small,dirty,start,desc)
    for row in (6,7):
        base=int(desc[3+10*row+3])
        assert int(plans[row,0]) in (2,3)
        bad=raw.copy()
        bad[base]=np.uint32(0)
        try:decode(rows(2051),start,desc,bad)
        except ValueError:pass
        else:raise AssertionError('Accepted a damaged directory.')


class PriceControls(unittest.TestCase):
    def test_source_checked_shared_c_decoder_against_independent_packets(self):
        import json
        from pathlib import Path
        import subprocess
        import sys
        import tempfile
        from build_native import _build_one,find_compiler
        from classic_frame_defaults import verified_restore
        try:compiler=find_compiler()
        except RuntimeError:self.skipTest('Optional frame decoder needs a C compiler.')
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(verified_restore(temp))
            _build_one(compiler=compiler,output_dir=temp,stem='r35_board_codec_decode',
                       manifest_name='physical-board-build.json')
            subprocess.run([sys.executable,'-B','-c',
                            'import sys; from test_r35_frame_defaults import check_native; check_native(sys.argv[1])',
                            temp],check=True)
            path=Path(temp,'physical-board-build.json')
            manifest=json.loads(path.read_bytes())
            manifest['source_sha256']='0'*64
            path.write_text(json.dumps(manifest))
            self.assertIsNone(verified_restore(temp))

    def test_warm_price_hint_reads_native_arrays_and_never_reuses_host_frame_bytes(self):
        from unittest.mock import Mock
        from classic_checkpoint_ranges import CheckpointRanges
        from systematic_jobs import JobLedger
        from test_replica_resize import host_dfs
        n=128
        gpu=host_dfs(n)
        gpu.cp.RawKernel=Mock(side_effect=AssertionError('Warm price must avoid default-value device summary'))
        gpu.maxdepths[:]=101
        gpu.floors[:]=40
        for field in FIELDS:getattr(gpu,field)[159,7]=0xf123
        expected=gpu.snapshot_checkpoint()
        reader=CheckpointRanges(gpu,JobLedger.empty(0,n))
        got=reader.capture()
        self.assertEqual(reader.frame_defaults.distribution,{'warm_direct':True})
        self.assertEqual((reader.frame_defaults.maps,reader.frame_defaults.read_bytes),(0,0))
        self.assertEqual(reader.maps,len(gpu.DEVICE_FIELDS))
        for k in expected:np.testing.assert_array_equal(got[k],expected[k])
        for field in FIELDS:self.assertEqual(int(got[field][159,7]),0xf123)
        device=FrameDefaults(gpu)
        depths=np.full(n,101,np.int16)
        floors=np.full(n,54,np.int16)
        self.assertFalse(device.prefer_direct(depths,floors))
        floors-=1
        self.assertTrue(device.prefer_direct(depths,floors))
        self.assertFalse(FrameDefaults(gpu,force=True).prefer_direct(depths,floors))
        self.assertFalse(device.prefer_direct(depths,depths))
        self.assertFalse(device.prefer_direct(depths[:5],floors))
        self.assertFalse(device.prefer_direct(None,None))

    def test_webgpu_high_water_price_falls_back_to_exact_physical_buffers(self):
        from classic_checkpoint_ranges import CheckpointRanges
        from systematic_jobs import JobLedger
        from test_webgpu_checkpoint import memory_engine,oracle
        n=128
        gpu=memory_engine(n)
        gpu.buffers[5][1]=116
        gpu.buffers[5][2]=20
        reader=CheckpointRanges(gpu,JobLedger.empty(0,n),board_mode='ranges')
        expected=oracle(gpu)
        got=reader.capture()
        self.assertEqual(reader.frame_defaults.distribution,{'warm_direct':True})
        self.assertEqual((reader.frame_defaults.maps,reader.frame_defaults.read_bytes),(0,0))
        self.assertEqual(reader.maps,6)
        for name in expected:np.testing.assert_array_equal(got[name],expected[name])
        gpu.buffers[1][159,91]=0xffd3b435
        reader.installed[:]=True
        again=reader.capture()
        self.assertEqual(int(again['firsts'][159,91]),0xb435)
        self.assertEqual(reader.maps,6)
        for field in FIELDS:np.testing.assert_array_equal(got[field],expected[field])

    def test_large_varying_warm_rows_avoid_large_cpu_unpack(self):
        from classic_frame_defaults import _worth_unpacking
        plans=np.zeros((160,10),np.uint32)
        plans[:90,0]=1
        plans[:90,2]=plans[:90,5]=131072
        self.assertTrue(_worth_unpacking(131072,plans))
        plans[:90,9]=36
        self.assertFalse(_worth_unpacking(131072,plans))
        plans[1:,9]=0
        self.assertTrue(_worth_unpacking(131072,plans))


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
