"""Independent bit-level and native readback controls for physical board delta checkpoints."""
import os
import unittest
from unittest.mock import patch

import numpy as np

import classic_checkpoint_delta as delta
from classic_checkpoint_ranges import CheckpointRanges
from systematic_jobs import JobLedger
from test_gpu_gold import gold_fixture
from test_r35_checkpoint_ranges import NativeControls, compare


def columns(n):
    return np.full((160,n),-1,np.int16),np.zeros((5,n),np.uint32)


def adversarial(n=271):
    """Ten column pairs select all five encodings, with and without invalid codes."""
    old=columns(n)
    new=columns(n)
    ids=np.array([0,1,2,5,31,32,63,64,126,127,128,129,255,256,n-1],np.int32)
    invalid=(-32768,-2,480,32767)
    for j,lane in enumerate(ids[:10]):
        mode=j%5
        old[0][:,lane]=12
        new[0][:,lane]=12
        if mode==0:
            new[0][159,lane]=479
        elif mode==1:
            new[0][:,lane]=-1
            new[0][159,lane]=479
        elif mode==2:
            new[0][1::4,lane]=479
        elif mode==3:
            new[0][:,lane]=-1
            new[0][1::4,lane]=479
        else:
            new[0][:,lane]=(3*np.arange(160)+1)%480
        if j>=5:
            row=int(np.flatnonzero(new[0][:,lane]!=-1)[-1])
            new[0][row,lane]=invalid[j%4]
        new[1][:,lane]=np.array([0x80000001,0xffffffff,0,0x12345678,j],np.uint32)
    new[1][4,ids[10]]=0x80000000
    old[0][:,ids[11]]=0
    new[0][[0,31,32,63,64,127,128,159],ids[12]]=np.arange(8)
    return old,new,ids


def oracle_packet(old,new,lanes):
    """Build exact packet words independently with integer bit concatenation."""
    tickets=[]
    payload=[]
    modes=[]
    for index,lane in enumerate(lanes):
        before,after=old[0][:,lane],new[0][:,lane]
        changes=np.flatnonzero(before!=after).tolist()
        present=np.flatnonzero(after!=-1).tolist()
        used=np.flatnonzero(old[1][:,lane]!=new[1][:,lane]).tolist()
        if not changes and not used:continue
        bits=16 if any(v < -1 or v >= 480 for v in after.tolist()) else 9
        choices=(changes,present,changes,present,range(160))
        def cost(i):
            return (len(choices[i])*(bits+(8 if i<2 else 0))+31)//32+(5 if i in (2,3) else 0)
        mode=min(range(5),key=cost)
        rows=choices[mode]
        if mode in (2,3):
            mask=sum(1<<i for i in rows)
            payload.extend((mask>>(32*i))&0xffffffff for i in range(5))
        width=bits+(8 if mode<2 else 0)
        value=0
        for rank,row in enumerate(rows):
            code=int(after[row])
            code=480 if bits==9 and code==-1 else code & 0xffff
            token=code | ((row<<bits) if mode<2 else 0)
            value|=token<<(width*rank)
        payload.extend((value>>shift)&0xffffffff for shift in range(0,len(rows)*width,32))
        payload.extend(int(new[1][row,lane]) for row in used)
        ticket=1+sum(1<<i for i in used)+32*(10*len(rows)+mode+(5 if bits==16 else 0))
        tickets.append((index,ticket))
        modes.append((mode,bits))
    count=len(lanes)
    form=1 if count<=65536 else 2
    if (count+1)//2<=form*len(tickets):form=0
    if form==0:
        directory=[0]*((count+1)//2)
        for i,t in tickets:directory[i//2]|=t<<((i%2)*16)
    elif form==1:directory=[(i<<16)|t for i,t in tickets]
    else:directory=[v for entry in tickets for v in entry]
    header=np.array([0,len(tickets),len(payload),len(directory),form],np.uint32)
    return header,np.array(directory+payload,np.uint32),modes


def assert_pair(test,expected,actual):
    test.assertIsNotNone(actual)
    for want,got in zip(expected,actual,strict=True):
        test.assertEqual(want.dtype,got.dtype)
        np.testing.assert_array_equal(want,got)


class PacketOracleTests(unittest.TestCase):
    def test_all_encodings_bit_widths_masks_and_sparse_lane_tables(self):
        old,new,ids=adversarial()
        for lanes,form in ((ids,0),(np.arange(old[0].shape[1]),1)):
            header,body,modes=oracle_packet(old,new,lanes)
            self.assertEqual(int(header[-1]),form)
            self.assertTrue({(i,b) for b in (9,16) for i in range(5)}<=set(modes))
            offsets,changed,layout,payload=delta._directory(len(lanes),header,body)
            for native in (False,True):
                decoder=None
                if native:
                    decoder=delta.verified_decoder()
                    if decoder is None:continue
                result=delta._decode(old,lanes[offsets],changed,layout,payload,decoder)
                assert_pair(self,new,result)
        self.assertEqual(old[0][159,ids[0]],12)
        self.assertEqual(old[1].sum(),0)

    def test_deterministic_scattered_changes_carry_every_matching_raw_value(self):
        rng=np.random.default_rng(42035)
        for n in (1,2,7,65,129):
            old=columns(n)
            old[0][:]=rng.integers(-1,480,size=old[0].shape,dtype=np.int16)
            new=tuple(value.copy() for value in old)
            for i in range(n):
                if i%3==0:
                    rows=rng.choice(160,(31*i)%161,replace=False)
                    new[0][rows,i]=rng.choice(np.array([-32768,-2,-1,0,7,479,480,32767],np.int16),len(rows))
                if i%5==0:new[1][i%5,i]=0xf0349871
            lanes=np.arange(n,dtype=np.int32)
            header,body,_=oracle_packet(old,new,lanes)
            ids,summary,layout,data=delta._directory(n,header,body)
            assert_pair(self,new,delta._decode(old,lanes[ids],summary,layout,data))

    def test_directory_width_at_65536_and_absolute_maximum(self):
        # A single used-word change in the last active lane, with no board token.
        for k,form in ((65535,1),(65536,1),(65537,2),(131072,2)):
            ticket=1+16
            directory=[((k-1)<<16)|ticket] if form==1 else [k-1,ticket]
            head=np.array([0,1,1,form,form],np.uint32)
            body=np.array([*directory,0x80000001],np.uint32)
            ids,changed,layout,payload=delta._directory(k,head,body)
            self.assertEqual(ids.tolist(),[k-1])
            self.assertFalse(changed.any())
            self.assertEqual((int(layout[3][0]),int(payload[0])),(16,0x80000001))

    def test_corrupt_physical_tokens_match_native_rejection(self):
        # Each malformed body fits the advertised byte budget, unlike truncation.
        old=columns(8)
        for mode,count,words in (
                (0,1,[481]), (0,1,[(160<<9)|1]),
                (0,2,[1|(1<<17),0]),
                (2,1,[0,0,0,0,0,0]), (2,1,[3,0,0,0,0,0])):
            ticket=1+32*(10*count+mode)
            header=np.array([0,1,len(words),1,1],np.uint32)
            packet=np.array([ticket,*words],np.uint32)
            indices,changed,layout,payload=delta._directory(8,header,packet)
            decoders=[None]
            native=delta.verified_decoder()
            if native:decoders.append(native)
            for decoder in decoders:
                with self.subTest(mode=mode,count=count,words=words,native=decoder is not None),self.assertRaises(ValueError):
                    delta._decode(old,indices,changed,layout,payload,decoder)

    def test_invalid_counts_duplicates_and_truncation_are_rejected(self):
        for k,meta,packet in (
                (10,[1,0,0,0,1],[]),(10,[0,11,0,0,1],[]),
                (10,[0,1,0,1,1],[10<<16|1]),(10,[0,2,0,2,1],[1,1]),
                (10,[0,1,2,1,1],[2,0]),(10,[0,1,0,1,1],[0]),
                (10,[0,1,0,1,1],[51521]),(10,[0,1,0,1,1],[1+32*4])):
            with self.subTest(meta=meta,packet=packet),self.assertRaises(ValueError):
                delta._directory(k,np.array(meta,np.uint32),np.array(packet,np.uint32))


class NativePacketChecks:
    pool=NativeControls.pool

    def put(self,gpu,fields):
        gpu.boards[:,:]=gpu.cp.asarray(fields[0])
        gpu.used[:,:]=gpu.cp.asarray(fields[1])

    def test_packets_match_independent_bits_readback_and_immutable_generations(self):
        old,new,ids=adversarial()
        gpu=self.pool(*gold_fixture(),old[0].shape[1])
        for lanes in (ids[:13],ids,np.arange(gpu.n,dtype=np.int32)):
            self.put(gpu,old)
            initial=gpu.snapshot_checkpoint()
            codec=delta.BoardDelta(gpu,device_plan=True)
            self.addCleanup(codec.close)
            codec.seed(initial)
            unchanged=codec.capture(lanes)
            assert_pair(self,old,unchanged)
            self.put(gpu,new)
            expected=tuple(a.get() for a in (gpu.boards,gpu.used))
            header,packet,_=oracle_packet(old,new,lanes)
            directory=delta._directory
            def audited(k,head,body):
                self.assertEqual(k,len(lanes))
                np.testing.assert_array_equal(head,header)
                np.testing.assert_array_equal(body,packet)
                return directory(k,head,body)
            with patch.object(delta,'_directory',side_effect=audited) as decoder:
                first=codec.capture(lanes)
                decoder.assert_called_once()
            assert_pair(self,expected,first)
            assert_pair(self,old,(initial['boards'],initial['used']))
            assert_pair(self,expected,codec.capture(lanes))
            empty=columns(gpu.n)
            self.put(gpu,empty)
            assert_pair(self,empty,codec.capture(lanes))
            assert_pair(self,expected,first)
            codec.close()
            restarted=delta.BoardDelta(gpu,device_plan=True)
            self.addCleanup(restarted.close)
            restarted.seed(gpu.snapshot_checkpoint())
            self.put(gpu,new)
            assert_pair(self,expected,restarted.capture(lanes))

    def test_sparse_final_lane_on_both_directory_widths_and_maximum(self):
        from contextlib import ExitStack
        from types import SimpleNamespace
        n=131072
        base=self.pool(*gold_fixture(),2)
        board,masks=columns(n)
        if self.web:
            usage=base.wgpu.BufferUsage
            device=base.adapter.request_device_sync(required_limits={'max-buffer-size':660*n,'max-storage-buffer-binding-size':660*n})
            self.addCleanup(device.destroy)
            source=device.create_buffer(size=660*n,usage=usage.STORAGE|usage.COPY_SRC|usage.COPY_DST)
            self.addCleanup(source.destroy)
            owned=ExitStack()
            self.addCleanup(owned.close)
            gpu=SimpleNamespace(n=n,backend='webgpu',gpu_device=device,
                queue=device.queue,wgpu=base.wgpu,buffers=[source],_resources=owned,readback_bytes=0)
            # The WebGPU physical representation has one uint32 word per int16 cell.
            row=np.full(n,0xffffffff,np.uint32)
            for i in range(160):gpu.queue.write_buffer(source,i*n*4,row)
            def change(row,lane,value):
                gpu.queue.write_buffer(source,(row*n+lane)*4,np.array([value],np.uint32))
            def readback():
                raw=np.frombuffer(gpu.queue.read_buffer(source),np.uint32).reshape(165,n)
                return raw[:160].astype(np.int16),raw[160:].copy()
        else:
            gpu=SimpleNamespace(n=n,backend='cuda',cp=base.cp,boards=base.cp.asarray(board),used=base.cp.asarray(masks))
            def change(row,lane,value):
                if row<160:gpu.boards[row,lane]=value
                else:gpu.used[row-160,lane]=value
            def readback():
                return gpu.boards.get(),gpu.used.get()
        codec=delta.BoardDelta(gpu,device_plan=True)
        self.addCleanup(codec.close)
        codec.seed(dict(boards=board,used=masks))
        for count in (65536,65537,131072):
            lane=count-1
            change(159,lane,479)
            change(164,lane,0x80000001)
            with self.subTest(count=count):
                expected=readback()
                observed=[]
                inspect=delta._directory
                def record(k,head,words):
                    observed.append((k,int(head[1]),int(head[4])))
                    return inspect(k,head,words)
                with patch.object(delta,'_directory',side_effect=record):
                    actual=codec.capture(np.arange(count,dtype=np.int32))
                assert_pair(self,expected,actual)
                self.assertEqual(observed,[(count,1,1 if count==65536 else 2)])
        self.assertEqual(int(board[159,-1]),-1)
        self.assertEqual(int(masks[4,-1]),0)

    def test_rejected_packet_after_mirror_advance_reads_full_state_and_retries(self):
        gpu=self.pool(*gold_fixture(),256)
        ids=np.array([0,31,32,127,128,255],np.int32)
        gpu.load_prefixes([[3*i] for i in range(len(ids))],lanes=ids)
        reader=CheckpointRanges(gpu,JobLedger.empty(0,gpu.n),board_mode='device')
        first=compare(self,gpu,reader)
        saved=first['boards'].copy(),first['used'].copy()
        gpu.step(7)
        with patch.object(delta,'_directory',side_effect=ValueError('Failed directory after mirror update')) as broken:
            second=compare(self,gpu,reader)
            broken.assert_called_once()
        frozen=second['boards'].copy(),second['used'].copy()
        self.assertGreaterEqual(reader.board_fallbacks,1)
        gpu.step(9)
        compare(self,gpu,reader)
        compare(self,gpu,reader)
        assert_pair(self,saved,(first['boards'],first['used']))
        assert_pair(self,frozen,(second['boards'],second['used']))


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','requires a CUDA device')
class CudaDeltaTests(NativePacketChecks,unittest.TestCase):
    web=False


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_WEBGPU')=='1','requires a WebGPU device')
class WebDeltaTests(NativePacketChecks,unittest.TestCase):
    web=True


if __name__=='__main__':
    unittest.main()
