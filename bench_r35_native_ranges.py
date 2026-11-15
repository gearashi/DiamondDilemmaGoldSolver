"""Validate checkpoint ranges implied by frozen lane ownership and attained frame depth."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from bench_r35_parked_public import inputs,save_pending_gold
from classic_checkpoint_ranges import CheckpointRanges
from dfs_gpu import DFS
from search_problem import SearchProblem
from systematic_jobs import JobLedger


FRAMES=('firsts','lengths','cursors','shifts')


class Ranges:
    def __init__(self,gpu,ledger):
        self.gpu=gpu
        self.saved=gpu.snapshot_checkpoint()
        self.bank=ledger.paused
        self.assignment=ledger.cursor,ledger.paused.offset
        self.web=getattr(gpu,'backend',None)=='webgpu'

    def capture(self,ledger):
        gpu=self.gpu
        start=perf_counter()
        maps=read=0
        prior=self.saved
        signature=ledger.cursor,ledger.paused.offset
        previous=np.flatnonzero(np.isin(prior['states'],(0,1)))
        if self.bank is not ledger.paused or signature!=self.assignment:
            fresh=gpu.snapshot_checkpoint()
            count=sum(fresh[k].nbytes for k in gpu.DEVICE_FIELDS) if not self.web else 816*gpu.n*4
            maps=6 if self.web else 12
            self.saved=fresh
            self.bank=ledger.paused
            self.assignment=signature
            return fresh,dict(read=count,maps=maps,reset=True,seconds=round(perf_counter()-start,6))
        if not len(previous):
            return prior,dict(read=0,maps=0,reset=False,seconds=0,span=0,rows=0)
        lo,hi=int(previous[0]),int(previous[-1])+1
        columns=np.arange(lo,hi)
        devices={}
        def put(name,part):
            old=prior[name]
            if part.shape==old.shape:
                value=part
            else:
                value=old.copy()
                if old.ndim==1:
                    value[lo:hi]=part
                else:
                    value[:len(part),lo:hi]=part
            value.setflags(write=False)
            devices[name]=value
        def physical(index,rows):
            nonlocal maps,read
            raw=gpu._read_rows(gpu.buffers[index],np.arange(rows),columns)
            maps+=1
            read+=raw.nbytes
            return raw
        def cuda(name,rows=None):
            nonlocal maps,read
            arr=getattr(gpu,name)
            if arr.ndim==1:
                part=arr[lo:hi].get()
            else:
                part=arr[:rows,lo:hi].get()
            maps+=1
            read+=part.nbytes
            put(name,part)
        if self.web:
            raw=physical(5,11)
            for i,(k,dtype) in enumerate((('depths',np.int16),('maxdepths',np.int16),('floors',np.int16),
                                         ('states',np.uint8),('rng',np.uint32))):
                put(k,raw[i].astype(dtype,copy=True))
            put('counters',(raw[6::2].astype(np.uint64)<<np.uint64(32))|raw[5::2].astype(np.uint64))
            raw=physical(0,165)
            put('boards',raw[:160].astype(np.int16))
            put('used',raw[160:].copy())
        else:
            for k in ('boards','used','depths','maxdepths','floors','states','rng','counters'):
                cuda(k)
        # Native frame writes only reach the current depth, including the
        # next uninitialized child cursor. Host job installation resets all
        # rows and is guarded by the ownership check above.
        extent=min(160,int(np.max(devices['maxdepths'][lo:hi]))+1)
        for index,name in enumerate(FRAMES,1):
            if self.web:
                put(name,physical(index,extent).astype(np.uint16))
            else:
                cuda(name,extent)
        result={**prior,**devices}
        self.saved=result
        return result,dict(read=read,maps=maps,reset=False,span=hi-lo,rows=extent,seconds=round(perf_counter()-start,6))


def run(p,n,used,api,consume,wide,production=False,scatter=False,budgets=None,board_mode=None,dynamic=False,trace=False):
    args,prefixes,lengths=inputs(p,used,consume)
    if api=='CUDA':
        factory,extra=DFS,{}
    else:
        import wgpu
        from dfs_webgpu import WebGPUDFS
        factory=WebGPUDFS
        adapter=next(a for a in wgpu.gpu.enumerate_adapters_sync()
                     if a.info['backend_type']==api and a.info['adapter_type']=='DiscreteGPU')
        extra=dict(adapter=adapter)
    if dynamic:
        args['gpu_features']={'dynamic_mrv':True,'forced':True}
    gpu=factory(**args,n=n,seed=35068,prefixes=[],**extra)
    ledger=JobLedger.empty(len(prefixes),n)
    if scatter and len(prefixes)<n:
        cols=np.linspace(0,n-1,len(prefixes),dtype=np.int32)
        gpu.load_prefixes([row[:size] for row,size in zip(prefixes,lengths)],lanes=cols)
        ledger.ids[cols]=np.arange(len(prefixes))
        ledger.cursor=len(prefixes)
        ledger.validate(gpu.states.get())
    else:
        ledger.refill(gpu,prefixes,lengths)
    if production:
        reader=CheckpointRanges(gpu,ledger,board_mode=board_mode,trace=trace)
        first=reader.capture()
        old={k:v.copy() for k,v in first.items()}
    else:
        reader=Ranges(gpu,ledger)
    result=[]
    try:
        for nodes in (budgets or ((0,1,8,64,256,4096,32768) if wide else (0,1,8,64,256))):
            left=nodes
            gpu_ms=0.0
            while left:
                k=min(512,left)
                gpu_ms+=gpu.step(k)
                ledger.refill(gpu,prefixes,lengths)
                left-=k
            start=perf_counter()
            expected=gpu.snapshot_checkpoint()
            elapsed=perf_counter()-start
            save_pending_gold(p,gpu,expected['states'],Path(__file__).parent/'audit/r35/checkpoint-gold.json',expected['boards'])
            if production:
                start=perf_counter()
                actual=reader.capture()
                metrics=dict(read=reader.read_bytes,maps=reader.maps,seconds=round(perf_counter()-start,6),
                             active=int(np.count_nonzero(expected['states']<=1)),board_fallbacks=reader.board_fallbacks,
                             gpu_ms=round(gpu_ms,3),frame_rows=reader.trace_rows)
                if reader.board_projection is not None:
                    metrics['board_upload']=reader.board_projection.upload_bytes
                    metrics['board_read']=reader.board_projection.read_bytes
                if reader.board_delta is not None:
                    metrics.update(board_upload=reader.board_delta.upload_bytes,board_read=reader.board_delta.read_bytes,
                                   modes=reader.board_delta.distribution)
                assert set(actual)==set(expected)
                for k in first:
                    np.testing.assert_array_equal(old[k],first[k],err_msg=k)
            else:
                actual,metrics=reader.capture(ledger)
            for k in (expected if production else gpu.DEVICE_FIELDS):
                assert actual[k].dtype==expected[k].dtype and np.array_equal(actual[k],expected[k]),k
            raw=sum(expected[k].nbytes for k in gpu.DEVICE_FIELDS) if api=='CUDA' else 816*n*4
            result.append(dict(nodes=nodes,baseline_bytes=raw,baseline_seconds=round(elapsed,6),**metrics))
    finally:
        if hasattr(gpu,'close'):
            gpu.close()
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--api',choices=('CUDA','Vulkan','D3D12','Metal'),required=True)
    cli.add_argument('--wide',action='store_true')
    cli.add_argument('--production',action='store_true')
    cli.add_argument('--scatter',action='store_true')
    cli.add_argument('--maximum',action='store_true')
    cli.add_argument('--budgets',nargs='+',type=int)
    cli.add_argument('--board-mode',choices=('device','device-native','physical','ranges'))
    cli.add_argument('--dynamic',action='store_true')
    cli.add_argument('--trace',action='store_true')
    cli.add_argument('--decoder-dir',type=Path)
    args=cli.parse_args()
    if args.decoder_dir:
        import classic_checkpoint_delta as board_module
        from functools import partial
        board_module.verified_decoder=partial(board_module.verified_decoder,args.decoder_dir)

    p=SearchProblem(json.loads(args.tiles.read_bytes()))
    cases=[(128,128,False),(2048,128,False),(2048,2048,False),(128,4096,True)]
    if args.wide:
        cases.extend(((32768,512,False),(32768,32768,False)))
    if args.scatter:
        cases=[(2048,32,False),(2048,256,False),(32768,512,False)]
    if args.maximum:
        cases=[(131072,512,False),(131072,131072,False)]
    out={}
    for n,used,consume in cases:
        out[f'{n}/{used}/{consume}']=run(p,n,used,args.api,consume,args.wide,args.production,args.scatter,args.budgets,args.board_mode,args.dynamic,args.trace)
        args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
