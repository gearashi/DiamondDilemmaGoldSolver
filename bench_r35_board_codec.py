"""Compare native board/used transfer and CPU decode on the same actual DFS snapshots."""
import argparse
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

from bench_r35_parked_public import inputs
from classic_checkpoint_boards import BoardProjection
from classic_checkpoint_ranges import CheckpointRanges
from dfs_gpu import DFS
from io_utils import atomic_json
from r28_exchange import read_json
from r35_board_codec import PhysicalBoardCodec,native_unpack
from search_problem import SearchProblem
from systematic_jobs import JobLedger


def run(problem,n,live,dynamic,api,steps,repeats,library):
    args,rows,lengths=inputs(problem,live,False)
    if dynamic:args['gpu_features']={'dynamic_mrv':True,'forced':True}
    if api=='CUDA':
        factory,extra=DFS,{}
    else:
        import wgpu
        from dfs_webgpu import WebGPUDFS
        adapter=next(a for a in wgpu.gpu.enumerate_adapters_sync() if a.info['backend_type']==api and
                     a.info['adapter_type'] in ('DiscreteGPU','IntegratedGPU'))
        factory,extra=WebGPUDFS,dict(adapter=adapter)
    gpu=factory(**args,n=n,prefixes=[],seed=35068,**extra)
    ids=np.linspace(0,n-1,live,dtype=np.int32)
    gpu.load_prefixes([r[:d] for r,d in zip(rows,lengths)],lanes=ids)
    ledger=JobLedger.empty(live,n)
    reader=CheckpointRanges(gpu,ledger)
    if library:
        import ctypes
        decoder=native_unpack(ctypes.CDLL(str(library.resolve())))
    else:decoder='numpy'
    codec=PhysicalBoardCodec(gpu,decoder)
    frames={name:BoardProjection(gpu) for name in ('native','direct','proof')}
    idle=np.zeros(n,bool)
    totals=[]
    def raw_read():
        reader.maps=reader.read_bytes=0
        if api=='CUDA':return reader._read('boards',ids),reader._read('used',ids)
        physical=reader._physical(0,0,165,ids)
        return physical[:160].astype(np.int16),physical[160:].copy()
    try:
        for nodes in steps:
            if nodes:gpu.step(nodes)
            fields=gpu.snapshot_checkpoint()
            live_depth=fields['depths'][ids]
            expected=(fields['boards'][:,ids],fields['used'][:,ids])
            records={m:[] for m in ('raw','physical','native','direct')}
            for rep in range(repeats+1):
                modes=list(records)
                if rep%2:modes.reverse()
                for mode in modes:
                    start=perf_counter()
                    if mode=='raw':
                        got=raw_read()
                        nmaps,nbytes,up,decode=reader.maps,reader.read_bytes,0,0
                    elif mode=='physical':
                        got=codec.capture(fields,ids)
                        nmaps,nbytes,up,decode=codec.maps,codec.read_bytes,codec.upload_bytes,codec.decode_seconds
                    else:
                        frame=frames[mode]
                        got=frame.capture(fields,ids,mode=mode,installed=idle)
                        nmaps,nbytes,up,decode=frame.maps,frame.read_bytes,frame.upload_bytes,None
                    fallback=got is None
                    if fallback:
                        got=raw_read()
                        nmaps+=reader.maps
                        nbytes+=reader.read_bytes
                    elapsed=perf_counter()-start
                    if not all(np.array_equal(a,b) for a,b in zip(got,expected)):raise AssertionError((mode,n,live,nodes))
                    if rep:records[mode].append(dict(seconds=elapsed,decode=decode,read_bytes=int(nbytes),
                                                     upload_bytes=int(up),maps=int(nmaps),fallback=fallback))
            bad_rows={}
            mutable=ids[live_depth>fields['floors'][ids]]
            if len(mutable):
                target=int(mutable[0])
                stale=dict(fields,depths=fields['depths'].copy())
                stale['depths'][target]=stale['floors'][target]
                for mode in ('physical','native','direct','proof'):
                    got=(codec.capture(stale,ids) if mode=='physical' else
                         frames[mode].capture(stale,ids,mode=mode,installed=idle))
                    bad_rows[mode]=('declined' if got is None else 'exact' if all(np.array_equal(a,b) for a,b in zip(got,expected))
                                    else 'different')
            metrics={key:{field:(None if rows[0][field] is None else median(v[field] for v in rows))
                          for field in rows[0]} for key,rows in records.items()}
            totals.append(dict(n=n,selected=live,dynamic=dynamic,nodes_per_lane=nodes,depth_min=int(live_depth.min()),
                               depth_max=int(live_depth.max()),depth_mean=float(live_depth.mean()),measured=metrics,
                               stale_depth_control=bad_rows,samples=records))
            print(api,n,live,dynamic,nodes,[(a,round(v['seconds']*1000,2),v['read_bytes']) for a,v in metrics.items()],flush=True)
    finally:
        if hasattr(gpu,'close'):gpu.close()
    return totals


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--api',choices=('CUDA','Vulkan','D3D12','Metal'),required=True)
    p.add_argument('--shape',nargs='+',default=['128/128','2048/128','2048/2048'])
    p.add_argument('--steps',nargs='+',type=int,default=[0,4,64])
    p.add_argument('--decoder',type=Path)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    report=[]
    for shape in args.shape:
        n,live=map(int,shape.split('/'))
        for dynamic in (False,True):
            report.extend(run(problem,n,live,dynamic,args.api,args.steps,args.repeats,args.decoder))
            atomic_json(args.output,dict(backend=args.api,fingerprint=problem.fingerprint,rows=report))


if __name__=='__main__':main()
