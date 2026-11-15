"""Check actual maximum-pool lane pages against independent small native replicas."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from bench_r35_parked_public import inputs,save_pending_gold
from dfs_gpu import DFS
from search_problem import SearchProblem


def verify(api,n,budgets,tiles,output,dynamic):
    problem=SearchProblem(json.loads(tiles.read_bytes()))
    args,prefixes,lengths=inputs(problem,n,False)
    if dynamic:args['gpu_features']={'forced':True,'dynamic_mrv':True}
    engine,extra=DFS,{}
    if api!='CUDA':
        import wgpu
        from dfs_webgpu import WebGPUDFS
        engine=WebGPUDFS
        extra={'adapter':next(a for a in wgpu.gpu.enumerate_adapters_sync()
                              if a.info['backend_type']==api and a.info['adapter_type']=='DiscreteGPU')}
    selected=np.unique(np.r_[0,1,126,127,128,129,256,n-2,n-1,np.linspace(0,n-1,34,dtype=np.int32)])
    selected=selected[(selected>=0)&(selected<n)]
    full=engine(**args,**extra,n=n,seed=31788,prefixes=[])
    small=None
    try:
        full.load_prefixes([row[:size] for row,size in zip(prefixes,lengths)])
        snapshot=full.snapshot_checkpoint()
        imported=snapshot.copy()
        for k in full.DEVICE_FIELDS:
            imported[k]=snapshot[k][...,selected].copy()
        imported['roots']=snapshot['roots'][selected]
        imported['prefix_codes']=snapshot['prefix_codes'][selected]
        small=engine(**args,**extra,n=len(selected),seed=8,prefixes=[])
        small.restore_checkpoint(imported)
        del snapshot,imported
        records=[]
        for nodes in budgets:
            times=[perf_counter()]
            def poll():
                times.append(perf_counter())
                return False
            full._page_stop=poll
            elapsed=full.step(nodes)
            times.append(perf_counter())
            flags=full.states.get()
            save_pending_gold(problem,full,flags,output.with_name('page-verified-gold.json'))
            small.step(nodes)
            raw=small.snapshot_checkpoint()
            for field in full.DEVICE_FIELDS:
                source=getattr(full,field)
                subset=source[:,selected].get() if len(source.shape)==2 else source[selected].get()
                np.testing.assert_array_equal(subset,raw[field],err_msg=field)
            durations=np.diff(times)*1000
            records.append(dict(replicas=n,budget=nodes,completed_pages=len(durations),elapsed_ms=elapsed,
                                page_max_ms=float(durations.max()),page_median_ms=float(np.median(durations)),
                                compared_lanes=selected.tolist(),active=int((flags==0).sum()),
                                native_endpoints_equal=True))
            output.write_text(json.dumps(records,indent=2)+'\n',encoding='utf8')
    finally:
        if small is not None and hasattr(small,'close'):small.close()
        if hasattr(full,'close'):full.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--api',required=True)
    p.add_argument('--lanes',type=int,default=131072);p.add_argument('--budgets',nargs='+',type=int,default=[1,8,64,512])
    p.add_argument('--tiles',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dynamic',action='store_true');a=p.parse_args()
    verify(a.api,a.lanes,a.budgets,a.tiles,a.output,a.dynamic)
