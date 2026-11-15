"""Exercise actual future-pose conflicts, prefix floors and existing native frontier controls."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from r28_exchange import read_json
from r35_physical_adjacency import geographic_inventory
from search_problem import SearchProblem
from validator import reverse_mask


def controls(p):
    rules=geographic_inventory(p)
    home=0;target=int(p.neighbors[home,0])
    outside=next(int(v) for v in p.neighbors[target] if v!=home)
    side=list(p.neighbors[target]).index(outside);opposite=int(p.sides[target,side])
    anchor_opposite=int(p.sides[home,0])
    owned={d//3 for pair in rules.required for d in pair}
    rows={}
    for x,y in sorted(rules.required):
        for a,b in ((x,y),(y,x)):
            tile,local=divmod(a,3);partner,physical=divmod(b,3)
            head=3*tile+(-local)%3;expected=3*partner+(anchor_opposite-physical)%3
            blocked=int(p.masks[expected,side])
            bad=next(code for code in range(3*p.n) if code//3 not in owned and
                     reverse_mask(int(p.masks[code,opposite]))!=blocked)
            good=next(code for code in range(3*p.n) if code//3 not in owned and
                      reverse_mask(int(p.masks[code,opposite]))==blocked)
            rows[head]=dict(remote=3*partner,external=bad,permitted=good)
    return (home,target,outside),rows


def native(p,api):
    from dfs_gpu import DFS,fill_order
    from gpu_reasoning import pack_frame
    import wgpu
    from dfs_webgpu import WebGPUDFS
    if api=='CUDA':factory,kwargs=DFS,{}
    else:
        adapter=next(x for x in wgpu.gpu.enumerate_adapters_sync()
                     if x.info['backend_type']==api and x.info['adapter_type']=='DiscreteGPU')
        factory,kwargs=WebGPUDFS,dict(adapter=adapter)
    (home,target,outside),rows=controls(p)
    origin=list(map(int,fill_order(p.neighbors)))
    order=[home,outside]+[i for i in origin if i not in (home,outside)]
    results={}
    def start(prefixes,path,gold=True,dynamic=False):
        return factory(p.masks,p.neighbors,p.sides,n=len(prefixes),order=path,prefixes=prefixes,
                       gold_data=p.tiles if gold else None,gpu_features={'dynamic_mrv':dynamic},seed=35,**kwargs)
    for kind in ('remote','external','permitted'):
        prefixes=[[head,child[kind]] for head,child in rows.items()]
        out=[]
        for active in (False,True):
            context=nullcontext() if active else patch('r35_gpu_mates.pose_table',return_value=np.empty(0,np.uint32))
            with context:gpu=start(prefixes,order)
            try:
                gpu.step(2 if kind=='permitted' else 64)
                gpu.verify()
                out.append(dict(exhausted=int((gpu.states.get()==2).sum()),tries=int(gpu.counters.get()[0].sum()),
                                lanes=len(prefixes),table_words=int(gpu.host_future_poses.size)))
            finally:
                if hasattr(gpu,'close'):gpu.close()
        if kind=='permitted':assert out[1]['exhausted']==out[0]['exhausted']==0,out
        else:assert out[1]['exhausted']==len(prefixes) and out[1]['tries']==0 and out[0]['tries']>0,out
        results[kind]=out
    heads=sorted(rows)
    reverse_runs=[]
    for active in (False,True):
        context=nullcontext() if active else patch('r35_gpu_mates.pose_table',return_value=np.empty(0,np.uint32))
        with context:gpu=start([[v] for v in heads],order)
        try:
            saved={name:val.copy() for name,val in gpu.snapshot_checkpoint().items()}
            for lane,head in enumerate(heads):
                child=rows[head]['external']
                shift,=np.flatnonzero(gpu.host_pool[:480]==child)
                saved['firsts'][1,lane]=0;saved['lengths'][1,lane]=480
                saved['cursors'][1,lane]=0;saved['shifts'][1,lane]=int(shift)
            gpu.restore_checkpoint(saved);gpu.step(1);gpu.step(1);gpu.verify()
            depth=gpu.depths.get()
            assert np.all(depth==1),depth
            reverse_runs.append(dict(tried=int(gpu.counters.get()[0].sum()),placed=int(gpu.counters.get()[1].sum())))
        finally:
            if hasattr(gpu,'close'):gpu.close()
    results['later_unrelated_candidate']=reverse_runs
    for dynamic in (False,True):
        if dynamic:order=[home,target]+[i for i in origin if i not in (home,target)]
        gpu=start([[v] for v in heads],order,dynamic=dynamic)
        try:
            saved={name:val.copy() for name,val in gpu.snapshot_checkpoint().items()}
            for lane,head in enumerate(heads):
                child=rows[head]['remote']
                shift,=np.flatnonzero(gpu.host_pool[:480]==child)
                firsts,shifts=pack_frame(0,int(shift),outside) if dynamic else (0,int(shift))
                saved['firsts'][1,lane]=firsts;saved['shifts'][1,lane]=shifts
                saved['lengths'][1,lane]=480;saved['cursors'][1,lane]=1
                saved['depths'][lane]=saved['maxdepths'][lane]=2
                saved['boards'][outside,lane]=child
                tile=child//3;saved['used'][tile//32,lane]|=np.uint32(1<<(tile%32))
            gpu.restore_checkpoint(saved);gpu.verify();gpu.step(1);gpu.verify()
            now=gpu.snapshot_checkpoint()
            assert np.all(now['depths']==1) and np.all(now['states']==0)
            assert np.all(now['boards'][outside]==-1) and np.array_equal(now['boards'][home],heads)
            for field in ('rng','firsts','lengths','shifts','cursors','prefix_codes'):
                np.testing.assert_array_equal(now[field],saved[field])
            results['restore_dynamic' if dynamic else 'restore_static']=dict(restored=len(heads),
                        unplaced=int(now['counters'][2].sum()),actual_child_rank=1,static_child_rank=order.index(outside))
        finally:
            if hasattr(gpu,'close'):gpu.close()
    order=[home,outside]+[i for i in origin if i not in (home,outside)]
    gpu=start([[h,c['remote']] for h,c in rows.items()],order,gold=False)
    try:
        gpu.step(64);gpu.verify()
        results['edges']=dict(table_words=int(gpu.host_future_poses.size),tries=int(gpu.counters.get()[0].sum()))
        assert not gpu.host_future_poses.size and results['edges']['tries']>0
    finally:
        if hasattr(gpu,'close'):gpu.close()
    return results


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--api',choices=('CUDA','D3D12','Vulkan','Metal'),required=True)
    cli.add_argument('--output',type=Path)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles))
    result=native(p,args.api)
    text=json.dumps(result,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':main()
