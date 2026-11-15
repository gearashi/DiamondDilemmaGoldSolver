"""Verify actual v1.14 native histories and count classic replay device/pipeline builds."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch

import numpy as np


def capture(tiles, work):
    # The caller puts a clean released checkout first on sys.path before any project import.
    from dfs_gpu import DFS
    from search_problem import SearchProblem
    from systematic_jobs import JobLedger
    p = SearchProblem(tiles)
    with np.load(work/'front.npz') as f:
        order, prefixes, lengths = f['order'],f['prefixes'],f['lengths']
    carry, history = None, []
    for i,(lanes, budget) in enumerate(((16,4),(4,7),(8,3))):
        gpu=DFS(p.masks,p.neighbors,p.sides,n=lanes,seed=35070+i,order=order,prefixes=[],gold_data=tiles)
        if carry is None:
            ledger=JobLedger.empty(len(prefixes),lanes)
            ledger.refill(gpu,prefixes,lengths)
        else:
            ledger=JobLedger.resume(gpu,carry,prefixes,lengths)
        for step in range(2):
            gpu.step(budget)
            states=gpu.states.get()
            assert not np.any(states==1)
            ledger.retire(states)
            if step==0 and np.any(ledger.ids<0):
                ledger.refill(gpu,prefixes,lengths)
        carry={**gpu.snapshot_checkpoint(),**ledger.fields()}
        history.append([dict(replicas=lanes,seed=35070+i,adaptive_gpu_batches=False),
                        dict(checkpoint={'completed_generation':2*(i+1)},gpu_work_budget=budget,
                             assigned_jobs=ledger.cursor,paused_jobs=ledger.paused_count)])
    np.savez(work/'saved.npz',**carry)
    (work/'history.json').write_text(json.dumps(history),encoding='utf8')


def experiment(tiles, args, work):
    from bench_r35_gpu_mates import controls
    from dfs_gpu import DFS, fill_order
    from dfs_webgpu import WebGPUDFS, _SearchProgram
    from search_problem import SearchProblem
    from systematic_evidence import verify_native_history
    from gpu_reasoning import normalize_features
    p=SearchProblem(tiles)
    (home,target,outside),cases=controls(p)
    order=[home,outside]+[int(i) for i in fill_order(p.neighbors) if i not in (home,outside)]
    rows=sorted((head,values[key]) for head,values in cases.items() for key in ('remote','permitted'))
    prefixes=np.full((len(rows),160),-1,np.int16)
    prefixes[:,:2]=rows
    lengths=np.full(len(rows),2,np.int16)
    np.savez(work/'front.npz',prefixes=prefixes,lengths=lengths,order=order)
    subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--capture',str(work),
        '--old-root',str(args.old_root.resolve()),'--tiles',str(args.tiles.resolve())],check=True)
    history=json.loads((work/'history.json').read_bytes())
    source=work/'saved.npz'
    with np.load(source,allow_pickle=False) as f:
        saved={k:f[k] for k in f.files}
    assert int(saved['counters'][0].sum())>0
    saved['rng'][0]^=np.uint32(1)
    np.savez(work/'forged.npz',**saved)
    base=dict(masks=p.masks,neighbors=p.neighbors,neighbor_sides=p.sides,order=order,gold_data=tiles)
    features=normalize_features({})
    out={'v114_fields':len(saved),'jobs':len(rows),'generations':6,
         'cuda_valid':verify_native_history(DFS,base,features,source,history,prefixes,lengths,lambda:False)}
    assert out['cuda_valid']
    import wgpu
    if args.dxc:
        from wgpu.backends.wgpu_native.extras import set_instance_extras
        dll=args.dxc.resolve(strict=True)
        set_instance_extras(dx12_compiler='Dxc',dxc_path=str(dll))
        out['dxc_sha256']=hashlib.sha256(dll.read_bytes()).hexdigest()
    for api in args.api:
        adapter=next(a for a in wgpu.gpu.enumerate_adapters_sync()
                     if a.info['backend_type']==api and a.info['adapter_type']=='DiscreteGPU')
        runs=[]
        for share in (False,True):
            builds=[]
            previous=_SearchProgram.__init__
            def build(instance,*args):
                previous(instance,*args)
                builds.append(instance)
            with patch.object(_SearchProgram,'__init__',build):
                start=perf_counter()
                live=WebGPUDFS(**base,n=64,seed=7,adapter=adapter,prefixes=[[int(x) for x in rows[0]]])
                try:
                    original=live.snapshot_checkpoint()
                    inputs=dict(base,adapter=adapter)
                    if share:
                        inputs['reuse_from']=live
                    verified=verify_native_history(WebGPUDFS,inputs,features,source,history,prefixes,lengths,lambda:False)
                    assert verified,(api,share)
                    count=len(builds)
                    after=live.snapshot_checkpoint()
                    assert set(original)==set(after) and all(np.array_equal(original[k],after[k]) for k in original)
                    elapsed=perf_counter()-start
                    if share:
                        assert not verify_native_history(WebGPUDFS,inputs,features,work/'forged.npz',history,prefixes,lengths,lambda:False)
                        assert not verify_native_history(WebGPUDFS,inputs,features,source,history,prefixes,lengths,lambda:True)
                        assert len(builds)==count
                        child=WebGPUDFS(**inputs,n=4,seed=1,prefixes=[])
                        assert child.pipeline is live.pipeline and child.parameters is not live.parameters
                        assert all(x is not y for x,y in zip(child.buffers,live.buffers))
                        live.close()
                        child.step(1)
                        child.verify()
                        child.close()
                        child.close()
                finally:
                    live.close()
                assert count==(1 if share else 4),(api,share,count)
                runs.append(dict(shared=share,devices=count,programs=count,native_histories=3,seconds=round(elapsed,4)))
        out[api]=runs
    return out


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True,type=Path)
    cli.add_argument('--old-root',required=True,type=Path)
    cli.add_argument('--capture',type=Path)
    cli.add_argument('--dxc',type=Path)
    cli.add_argument('--api',action='append',choices=('Vulkan','D3D12','Metal'),default=[])
    cli.add_argument('--output',type=Path)
    args=cli.parse_args()
    data=json.loads(args.tiles.read_bytes())
    if args.capture:
        sys.path.insert(0,str(args.old_root.resolve()))
        capture(data,args.capture)
    else:
        with TemporaryDirectory() as tmp:
            out=experiment(data,args,Path(tmp))
        args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
