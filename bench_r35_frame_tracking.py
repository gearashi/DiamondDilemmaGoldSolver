"""Paired native search overhead and full checkpoint equality for per-lane write tracking."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from bench_r35_parked_public import inputs
from classic_checkpoint_ranges import CheckpointRanges
from dfs_gpu import DFS
from search_problem import SearchProblem
from systematic_jobs import JobLedger


def trial(api,profile,before,n,setup,budgets,reference):
    opts,prefixes,lengths=setup
    shader='dfs_kernels.cu' if api=='CUDA' else 'dfs_kernels.wgsl'
    original=Path.read_text
    def image(path,*args,**kwargs):
        if profile=='previous' and path.name==shader:
            path=before/shader
        return original(path,*args,**kwargs)
    extra={}
    engine=DFS
    if api!='CUDA':
        import wgpu
        from dfs_webgpu import WebGPUDFS
        engine=WebGPUDFS
        extra['adapter']=next(a for a in wgpu.gpu.enumerate_adapters_sync()
                              if a.info['backend_type']==api and a.info['adapter_type']=='DiscreteGPU')
    with patch.object(Path,'read_text',image):
        gpu=engine(**opts,**extra,n=n,seed=71982,prefixes=[])
    try:
        if api=='CUDA' and profile=='previous':
            old=gpu.kernel
            def old_kernel(grid,block,args):
                if int(args[-1])!=0:
                    raise ValueError('Previous kernel requires an unsplit comparison budget.')
                return old(grid,block,args[:-3])
            gpu.kernel=old_kernel
        jobs=JobLedger.empty(len(prefixes),n)
        jobs.refill(gpu,prefixes,lengths)
        observer=CheckpointRanges(gpu,jobs,trace=profile=='tracked',board_mode='ranges')
        observer.capture()
        rows=[]
        for steps in budgets:
            left=steps
            gpu_ms=0.
            while left:
                batch=min(512,left)
                if profile=='previous':
                    module='dfs_gpu' if api=='CUDA' else 'dfs_webgpu'
                    with patch(module+'.dispatch_pages',return_value=[(0,n)]):
                        gpu_ms+=gpu.step(batch)
                else:
                    gpu_ms+=gpu.step(batch)
                left-=batch
            direct=gpu.snapshot_checkpoint()
            exact=observer.capture()
            for field in direct:
                np.testing.assert_array_equal(exact[field],direct[field],err_msg=field)
            digest=sha256()
            for field in gpu.DEVICE_FIELDS:
                digest.update(np.ascontiguousarray(direct[field]).view(np.uint8))
            signature=digest.hexdigest()
            if steps in reference:
                assert signature==reference[steps],(api,profile,steps,signature)
            reference[steps]=signature
            rows.append(dict(budget=steps,search_gpu_ms=round(gpu_ms,3),read=observer.read_bytes,
                             rank_spans=observer.trace_rows,physical_sha256=signature,
                             candidates=int(direct['counters'][0].sum())))
        return rows
    finally:
        if hasattr(gpu,'close'):gpu.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--api',required=True,choices=('CUDA','Vulkan','D3D12','Metal'))
    parser.add_argument('--previous',type=Path,required=True)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--lanes',type=int,default=32768)
    parser.add_argument('--budgets',nargs='+',type=int,default=[8,64,512,4096,16384])
    parser.add_argument('--rounds',type=int,default=2)
    args=parser.parse_args()
    p=SearchProblem(json.loads(args.tiles.read_bytes()))
    setup=inputs(p,args.lanes,False)
    ref={}
    result=[]
    for cycle in range(args.rounds):
        profiles=('previous','untracked','tracked')
        for profile in profiles if cycle%2==0 else profiles[::-1]:
            rows=trial(args.api,profile,args.previous.resolve(),args.lanes,setup,args.budgets,ref)
            result.append(dict(cycle=cycle,profile=profile,observations=rows))
            args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
