"""Exercise the production parked writer and cold resize on original CUDA-owned work."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bench_r35_checkpoint_storage import io_written
from bench_r35_gpu_mates import controls
from classic_parked_storage import ParkedCheckpointWriter, REFERENCE, restore_parked
from dfs_gpu import DFS, fill_order
from exact_frontier import plan_frontier
from search_problem import SearchProblem
from systematic_jobs import JobLedger
from systematic_search import atomic_npz
import systematic_evidence as proof


def inputs(p, total, consume):
    order=list(map(int,fill_order(p.neighbors)))
    if consume:
        (home,target,outside), cases=controls(p)
        far=next(i for i in order if i not in {home,target,outside,*p.neighbors[home],*p.neighbors[outside]})
        order=[home,outside,far]+[i for i in order if i not in (home,outside,far)]
        rows=[]
        for head,other in sorted(cases.items()):
            remote=other['remote']
            rows.extend((head,remote,code) for code in range(480) if code//3 not in (head//3,remote//3))
        assert total<=len(rows)
        prefixes=np.full((total,160),-1,np.int16)
        prefixes[:,:3]=rows[:total]
        lengths=np.full(total,3,np.int16)
    else:
        plan=plan_frontier(p.masks,p.neighbors,p.sides,order,total)
        prefixes,lengths=plan.prefixes[:total],plan.lengths[:total]
    return dict(masks=p.masks,neighbors=p.neighbors,neighbor_sides=p.sides,order=order,gold_data=p.tiles),prefixes,lengths


def api_restore(api, inputs, prefixes, lengths, files, lanes=16):
    if api=='CUDA':
        factory,extras=DFS,{}
    else:
        import wgpu
        from dfs_webgpu import WebGPUDFS
        factory=WebGPUDFS
        adapter=next(a for a in wgpu.gpu.enumerate_adapters_sync()
            if a.info['backend_type']==api and a.info['adapter_type']=='DiscreteGPU')
        extras={'adapter':adapter}
    before,after=[],[]
    first=None
    try:
        for path in files:
            local=dict(extras)
            if api!='CUDA' and first is not None:
                local['reuse_from']=first
            gpu=factory(**inputs,n=lanes,seed=9,prefixes=[],**local)
            if first is None:
                first=gpu
            try:
                with np.load(path,allow_pickle=False) as archive, restore_parked(path,archive) as full:
                    ledger=JobLedger.resume(gpu,full,prefixes,lengths)
                before.append({**gpu.snapshot_checkpoint(),**ledger.fields()})
                gpu.step(3)
                ledger.refill(gpu,prefixes,lengths)
                after.append({**gpu.snapshot_checkpoint(),**ledger.fields()})
            finally:
                if gpu is not first and hasattr(gpu,'close'):
                    gpu.close()
    finally:
        if first is not None and hasattr(first,'close'):
            first.close()
    for pair in (before,after):
        assert pair[0].keys()==pair[1].keys()
        for k in pair[0]:
            np.testing.assert_array_equal(pair[0][k],pair[1][k],err_msg=k)
    return dict(lanes=lanes,fields=len(after[0]),matched_before_and_after=True)


def run(p, folder, initial, active, consume, apis, *, long=False):
    args,prefixes,lengths=inputs(p,initial,consume)
    def pool(n):
        return DFS(**args,n=n,prefixes=[],seed=35068)
    gpu=pool(initial)
    ledger=JobLedger.empty(len(prefixes),initial)
    ledger.refill(gpu,prefixes,lengths)
    if not consume:
        gpu.step(32)
        ledger.refill(gpu,prefixes,lengths)
    snapshot={**gpu.snapshot_checkpoint(),**ledger.fields()}
    gpu=pool(active)
    ledger=JobLedger.resume(gpu,snapshot,prefixes,lengths)
    origin='20261115T000000000000Z'
    roots=[folder/'flat',folder/'parked']
    for root in roots:
        (root/'runs'/origin).mkdir(parents=True)
    paths=[root/'systematic/checkpoint.npz' for root in roots]
    writer=ParkedCheckpointWriter(paths[1],atomic_npz)
    rows=[]
    crossed={}
    budgets=(0,)+(1,8,64,256)*(8 if long else 1)
    for gen,steps in enumerate(budgets):
        if steps:
            gpu.step(steps)
            ledger.refill(gpu,prefixes,lengths)
        payload={**gpu.snapshot_checkpoint(),**ledger.fields(),
                 'exact_source_origin':np.array(origin),'exact_source_binding':np.array('benchmark'),
                 'exact_frontier_sha256':np.array(hashlib.sha256(prefixes.tobytes()+lengths.tobytes()).hexdigest()),
                 'exact_data_sha256':np.array(p.fingerprint),'exact_generation':np.int64(gen),
                 'exact_terminal':np.array('')}
        states=payload['states']
        for lane in np.flatnonzero(states==1):
            codes=list(map(int,payload['boards'][:,lane]))
            validation=p.validate(codes)
            if validation['valid']:
                (Path(__file__).parent/'audit/r35/checkpoint-gold.json').write_text(
                    json.dumps({'codes':codes,'validation':validation}),encoding='utf8')
                raise SystemExit('Verified Gold candidate saved; search stopped.')
        before=io_written()
        flat=atomic_npz(paths[0],payload)
        mid=io_written()
        stats=writer.write((payload,ledger.paused,ledger.paused.offset,True))
        end=io_written()
        external=stats['parked_bank_external']
        proof.check_receipt(roots[1],origin,gen,paths[1],'benchmark',parked=external)
        with np.load(paths[1],allow_pickle=False) as archive, restore_parked(paths[1],archive) as opened:
            for k,v in payload.items():
                np.testing.assert_array_equal(opened[k],v,err_msg=k)
        if gen==3:
            for api in apis:
                crossed[api]=api_restore(api,args,prefixes,lengths,paths)
        rows.append(dict(gen=gen,paused=ledger.paused_count,offset=ledger.paused.offset,external=external,
            full_write=mid-before if before is not None else None,actual_write=end-mid if end is not None else None,
            full_size=paths[0].stat().st_size,active_size=paths[1].stat().st_size,bank_compressed=flat['parked_member_bytes']))
    materialize=0
    if rows[-1]['external']:
        before=io_written()
        writer.portable()
        materialize=io_written()-before if before is not None else None
    proof.check_receipt(roots[1],origin,len(budgets)-1,paths[1],'benchmark')
    with np.load(paths[1],allow_pickle=False) as final:
        assert REFERENCE not in final and set(final.files)==set(payload)
        for k,v in payload.items():
            np.testing.assert_array_equal(final[k],v)
    assert not list(paths[1].parent.glob('parked-bank-*.npz'))
    if consume:
        assert ledger.paused.offset>0
    return dict(generations=rows,portable_same_gen_write=materialize,cold_resizes=crossed)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--api',action='append',default=[],choices=('CUDA','Vulkan','D3D12','Metal'))
    cli.add_argument('--small',action='store_true')
    cli.add_argument('--adverse',action='store_true')
    cli.add_argument('--long',action='store_true')
    args=cli.parse_args()
    problem=SearchProblem(json.loads(args.tiles.read_bytes()))
    out={}
    with TemporaryDirectory(dir=Path(__file__).parent/'audit/r35') as tmp:
        corpus=[(2048,1,False),(8192,128,False),(2304,128,True)] if not args.small else [
            (32,1,False),(128,8,False),(512,8,False),(1024,8,False),(1536,8,False)]
        if args.adverse:
            corpus=[(2,1,False),(4,1,False),(8,4,False),(32,28,False),(128,96,False),
                    (512,480,False),(2048,1500,False),(4096,2560,False)]
        for initial,active,consume in corpus:
            label=f'{initial}-to-{active}/consume-{consume}'
            out[label]=run(problem,Path(tmp)/label,initial,active,consume,args.api if consume else [],long=args.long)
            args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
