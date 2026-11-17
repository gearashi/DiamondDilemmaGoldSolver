"""Measure changed bytes in consecutive genuine classic GPU snapshots on original tiles."""
import argparse
import json
from pathlib import Path
import numpy as np

from dfs_gpu import DFS, fill_order
from exact_frontier import plan_frontier
from search_problem import SearchProblem


def compare(first, second):
    keys = sorted(first.keys() | second.keys())
    widths = (4096,16384,65536,262144,1048576)
    result = dict(raw_bytes=0,changed_fields=0,field_bytes=0,changed_bytes=0)
    chunks = dict.fromkeys(widths,0)
    for key in keys:
        old = np.asarray(first[key])
        new = np.asarray(second[key])
        raw = new.tobytes()
        previous = old.tobytes()
        result['raw_bytes'] += len(raw)
        if raw == previous and new.shape==old.shape and new.dtype==old.dtype:
            continue
        result['changed_fields'] += 1
        result['field_bytes'] += len(raw)
        if new.shape != old.shape or new.dtype != old.dtype:
            result['changed_bytes'] += len(raw)
        else:
            result['changed_bytes'] += int(np.count_nonzero(np.frombuffer(previous,np.uint8) != np.frombuffer(raw,np.uint8)))
        for width in widths:
            for start in range(0,len(raw),width):
                part = raw[start:start+width]
                if part != previous[start:start+width]:
                    chunks[width] += len(part)
    result['changed_chunk_bytes'] = chunks
    return result


def probe(problem, api, lanes, occupied, output):
    order = fill_order(problem.neighbors)
    cover = plan_frontier(problem.masks,problem.neighbors,problem.sides,order,occupied)
    rows = [row[:int(n)].tolist() for row,n in zip(cover.prefixes[:occupied],cover.lengths[:occupied])]
    if api == 'CUDA':
        engine,opts = DFS,{}
    else:
        import wgpu
        from dfs_webgpu import WebGPUDFS
        adapter = next(a for a in wgpu.gpu.enumerate_adapters_sync()
                       if a.info['backend_type']==api and a.info['adapter_type']=='DiscreteGPU')
        engine,opts = WebGPUDFS,{'adapter':adapter}
    gpu = engine(problem.masks,problem.neighbors,problem.sides,n=lanes,order=order,seed=35068,
                 prefixes=rows,gold_data=problem.tiles,**opts)
    try:
        def snapshot():
            saved=gpu.snapshot_checkpoint()
            for lane in np.flatnonzero(saved['states']==1):
                codes=saved['boards'][:,lane].astype(int).tolist()
                validation=problem.validate(codes)
                if validation['valid']:
                    output.with_suffix('.gold.json').write_text(json.dumps({'codes':codes,'validation':validation}),encoding='utf8')
                    raise SystemExit('Validated Gold board saved; search stopped.')
            return saved
        initial=snapshot()
        rows=[dict(nodes=0,**compare(initial,snapshot()))]
        before=initial
        for amount in (1,8,64,256):
            gpu.step(amount)
            after=snapshot()
            rows.append(dict(nodes=amount,**compare(before,after)))
            before=after
        replay_start=before
        gpu.step(11)
        uninterrupted=snapshot()
        gpu.restore_checkpoint(replay_start)
        gpu.step(11)
        expected=snapshot()
        # Both the released NPZ restore and a chunk reader must start with
        # the kernel's deliberately cold Gold prefix validation cursor.
        gpu.restore_checkpoint(replay_start)
        gpu.step(11)
        again=snapshot()
        assert all(np.array_equal(expected[k],again[k]) for k in expected)
        cold_delta = [k for k in expected if not np.array_equal(expected[k],uninterrupted[k])]
        gpu.verify(indices=list(range(min(occupied,8))))
        return dict(api=api,lanes=lanes,assigned=occupied,transitions=rows,identical_cold_restores=True,uninterrupted_difference=cold_delta)
    finally:
        if hasattr(gpu,'close'):
            gpu.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    args=cli.parse_args()
    p=SearchProblem(json.loads(args.tiles.read_bytes()))
    out=[]
    for api in ('CUDA','Vulkan','D3D12'):
        for lanes,occupied in ((128,128),(2048,128),(2048,2048)):
            out.append(probe(p,api,lanes,occupied,args.output))
            args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
