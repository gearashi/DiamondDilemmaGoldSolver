"""Test possible board-cell read ranges against original, independently captured device bytes."""
import argparse
import json
from pathlib import Path

import numpy as np

from bench_r35_parked_public import inputs
from dfs_gpu import DFS
from gpu_reasoning import frame_cells
from search_problem import SearchProblem
from systematic_jobs import JobLedger


def audit(old,current,changed,order,dynamic):
    living=np.isin(old['states'],(0,1))|changed
    ranks=np.argsort(order)[:,None]
    frames=np.arange(160)[:,None]
    caps=[0,0]
    lanes=np.flatnonzero(living)
    for start in range(0,len(lanes),1024):
        ids=lanes[start:start+1024]
        floors=current['floors'][ids]
        pad=changed[ids]|(current['states'][ids]>=2)
        movable=(ranks>=floors)
        strict=movable.copy() if dynamic else movable&(ranks<=current['maxdepths'][ids])
        if dynamic:
            shape=(160,len(ids))
            selected=np.zeros(shape,bool)
            depth,lane=np.nonzero((frames>=floors)&(frames<current['depths'][ids]))
            native=frame_cells(current['firsts'][:,ids],current['shifts'][:,ids])
            cells=native[depth,lane]
            assert (cells<160).all()
            selected[cells,lane]=True
        else:
            selected=ranks<current['depths'][ids]
        actual_change=old['boards'][:,ids]!=current['boards'][:,ids]
        narrow=((old['boards'][:,ids]>=0)|selected)&movable
        for i,mask in enumerate((strict,narrow)):
            mask[:,pad]=True
            assert not (actual_change&~mask).any()
            caps[i]+=int(mask.sum())
    return dict(previously_mutable_lanes=len(lanes),existing_board_values=len(lanes)*160,
                structural_board_values=caps[0],occupancy_board_values=caps[1],
                reset_lanes=int(changed.sum()),observed_changed_cells=int(np.count_nonzero(old['boards']!=current['boards'])))


def run(problem,n,used,consume,dynamic,budgets):
    args,rows,lengths=inputs(problem,used,consume)
    if dynamic:
        args['gpu_features']={'dynamic_mrv':True,'forced':True}
    gpu=DFS(**args,n=n,prefixes=[],seed=35068)
    ledger=JobLedger.empty(len(rows),n)
    ledger.refill(gpu,rows,lengths)
    ledger.checkpoint_installs=np.zeros(n,bool)
    old=gpu.snapshot_checkpoint()
    metrics=[]
    for nodes in budgets:
        for start in range(0,nodes,512):
            gpu.step(min(512,nodes-start))
            ledger.refill(gpu,rows,lengths)
        current=gpu.snapshot_checkpoint()
        for idx in np.flatnonzero(current['states']==1):
            board=current['boards'][:,idx].tolist()
            report=problem.validate(board)
            if report['valid']:
                (Path(__file__).parent/'audit/r35/checkpoint-gold.json').write_text(
                    json.dumps({'codes':board,'validation':report}),encoding='utf8')
                raise SystemExit('Verified Gold saved; search stopped.')
        metrics.append(dict(nodes_per_lane=nodes,**audit(old,current,ledger.checkpoint_installs,gpu.order,dynamic)))
        ledger.checkpoint_installs.fill(False)
        old=current
    return metrics


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--tiles',required=True,type=Path)
    args=cli.parse_args()
    p=SearchProblem(json.loads(args.tiles.read_bytes()))
    out={}
    for n,used,consume in ((128,128,False),(2048,128,False),(2048,2048,False),(128,4096,True),(32768,32768,False)):
        for dynamic in (False,True):
            out[f'{n}/{used}/refill-{consume}/dynamic-{dynamic}']=run(p,n,used,consume,dynamic,(1,64,512,4096))
            args.output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
