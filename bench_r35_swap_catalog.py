"""Check cached physical swap rankings against the released full-matrix computation."""
import argparse
import hashlib
from collections import Counter, OrderedDict
from pathlib import Path
from unittest.mock import patch

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r28_exchange_patches import AssignmentPrices
from r30_fresh import fresh_start
from r31_fresh import Engine
from r35_swap_catalog import SwapCatalog
from search_problem import SearchProblem


def full_ranking(model,prices,board,cells):
    cells=np.sort(np.asarray(cells,dtype=int))
    tiles=board[cells]//3
    grid=prices.prices[cells][:,tiles]
    maximum=grid.max(axis=2).astype(int)
    prior=grid[np.arange(len(cells)),np.arange(len(cells)),board[cells]%3]
    gain=maximum-prior[:,None]
    total=gain+gain.T
    legal=maximum>=0
    total[~(legal&legal.T)] = 0
    for i,c in enumerate(cells):
        total[i,np.isin(cells,model.problem.neighbors[c])] = 0
    x,y=np.nonzero(np.triu(total,1)>0)
    order=np.argsort(-total[x,y],kind='stable')
    i,j=x[order],y[order]
    return (cells[i],cells[j],3*tiles[j]+grid[i,j].argmax(axis=1),
            3*tiles[i]+grid[j,i].argmax(axis=1),total[i,j])


class Comparison:
    def __init__(self,model):
        self.model=model
        self.catalog=SwapCatalog(model)
        self.dense_fallback=SwapCatalog(model,.6)
        self.metrics=Counter()
        self.previous=None
        self.graph_lru={cap:OrderedDict() for cap in (1,2,4,8,16,32,64,320)}
        self.graph_counts=Counter()

    def check(self,prices,board,cells):
        prices.update(board)
        old=full_ranking(self.model,prices,board,cells)
        ranked=self.catalog.rank(prices,board,cells)
        fallback=self.dense_fallback.rank(prices,board,cells)
        if not all(np.array_equal(a,b) and np.array_equal(a,c) for a,b,c in zip(old,ranked,fallback)):
            raise ArithmeticError('Full and incremental oriented swap lists differ.')
        self.metrics['rank_entries_compared']+=len(old[0])
        if self.previous is not None and len(old[0]):
            if all(np.array_equal(old[k],self.previous[k]) for k in (0,1,4)):
                self.metrics['unchanged_nonempty_conflict_lists']+=1
        self.previous=old
        if len(old[0])>1:
            key=tuple(array.astype('<i4').tobytes() for array in (old[0],old[1],old[4]))
            self.graph_counts['nontrivial_lists']+=1
            for cap,lru in self.graph_lru.items():
                if key in lru:
                    lru.move_to_end(key)
                    self.graph_counts[f'hits_{cap}']+=1
                    self.graph_counts[f'moves_{cap}']+=len(old[0])
                    if len(old[0])>22:
                        self.graph_counts[f'large_hits_{cap}']+=1
                else:
                    lru[key]=None
                    if len(lru)>cap:lru.popitem(last=False)
        elif len(old[0]):
            self.graph_counts['singleton_lists']+=1

    def report(self):
        return dict(self.catalog.work,**self.metrics,graph_reuse=dict(self.graph_counts),
                    dense_fallback_work=dict(self.dense_fallback.work))


def modified(model,board,rng,steps):
    changed=board.copy()
    for step in range(steps):
        for attempt in range(200):
            a,b=rng.choice(model.free,2,replace=False)
            source=changed[[b,a]]//3
            valid=[np.flatnonzero(model.legal[cell,3*tile:3*tile+3]) for cell,tile in zip((a,b),source)]
            if all(len(v) for v in valid):
                changed[[a,b]]=3*source+np.array([rng.choice(v) for v in valid])
                break
        else:
            raise ValueError('No legal exchange sampled.')
    return changed


def scenarios(problem,seed):
    rng=np.random.default_rng(seed)
    model=BoardModel(problem)
    records={}
    for group in ('rotation','one_swap','four_swaps','protection','unrelated','unchanged'):
        probe=Comparison(model)
        board=fresh_start(model,rng)
        prices=AssignmentPrices(model)
        for index in range(32):
            cells=model.free
            if group=='protection':
                cells=np.sort(rng.choice(model.free,len(model.free)-8,replace=False))
            probe.check(prices,board,cells)
            if group=='rotation':
                options=[(i,3*(board[i]//3)+r) for i in model.free for r in range(3)
                         if r!=board[i]%3 and model.legal[i,3*(board[i]//3)+r]]
                cell,code=options[rng.integers(len(options))]
                board[cell]=code
            elif group in ('one_swap','four_swaps'):
                board=modified(model,board,rng,1 if group=='one_swap' else 4)
            elif group=='unrelated':
                board=fresh_start(model,rng)
        records[group]=probe.report()
    return records


def trajectory(problem,seed,rounds,mode,production_only=False):
    import r31_fresh
    engine=Engine(problem,bonds=True,cycle_budget=3,exact_packing=True,
                  separator_packing=mode=="stable",touch_packing="cp" if mode=="cp" else None)
    probe=Comparison(engine.model)
    released=r31_fresh.simultaneous_swaps
    production=None
    def check(model,board,cells,prices,**kwargs):
        nonlocal production
        if not production_only:
            probe.check(prices,board,cells)
        row = released(model,board,cells,prices,**kwargs)
        production = prices.swap_catalog
        return row
    with patch('r31_fresh.simultaneous_swaps',check):
        result=engine.run(seed,rounds=rounds)
    digest = hashlib.sha256(b''.join(int(v).to_bytes(2,'little') for v in result['codes'])).hexdigest()
    released_rows = read_json(Path(__file__).parent/'docs'/'R35_SWAP_PACKING.json')['trajectories']
    old = released_rows['pack-'+mode].get(str(seed),{}).get(str(rounds))
    if old and (old['final_hash']!=digest or old['work']!=result['work']):
        raise ArithmeticError("Repaired codes or search work changed from the pre-catalog receipt.")
    return dict(seed=seed,rounds=rounds,profile=mode,score=result['score'],components=result['components'],
                final_hash=digest,work_and_codes_match_original=bool(old),physical_operations=dict(production.work),
                milestones={k:v for k,v in result['work'].items() if k.startswith('round_to_')},
                work=probe.report())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('base','stable','cp'),default='base')
    parser.add_argument('--trajectory-only',action='store_true')
    parser.add_argument('--production-only',action='store_true',help='Check final receipts without running the full per-round oracle.')
    parser.add_argument('--rounds',type=int,default=320)
    parser.add_argument('--seeds',type=int,nargs='+',default=[3500,3501,3502,3503])
    args=parser.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    report=dict(input=problem.fingerprint,isolated={} if args.trajectory_only else scenarios(problem,35350),trajectories=[])
    atomic_json(args.output,report)
    for seed in args.seeds:
        row=trajectory(problem,seed,args.rounds,args.mode,args.production_only)
        report['trajectories'].append(row)
        atomic_json(args.output,report)
        print(row,flush=True)


if __name__=='__main__':
    main()
