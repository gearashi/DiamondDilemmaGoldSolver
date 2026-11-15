"""Measure unchanged-board duals on the actual regional inputs of sustained fresh repair."""
import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from io_utils import atomic_json
from r28_exchange import read_json
from r31_fresh import Engine
from r35_restricted_assignment import identity_dual
from search_problem import SearchProblem


def run(problem,seed,rounds):
    import r31_fresh
    solve=r31_fresh.pack_cycles
    counts=Counter()
    def observed(model,board,cells,prices,**kwargs):
        if kwargs.get('partition')!='neighbors':
            return solve(model,board,cells,prices,**kwargs)
        cells=np.sort(cells)
        prices.update(board)
        grid=prices.prices[cells][:,board[cells]//3]
        maximum=grid.max(axis=2).astype(int)
        current=grid[np.arange(len(cells)),np.arange(len(cells)),board[cells]%3]
        result,step=identity_dual(maximum-current[:,None],maximum>=0,len(cells))
        kwargs['identity_check']=False
        answer,info=solve(model,board,cells,prices,**kwargs)
        counts['queries']+=1
        if result:
            if not info['complete'] or np.any(answer!=board):
                raise AssertionError('Certified unchanged assignment differs from the solved regional witness.')
            counts['identity_'+str(step)]+=1
        else:
            counts['open']+=1
        return answer,info
    with patch('r31_fresh.pack_cycles',observed):
        row=Engine(problem,bonds=True,cycle_budget=3,exact_packing=True,regional_cycle_budget=2).run(seed,rounds=rounds)
    return dict(seed=seed,rounds=rounds,score=row['score'],work=dict(counts))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',nargs='+',type=int,default=[3500,3501,3502,3503])
    ap.add_argument('--rounds',type=int,default=320)
    a=ap.parse_args()
    p=SearchProblem(read_json(a.tiles))
    records=[]
    for seed in a.seeds:
        records.append(run(p,seed,a.rounds))
        atomic_json(a.output,dict(input=p.fingerprint,records=records))
        print(records[-1],flush=True)


if __name__=='__main__':
    main()
