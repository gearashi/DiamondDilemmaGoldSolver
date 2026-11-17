"""Held-out continuations from the completed real-board checkpoints and exact tie comparisons."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel, polish_step
from r28_exchange import read_json
from r31_fresh import joint_repair, quality, result
from r31_fresh_ties import optimize_ties
from search_problem import SearchProblem


def race(model,state,seed,budget,mode,partner):
    current=state['current']
    incumbent=state['best']
    rng=np.random.default_rng()
    rng.bit_generator.state=state['rng']
    arms=[(current,rng)]
    if mode!='continue':
        other=incumbent['codes'] if mode=='fork' else partner['best']['codes']
        arms.append((other,np.random.default_rng(np.random.SeedSequence([seed,4455]))))
    best=incumbent
    for i in range(budget):
        if best['gold']:
            break
        arm=i%len(arms)
        row,rng=arms[arm]
        candidate,_=polish_step(model,model.legal_board(row),rng)
        check=result(model,candidate)
        arms[arm]=(candidate.tolist(),rng)
        if quality(check)>quality(best):
            best=check
    return {k:v for k,v in best.items() if k!='codes'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--input',nargs='+',required=True,type=Path)
    p.add_argument('--budget',default=32,type=int)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    model=BoardModel(SearchProblem(read_json(args.tiles)))
    report={'budget':args.budget,'results':[]}
    for path in args.input:
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            states=[json.loads(s) for (s,) in db.execute('SELECT state FROM r30_trajectory WHERE rounds=320 ORDER BY seed')]
        for i,state in enumerate(states):
            seed=state['seed']
            board=model.legal_board(state['best']['codes'])
            initial=result(model,board)
            row=dict(source=path.name,seed=seed,initial=initial['score'],before_components=initial['components'])
            for mode in ('continue','fork','independent'):
                row[mode]=race(model,state,seed,args.budget,mode,states[(i+1)%len(states)])
            independent=model.independent_set(board,np.random.default_rng(seed))[:10]
            row['layers']=[{k:v for k,v in layer.items() if k!='codes'}
                           for layer in optimize_ties(model,board,independent)]
            if i==0:
                _,row['all_adjacent']=joint_repair(model,board,model.free,adjacent=True,effort=.05)
            report['results'].append(row)
            print(path.name,seed,row['continue']['score'],row['fork']['score'],flush=True)
            atomic_json(args.output,report)


if __name__=='__main__':
    main()
