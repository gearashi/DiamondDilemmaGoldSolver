"""Paired independent-board experiments on identical full inputs and charged repair budgets."""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel, polish_step
from r28_exchange import read_json
from r28_exchange_rotations import RotationOracle
from r30_fresh import _consider, _sample_start
from r31_fresh import Assemblies, Engine, physical_bonds
from r31_rotation_messages import CyclicRotationOracle
from search_problem import SearchProblem


def rotation_counts(model,boards):
    report={}
    winners=[]
    for cls in (RotationOracle,CyclicRotationOracle):
        oracle=cls(model)
        scores=[]
        for row in boards:
            out=oracle.optimum(row,max_entries=59049)
            if out is None:
                scores.append(None)
            else:
                score,choices=out
                chosen=next(choices)
                if model.problem.validate(chosen)['matched_edges']!=score:
                    raise AssertionError('Rotation witness failed.')
                scores.append(score)
        winners.append(scores)
        report[cls.__name__]=dict(scores=scores,**oracle.work)
    if winners[0]!=winners[1]:
        raise AssertionError('Rotation frame changed the optimum.')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--backend',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--rounds',type=int,default=32)
    p.add_argument('--dispatches',type=int,default=64)
    p.add_argument('--replicas',type=int,default=512)
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3])
    p.add_argument('--modes',nargs='+',default=['released','plain','bonds','slate','joint','bond-joint','adjacent','fork','ties','slack1','slack2','relink'])
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--rotation-input',nargs='+',type=Path)
    args=p.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    model=BoardModel(problem)
    bonds=Assemblies(model).bonds
    settings={'plain':{},'bonds':dict(bonds=True),'slate':dict(slate=3),'fork':dict(fork=True),
              'joint':dict(joint=True),'bond-joint':dict(joint=True,bonds=True),'adjacent':dict(adjacent=True),
              'ties':dict(ties=8),'slack1':dict(ties=8,tie_slack=1),'slack2':dict(ties=8,tie_slack=2),
              'relink':dict(relink=True,slate=3),'combined':dict(bonds=True,joint=True,slate=3),'global':dict(cycle_budget=3),
              'bond-global':dict(bonds=True,cycle_budget=3)}
    result={'domain':problem.fingerprint,'backend':args.backend,'rounds':args.rounds,'dispatches':args.dispatches,
            'replicas':args.replicas,'runs':{}}
    if args.rotation_input:
        boards=[]
        for path in args.rotation_input:
            with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True)) as db:
                for (state,) in db.execute('SELECT state FROM r30_trajectory WHERE rounds=320 ORDER BY seed'):
                    boards.append(json.loads(state)['best']['codes'])
        result['rotation']=rotation_counts(model,boards)
    for mode in args.modes:
        engine=Engine(problem,**settings[mode]) if mode!='released' else None
        result['runs'][mode]=[]
        for seed in args.seeds:
            before=time.perf_counter()
            if engine:
                row=engine.run(seed,backend=args.backend,replicas=args.replicas,dispatches=args.dispatches,rounds=args.rounds)
            else:
                state=_sample_start(model,seed,args.backend,args.replicas,args.dispatches)
                board=model.legal_board(state['current'])
                rng=np.random.default_rng()
                rng.bit_generator.state=state['rng']
                best=state['best']
                work=Counter()
                for step in range(args.rounds):
                    if best['gold']:
                        break
                    board,validation=polish_step(model,board,rng)
                    best=_consider(best,board,validation)
                    work['repair_steps']+=1
                    work['block_assignments']+=1
                    for threshold in (150,160,170,175,180,185,190,200,208):
                        if best['score']>=threshold:
                            work.setdefault(f'round_to_{threshold}',step+1)
                row=dict(**best,seed=seed,initial=state['initial'],work=dict(work),
                         gpu_proposals=state['gpu_proposals'])
            check=problem.validate(row['codes'])
            assert check['valid']==row['gold'] and check['matched_edges']==row['score']
            summary={k:v for k,v in row.items() if k!='codes'}
            summary['seconds']=round(time.perf_counter()-before,4)
            summary['bonds_kept']=len(bonds&physical_bonds(model,np.array(row['codes'])))
            result['runs'][mode].append(summary)
            print(mode,seed,summary['score'],summary['bonds_kept'],flush=True)
            atomic_json(args.output,result)
    atomic_json(args.output,result)


if __name__=='__main__':
    main()
