"""Measure zero-gain certificates on physical repair pools and the unchanged saved incumbent."""
import argparse
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r31_fresh import Engine,joint_repair,forest_pool
from r35_contact_bound import no_contact_gain
from search_problem import SearchProblem


def compare(model,board,cells,options,rows):
    disjoint=bool(options.get('disjoint',False))
    contacts=options.get('contacts')
    if contacts is None:contacts=-1 if disjoint else int(options.get('adjacent',False))
    proved,why=no_contact_gain(model,board,cells,contacts=contacts,disjoint=disjoint,forbidden=options.get('forbidden'))
    result,info=joint_repair(model,board,cells,**options)
    gain=int(model.matches(result).sum()-model.matches(board).sum())
    if proved and gain>0:raise AssertionError('Refuted claimed local optimum.')
    rows.append(dict(proved=bool(proved),work=why,cp=info,gain=gain,pool=list(map(int,cells)),board=board.tolist()))
    return result,info


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--incumbent',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rounds',type=int,default=32)
    ap.add_argument('--seeds',type=int,nargs='+',default=[3500,3501,3502,3503])
    ap.add_argument('--incumbent-only',action='store_true')
    args=ap.parse_args()
    p=SearchProblem(read_json(args.tiles))
    m=BoardModel(p)
    center=m.legal_board(read_json(args.incumbent)['codes'])
    check=p.validate(center.tolist())
    if check['matched_edges']!=207:raise ValueError('Require the unchanged independently revalidated 207 center.')
    cases=[]
    modes=('207-random','207-forest') if args.incumbent_only else ('random-contact','forest-disjoint','guarded-contact','207-random','207-forest')
    for mode in modes:
        for seed in args.seeds:
            rows=[]
            if mode.startswith('207'):
                rng=np.random.default_rng(seed)
                for step in range(args.rounds):
                    if mode=='207-random':cells=rng.choice(m.free,10,replace=False)
                    else:cells=forest_pool(m,center,m.free,8,rng)
                    options=dict(contacts=1 if mode=='207-random' else -1,disjoint=mode=='207-forest')
                    compare(m,center,cells,options,rows)
                score=check['matched_edges']
            else:
                forest=mode=='forest-disjoint'
                engine=Engine(p,bonds=True,joint=True,cycle_budget=3,exact_packing=True,forest=forest,pool=8 if forest else 10,
                              disjoint=forest,contacts=None if forest else 1,forbidden_repair='guarded' if mode=='guarded-contact' else None)
                def observed(model,board,cells,**options):return compare(model,board,cells,options,rows)
                with patch('r31_fresh.joint_repair',observed):score=engine.run(seed,rounds=args.rounds)['score']
            cases.append(dict(mode=mode,seed=seed,score=score,rows=rows))
            atomic_json(args.output,dict(input=p.fingerprint,cases=cases))
            print(mode,seed,Counter(next((k for k in ('root_dual_proofs','support_proofs') if k in r['work']),'open') for r in rows),flush=True)


if __name__=='__main__':
    main()
