"""Matched deterministic signed refit, endpoint-cut, incremental-price and tied-donor trajectories."""
from argparse import ArgumentParser
from pathlib import Path
import hashlib
import json

from io_utils import atomic_json
from r32_fresh import fresh
from r32_gold_loss import read_json
from search_problem import SearchProblem


def digest(board):
    return hashlib.sha256(b''.join(int(x).to_bytes(2,'little') for x in board)).hexdigest()


if __name__=='__main__':
    parser=ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='../data/tiles.json')
    parser.add_argument('--effort',type=float,default=.15)
    parser.add_argument('--high-effort',action='store_true')
    parser.add_argument('--rounds',type=int,default=64)
    parser.add_argument('--seeds',nargs='+',type=int,default=[4,5,6,7,11,19,31,51])
    parser.add_argument('--diverse-only',action='store_true')
    parser.add_argument('--recommended',action='store_true')
    parser.add_argument('--start-only',action='store_true')
    parser.add_argument('--delayed-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    configs=[
        ('baseline',dict(refit_bounds=False,delta_prices=False)),
        ('ceilings-and-delta',dict()),
        ('pooled-cuts',dict(endpoint_reuse=True)),
        ('raw-ties',dict(raw_ties=True)),
    ]
    if args.diverse_only:
        configs=[("diverse-plateau",dict(diverse_ties=True))]
    if args.high_effort:
        configs=configs[:2]
    if args.delayed_only:
        configs=[("delayed-raw-ties",dict(raw_ties="plateau"))]
    if args.start_only:
        configs=[("fixed-start-cut",dict(phase_start_cut=True))]
    if args.recommended:
        configs=[("raw-without-refit",dict(raw_ties=True,refit=0)),
                 ("raw-start-cut",dict(raw_ties=True,refit=0,phase_start_cut=True)),
                 ("delayed-start-cut",dict(raw_ties="plateau",refit=0,phase_start_cut=True))]
    rows=[]
    for seed in args.seeds:
        base=None
        for name,options in configs:
            p=SearchProblem(read_json(args.tiles))
            states=[]
            def capture(state):
                states.append(dict(cursor=state['cursor'],current=digest(state['current']),
                                   phases=digest(state['phases']),raw=state['best']['score'],signed=state['best']['signed']))
            out=fresh(p,seed,rounds=args.rounds,effort=args.effort,cube_bits=2,save=capture,**dict({"refit":8},**options))
            v=p.validate(out['codes'])
            assert v['matched_edges']==out['score']
            row=dict(seed=seed,arm=name,raw=out['score'],signed=out.get('signed'),gold=out['gold'],
                     board=digest(out['codes']),states=states,work=out['work'])
            if name=='baseline':
                base=row
            if name=='ceilings-and-delta' and (base['states']!=states or base['board']!=row['board']):
                raise AssertionError(('Optimized deterministic trajectory changed',seed,base,row))
            rows.append(row)
            atomic_json(args.output,dict(rounds=args.rounds,rows=rows))
            print(seed,name,row['raw'],row['signed'],row['work'].get('refits_skipped_by_ceiling'),
                  row['work'].get('refit_calls'),row['work'].get('improved_refits'),flush=True)
