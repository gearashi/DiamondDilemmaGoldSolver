"""Compare persistent parent stars with the earlier two-center bound and measure stop granularity."""
import argparse
from collections import Counter
from contextlib import closing
import json
from pathlib import Path
import sqlite3
from time import monotonic

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r29_plateau import SwapGraph,walk
from r30_multicenter import SharedSupports
from r35_plateau_pages import stored_vector
from search_problem import SearchProblem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('tiles','incumbent','database','stops','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    args=parser.parse_args()
    if args.stops.exists():raise ValueError('Use a fresh SQLite file for stop timing.')
    problem=SearchProblem(read_json(args.tiles))
    center=read_json(args.incumbent)['codes']
    model=BoardModel(problem)
    stats=Counter()
    with closing(sqlite3.connect(args.database)) as db:
        graph=SwapGraph(db,model)
        root=graph.key(center)
        for key,raw in db.execute('SELECT board,result FROM walk_vectors JOIN optima ON key=board WHERE board!=?',(root,)):
            child=json.loads(raw)['codes']
            both=SharedSupports(model,[center,child],[(2,)])
            wanted=stored_vector(graph,key)[1]
            for i in np.linspace(0,len(graph.pairs)-1,128,dtype=int):
                cells=tuple(map(int,graph.pairs[i]))
                values={}
                terms={}
                both.upper(0,cells,values,terms,stats)
                got=both.upper(1,cells,values,terms,stats)
                if got!=wanted[i]:raise ArithmeticError('Two-center and stored parent bounds differ.')
                stats['equality_checks']+=1
    stops=[]
    for cap in (0.00001,0.05,0.25,1):
        start=monotonic()
        result=walk(problem,center,args.stops,206,208,max_centers=64,seconds=cap,tie_limit=1)
        duration=monotonic()-start
        stops.append(dict(seconds=cap,wall=duration,score=result['score'],open=result['open'],closed=result['closed'],
                          candidates=result.get('swap_candidates',0),root_pending=result.get('initialization_pending',False)))
        assert len(result['swaps'])==0 and problem.validate(result['codes'])['matched_edges']==result['score']
    atomic_json(args.output,dict(two_center_supports=stats,stops=stops))
    print(stats,stops)


if __name__=='__main__':main()
