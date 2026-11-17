"""Compare independently resumed BFS searches on actual or planted plateaus."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from io_utils import atomic_json
from r28_exchange import read_json
from r29_plateau import walk
from search_fixtures import planted_gold
from search_problem import SearchProblem


def compare(problem,start,folder,label,floor,goal,limit):
    rows={}
    keys=('centers','edge_rows','swap_candidates','solved','star_evaluations','reused_arcs',
          'path_propagations','band_expansions','activations','depth_closed','potential_closed')
    for use in (False,True):
        name='activate' if use else 'rebuild'
        db=folder/(label+'-'+name+'.sqlite')
        if db.exists():raise ValueError('Use a fresh directory for the differential.')
        def call(budget):
            return walk(problem,start,db,floor,goal,widest=True,max_centers=budget,tie_limit=1,
                        max_swaps=limit,potentials=False,incremental=use)
        partial=call(1)
        done=call(100000)
        repeat=call(100000)
        proof={k:done[k] for k in ('score','floor','widest_floor','shortest_at_floor','reached','swaps')}
        codes=done['codes']
        assert problem.validate(codes)['matched_edges']==proof['score']
        board=list(start)
        for a,b in proof['swaps']:board[a],board[b]=board[b],board[a]
        assert [v//3 for v in board]==[v//3 for v in codes]
        with closing(sqlite3.connect(db)) as conn:
            table='walk_distances' if use else 'walk_queue'
            positions=conn.execute('SELECT count(*) FROM '+table).fetchone()[0]
        rows[name]=dict(proof=proof,stored_distance_rows=positions,
                        work={k:partial.get(k,0)+done.get(k,0) for k in keys},
                        repeated={k:repeat.get(k,0) for k in keys})
        print(label,name,rows[name],flush=True)
    for k in ('score','floor','widest_floor','shortest_at_floor','reached'):
        assert rows['activate']['proof'][k]==rows['rebuild']['proof'][k],(label,k,rows)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--tiles',type=Path)
    p.add_argument('--incumbent',type=Path)
    p.add_argument('--min-floor',type=int,default=205)
    args=p.parse_args()
    args.directory.mkdir(parents=True,exist_ok=True)
    reports={}
    if args.tiles:
        problem=SearchProblem(read_json(args.tiles))
        reports['actual']=compare(problem,read_json(args.incumbent)['codes'],args.directory,'actual',args.min_floor,208,1)
    else:
        for seed in (19,31,83,310):
            data,board,codes=planted_gold(seed,1,3)
            problem=SearchProblem(data,board,{i:codes[i] for i in range(4,len(codes))})
            start=codes.copy()
            start[0],start[1]=start[1],start[0]
            start[2],start[3]=start[3],start[2]
            for limit in (1,2,None):
                name=f'planted-{seed}-{limit}'
                reports[name]=compare(problem,start,args.directory,name,0,len(problem.edges),limit)
    atomic_json(args.directory/'comparison.json',reports)


if __name__=='__main__':
    main()
