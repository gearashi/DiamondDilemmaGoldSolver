"""Check whether parallel same-depth prefetch cuts proof work on positive physical routes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r29_plateau import SwapGraph,walk
from r35_plateau_pages import advance,schema
from search_fixtures import planted_gold
from search_problem import SearchProblem


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory',type=Path,required=True)
    folder=ap.parse_args().directory
    folder.mkdir(parents=True,exist_ok=True)
    out=[]
    for seed in (19,31,83,310):
        data,shape,ref=planted_gold(seed,1,3)
        p=SearchProblem(data,shape,{i:ref[i] for i in range(4,len(ref))})
        start=ref.copy()
        start[0],start[1]=start[1],start[0]
        start[2],start[3]=start[3],start[2]
        for workers in (1,2,4):
            dbpath=folder/f'frontier-{seed}-{workers}.sqlite'
            if dbpath.exists():raise ValueError('Use an empty directory for the frontier comparison.')
            def search(cap):
                return walk(p,start,dbpath,0,15,max_centers=cap,tie_limit=1)
            first=search(1)
            extra=[]
            if workers>1:
                with closing(sqlite3.connect(dbpath)) as db:
                    jobs=db.execute("""SELECT q.board,q.parent,o.result FROM walk_queue q JOIN optima o ON o.key=q.board
                                    WHERE closed=0 ORDER BY depth,q.rowid LIMIT ?""",(workers,)).fetchall()
                def pref(row):
                    key,parent,raw=row
                    with closing(sqlite3.connect(dbpath,timeout=60)) as db:
                        g=SwapGraph(db,BoardModel(p))
                        schema(db)
                        g.work['swap_candidates']=0
                        advance(g,dbpath,key,json.loads(raw)['codes'],0,parent=parent)
                        return g.work
                with ThreadPoolExecutor(workers) as ex:
                    extra=list(ex.map(pref,jobs))
            end=search(128)
            work={k:sum(r.get(k,0) for r in [first,*extra,end]) for k in ('solved','star_evaluations','swap_candidates')}
            assert end['reached'] and p.validate(end['codes'])['valid']
            if workers==1:control=end
            else:assert (end['swaps'],end['codes'])==(control['swaps'],control['codes'])
            out.append(dict(seed=seed,workers=workers,work=work,shortest=end['shortest_at_floor'],swaps=end['swaps']))
    atomic_json(folder/'comparison.json',out)
    print(out)


if __name__=='__main__':main()
