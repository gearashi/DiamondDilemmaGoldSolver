"""Compare saved exact star vectors over every pair at actual neighboring 206-floor vertices."""
import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import time
import zlib

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r29_plateau import SwapGraph
from r35_plateau_pages import advance,materialize,schema,stored_vector
from search_problem import SearchProblem


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--incumbent',type=Path,required=True)
    ap.add_argument('--database',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--centers',type=int,default=12)
    args=ap.parse_args()
    p=SearchProblem(read_json(args.tiles))
    m=BoardModel(p)
    center=m.legal_board(read_json(args.incumbent)['codes'])
    if args.database.exists():raise ValueError('Use a fresh vector comparison database.')
    with closing(sqlite3.connect(args.database)) as db:
        graph=SwapGraph(db,m)
        schema(db)
        db.execute('CREATE TABLE IF NOT EXISTS walk_legacy(board TEXT PRIMARY KEY,score INTEGER,seed TEXT)')
        root=graph.key(center)
        ri=graph.optimum(root,center,0)
        db.commit()
        graph.work['swap_candidates']=0
        assert advance(graph,args.database,root,center,206)
        children=list(graph.known_neighbors(root,206))
        rootbound=ExchangeBound(m,center)
        prior=rootbound.bounds(graph.pairs,(1,0),doubled=True)
        # Cover shared-even and changed-odd baselines and spread over the entire neighbor list.
        selected=[children[i] for i in np.linspace(0,len(children)-1,min(args.centers,len(children)),dtype=int)]
        recs=[]
        for key,item in selected:
            board=m.legal_board(item['codes'])
            assert int(m.matches(board).sum())>=206 and len(np.flatnonzero(center//3!=board//3))==2
            cold=ExchangeBound(m,board)
            before=graph.work.copy()
            _,floor=materialize(graph,key,board,lambda:None,float('inf'),parent=root)
            base,result=stored_vector(graph,key)
            total_bytes=db.execute('SELECT length(deltas) FROM walk_vectors WHERE board=?',(key,)).fetchone()[0]
            raw=zlib.compress(np.packbits(result&1,bitorder='little').tobytes())
            check=cold.bounds(graph.pairs,(1,0),doubled=True)
            if not np.array_equal(result,check):raise ArithmeticError('An actual transported doubled star differs.')
            changed=set(np.flatnonzero(board//3!=center//3).tolist())
            locations=tuple(sorted(changed))
            for _ in range(2):changed.update(int(q) for c in list(changed) for q in m.problem.neighbors[c] if q>=0)
            unaffected=~np.isin(graph.pairs,list(changed)).any(axis=1)
            lossy=(2*(prior//2)-rootbound.total+cold.total)//2
            legacy_bytes=len(zlib.compress((result//2).astype('<i4').tobytes()))
            recs.append(dict(swap=locations,score=item['score'],components=item['components'],base=base,pairs=len(check),
                             recomputed=graph.work['star_evaluations']-before['star_evaluations'],
                             reused=graph.work.get('transported_stars',0)-before.get('transported_stars',0),
                             lossy_floor_errors=int(np.count_nonzero(unaffected&(lossy!=check//2))),
                             floor_bytes=legacy_bytes,extra_parity_bytes=len(raw),stored_delta_bytes=total_bytes,
                             compressed_i2_bytes=len(zlib.compress(check.astype('<i2').tobytes())),
                             compressed_delta_bytes=len(zlib.compress((check-cold.total).astype('i1').tobytes()))))
            atomic_json(args.output,dict(input=p.fingerprint,source_score=ri['score'],source_base=rootbound.total,
                                        all_206_neighbors=len(children),records=recs))
            print(locations,recs[-1]['reused'],recs[-1]['lossy_floor_errors'],flush=True)


if __name__=='__main__':
    main()
