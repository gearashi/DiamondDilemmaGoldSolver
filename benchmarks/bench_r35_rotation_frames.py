"""Measure the extra disconnected components identified under local rotation-rank relabeling."""
import argparse
from collections import Counter
import itertools
from pathlib import Path

import numpy as np

from bench_r35_rotation_components import cases
from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r28_exchange_rotations import RotationOracle
from search_problem import SearchProblem


def orientations(key,exhaustive):
    unaries,raw,scope=key
    tables=[(a,b,np.frombuffer(data,dtype='<i4').reshape(len(unaries[a]),len(unaries[b]))) for a,b,data in raw]
    choices=[]
    for c,u in enumerate(unaries):
        edges=[table if a==c else table.T for a,b,table in tables if c in (a,b)]
        labels=[(value,tuple(tuple(sorted(map(int,tab[r]))) for tab in edges)) for r,value in enumerate(u)]
        ordered=sorted(range(len(u)),key=labels.__getitem__)
        blocks=[tuple(ids) for _,ids in itertools.groupby(ordered,key=labels.__getitem__)]
        if exhaustive and len(unaries)<=3:
            variants=[tuple(j for block in options for j in block)
                      for options in itertools.product(*(itertools.permutations(block) for block in blocks))]
        else:variants=[tuple(ordered)]
        choices.append(variants)
    best=None
    count=0
    for perms in itertools.product(*choices):
        count+=1
        candidate=(tuple(tuple(u[i] for i in perm) for u,perm in zip(unaries,perms)),
                   tuple((a,b,tab[np.ix_(perms[a],perms[b])].tobytes()) for a,b,tab in tables),scope)
        if best is None or candidate<best:best=candidate
    return best,count


def vertex_orbit(key):
    unary,rows,scope=key
    if len(unary)>3:return orientations(key,False)[0],1
    winners=[]
    for order in itertools.permutations(range(len(unary))):
        indices={q:i for i,q in enumerate(order)}
        translated=[]
        for a,b,raw in rows:
            x,y=indices[a],indices[b]
            table=np.frombuffer(raw,dtype='<i4').reshape(len(unary[a]),len(unary[b]))
            if x>y:x,y,table=y,x,table.T
            translated.append((x,y,table.tobytes()))
        # Ignore elimination order to measure the most an exact cached complete
        # table of joint optima could add over the ordered response masks.
        moved=tuple(unary[i] for i in order),tuple(sorted(translated)),()
        winners.append(orientations(moved,True)[0])
    return min(winners),len(winners)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    model=BoardModel(SearchProblem(read_json(args.tiles)))
    source=model.legal_board(read_json(args.board)['codes'])
    result={}
    for seed in (851,7407):
        for name,queries in cases(model,source,seed).items():
            oracle=RotationOracle(model,components=True)
            oracle.component_cache.frames=0
            oracle.component_cache.vertices=0
            sequence=[]
            existing=oracle.component_cache.get
            def recorded(key):
                sequence.append(key)
                return existing(key)
            oracle.component_cache.get=recorded
            for board,cap in queries:
                call=oracle.optimum(board,max_entries=cap,ties=True,limit=4)
                if call is not None:list(call[1])
            unique=set(sequence)
            out={}
            for mode in (False,True):
                rows=[(k,*orientations(k,mode)) for k in unique]
                classes={v for _,v,_ in rows}
                # Every distinct new raw key incurs one canonicalization. Cache
                # raw aliases for future queries to give the alternative its best case.
                out['exact3' if mode else 'ordered']=dict(raw=len(unique),classes=len(classes),queries=len(sequence),
                    saved_component_solves=len(unique)-len(classes),
                    saved_factor_keys=sum(len(k[0]) for k in unique)-sum(len(v[0]) for v in classes),
                    reindex_trials=sum(n for _,_,n in rows),
                    changed_shapes=dict(Counter(str(len(k[0])) for k,v,n in rows if k!=v)))
            vertex=[vertex_orbit(k) for k in unique]
            classes={v for v,_ in vertex}
            out['vertex3']=dict(classes=len(classes),vertex_orders=sum(n for _,n in vertex),
                                saved_factor_keys=sum(len(k[0]) for k in unique)-sum(len(k[0]) for k in classes))
            result[str(seed)+'-'+name]=out
            atomic_json(args.output,dict(input=model.problem.fingerprint,groups=result))
            print(seed,name,out,flush=True)


if __name__=='__main__':main()
