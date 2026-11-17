"""Compare donor distance models, geographic pool choice, and disconnected shape proof work."""
from collections import Counter
from itertools import combinations, permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import argparse
import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r29_exchange_clusters import StarClusters
from r30_exact_routes import CycleModel
from r30_multicenter import run
from r31_fresh import joint_repair,forest_pool
from r31_routes import rank
from r34_cycle_orders import CycleOrders,FIVE_FAMILIES
from r34_exchange_undo import CoveredUndo
from r35_route_shapes import ContactShapes
from r35_remote_donors import response_pool
from r35_route_distance import DistanceModel
from search_problem import SearchProblem


def shape_trials(model,board,out):
    b=ExchangeBound(model,board)
    stars=StarClusters(b)
    rng=np.random.default_rng(3521)
    rows=[]
    for family in (*FIVE_FAMILIES,(2,2,2,4),(2,2,3,3)):
        for i in range(8):
            k=sum(family)
            cells=tuple(sorted(map(int,rng.choice(model.free,k,replace=False))))
            evidence=ContactShapes(stars)
            choices=CycleOrders(family)
            pool=set(evidence.allowed_orders(cells,choices,208))
            surviving=0
            for order in choices:
                if stars.price(cells,order)>=416:
                    assert order in pool
                    surviving+=1
            rows.append(dict(cells=cells,family=family,total=len(choices),remaining=len(pool),
                             above_star=surviving,work=dict(evidence.work)))
        atomic_json(out,rows)
        print(family,flush=True)


def distance_trials(model,board,out):
    rows=[]
    for pool in ((0,1,2,3),(4,9,16,19,22,43),(2,6,9,17,43,66),
                 (18,30,42,44,51,62,118,122)):
        for target in (207,208):
            d=DistanceModel(model,board,pool)
            candidate=d.solve(target,maximum=6,seconds=20,optimize=True)
            # Identify physical feasibility against all fixed families including
            # identity. Negative distance-k claims need every family up to k.
            candidates=[]
            for k in range(7):
                def families(total,lo):
                    if not total:
                        yield ()
                    for cost in range(lo,total+1):
                        for tail in families(total-cost,cost):
                            yield (cost+1,*tail)
                witness=False
                for fam in families(k,1):
                    if sum(fam)>len(pool):
                        continue
                    control=CycleModel(model,board,pool,fam).solve(target=target,seconds=4)
                    candidates.append(dict(family=fam,status=control['status'],branches=control.get('branches'),score=control.get('score')))
                    if control.get('score') is not None and control['score']>=target:
                        witness=True
                        break
                if witness:
                    break
            if candidate.get('distance') is not None:
                assert candidate['score']>=target
                assert not all(r['status']=='INFEASIBLE' for r in candidates)
            rows.append(dict(pool=pool,target=target,candidate=candidate,fixed=candidates))
            atomic_json(out,rows)
            print(pool,target,candidate['status'],candidate['branches'],flush=True)


def donor_trials(model,reference,out):
    rows=[]
    for case,board in [('incumbent',reference),('moderate',np.array(reference)),('fresh',np.array(reference))]:
        if case!='incumbent':
            rng=np.random.default_rng(3523)
            n=18 if case=='moderate' else 75
            cells=rng.choice(model.free,n,replace=False)
            board[cells]=board[rng.permutation(cells)]
        start=model.problem.validate(list(map(int,board)))['matched_edges']
        for seed in range(8):
            for size in (8,):
                for imports in (0,1,2,3,4,-1):
                    rng=np.random.default_rng(seed)
                    if imports<0:
                        pool=list(map(int,rng.choice(model.free,size,replace=False)));w={}
                    elif imports==0:
                        pool=list(map(int,forest_pool(model,np.array(board),model.free,size,rng)));w={}
                    else:
                        pool,w=response_pool(model,board,size,rng,imports=min(3,imports),adaptive=imports==4)
                    candidate,result=joint_repair(model,board,pool,contacts=-1,effort=.5)
                    score=model.problem.validate(list(map(int,candidate)))['matched_edges']
                    assert len(set(pool))==size and score>=start
                    rows.append(dict(case=case,seed=seed,size=size,imports=imports,pool=pool,start=start,
                                     score=score,work=w,native=result))
            atomic_json(out,rows)
            print(case,seed,[(r['imports'],r['score']) for r in rows[-6:]],flush=True)


def pages(problem,board,out):
    data=[]
    with TemporaryDirectory(dir=out.parent) as path:
        for family in ((6,),(2,5),(3,4),(2,2,4)):
            size=sum(family)
            for cells in ((30,62,105,104,107,106,43,44),(43,123,30,51,62,42,44,103)):
                start=rank(cells[:size],BoardModel(problem).free)
                results={}
                for mode in ('old','new'):
                    db=Path(path)/(str(family)+mode+'.sqlite')
                    with patch('r35_route_splits.split_reject',return_value=False) if mode=='old' else __import__('contextlib').nullcontext():
                        arr=run(problem,[board],db,families=(family,),start=start,stop=start+2,page=2)
                    assert arr['range_finished']
                    results[mode]=dict(best=arr['best'],work=arr['work'])
                assert results['old']['best']==results['new']['best']
                data.append(dict(family=family,start=start,modes=results))
                atomic_json(out,data)
                print(family,start,[(k,v['work'].get('first_predecessor',0)) for k,v in results.items()],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',choices=('distance','donor','shapes','pages'))
    parser.add_argument('--tiles',default='../data/tiles.json')
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    model=BoardModel(SearchProblem(read_json(a.tiles)))
    board=read_json('docs/R29_INCUMBENT.json')['codes']
    if a.experiment=='pages':
        pages(model.problem,board,a.output)
    else:
        {'distance':distance_trials,'donor':donor_trials,'shapes':shape_trials}[a.experiment](model,board,a.output)
