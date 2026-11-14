"""Reproduce all six full directed-cycle joins and bounded larger-family controls."""
import argparse
import itertools
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r28_exchange_rotations import RotationOracle
from r29_exchange_route import ROOT,domain_binding,rotation_optimum,swap_orders
from r30_cycles import Predecessors,canonical_cycles
from r30_cycle_joins import TypedJoin
from r30_route import search
from search_fixtures import planted_gold
from search_problem import SearchProblem


def pool_comparison(bound,pool,family,threshold):
    summary={}
    ref=None
    for mode in ('typed_dfs','indexed_halves'):
        join=TypedJoin(bound,pool,family,threshold)
        start=time.perf_counter()
        work=join.survivors() if mode=='typed_dfs' else join.half_survivors()
        rows={(cells,order) for _,cells,order in work}
        summary[mode]=dict(join.cost,wall=time.perf_counter()-start)
        if ref is None:ref=rows
        elif ref!=rows:raise ArithmeticError('Half join and direct typed join disagree.')
    orders=[o for o in swap_orders(sum(family),sum(family)-len(family))
            if sorted(map(len,canonical_cycles(range(sum(family)),o)))==sorted(family)]
    subsets=np.asarray(list(itertools.combinations(sorted(pool),sum(family))),dtype=int)
    direct={(tuple(r),o) for o in orders for r in subsets[bound.bounds(subsets,o)>=threshold]}
    if direct!=ref:raise ArithmeticError('Join disagrees with independent donor enumeration.')
    summary['direct_star_queries']=len(orders)*len(subsets)
    summary['expected_star_survivors']=len(ref)
    return summary


def class_bounds(bound):
    """Measure the additional supports closed by the family-specific donor assignment."""
    from r29_exchange_clusters import StarClusters
    clusters=StarClusters(bound)
    rng=np.random.default_rng(3007)
    rows=[]
    for family in ((4,2),(3,3),(3,2,2),(2,2,2,2)):
        n=sum(family)
        # Mix independent samples with dense high-scoring donor regions.
        pools=[[18,30,42,44,51,62,118,122,146,155],list(range(12)),list(range(bound.n))]
        lower=0;extra=0;screened=0
        random_cells=[tuple(sorted(rng.choice(pools[j%3],n,replace=False).tolist())) for j in range(360)]
        hard_cells=set()
        for base in ((30,62,105,104,107,106),(43,123,30,51,62,42)):
            near=sorted({int(v) for c in base for v in bound.neighbors[c]}-set(base)-{bound.n})
            # One-cell substitutions about each observed six-cell star survivor.
            for removed in (None,*base):
                seed=set(base)-{removed}
                for add in itertools.combinations(near,n-len(seed)):
                    hard_cells.add(tuple(sorted(seed|set(add))))
        # Balanced reproducible upper limit for dense eight-row neighborhoods.
        hard_cells=sorted(hard_cells)
        if len(hard_cells)>400:
            hard_cells=[hard_cells[i] for i in sorted(rng.choice(len(hard_cells),400,replace=False))]
        for cells in random_cells+hard_cells:
            unconstrained=clusters.support_upper_twice(cells)
            typed=clusters.support_upper_twice(cells,family=family)
            if unconstrained<typed:raise ArithmeticError('Family relaxation unexpectedly weaker.')
            lower+=typed<unconstrained
            extra+=typed<416<=unconstrained
            screened+=unconstrained<416
        rows.append(dict(family=family,samples=len(random_cells)+len(hard_cells),targeted=len(hard_cells),stricter=lower,additional_closed_at_208=extra,
                         generic_closed_at_208=screened))
    return rows


def ordering_controls():
    """Plant a reachable exact endpoint inside each cycle family."""
    result=[]
    for family in ((4,2),(3,3),(3,2,2),(2,2,2,2)):
        size=sum(family)
        for seed in (21,37):
            data,geo,codes=planted_gold(seed,1,3)
            p=SearchProblem(data,geo)
            model=BoardModel(p)
            rng=np.random.default_rng(seed)
            indices=rng.permutation(size).tolist()
            row=np.asarray(codes).copy()
            offset=0
            for k in family:
                cycle=indices[offset:offset+k];offset+=k
                row[cycle]=row[cycle[1:]+cycle[:1]]
            bound=ExchangeBound(model,row)
            outcomes=[]
            for ranked in (False,True):
                searcher=TypedJoin(bound,range(size),family,len(p.edges))
                oracle=RotationOracle(model)
                exact=0;score=None
                for cycles,cells,order in searcher.survivors(order_by_gain=ranked):
                    item=rotation_optimum(bound,oracle,bound.witness_seed(cells,order));exact+=1
                    if item['score']==len(p.edges):
                        score=item['score'];break
                if score!=len(p.edges):raise ArithmeticError('Missed planted witness.')
                outcomes.append(dict(order='gain' if ranked else 'cell',prefixes=searcher.cost['prefixes'],
                                     stars=searcher.cost['star_queries'],exact=exact,score=score))
            result.append(dict(family=family,seed=seed,modes=outcomes))
    return result


def benchmark(problem,database,out):
    centers=read_json(ROOT/'docs/R29_EXCHANGE_CENTERS.json')
    results=[]
    model=BoardModel(problem)
    if domain_binding(model)!=centers['domain']:raise ValueError('Requires the release tile set and domain.')
    for j in (1,0,2):
        row=centers['centers'][j]
        board=[3*t for t in row['pieces']]
        for fam in ((3,3),(4,2)):
            result=search(problem,board,208,database,fam,batch=64)
            results.append(result)
            atomic_json(out,dict(domain=domain_binding(model),six=results))
    bound=ExchangeBound(model,[3*x for x in centers['centers'][1]['pieces']])
    pred=Predecessors(bound,208)
    candidates=[]
    # Validate only actual final physical survivors within the chosen scopes.
    from contextlib import closing
    with closing(sqlite3.connect(database)) as db:
        for encoded, in db.execute("SELECT result FROM optima WHERE key LIKE ?",(domain_binding(model)+':%',)):
            item=json.loads(encoded)
            donors=np.argsort(bound.tiles)[np.asarray(item['codes'])//3]
            cs=canonical_cycles(range(bound.n),donors)
            if tuple(sorted(map(len,cs))) not in ((2,4),(3,3)):continue
            contacts=sum(b in pred.close[a] for a in cs[0] for b in cs[1])
            candidates.append(dict(cycles=cs,contacts=contacts,score=item['score'],
                                   shallow=pred.reject(cs),deep=pred.reject(cs,deep=True)))
    controls=[]
    for family in ((3,2,2),(2,2,2,2)):
        for pool in ([18,30,42,44,51,62,118,122,146,155], list(range(10))):
            result=pool_comparison(bound,pool,family,208)
            controls.append(dict(pool=pool,family=family,work=result))
    report=dict(domain=domain_binding(model),six=results,middle_candidates=candidates,large_pools=controls,ordering=ordering_controls(),
                family_bounds=class_bounds(bound))
    atomic_json(out,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--database',required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    benchmark(SearchProblem(read_json(a.tiles)),a.database,a.output)
