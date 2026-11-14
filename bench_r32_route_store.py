"""Paired real-data trials for route coverage and exterior and partner reuse."""
import argparse
import json
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, domain_binding,read_json
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r29_exchange_route import physical_hash,rotation_optimum
from r30_cycle_joins import TypedJoin
from r30_multicenter import orders,run
from r31_route_geometry import eligible_orders
from r31_route_price import Exterior
from r31_routes import search_roots
from r32_exterior import CompactExterior
from r32_route_store import RootCoverage
from search_problem import SearchProblem


def measure(problem,directory):
    model=BoardModel(problem)
    codes=read_json(Path(__file__).parent/'docs/R29_INCUMBENT.json')['codes']
    centers=[[3*t for t in row['pieces']] for row in read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')['centers']]
    bound=ExchangeBound(model,codes)
    oracle=RotationOracle(model)
    stars=StarClusters(bound)
    binding=domain_binding(model)
    owner=physical_hash(bound.tiles)
    report={'fingerprint':problem.fingerprint,'families':[]}
    for family in ((2,2,3),(2,2,2,2)):
        k=sum(family)
        path=str(directory/('rank-'+str(k)+'.sqlite'))
        ranked=run(problem,centers,path,families=(family,),start=307200000000,stop=307200000128,page=32)
        cross=run(problem,[centers[1]],path,families=(family,),start=307200000000,stop=307200000128,page=32)
        cold=run(problem,[centers[1]],':memory:',families=(family,),start=307200000000,stop=307200000128,page=32)
        assert cross['best']==cold['best']==ranked['best']
        rootpath=str(directory/('roots-'+str(k)+'.sqlite'))
        pool=tuple(range(10))
        kwargs=dict(pool=pool,quantum=128)
        root1=search_roots(problem,codes,family,208,rootpath,start=0,stop=12,**kwargs)
        root2=search_roots(problem,codes,family,208,rootpath,start=12,stop=24,**kwargs)
        with closing(sqlite3.connect(rootpath)) as db:
            rc=RootCoverage(db,binding,owner,family,208)
            from itertools import combinations
            total=saved=0
            for cells in combinations(pool,k):
                for o in eligible_orders(stars.close,cells,family):
                    total+=1
                    saved+=rc.covered(cells,o,stars.close)
        runs=[]
        for cells in (tuple(range(k)),(18,30,42,44,51,62,118,122)[:k]):
            legacy=Exterior(bound,cells)
            compact=CompactExterior(bound,cells)
            comparisons=[]
            for order in orders(family)[::25]:
                original=legacy.score(order)
                current=compact.score(order)
                truth=rotation_optimum(bound,oracle,bound.witness_seed(cells,order))['score']
                assert (original is None or original==truth) and (current is None or current==truth)
                comparisons.append([original,current,truth])
            runs.append(dict(cells=cells,optima=comparisons,old_entries=legacy.work['table_entries'],
                             new_entries=compact.work['table_entries']))
        halves=[]
        for p in (pool,(18,30,42,44,51,62,118,122,146,155)):
            a=TypedJoin(bound,p,family,208,contact=True)
            b=TypedJoin(bound,p,family,208,contact=True)
            assert sorted(a.half_survivors())==sorted(b.half_survivors(True))
            halves.append(dict(pool=p,baseline=dict(a.cost),grouped=dict(b.cost)))
        report['families'].append(dict(family=family,ranked=ranked['work'],cold=cold['work'],inherited=cross['work'],
                root1=root1['work'],root2=root2['work'],connected_orders=total,covered_orders=saved,
                exterior=runs,halves=halves))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with tempfile.TemporaryDirectory() as temp:
        result=measure(SearchProblem(read_json(args.tiles)),Path(temp))
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
