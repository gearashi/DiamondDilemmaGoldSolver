"""Adaptive residuals, comparison witnesses, CP replay and root/pool interoperability."""
import argparse
from collections import Counter
from itertools import combinations
import json
from math import comb
from pathlib import Path
import tempfile

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r29_exchange_clusters import StarClusters
from r30_cycles import canonical_cycles
from r30_cycle_joins import TypedJoin
from r30_exact_routes import CycleModel,learn,rules_in
from r30_multicenter import orders,run
from r31_route_geometry import eligible_orders
from r31_route_price import DefectPrices,Exterior,SharedStarPrice,comparison_relations
from r31_routes import search,search_roots,rank
from search_problem import SearchProblem


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    problem=SearchProblem(read_json(a.tiles))
    model=BoardModel(problem)
    codes=read_json(Path(__file__).parent/'docs/R29_INCUMBENT.json')['codes']
    b=ExchangeBound(model,codes)
    stars=StarClusters(b)
    defects=DefectPrices(b)
    rng=np.random.default_rng(3131)
    report={'seed':3131,'geometry_cp':[],'joint':[],'comparisons':[]}
    def save():
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    report['legacy_half']=[]
    for pool in [(18,30,42,44,51,62,118,122,146,155),tuple(range(10))]:
        for f in ((2,2,3),(2,2,2,2)):
            old=TypedJoin(b,pool,f,208,contact=True)
            rows=list(old.half_survivors())
            report['legacy_half'].append(dict(pool=pool,family=f,work=dict(old.cost),hits=len(rows)))
    family=(2,2,2,2)
    for _ in range(100000):
        cells=tuple(sorted(map(int,rng.choice(160,8,replace=False))))
        if sum(stars.close[x,y] for x,y in combinations(cells,2))<3 or tuple(eligible_orders(stars.close,cells,family)):
            continue
        query=CycleModel(model,codes,cells,family)
        row={'cells':cells}
        row['one_contact']=query.solve(target=208,seconds=3,contact=True,connected=False)
        row['all_cycles']=query.solve(target=208,seconds=3,contact=True,connected=True)
        assert row['all_cycles']['status']=='INFEASIBLE'
        report['geometry_cp'].append(row)
        if len(report['geometry_cp'])==8:
            break
    save()
    for fam in ((2,2,3),family):
        k=sum(fam)
        count=Counter()
        hard=[]
        for i in range(1200):
            # Mix dispersed cells with a forced connected geographical patch.
            if i%3==0:
                reached={int(rng.integers(160))}
                while len(reached)<k:
                    options=sorted({int(a) for c in reached for a in b.neighbors[c]}-reached-{b.n})
                    reached.add(int(rng.choice(options)))
                cells=tuple(sorted(reached))
            else:
                cells=tuple(sorted(map(int,rng.choice(160,k,replace=False))))
            current=tuple(eligible_orders(stars.close,cells,fam))
            if not current:
                continue
            count['connected_sets']+=1
            old=stars.support_upper_twice(cells,family=fam)
            if old<416:
                continue
            count['old_typed_open']+=1
            dynamic=SharedStarPrice(b,cells).upper(current)
            assert dynamic==max(stars.price(cells,o) for o in current)
            count['joint_closed']+=dynamic<416
            if dynamic>=416:
                for o in current:
                    cap=stars.price(cells,o)
                    if cap>=416:
                        hard.append((cap,cells,o))
        report['joint'].append(dict(family=fam,sets=1200,**count))
        if not hard:
            # Probe tightest candidates among the first dense patch for a
            # lattice of thresholds, keeping Gold/208 proof scope unchanged.
            cells=tuple(range(k))
            hard=sorted(((stars.price(cells,o),cells,o) for o in eligible_orders(stars.close,cells,fam)),reverse=True)[:6]
        for cap,cells,o in sorted(hard,reverse=True)[:8]:
            keep,state=comparison_relations(b,canonical_cycles(cells,o),max_comparisons=360,max_entries=65536)
            report['comparisons'].append(dict(family=fam,cells=cells,order=o,star_twice=cap,
                                              defect_cap=defects.endpoint_upper(cells,o),survived=keep,**state))
        save()
    report['exterior_controls'] = []
    from search_fixtures import planted_gold
    for size in (1,2):
        tiles,geo,planted = planted_gold(131,size,4)
        small = ExchangeBound(BoardModel(SearchProblem(tiles,geo)),planted)
        for fam in ((2,2,3),(2,2,2,2)):
            cells = tuple(range(sum(fam)))
            shared = Exterior(small,cells)
            cold = Exterior(small,cells)
            cold.work.clear()
            chosen = orders(fam)[:12]
            first = [shared.score(o) for o in chosen]
            second = [cold.score(o,shared=False) for o in chosen]
            assert first==second
            report['exterior_controls'].append(dict(n=small.n,family=fam,scores=first,shared=dict(shared.work),cold=dict(cold.work)))
    with tempfile.TemporaryDirectory() as temp:
        db=str(Path(temp)/'transfer.sqlite')
        f=(2,2,3)
        cells=(18,30,42,44,51,62,118)
        lo=rank(cells,model.free)
        baseline=run(problem,[codes],db,families=[f],start=lo,stop=lo+1)
        replay=search(problem,codes,cells,f,208,database=db)
        assert replay['work']=={'whole_pool_reused':1}
        query=CycleModel(model,codes,cells,f)
        pins=list(zip(cells[:2],cells[1:3]))
        rule=learn(query,pins,208,seconds=4,database=db,contact=True,connected=True)
        again=query.solve(target=208,pins=pins,seconds=4,rules=rules_in(db,query.binding))
        report['transfer']=dict(ranked=baseline['work'],connected=replay['work'],learn=rule,repeat=again)
        from time import monotonic
        from sqlite3 import connect
        timed=[]
        for f in ((2,2,3),(2,2,2,2)):
            for i in range(2):
                now=monotonic()
                result=search_roots(problem,codes,f,208,db,start=0,stop=1,seconds=0.15,quantum=128)
                elapsed=monotonic()-now
                with connect(db) as check:
                    backlog=check.execute('SELECT count(*) FROM r31_root_cursor').fetchone()[0]
                assert not result['range_finished'] and result['work'].get('cursor_steps',0)>0 and backlog>0
                timed.append(dict(family=f,attempt=i+1,allowance=0.15,elapsed=elapsed,result=result))
        report['bounded_full_roots']=timed
    save()

if __name__=='__main__':
    main()
