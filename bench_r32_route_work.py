"""Rerun original-tile paired experiments for route workstreams 1 through 7."""
from collections import Counter
from itertools import combinations
from time import monotonic
from pathlib import Path
import argparse,json,tempfile
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, read_json,domain_binding
from r29_exchange_clusters import StarClusters
from r30_multicenter import run
from r31_route_geometry import ConnectedJoin
from r31_route_price import DefectPrices
from r30_cycle_joins import solve_pool
from r32_route_join import traverse
from r32_route_pool import defect_cover
from search_problem import SearchProblem


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--output',required=True)
    p.add_argument('--ranks',type=int,default=4096)
    a=p.parse_args()
    problem=SearchProblem(read_json(a.tiles))
    model=BoardModel(problem)
    archive=read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')
    boards=[[3*t for t in row['pieces']] for row in archive['centers']]
    bounds=[ExchangeBound(model,b) for b in boards]
    stars=[StarClusters(b) for b in bounds]
    report=dict(domain=domain_binding(model),ranks=[],joins=[],covers=[])
    def save():
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    old=('contact','core','connected','star','family')
    with tempfile.TemporaryDirectory() as tmp:
        for family, starts in [((2,2,3),[200000000000,301414649175,390001234567]),
                               ((2,2,2,2),[5360481967048,1014012345678,7000012345678])]:
            for start in starts:
                row=dict(family=family,start=start,width=a.ranks)
                for name,filters in [('v112',old),('ranges',('ranges',*old)),
                                     ('new',('ranges',*old[:3],'orderprice',*old[3:]))]:
                    t=monotonic()
                    result=run(problem,boards,str(Path(tmp)/f'{name}-{start}.sqlite'),start=start,stop=start+a.ranks,
                               page=256,families=[family],filters=filters)
                    assert result['range_finished']
                    row[name]=dict(work=result['work'],seconds=round(monotonic()-t,3),best=result['best'])
                assert row['new']['best']==row['v112']['best']==row['ranges']['best']
                report['ranks'].append(row)
                save()
            for pool in [(18,30,42,44,51,62,118,122,146,155),tuple(range(10)),(9,10,11,12,13,21,26,27,28,29)]:
                joins=[ConnectedJoin(s,pool,family,208) for s in stars]
                row=dict(family=family,pool=pool)
                t=monotonic()
                oldrows=set()
                count=Counter()
                for i,j in enumerate(joins):
                    oldrows.update((i,c,o) for c,o in j.survivors())
                    count.update(j.work)
                row['last']=dict(work=dict(count),seconds=round(monotonic()-t,3))
                t=monotonic()
                half = [solve_pool(problem,b,pool,family,208,halves=True) for b in boards]
                row['half']=dict(seconds=round(monotonic()-t,3),
                                 counts=[{k:v for k,v in h.items() if isinstance(v,(int,float,dict)) and k!='best'} for h in half])
                for name,kwargs in [('shared',{}),('individual',dict(together=False)),('bridge',dict(bridge=True))]:
                    t=monotonic()
                    got,work=traverse(joins,**kwargs)
                    values=list(got)
                    assert len(values)==len(set(values)) and set(values)==oldrows
                    row[name]=dict(work=dict(work),seconds=round(monotonic()-t,3))
                row['surviving_stars']=len(oldrows)
                report['joins'].append(row)
                save()
    pool=tuple(range(10))
    defects=DefectPrices(bounds[0])
    for g,k in [(((0,1,2),),4),(((0,1),(2,3)),4),(((18,30),(42,44)),3),(((51,62),(118,122)),3)]:
        for allowed in (tuple(sorted(set(pool)|{x for c in g for x in c})),tuple(range(160))):
            upper=defects.actor_upper(g,k,pool=allowed)
            if upper>=208:
                o=defect_cover(bounds[0],g,allowed,k,208)
                report['covers'].append(dict(group=g,available=len(allowed),remaining=k,old_upper=upper,result=o))
    save()


if __name__=='__main__':
    main()
