"""Paired production-rank, connected-pool and scoped four-swap experiments."""
import argparse
from collections import Counter
from itertools import combinations,permutations
import json
from math import comb
from pathlib import Path
import tempfile

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,combination_batch,read_json
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r29_exchange_route import physical_hash
from r30_cycles import canonical_cycles
from r30_exact_routes import CycleModel,learn
from r30_multicenter import run,orders
from r31_route_geometry import ConnectedJoin,connected,eligible_orders
from r31_route_price import DefectPrices,Exterior,SharedStarPrice,comparison_relations
from r31_routes import search,search_roots,rank
from search_problem import SearchProblem


def actor_bounds(defects,family,length,first=None):
    """Vectorized instance of actor_upper over the entire unrestricted set of swaps/triples."""
    b=defects.b
    n=b.n
    if len(b.model.free)!=n:
        raise ValueError("Vectorized benchmark expects unrestricted movable cells.")
    edge=np.asarray([[a,z] for a,_,z,_ in defects.edges],dtype=int)
    incident=np.full((n,3),len(edge),dtype=int)
    for c,row in enumerate(defects.incident):
        incident[c,:len(row)]=[e for e,s,d in row]
    pending=[]
    counts=Counter()
    checked=0
    def consume():
        nonlocal checked
        cycles=np.asarray(pending)
        batch=len(cycles)
        ids=np.broadcast_to(b.tiles,(batch,n)).copy()
        ix=np.arange(batch)[:,None]
        ids[ix,cycles]=b.tiles[np.roll(cycles,-1,axis=1)]
        held=np.zeros((batch,n+1),dtype=bool)
        held[ix,cycles]=True
        if first:
            if np.any(held[:,list(first)]):
                raise AssertionError("Overlapping anchored cycles.")
            ids[:,list(first)]=b.tiles[list(first[1:]+first[:1])]
            held[:,list(first)]=True
        bad=~defects.fit[ids[:,edge[:,0]],ids[:,edge[:,1]]]
        bad=np.pad(bad,((0,0),(0,1)))
        degrees=bad[:,incident].sum(axis=2)
        degrees[held[:,:n]]=0
        left=sum(family)-length-(len(first) if first else 0)
        limit=np.partition(degrees,n-left,axis=1)[:,-left:].sum(1)
        repairable=(bad[:,:-1] & ~(held[:,edge[:,0]] & held[:,edge[:,1]])).sum(1)
        caps=len(edge)-bad.sum(1)+np.minimum(limit,repairable)
        for num,count in zip(*np.unique(caps,return_counts=True)):
            counts[int(num)]+=int(count)
        if checked==0:
            test=tuple(map(int,cycles[0]))
            exact=defects.actor_upper(([first] if first else [])+[test],left)
            assert exact==caps[0]
        checked+=batch
        pending.clear()
    for group in combinations([c for c in range(n) if not first or c not in first],length):
        for tail in permutations(group[1:]):
            pending.append((group[0],*tail))
            if len(pending)==512:
                consume()
    if pending:
        consume()
    return dict(actors=checked,minimum=min(counts),maximum=max(counts),
                closed_at_208=sum(num for cap,num in counts.items() if cap<208))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--output',required=True)
    p.add_argument('--triples',action='store_true')
    a=p.parse_args()
    model=BoardModel(SearchProblem(read_json(a.tiles)))
    archive=read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')
    boards=[[3*t for t in row['pieces']] for row in archive['centers']]
    center=read_json(Path(__file__).parent/'docs/R29_INCUMBENT.json')['codes']
    b=ExchangeBound(model,center)
    clusters=StarClusters(b)
    defects=DefectPrices(b)
    report=dict(domain=archive['domain'],center=physical_hash(b.tiles),target=208)
    def save():
        Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    scopes=report['ranks']=[]
    filters=('contact','core','star','family')
    rng=np.random.default_rng(131)
    pools=[(18,30,42,44,51,62,118,122,146,155),tuple(range(10))]
    with tempfile.TemporaryDirectory() as temp:
        for f in ((2,2,3),(2,2,2,2)):
            size=sum(f)
            for start in (200000000000,int(rng.integers(0,comb(160,size)-1024))):
                pair=[]
                for new in (False,True):
                    result=run(model.problem,boards,str(Path(temp)/f'{size}-{start}-{new}.sqlite'),
                        families=[f],target=208,start=start,stop=start+1024,page=256,
                        filters=filters[:2]+(('connected',) if new else ())+filters[2:])
                    assert result['range_finished']
                    pair.append(result)
                assert pair[0]['best']==pair[1]['best']
                scopes.append(dict(family=f,start=start,stop=start+1024,baseline=pair[0]['work'],connected=pair[1]['work']))
                save()
    report['independent_geometry']=[]
    for f in ((2,2,3),(2,2,2,2)):
        sets=[tuple(sorted(map(int,rng.choice(b.n,sum(f),replace=False)))) for _ in range(4096)]
        c=Counter()
        for cells in sets:
            possible=sum(bool(clusters.close[a,z]) for a,z in combinations(cells,2))
            if possible<len(f)-1:
                continue
            c['sets_passing_count']+=1
            c['orders_passing_count']+=len(orders(f))
            keep=tuple(eligible_orders(clusters.close,cells,f))
            c['sets_passing_connected']+=bool(keep)
            c['orders_passing_connected']+=len(keep)
        report['independent_geometry'].append(dict(family=f,sets=len(sets),**c))
    save()
    report['pools']=[]
    report['six_tests']=[]
    for pool in pools:
        for f in ((2,2,3),(2,2,2,2)):
            ref=ConnectedJoin(clusters,pool,f,208,actor_bounds=False)
            new=ConnectedJoin(clusters,pool,f,208)
            ref_rows=set(ref.survivors(price=False))
            new_rows=set(new.survivors())
            assert ref_rows==new_rows
            local=tuple(pool[:sum(f)])
            dirs=tuple(eligible_orders(clusters.close,local,f))
            price=SharedStarPrice(b,local)
            cap=price.upper(dirs)
            assert cap==max((clusters.price(local,o) for o in dirs),default=-1)
            cheap_caps=[defects.endpoint_upper(local,o) for o in dirs]
            stars=[clusters.price(local,o)//2 for o in dirs]
            item=dict(pool=pool,family=f,unpriced=dict(ref.work),priced=dict(new.work),surviving=len(ref_rows),
                      support=local,connected_donors=len(dirs),joint_star_twice=cap,
                      new_defect_closures=sum(s>=208 and d<208 for s,d in zip(stars,cheap_caps)),
                      defect_upper_rejections=sum(d<208 for d in cheap_caps))
            report['pools'].append(item)
            for o in sorted(dirs,key=lambda o:-clusters.price(local,o))[:3]:
                keep,w= comparison_relations(b,canonical_cycles(local,o),max_entries=65536)
                report['six_tests'].append(dict(cells=local,order=o,family=f,keep=keep,**w))
            save()
    report['actors']=[]
    for f in ((2,2,3),(2,2,2,2)):
        for length in sorted(set(f)):
            if length==3 and not a.triples:
                continue
            report['actors'].append(dict(family=f,length=length,**actor_bounds(defects,f,length)))
            save()
        for anchor in ((18,30),(51,62)):
            report['actors'].append(dict(family=f,length=2,anchor=anchor,**actor_bounds(defects,f,2,anchor)))
            save()
    report['exterior']=[]
    for f in ((2,2,3),(2,2,2,2)):
        cells=pools[0][:sum(f)]
        candidates=orders(f)[:6]
        ext=Exterior(b,cells)
        oracle=RotationOracle(model)
        results=[]
        for o in candidates:
            value=ext.score(o)
            actual=oracle.optimum(b.witness_seed(cells,o))
            if actual is not None and value is not None:
                assert value==actual[0]
            results.append((value,actual[0] if actual else None))
        report['exterior'].append(dict(family=f,cells=cells,values=results,shared=dict(ext.work),reference=oracle.work))
        save()
    report['cp']=[]
    for f in ((2,2,3),(2,2,2,2)):
        for pool in pools:
            query=CycleModel(model,center,pool,f)
            row=dict(family=f,pool=pool)
            for label,contact,connected_cut in (('ordinary',False,False),('old_contact',True,False),('connected',True,True)):
                row[label]=query.solve(target=208,seconds=8,contact=contact,connected=connected_cut)
            assert not ({'INFEASIBLE','OPTIMAL'}<={row['connected']['status'],row['ordinary']['status']})
            report['cp'].append(row)
            save()
    with tempfile.TemporaryDirectory() as temp:
        db=str(Path(temp)/'replay.sqlite')
        f=(2,2,3)
        cells=pools[0][:7]
        completed=search(model.problem,center,cells,f,208,database=db)
        lo=rank(cells,model.free)
        replay=run(model.problem,[center],db,families=[f],start=lo,stop=lo+1,target=208)
        higher=run(model.problem,[center],db,families=[f],start=lo,stop=lo+1,target=209)
        assert replay['new_pages']==0 and higher['work'].get('first_prior_page')==210
        roots=search_roots(model.problem,center,f,208,db,pool=pools[1],start=0,stop=32)
        warm=search_roots(model.problem,center,f,208,db,pool=pools[1],start=0,stop=32)
        assert roots['work']['roots_completed']==32 and warm['work']['roots_reused']==32
        report['coverage']=dict(connected=completed['work'],same_rank=replay['work'],higher=higher['work'],roots=roots,warm_roots=warm)
    save()
    print(json.dumps(dict(output=a.output,covered_ranges=len(scopes),pools=len(report['pools']))))

if __name__=='__main__':
    main()
