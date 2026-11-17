"""Independent follow-ups: connected-price discriminators, bridge bounds and full-board prefix coverage."""
import argparse,json,tempfile
from itertools import combinations
from pathlib import Path
from collections import Counter
import numpy as np

from r28_exchange import read_json,ExchangeBound,domain_binding
from r27_incumbent import BoardModel
from r29_exchange_clusters import StarClusters
from r29_exchange_route import physical_hash
from r30_multicenter import orders
from r31_route_geometry import eligible_orders,cycle_options
from r31_route_price import DefectPrices
from r32_route_join import donor_bridge_upper
from r32_route_pool import stream_prefixes,defect_cover
from search_problem import SearchProblem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--output',required=True)
    a=parser.parse_args()
    p=SearchProblem(read_json(a.tiles))
    model=BoardModel(p)
    boards=[[3*t for t in row['pieces']] for row in read_json('docs/R29_EXCHANGE_CENTERS.json')['centers']]
    bounds=[ExchangeBound(model,b) for b in boards]
    stars=[StarClusters(b) for b in bounds]
    rep=dict(domain=domain_binding(model),order_filter=[],bridge=[],stream=[],defects=[])
    rng=np.random.default_rng(320002)
    for f in ((2,2,3),(2,2,2,2)):
        rec=Counter()
        for _ in range(900):
            k=sum(f)
            if _%2:
                cells=tuple(sorted(map(int,rng.choice(160,k,replace=False))))
            else:
                source=set(map(int,rng.choice(160,3,replace=False)))
                while len(source)<k:
                    candidate=int(rng.choice(sorted(source)))
                    source.add(int(rng.choice(np.flatnonzero(stars[0].close[candidate]))))
                cells=tuple(sorted(source))
            eligible=tuple(eligible_orders(stars[0].close,cells,f))
            if not eligible:continue
            for s in stars:
                typed=s.support_upper_twice(cells,family=f)
                coupled=s.support_upper_twice(cells,orders=eligible)
                assert coupled<=typed
                rec['queries']+=1
                rec['strictly_tighter']+=coupled<typed
                rec['additional_208_closures']+=coupled<416<=typed
        rep['order_filter'].append(dict(family=f,work=dict(rec)))
        with tempfile.TemporaryDirectory() as tmp:
            for size,pool in [(10,tuple(range(10))),(160,tuple(range(160)))]:
                args=dict(pool=pool,family=f,target=208,database=str(Path(tmp)/f'{size}.sqlite'),start=12,stop=44)
                cold=stream_prefixes(p,boards,**args,max_pages=12)
                rest=stream_prefixes(p,boards,**args)
                warm=stream_prefixes(p,boards,**args)
                assert cold['new_pages']>0 and rest['range_finished'] and warm['new_pages']==0 and not warm['work']
                rep['stream'].append(dict(size=size,family=f,partial=cold,resume=rest,warm=warm))
        pool=set(range(12))
        start=((0,1),(2,3)) if len(f)==3 else ((0,1),(2,3),(4,5))
        width=sum(f)-sum(map(len,start))
        for s in stars:
            rest=pool-{c for g in start for c in g}
            options=list(cycle_options(rest,width,rest))
            upper,work=donor_bridge_upper(s,start,options)
            rep['bridge'].append(dict(family=f,center=physical_hash(s.bound.tiles),prefix=start,
                                      upper_twice=upper,closed=upper<416,work=work))
    # Evaluate several completely free prefixes in every center. Only checked INFEASIBLE closes.
    for s in stars:
        scorer=DefectPrices(s.bound)
        for left,right in combinations([(18,30),(42,44),(51,62),(118,122),(146,155),(0,1),(2,3),(4,5)],2):
            g=(left,right)
            if scorer.actor_upper(g,3,pool=range(160))<208:
                continue
            result=defect_cover(s.bound,g,range(160),3,208)
            rep['defects'].append(dict(center=physical_hash(s.bound.tiles),prefix=g,result=result))
    Path(a.output).write_text(json.dumps(rep,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
