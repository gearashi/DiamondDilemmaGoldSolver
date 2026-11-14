"""Prove or refute a second cross-cycle contact on the seven- and eight-cell routes."""
import argparse
import itertools
import math
from collections import Counter
from pathlib import Path

import numpy as np

from bench_r30_contacts import physical_gains
from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, combination_batch, domain_binding, read_json
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r29_exchange_route import ROOT, physical_hash, rotation_optimum
from search_problem import SearchProblem


def prices(bound, clusters, size, threshold):
    orders=((1,0),) if size==2 else ((1,2,0),(2,0,1))
    free=tuple(map(int,bound.model.free))
    count=math.comb(len(free),size)
    rows=[]
    for pos in range(0,count,4096):
        groups=combination_batch(free,size,pos,min(4096,count-pos))
        mixture=np.any([clusters.close[groups[:,a],groups[:,b]] for a,b in
                        itertools.combinations(range(size),2)],axis=0)
        for order in orders:
            gain=clusters.delta[groups,groups[:,order]].sum(axis=1)
            if np.any(mixture):
                gain[mixture]=physical_gains(bound,groups[mixture],order)
            for ix in np.flatnonzero(gain>=threshold):
                cells=tuple(map(int,groups[ix]))
                cycle=(cells[0],cells[order[0]]) if size==2 else (cells[0],cells[order[0]],cells[order[order[0]]])
                used=sum(1<<c for c in cycle)
                rows.append((int(gain[ix]),used,cycle))
    return sorted(rows,key=lambda row:(-row[0],row[2]))


def one_contact(bound,target,family,top,clusters,cache):
    # With just one cross pair, its mixed star difference is at most twice
    # the number of affected stars shared by these two distinct cells.
    affected=[set(a) for a in clusters.affected]
    overlap=max(len(a.intersection(b)) for i,a in enumerate(affected) for b in affected[:i])
    floor=2*target-bound.total
    limit=floor-2*overlap
    catalog={}
    for k in set(family):
        others=list(family);others.remove(k)
        threshold=limit-sum(sum(top[n][:q]) for n,q in Counter(others).items())
        if (k,threshold) not in cache:cache[k,threshold]=prices(bound,clusters,k,threshold)
        catalog[k]=cache[k,threshold]
    stats=Counter();survivors=[]
    oracle=RotationOracle(bound.model)
    def visit(pos,last,gain,used,cycles,contacts):
        length=family[pos]
        suffix=family[pos+1:]
        possible=sum(sum(top[n][:q]) for n,q in Counter(suffix).items())
        for ix in range(last,len(catalog[length])):
            g,mask,c=catalog[length][ix]
            if gain+g+possible<limit:break
            if used&mask:continue
            pairs=sum(clusters.close[a,b] for other in cycles for a in c for b in other)
            total=contacts+int(pairs)
            if total>1:continue
            if suffix:
                visit(pos+1,ix+1 if suffix[0]==length else 0,gain+g,used|mask,(*cycles,c),total)
                continue
            if total!=1:continue
            stats['standalone_pass']+=1
            # Apply the geometry-specific interaction cap of the one present pair.
            cap=2*sum(len(affected[a]&affected[b]) for other in (*cycles,c)
                      for second in (*cycles,c) if other<second for a in other for b in second)
            if gain+g+cap<floor:continue
            cs=(*cycles,c)
            cells=tuple(sorted(y for z in cs for y in z));index={y:j for j,y in enumerate(cells)}
            moves={a:b for z in cs for a,b in zip(z,z[1:]+z[:1])}
            order=tuple(index[moves[y]] for y in cells)
            stats['star_queries']+=1
            if clusters.price(cells,order)<2*target:continue
            seed=bound.witness_seed(cells,order)
            if seed is None:continue
            result=rotation_optimum(bound,oracle,seed)
            stats['exact']+=1
            survivors.append(dict(cycles=cs,score=result['score']))
    visit(0,0,0,0,(),0)
    return dict(family=family,catalog_counts={k:len(v) for k,v in catalog.items()},
                max_common_stars=overlap,work=dict(stats),star_survivors=survivors,
                second_contact_required=all(s['score']<target for s in survivors))


def benchmark(problem,out):
    model=BoardModel(problem)
    data=read_json(ROOT/'docs/R29_EXCHANGE_CENTERS.json')
    first=read_json(ROOT/'docs/R30_CONTACT_RESULTS.json')
    if domain_binding(model)!=data['domain']:raise ValueError('This certificate requires the published domain.')
    result=[]
    for center in data['centers']:
        fact=next(c for c in first if c['center']==center['sha256'] and c['cross_cycle_required'])
        bound=ExchangeBound(model,[3*t for t in center['pieces']])
        if bound.total!=fact['star'] or physical_hash(bound.tiles)!=fact['center']:raise ValueError('Scope mismatch.')
        top={k:fact['constituents'][str(k)]['top'] for k in (2,3)}
        clusters=StarClusters(bound)
        cache={}
        for family in ((2,2,2,2),(3,2,2)):
            row=one_contact(bound,208,family,top,clusters,cache)
            result.append(dict(center=center['sha256'],**row))
            atomic_json(out,dict(domain=data['domain'],target=208,proofs=result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    benchmark(SearchProblem(read_json(args.tiles)),args.output)
