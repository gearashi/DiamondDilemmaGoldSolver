"""Exhaust disconnected cycle-contact graphs in the seven- and eight-cell four-swap families."""
import argparse
import itertools
from collections import Counter
from pathlib import Path

import numpy as np

from bench_r30_single_contact import prices
from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, domain_binding, read_json
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r29_exchange_route import ROOT, physical_hash, rotation_optimum
from search_problem import SearchProblem


def joined_order(cycles):
    cells=tuple(sorted(itertools.chain.from_iterable(cycles)))
    loc={v:i for i,v in enumerate(cells)}
    move={a:b for c in cycles for a,b in zip(c,c[1:]+c[:1])}
    return cells,tuple(loc[move[a]] for a in cells)


def coupled(bound,clusters,left_length,threshold,engine,work):
    """All contacting 2+2 or 3+2 components above the given standalone gain."""
    import ctypes
    integer=ctypes.c_int
    arr=lambda dtype:np.ctypeslib.ndpointer(dtype,flags='C_CONTIGUOUS')
    fn=engine.dll.group_page
    fn.argtypes=[ctypes.c_void_p,integer,integer,integer,integer,arr(np.uint64),arr(np.int32),integer]
    fn.restype=integer
    result=[]
    def page(lo,hi,cap):
        counts=np.zeros(4,dtype=np.uint64)
        rows=np.zeros((cap,6),dtype=np.int32)
        total=fn(engine.ptr,lo,hi,left_length,bound.total+threshold,counts,rows,cap)
        if total==-1:
            if hi-lo==1:page(lo,hi,cap*2)
            else:
                mid=(hi+lo)//2
                page(lo,mid,cap)
                page(mid,hi,cap)
            return
        if total<0:raise ValueError('Invalid component page.')
        work['left_cycles']+=int(counts[0]);work['star_queries']+=int(counts[2])
        for row in rows[:total]:
            first=tuple(map(int,row[:left_length]))
            other=tuple(map(int,row[left_length:left_length+2]))
            cs=(first,other)
            cells,order=joined_order(cs)
            value=clusters.price(cells,order)-bound.total
            if value<threshold:raise AssertionError('Native joint star pricing mismatch.')
            result.append((value,sum(1<<c for c in cells),cs))
    page(0,bound.n**2,512)
    return sorted(result,reverse=True)

def group_audit(bound,top):
    target=208
    floor=2*target-bound.total
    clusters=StarClusters(bound)
    work=Counter()
    from r30_route import NativeCycles
    engine=NativeCycles(bound,target,(4,2),component=True)
    all_swaps=prices(bound,clusters,2,-100)
    # A pair-of-pairs component cannot gain more than the two best swap gains
    # plus the largest possible cross interaction among its four distinct cells.
    # Use the universal 4*4 arm first, then tighten with the measured maximum.
    pair_upper=sum(top[2][:2])+16
    detect=min(floor-pair_upper,floor-top[3][0],floor-sum(top[2][:2]))
    # Find the true best swap pair. Starting at -2 is optional. Widen until
    # at least one appears, or all gains relevant to the final threshold are included.
    cut=max(-2,detect)
    while True:
        pairs=coupled(bound,clusters,2,cut,engine,work)
        if pairs or cut==detect:break
        cut=max(detect,cut-3)
    max_pair=pairs[0][0] if pairs else cut-1
    needed=min(floor-max_pair,floor-top[3][0],floor-sum(top[2][:2]))
    if cut>needed:pairs=coupled(bound,clusters,2,needed,engine,work)
    slot={(2,2):pairs}
    # triple + swap paired, residual isolated swap.
    slot[3,2]=coupled(bound,clusters,3,floor-top[2][0],engine,work)
    engine.close()
    max_gain=lambda vals: vals[0][0] if vals else -10000
    # Only retrieve standalone cycles capable of topping up a surviving pair.
    single3=prices(bound,clusters,3,floor-max_gain(pairs)) if pairs else []
    best=[group[0][0] if group else None for group in (pairs,slot[3,2])]
    oracle=RotationOracle(bound.model)
    answer=[]
    # Each record is one connected component. Distinct components have disjoint
    # star neighborhoods, so their exact doubled star gains add.
    def candidates(first,rest,stats):
        for gain,used,cycles in first:
            def finish(pos,g,owned,cs,last):
                if pos==len(rest):
                    if g<floor:return
                    stats['star_pass']+=1
                    cells,order=joined_order(cs)
                    if clusters.price(cells,order)!=bound.total+g:raise AssertionError('Component gains failed to add.')
                    seed=bound.witness_seed(cells,order)
                    if seed is None:return
                    row=rotation_optimum(bound,oracle,seed)
                    stats['exact']+=1
                    score=row['score']
                    stats['max_score']=max(stats['max_score'],score)
                    if score>=target:
                        answer.append(dict(cycles=cs,score=score))
                    return
                group=rest[pos]
                extra=sum(max_gain(L) for L in rest[pos+1:])
                for ix,(value,mask,members) in enumerate(group):
                    if g+value+extra<floor:break
                    if last is not None and ix<=last:continue
                    if mask&owned or any(clusters.close[a,b] for C in cs for D in members for a in C for b in D):continue
                    finish(pos+1,g+value,owned|mask,cs+members,ix if pos+1<len(rest) and rest[pos+1] is group else None)
            finish(0,gain,used,cycles,None)
    swapgroups=[(g,m,(c,)) for g,m,c in all_swaps]
    tripgroups=[(g,m,(c,)) for g,m,c in single3]
    cases={}
    for name,first,rest in [('322_pair_swap_swap',pairs,[tripgroups]),('322_pair_triple_swap',slot[3,2],[swapgroups]),
                            ('2222_one_pair',pairs,[swapgroups,swapgroups])]:
        counts=Counter()
        candidates(first,rest,counts)
        cases[name]=dict(counts)
    # Order two independent pairs by their indices to remove the alternative order.
    counts=Counter()
    for i,row in enumerate(pairs):candidates([row],[pairs[i+1:]],counts)
    cases['2222_two_pairs']=dict(counts)
    # One connected group of three swaps and a remote fourth. The older
    # three-swap engine prices all physical permutations with ownership OFF.
    # Round the doubled-star threshold down because its API accepts integers.
    from r29_exchange_route import NativeTriples
    threshold=(2*target-top[2][0])//2
    triples=[]
    three=NativeTriples(bound,threshold,[])
    try:
        for lo in range(0,len(three.pairs),256):
            counters,rows=three.page(lo,min(lo+256,len(three.pairs)))
            work['triple_star_pass']+=int(counters['star_pass'])
            work['triple_seam_pass']+=int(counters['seam_pass'])
            for row in rows:
                cs=tuple(tuple(map(int,row[i:i+2])) for i in (0,2,4))
                if sum(any(clusters.close[a,b] for a in A for b in B) for A,B in itertools.combinations(cs,2))<2:
                    continue
                cells,order=joined_order(cs)
                gain=clusters.price(cells,order)-bound.total
                if gain+top[2][0]>=floor:
                    triples.append((gain,sum(1<<c for c in cells),cs))
    finally:three.close()
    counts=Counter()
    candidates(triples,[swapgroups],counts)
    cases['2222_three_plus_one']=dict(counts)
    return dict(component_queries=dict(work),best_retained_pair_gains=best,
                minimum_connected_cycles=3 if not answer else None,
                four_swaps_all_connected=not any(len(hit['cycles'])==4 for hit in answer),retained={str(k):len(v) for k,v in slot.items()},
                cases=cases,hits=answer)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    model=BoardModel(SearchProblem(read_json(a.tiles)))
    archive=read_json(ROOT/'docs/R29_EXCHANGE_CENTERS.json')
    contact=read_json(ROOT/'docs/R30_CONTACT_RESULTS.json')
    if domain_binding(model)!=archive['domain']:raise ValueError('Expected the release puzzle.')
    output=[]
    for rec in archive['centers']:
        bound=ExchangeBound(model,[3*t for t in rec['pieces']])
        raw=next(c for c in contact if c['center']==physical_hash(bound.tiles) and c['cross_cycle_required'])
        top={k:raw['constituents'][str(k)]['top'] for k in (2,3)}
        output.append(dict(center=rec['sha256'],**group_audit(bound,top)))
        atomic_json(a.output,dict(domain=archive['domain'],target=208,proofs=output))
