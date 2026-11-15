"""Minimize repeated component occupancy work while replaying compound geometry conflicts."""
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
import argparse
import json

from bench_r35_loss_boundary344 import problem_scope
from r35_geographic_packing_check import check_packing


def parts(bonds):
    neighbors=defaultdict(set)
    for a,b in bonds:
        a,b=a//3,b//3
        neighbors[a].add(b);neighbors[b].add(a)
    unseen=set(neighbors)
    result=[]
    while unseen:
        root=min(unseen);stack=[root];component={root}
        while stack:
            for t in neighbors[stack.pop()]-component:
                component.add(t);stack.append(t)
        unseen.difference_update(component)
        result.append(frozenset((a,b) for a,b in bonds if a//3 in component))
    return result


def finite_pack(problem,bonds,pose_cache,max_steps):
    masks=[pose_cache(component) for component in parts(bonds)]
    masks.sort(key=len)
    steps=0
    class Limit(Exception):pass
    @lru_cache(maxsize=4096)
    def pack(i,used):
        nonlocal steps
        steps+=1
        if steps>max_steps:raise Limit
        if i==len(masks):return True
        return any(not old&used and pack(i+1,used|old) for old in masks[i])
    try:return pack(0,0)
    except Limit:return None


def compact(problem, original, pose_cache, stats, limit=3000):
    bonds=set(original)
    components=parts(bonds)
    for component in sorted(components,key=len,reverse=True):
        smaller=bonds-component
        if finite_pack(problem,smaller,pose_cache,limit) is False:
            bonds=smaller;stats['whole_components_removed']+=1
    # The native old minimizer can return a larger core when its finite
    # branching cap expires on a deletion. Retain every deletion only if
    # a completely independent occupancy search actually finishes.
    stride=max(1,len(bonds)//2)
    while stride:
        ordered=sorted(bonds)
        j=0
        while j<len(ordered):
            smaller=set(ordered[:j]+ordered[j+stride:])
            if finite_pack(problem,smaller,pose_cache,limit) is False:
                ordered=sorted(smaller)
            else:j+=stride
        bonds=set(ordered);stride//=2
    if check_packing(problem,bonds)['feasible'] is not False:
        raise ArithmeticError('A compacted geometry conflict lost its independent certificate.')
    return frozenset(bonds)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True);cli.add_argument('--claims',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path);cli.add_argument('--limit',type=int,default=3000)
    cli.add_argument('--count',type=int,default=0)
    args=cli.parse_args()
    p,_,_,_=problem_scope(args.tiles)
    original=json.loads(args.claims.read_bytes())['geographic_cores']
    if args.count:original=original[:args.count]
    pairs={};stats=Counter()
    from r35_geographic_packing_check import component_masks
    incidence={}
    for a,sa,b,sb in p.edges:incidence[a,sa]=(b,sb);incidence[b,sb]=(a,sa)
    @lru_cache(maxsize=1024)
    def poses(component):
        return component_masks(p.n,incidence,component)
    for i,row in enumerate(original):
        source=frozenset(map(tuple,row));reduced=compact(p,source,poses,stats,limit=args.limit)
        pairs[i]={'original':len(source),'size':len(reduced),'core':sorted(reduced)}
        info=poses.cache_info()
        result=dict(puzzle=p.fingerprint,work=dict(stats,hits=info.hits,misses=info.misses),
                    ordered=list(pairs.values()),geographic_cores=[x['core'] for x in pairs.values()])
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
        if i%16==0:print(i+1,len(source),len(reduced),flush=True)

if __name__=='__main__':main()
