"""Compare fallback cost on successively weakened published physical references."""
import argparse
from collections import Counter
from math import inf
from pathlib import Path
from random import Random
from time import monotonic
import json
import time
from itertools import combinations

from r32_gold_partition import insert

from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r35_group_circuits import GroupCircuits
from r35_group_images import GroupImages
from r35_group_drops import reuse_drops
from search_problem import SearchProblem


def legacy_drops(query, flipped, budget, deadline):
    errors = 1 | sum(1 << (i+1) for i in flipped)
    rows = {bond: row | ((symbol & errors).bit_count() & 1)
            for bond, (row, symbol) in query._joint_bonds.items()}
    basis = {}
    def resolve(base,groups,left):
        if time.monotonic()>=deadline:raise TimeoutError
        if not groups:return True
        group=groups[0]
        available=sorted(group-query.required)
        for size in range(1,min(left-len(groups)+1,len(available))+1):
            for omitted in combinations(available,size):
                child=base.copy()
                if all(insert(child,rows[p]) for p in group-set(omitted)):
                    if resolve(child,groups[1:],left-size):return True
        return False
    # Try the group with the fewest admissible single drops first.
    def difficulty(group):
        return sum(all(insert(child,rows[p]) for p in group-{q})
                   for q in group-query.required for child in [basis.copy()])
    return resolve(basis,sorted(query.groups,key=difficulty),budget-len(flipped))

def run(tiles, timeout, cuts, full):
    problem = SearchProblem(read_json(tiles))
    docs = Path(__file__).parent/'docs'
    ref,_,groups = reference_groups(problem,read_json(docs/'R29_INCUMBENT.json')['codes'])
    req = inventory(problem).required
    query = OutsideSyndromes(problem,ref,groups,req)
    order = list(query.outside)
    Random(35).shuffle(order)
    positives = read_json(docs/'R34_INTEGER_CORES_FINAL.json')['arrivals']
    patterns = [[],*positives] if full else [[],*positives[:3]]
    result = []
    for n in cuts:
        q = OutsideSyndromes(problem,ref-set(order[:n]),query.groups,req)
        # Bound memory in the comparison oracle. Direct tests enumerate all
        # labelled free phases and do not depend on this implementation.
        images = GroupImages.compile(q.groups,q.required,q._joint_bonds,len(q.outside),max_rank=15)
        q._joint_images = None
        circuits = GroupCircuits(q)
        cases=[]
        for indices in patterns:
            local = [q.outside.index(query.outside[i]) for i in indices if query.outside[i] in q.outside]
            for cap in (15,17):
                results={}
                funcs = [('drop',lambda d,w:legacy_drops(q,local,cap,d)),
                         ('reused_drop',lambda d,w:reuse_drops(q,local,cap,d,w)),
                         ('circuit',lambda d,w:circuits.solve(local,cap,d,w))]
                if images is not None:
                    funcs.append(('image',lambda d,w:images.minimum(local,d)<=cap-len(local)))
                discriminator = 1 | sum(2<<i for i in local)
                if any((c&discriminator).bit_count()&1 for c in q._joint_dependencies):
                    continue
                for name,fn in funcs:
                    w=Counter()
                    start=monotonic()
                    try:
                        val=fn(start+timeout,w)
                    except TimeoutError:
                        val=None
                    results[name]=dict(answer=val,sec=round(monotonic()-start,5),**w)
                if len({r['answer'] for r in results.values() if r['answer'] is not None})>1:
                    raise AssertionError((n,local,cap,results))
                cases.append(dict(outside=local,cap=cap,arms=results))
        counts={name:sum(k['arms'][name]['answer'] is not None for k in cases) for name in results}
        print(n, len(images.images) if images else None, len(cases), counts,flush=True)
        result.append(dict(omitted=n, image_count=len(images.images) if images else None,cases=cases))
    return dict(timeout=timeout,layout_seed=35,cases=result)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    parser.add_argument('--timeout',type=float,default=.5)
    parser.add_argument('--cuts',type=int,nargs='+',default=[4,8,16,32,80])
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.tiles,args.timeout,args.cuts,args.full)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
