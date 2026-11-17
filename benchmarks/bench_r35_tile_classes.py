"""Screen exact physical mask rotations before constructing a soft-score tile quotient."""
import argparse
from collections import defaultdict
from copy import deepcopy
from itertools import permutations,product
from pathlib import Path

from io_utils import atomic_json
from r28_exchange import read_json
from search_fixtures import planted_gold
from search_problem import SearchProblem


def classes(problem):
    rows=defaultdict(list)
    for tile in range(problem.n):
        signature=min(tuple(map(int,row)) for row in problem.masks[3*tile:3*tile+3])
        rows[signature].append(tile)
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    p=SearchProblem(read_json(args.tiles))
    groups=classes(p)
    repeats=[ids for ids in groups.values() if len(ids)>1]
    data,geo,ref=planted_gold(3547,1,4)
    data=deepcopy(data)
    data['tiles'][1]['segments']=[[[int((side+1)%3),port] for side,port in segment] for segment in data['tiles'][0]['segments']]
    free=[i for i,v in enumerate(ref) if v//3 in (0,1)]
    pins={i:v for i,v in enumerate(ref) if i not in free}
    q=SearchProblem(data,geo,fixed=pins)
    extra=SearchProblem(data,geo,fixed=dict(pins)|{free[0]:0})
    assert any({0,1}<=set(ids) for ids in classes(q).values())
    count=0
    fixed_legal=0
    broken=0
    orbit=set()
    for ordering in permutations((0,1)):
        for rotations in product(range(3),repeat=2):
            codes=list(ref)
            for c,t,r in zip(free,ordering,rotations):
                codes[c]=3*t+r
            swapped=list(codes)
            for c in free:
                t,r=divmod(codes[c],3)
                swapped[c]=3*(1-t)+(r+(-1 if t==0 else 1))%3
            assert all(tuple(q.masks[u])==tuple(q.masks[v]) for u,v in zip(codes,swapped))
            assert q.validate(codes)['matched_edges']==q.validate(swapped)['matched_edges']
            orbit.add(min(tuple(codes),tuple(swapped)))
            count+=1
            if all(code in domain for code,domain in zip(codes,extra.domains)):
                fixed_legal+=1
                if any(code not in domain for code,domain in zip(swapped,extra.domains)):broken+=1
    assert (count,len(orbit),fixed_legal,broken)==(18,9,3,3)
    atomic_json(args.output,dict(input=p.fingerprint,pieces=p.n,cyclic_classes=len(groups),
                                 distinct_oriented_triples=len(set(tuple(map(int,row)) for row in p.masks)),
                                 repeated_classes=repeats,planted_labelled=count,planted_orbits=len(orbit),
                                 pinned_placements=fixed_legal,pinned_exchanges_invalid=broken))


if __name__=='__main__':
    main()
