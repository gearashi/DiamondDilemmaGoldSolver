"""Find and independently validate held-out placement-level replay units on labelled small domains."""
import json
from itertools import combinations,product
from pathlib import Path
from random import Random

from r27_structure_filters import bits
from r27_structure_fixtures import independent,planted
from r32_gold_watch import BondWatcher
from r35_geographic_replay import replay_for
from r31_geographic import physical_bonds
from validator import reverse_mask


def scan(seed,max_scopes=80):
    rng=Random(seed);p,initial=planted(1,4,seed,False,True)
    witness=[3*t for t in range(p.n)]
    for attempt in range(max_scopes):
        ds=tuple(sum(1<<v for v in bits(mask) if v==witness[c] or rng.random()<.25) for c,mask in enumerate(initial))
        gauge=BondWatcher(p,());baseline=set(physical_bonds(p,witness))
        optional=[];forced=[]
        for c,s,d,o in p.edges:
            for v in bits(ds[c]):
                for w in bits(ds[d]):
                    if v//3==w//3 or int(p.masks[v,s])!=reverse_mask(int(p.masks[w,o])):continue
                    e=tuple(sorted((3*(v//3)+(s-v%3)%3,3*(w//3)+(o-w%3)%3)))
                    left,ls=gauge._forced_poses(ds,*e);right,rs=gauge._forced_poses(ds,*reversed(e))
                    if left or right:forced.append(e)
                    elif (ls or rs) and e not in baseline:optional.append(e)
        options=sorted(set(optional));must=sorted(set(forced))
        rng.shuffle(options);rng.shuffle(must)
        checker=replay_for(p)
        for e in options:
            for k in (0,1,2):
                for f in list(combinations(must,k))[:45]:
                    row=(e,*f)
                    if len({d for pair in row for d in pair})!=2*len(row):continue
                    if not checker.check(ds,[row]):continue
                    narrowed=list(ds)
                    watch=BondWatcher(p,(),geographic=[row],domains=ds)
                    watch.prune(narrowed)
                    if narrowed==list(ds):continue
                    expected=independent(p,ds)
                    assert expected==independent(p,narrowed) and expected
                    def edges(domains):
                        return {layout for layout in product(*(list(bits(d)) for d in domains))
                                if len({v//3 for v in layout})==p.n and len(physical_bonds(p,layout))==len(p.edges)}
                    all_edges=edges(ds)
                    assert all_edges==edges(narrowed)
                    return dict(reference_edge_perfect=len(all_edges),seed=seed,attempt=attempt,domains=[hex(d) for d in ds],core=row,
                                removals=watch.removed,reference_gold=len(expected))
    return dict(seed=seed,found=False)


if __name__=='__main__':
    rows=[scan(seed) for seed in (35,71,103,157)]
    Path('docs/R35_GEOGRAPHIC_UNITS.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
