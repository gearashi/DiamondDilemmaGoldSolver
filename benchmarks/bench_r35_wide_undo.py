"""Probe larger geographic same-boundary cases from genuine five-swap endpoints."""
from collections import Counter
from itertools import combinations,product
import argparse
from pathlib import Path
from random import Random
import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r29_exchange_route import physical_hash,existing_balls,owner_copies,is_owned
from r30_multicenter import released_six
from r28_exchange import domain_binding
from r34_exchange_undo import CoveredUndo
from r35_route_difference import differential
from io_utils import atomic_json
from search_problem import SearchProblem


class SmallTable(CoveredUndo):
    def _region_gain(self,cells,donors,changed_ports=()):
        boundary=set().union(*(self.near[c] for c in cells))-set(cells)
        if len(boundary)+len(cells)>12:
            return None
        return super()._region_gain(cells,donors,changed_ports)


def dense(bound,cells,donors,common):
    """Brute-force interiors and exteriors in independent blocks (all rotations)."""
    m=bound.model
    neighbors={int(d) for c in cells for d in bound.neighbors[c] if d!=bound.n}
    outside=sorted(neighbors-set(cells))
    seams=[(a,s,b,t) for a,s,b,t in m.problem.edges if a in cells or b in cells]
    positions={c:j for j,c in enumerate(cells)}
    ext=list(product(range(3),repeat=len(outside)))
    best=-10**9
    for start in range(0,len(ext),16):
        batch=np.array(ext[start:start+16])
        outer={c:3*bound.tiles[common.get(c,c)]+batch[:,j,None] for j,c in enumerate(outside)}
        prices=[]
        for arrangement in (cells,donors):
            domains=[tuple(v for v in range(3*bound.tiles[d],3*bound.tiles[d]+3) if m.legal[c,v])
                     for c,d in zip(cells,arrangement)]
            interior=np.asarray(list(product(*domains)),dtype=int)
            codes=outer|{c:interior[:,j] for j,c in enumerate(cells)}
            cost=0
            for a,s,b,t in seams:
                cost=cost+(m.masks[codes[a],s]==m.reverse[m.masks[codes[b],t]]).astype(np.int16)
            prices.append(cost.max(axis=1))
        best=max(best,int((prices[1]-prices[0]).max()))
    return best


def run(tiles,out):
    m=BoardModel(SearchProblem(read_json(tiles)))
    b=ExchangeBound(m,read_json('docs/R29_INCUMBENT.json')['codes'])
    pool=sorted(map(int,m.free))
    centers=existing_balls(m,208)
    region=CoveredUndo(b,lambda g:False)
    rng=Random(352019)
    records=[]
    for family in ((2,2,4),(2,2,2,3),(2,2,2,2,2),(2,3,3)):
        owners=list(owner_copies(m,centers,b.tiles,sum(family)))
        def covered(groups):
            cells=sorted(c for row in groups for c in row)
            pos={c:i for i,c in enumerate(cells)}
            changes={c:d for row in groups for c,d in zip(row,(*row[1:],row[0]))}
            return is_owned(owners,cells,tuple(pos[changes[c]] for c in cells))
        counters=Counter()
        examples=[]
        for trial in range(24):
            # The final small predecessor (one physically covered swap) keeps
            # the remaining adjacent cycles in a single varied donor region.
            sizes=family[1:]
            changed=[rng.choice(pool)]
            while len(changed)<sum(sizes):
                near=set().union(*(region.near[c] for c in changed))-set(changed)
                changed.append(rng.choice(sorted(near & set(pool))))
            cols=list(changed);rng.shuffle(cols)
            cycles=[]
            for size in sizes:
                cycles.append(tuple(cols[:size]));cols=cols[size:]
            remaining=tuple(rng.sample(sorted(set(pool)-set(changed)),family[0]))
            group=tuple(sorted((*cycles,remaining),key=lambda v:(len(v),v)))
            before=SmallTable(b,covered).rejects(group)
            after=CoveredUndo(b,covered).rejects(group)
            counters['all_checked']+=1
            counters['previous_first']+=before
            counters['new_first']+=after
            # A sparse valid owner may have only the specific predecessor page.
            specific=lambda groups:tuple(groups)==(remaining,) and covered(groups)
            original=SmallTable(b,specific).rejects(group)
            refined=CoveredUndo(b,specific).rejects(group)
            counters['sparse_old']+=original
            counters['sparse_new']+=refined
            assert not original or refined
            assert not before or after
            boundary=set().union(*(region.near[c] for c in changed))-set(changed)
            if len(boundary)+len(changed)>12 and len(examples)<2 and len(changed)==6:
                donors={c:d for cycle in cycles for c,d in zip(cycle,(*cycle[1:],cycle[0]))}
                common=dict(zip(remaining,(*remaining[1:],remaining[0])))
                price=differential(b,dict(zip(changed,changed)),donors,common)
                exact=dense(b,changed,[donors[c] for c in changed],common)
                assert price==exact
                examples.append(dict(cells=changed,cycles=cycles,predecessor=remaining,price=price,dense=exact))
        records.append(dict(family=family,work=dict(counters),independent=examples))
        atomic_json(out,records)
        print(family,counters,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='../data/tiles.json')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    run(a.tiles,a.output)
