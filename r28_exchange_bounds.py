"""Bound an entire group of physical exchanges before choosing its internal permutation."""
import itertools

import numpy as np

from constraint_connectivity_round26 import assignment_upper_bound
from r28_exchange import ExchangeBound


def partial_bound(bound, cells, *, assignment=True, distinct=True, weights=(1,1)):
    """Admissible bound for permuting exactly the listed physical inventory.

    An oriented seam contributes weights[0] at its listed first endpoint and
    weights[1] at its second. Their sum must be positive. Neighbor choices at a
    single star can be required to use distinct physical pieces.
    """
    cells = tuple(map(int,cells))
    if (len(set(cells))!=len(cells) or any(c<0 or c>=bound.n for c in cells) or
            len(weights)!=2 or any(type(v)is not int or v<0 for v in weights) or not sum(weights)):
        raise ValueError("Invalid exchange scope or weights.")
    active = set(cells)
    tiles = bound.tiles
    available = tuple(int(tiles[c]) for c in cells)
    edge_weight = np.zeros((bound.n,3),dtype=int)
    for a,sa,b,sb in bound.model.problem.edges:
        edge_weight[a,sa],edge_weight[b,sb] = weights
    rows = {c:[] for c in cells}
    constant = 0
    for c in range(bound.n):
        choices = available if c in active else (int(tiles[c]),)
        for t in choices:
            best = None
            for r in range(3):
                if not bound.model.legal[c,3*t+r]:
                    continue
                terms = []
                price = 0
                for side,nb in enumerate(bound.neighbors[c]):
                    if nb==bound.n:
                        continue
                    mask = bound.masks[t,r,side]
                    w = int(edge_weight[c,side])
                    if nb in active:
                        candidates = [(other,int(bound.support[mask,other])*w) for other in available
                                      if other!=t and any(bound.model.legal[nb,3*other:3*other+3])]
                        terms.append(candidates)
                    else:
                        price += int(bound.support[mask,tiles[nb]]) * w
                if any(not values for values in terms):
                    continue
                if distinct and len(terms)>1:
                    # Only three incident edges exist. Evaluate the local injection exactly.
                    bonus = max((sum(w for _,w in row) for row in itertools.product(*terms)
                                 if len({other for other,_ in row})==len(row)),default=None)
                    if bonus is None:
                        continue
                else:
                    bonus = sum(max(w for _,w in row) for row in terms)
                best = max(price+bonus,best) if best is not None else price+bonus
            if c in active:
                rows[c].append(best)
            elif best is None:
                return -1
            else:
                constant += best
    if assignment:
        result = assignment_upper_bound(list(rows.values()))
        if result is None:
            return -1
        variable = result["bound"]
    else:
        if any(all(v is None for v in row) for row in rows.values()):
            return -1
        variable = sum(max(v for v in row if v is not None) for row in rows.values())
    return (constant+variable)//sum(weights)


def patch_dominance(model, exterior_domains, patch, first, second):
    """Strict replacement in one patch. Both placements use the same inventory."""
    patch = tuple(patch)
    if (len(set(patch))!=len(patch) or len(first)!=len(second) or len(first)!=len(patch)
            or len({x//3 for x in first})!=len(patch) or sorted(x//3 for x in first)!=sorted(x//3 for x in second)
            or any(not model.legal[c,x] or not model.legal[c,y] for c,x,y in zip(patch,first,second))):
        raise ValueError("Replacement must preserve the patch's complete legal inventory.")
    before,after = dict(zip(patch,first)),dict(zip(patch,second))
    exterior = {}
    constant = 0
    for a,sa,b,sb in model.problem.edges:
        if a not in before and b not in before:
            continue
        if b in before and a not in before:
            a,sa,b,sb = b,sb,a,sa
        mask = model.masks[before[a],sa]
        other = model.masks[after[a],sa]
        if b in before:
            constant += int(other==model.reverse[model.masks[after[b],sb]])-int(mask==model.reverse[model.masks[before[b],sb]])
        else:
            codes = exterior_domains[b]
            values = exterior.setdefault(b,np.zeros(len(codes),dtype=int))
            neighbor = model.reverse[model.masks[list(codes),sb]]
            values += (other==neighbor).astype(int)-(mask==neighbor)
    return constant+sum(int(v.min()) for v in exterior.values())>0
