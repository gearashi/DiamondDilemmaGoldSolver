"""Distance-minus-one predecessor tests for arbitrary nontrivial permutation cycles."""
from itertools import combinations

from r30_cycles import canonical_cycles


def predecessor_splits(cycles):
    cells=sorted(c for group in cycles for c in group)
    index={c:i for i,c in enumerate(cells)}
    final={c:d for group in cycles for c,d in zip(group,(*group[1:],group[0]))}
    for group in cycles:
        for a,b in combinations(group,2):
            old=final.copy()
            old[a],old[b]=old[b],old[a]
            order=tuple(index[old[c]] for c in cells)
            yield canonical_cycles(cells,order),(a,b),(old[a],old[b]),(final[a],final[b]),final


def split_reject(engine,cycles):
    for pred,cells,old,new,final in predecessor_splits(cycles):
        upper=engine.predecessor_upper(pred)
        if upper is None:
            continue
        boundary=(engine.near[cells[0]]|engine.near[cells[1]])-set(cells)
        ports=tuple((c,final[c]) for c in sorted(boundary) if c in final)
        change=engine.split_gain(cells,old,new,ports)
        if change is not None and upper+change<engine.target:
            return True
    return False


def score_change(engine,cells,old,new,ports):
    from r35_route_difference import differential
    return differential(engine.bound,dict(zip(cells,old)),dict(zip(cells,new)),dict(ports))
