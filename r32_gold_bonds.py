"""Physical side supports that preserve the possibility of one Gold loop."""
from collections import defaultdict
from dataclasses import dataclass
from r27_structure_filters import _perfect_supports, Work, WorkLimit
from validator import reverse_mask


def endpoints(problem):
    arcs = {}
    for i, tile in enumerate(problem.tiles):
        for (s, p), (t, q) in tile['segments']:
            a, b = (3*i+s, p), (3*i+t, q)
            arcs[a], arcs[b] = b, a
    return arcs


def loop_groups(arcs, sides, pairs, total):
    """Find proper closed circuits whose joins are all among these selected bonds."""
    mate = {}
    for a, b in pairs:
        if a == b or a in mate or b in mate:
            raise ValueError("Side appears twice in a proposed sewing")
        mate[a], mate[b] = b, a
    seen = set()
    groups = set()
    for a in mate:
        for p in sides[a]:
            start = a, p
            if start in seen:
                continue
            here = start
            used = set()
            length = 0
            while here not in seen and here[0] in mate:
                seen.add(here)
                x, k = here
                y = mate[x]
                used.add((min(x,y), max(x,y)))
                here = arcs[y,12-k]
                length += 1
            if here == start and length < total:
                groups.add(frozenset(used))
    return groups


@dataclass(frozen=True)
class BondInventory:
    possible: frozenset
    forbidden: frozenset
    required: frozenset
    loops: frozenset
    closed_surface: bool


def inventory(problem):
    saved = getattr(problem, '_gold_bond_inventory', None)
    if saved is not None:
        return saved
    arcs = endpoints(problem)
    masks = [int(mask) for row in problem.masks[::3] for mask in row]
    by_mask = defaultdict(list)
    for a, m in enumerate(masks):
        by_mask[m].append(a)
    candidates = {(a,b) for a,m in enumerate(masks) for b in by_mask[reverse_mask(m)]
                  if a < b and a//3 != b//3}
    sides = [tuple(k+1 for k in range(11) if m >> k & 1) for m in masks]
    # A two-tile closed circuit must use same-side arcs on both participating tiles.
    returns = {a for (a,k),(b,j) in arcs.items() if a == b}
    loops = {pair for pair in candidates if pair[0] in returns and pair[1] in returns
             and loop_groups(arcs, sides, [pair], problem.total_segments)}
    allowed = candidates - loops
    closed = all(int(v) >= 0 for row in problem.neighbors for v in row)
    required = set()
    if closed:
        rows = [set() for _ in masks]
        for a,b in allowed:
            rows[a].add(b)
            rows[b].add(a)
        try:
            rows = _perfect_supports(rows, Work(200000))
        except WorkLimit:
            # These rows still contain every perfect sewing.
            pass
        if rows is None:
            allowed = set()
        else:
            allowed = {(a,b) for a,row in enumerate(rows) for b in row if a < b}
            required = {(min(a,b),max(a,b)) for a,row in enumerate(rows)
                        if len(row)==1 for b in row}
    saved = BondInventory(frozenset(allowed), frozenset(candidates - allowed),
                          frozenset(required), frozenset(loops), closed)
    problem._gold_bond_inventory = saved
    return saved


def forbidden_codes(pairs, side, other):
    for a,b in pairs:
        for x,y in ((a,b),(b,a)):
            yield 3*(x//3)+(side-x%3)%3, 3*(y//3)+(other-y%3)%3


def bounded_loop_cores(problem, selected, limit):
    """Trace already chosen joins, then optionally probe small connected extensions."""
    from itertools import combinations
    arcs = endpoints(problem)
    sides = [tuple(k for k in range(1,12) if (a,k) in arcs) for a in range(3*problem.n)]
    groups = loop_groups(arcs, sides, selected, problem.total_segments)
    local = inventory(problem)
    occupied = {s for pair in selected for s in pair}
    rows = sorted(p for p in local.possible if not occupied.intersection(p))
    # Two additions can close a loop only if they share a physical component.
    comp = list(range(problem.n))
    def root(i):
        while i != comp[i]:
            comp[i] = comp[comp[i]]
            i = comp[i]
        return i
    for a,b in selected:
        comp[root(a//3)] = root(b//3)
    by_component = defaultdict(list)
    for p in rows:
        for x in {root(p[0]//3),root(p[1]//3)}:
            by_component[x].append(p)
    seen = set()
    visited = 0
    for size in (1,2,3):
        for bucket in by_component.values():
            for extra in combinations(bucket,size):
                if visited >= limit:
                    return groups, visited
                if extra in seen or len({a for p in extra for a in p}) != 2*size:
                    continue
                seen.add(extra)
                visited += 1
                groups.update(loop_groups(arcs,sides,[*selected,*extra],problem.total_segments))
    return groups,visited
