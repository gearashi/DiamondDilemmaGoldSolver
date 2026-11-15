"""Joint physical contact and compulsory-mover patterns for a small colex suffix."""
from itertools import combinations, combinations_with_replacement
from math import comb

from r31_route_geometry import connected_mask


def _classes(low, tests):
    rows = [(low, 0)]
    for i, test in enumerate(tests):
        rows = [(part, value) for subset, tag in rows
                for part, value in ((subset & test, tag | (1 << i)), (subset & ~test, tag)) if part]
    return rows


def _contact_exists(left, right, contacts):
    if left.bit_count() > right.bit_count():
        left, right = right, left
    while left:
        bit = left & -left
        if contacts[bit.bit_length() - 1] & right:
            return True
        left ^= bit
    return False


def _realizable(parts, pairs, contact_mask, contacts):
    """Test required edges and injective representatives. Extra edges are allowed."""
    if not contact_mask:
        return True  # multiplicities per disjoint class were checked by the caller
    if len(parts) == 2:
        return _contact_exists(*parts, contacts)
    edges = [pair for j, pair in enumerate(pairs) if contact_mask >> j & 1]
    if len(edges) == 1:
        a, b = edges[0]
        # If the third cell shares a class, multiplicity guarantees another
        # representative after choosing the edge endpoints.
        return _contact_exists(parts[a], parts[b], contacts)
    pivot = next(c for c in range(3) if all(c in edge for edge in edges[:2]))
    left, right = (j for j in range(3) if j != pivot)
    roots = parts[pivot]
    while roots:
        v = roots & -roots
        roots ^= v
        near = contacts[v.bit_length() - 1]
        a, b = near & parts[left], near & parts[right]
        if a and b and ((a | b).bit_count() >= 2):
            if len(edges) == 2 or _contact_exists(a, b, contacts):
                return True
    return False


def attainable(engine, fixed, width, limit, budget=4096):
    """Return exact feasibility, or a conservative positive on the class-work limit."""
    if width not in (2, 3):
        raise ValueError('Expected two or three free suffix positions.')
    low = engine.low[limit]
    if low.bit_count() < width:
        return False
    fixed_mask = sum(1 << c for c in fixed)
    fixed_tests = tuple(engine.contacts[c] for c in fixed)
    tags_mask = (1 << len(fixed)) - 1
    pairs = tuple(combinations(range(width + len(fixed)), 2))
    inner = tuple(combinations(range(width), 2))
    positions = {pair:i for i, pair in enumerate(pairs)}
    fixed_edges = tuple(positions[i+width,j+width] for i,j in combinations(range(len(fixed)),2)
                        if engine.close[fixed[i],fixed[j]])
    possible_graphs = sorted(range(1 << len(inner)), key=int.bit_count)
    for cores, required, geometry, cap in engine.centers:
        if cap < engine.target:
            continue
        missing = tuple(sorted({low & core for core in cores if not fixed_mask & core}))
        if (not missing and len(fixed_edges)>=required
                and any(f not in geometry for f in engine.families)):
            return True
        if 0 in missing:
            continue
        rows = _classes(low, (*fixed_tests, *missing))
        if comb(len(rows) + width - 1, width) > budget:
            return True
        full = ((1 << len(missing)) - 1) << len(fixed)
        use_geometry = all(f in geometry for f in engine.families)
        for choices in combinations_with_replacement(range(len(rows)), width):
            if any(rows[c][0].bit_count() < choices.count(c) for c in set(choices)):
                continue
            tag_union = 0
            for c in choices:
                tag_union |= rows[c][1]
            if tag_union & full != full:
                continue
            external = [*fixed_edges]
            parts = []
            for j,c in enumerate(choices):
                cells, tag = rows[c]
                parts.append(cells)
                tag &= tags_mask
                while tag:
                    bit = tag & -tag
                    tag ^= bit
                    external.append(positions[j,width+bit.bit_length()-1])
            if len(external) + len(inner) < required:
                continue
            for mask in possible_graphs:
                if len(external) + mask.bit_count() < required:
                    continue
                if use_geometry:
                    edges = tuple(sorted((*external, *(positions[p] for j,p in enumerate(inner) if mask >> j & 1))))
                    if not any(connected_mask(f, edges) for f in engine.families):
                        continue
                if _realizable(parts, inner, mask, engine.contacts):
                    return True
    return False
