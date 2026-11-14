"""Semantic identities for exact soft patch queries in one fixed problem.

The outgoing codes, rather than a frozen full witness, are transplanted into
the new exterior. Global Gold acceptance always uses the independent validator.
"""
from collections import Counter
from validator import reverse_mask


def exterior(problem, board, patch, gold=True):
    cells = tuple(sorted(patch))
    inside = set(cells)
    if any(board[c] not in problem.domains[c] for c in range(problem.n) if c not in inside):
        return None
    fixed_matches = 0
    fixed_edges = 0
    boundary = []
    for a, sa, b, sb in problem.edges:
        if a in inside and b in inside:
            continue
        if b in inside:
            a, sa, b, sb = b, sb, a, sa
        if a in inside:
            boundary.append((a, sa, b, sb, reverse_mask(problem.masks[board[b], sb])))
        else:
            fixed_edges += 1
            fixed_matches += int(problem.masks[board[a], sa]) == reverse_mask(problem.masks[board[b], sb])
    interface = tuple((a, s, mask) for a, s, _, _, mask in boundary)
    # The local query forbids the current inside vector. That vector also
    # identifies the complete inventory, even when rotations are interchangeable.
    # An all-outside mismatch makes a Gold cut impossible for every inside
    # completion. Any differing outside score is a constant objective offset.
    scope = (cells, tuple(board[c] for c in cells), interface, gold and fixed_matches == fixed_edges)
    if not gold or fixed_matches < fixed_edges:
        return scope
    parents = {}
    def find(v):
        while parents[v] != v:
            parents[v] = parents[parents[v]]
            v = parents[v]
        return v
    for c, code in enumerate(board):
        if c not in inside:
            for i in range(problem._segment_counts[code//3]):
                parents[c, i] = (c, i)
    for a, sa, b, sb in problem.edges:
        if a in inside or b in inside:
            continue
        for i, j in zip(problem._segment_ports[board[a]][sa], reversed(problem._segment_ports[board[b]][sb])):
            if min(i, j) >= 0:
                parents[find((a, i))] = find((b, j))
    sizes = Counter(find(v) for v in parents)
    ends = {v:[] for v in sizes}
    for a, s, b, sb, _ in boundary:
        for pos, seg in enumerate(reversed(problem._segment_ports[board[b]][sb])):
            if seg >= 0:
                ends[find((b, seg))].append((a, s, pos))
    return scope + (tuple(sorted((sizes[g], tuple(sorted(ends[g]))) for g in sizes)),)


def transplant(problem, pending, records, *, target, loop_aware):
    board = pending['board']
    patch = pending['patch']
    # Only compare completed, conclusive queries with identical inside values.
    rows = [r for r in records if r['patch'] == patch and r['board'] != board and
            r['status'] == 'OPTIMAL' and r['candidate'] is not None and
            all(board[c] == r['board'][c] for c in patch)]
    if not rows:
        return None
    gold = target == 'gold' and loop_aware
    identity = exterior(problem, board, patch, gold)
    if identity is None:
        return None
    for row in rows:
        if exterior(problem, row['board'], patch, gold) != identity:
            continue
        result = list(board)
        for c in patch:
            result[c] = row['candidate'][c]
        if gold:
            checked = problem.validate(result)
            # For a fully matched exterior, lazy Gold cuts may also exclude a
            # nonperfect soft alternative. Transfer only an accepted Gold witness.
            if len(identity) == 5 and not checked['valid']:
                continue
        return {'status':'OPTIMAL', 'candidate':result}
    return None
