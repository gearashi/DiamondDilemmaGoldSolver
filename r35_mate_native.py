"""Post small reciprocal matching prices when the phasewise price is exact."""
from itertools import combinations

from r29_gold_phase import SignedParity


def always_exact(problem, groups, required, limit=12):
    """Every consistent phase has at most this many undominated open groups."""
    groups = sorted({frozenset(group) - required for group in groups}, key=lambda g: (len(g), sorted(g)))
    minimal = []
    for group in groups:
        if not any(prior <= group for prior in minimal):
            minimal.append(group)
    if len(minimal) <= limit:
        return True
    basis = SignedParity(problem)
    for a, b in required:
        if not basis.bond(a, b):
            return False
    mark = len(basis.base.trail)
    possible = 0
    for group in minimal:
        # A contradicted conjunction already loses a named bond on every signed
        # phase. This check changes neither the group's clauses nor its cost.
        if all(basis.bond(a, b) for a, b in sorted(group)):
            possible += 1
        basis.base.rollback(mark)
        if possible > limit:
            return False
    return True


def named_pairs(prices):
    sides = {side for family, _ in prices.classes if len(family) <= prices.limit for side in family}
    return frozenset(pair for pair in prices.ref - prices.required if pair[0] in sides or pair[1] in sides)


def clauses(prices, losses, top):
    """Use a retained named seam as the exact complement of its charged loss."""
    result = []
    covered = set()
    modeled_sides = set()
    for family, edges in prices.classes:
        if len(family) > prices.limit:
            continue
        modeled_sides.update(family)
        by_side = {side: [] for side in family}
        for a, b, parity in edges:
            if (a, b) in losses:
                chosen = -losses[a, b]
            else:
                top += 1
                chosen = top
            x, y = a // 3 + 1, b // 3 + 1
            compatible = ((x, y), (-x, -y)) if parity else ((x, -y), (-x, y))
            result.extend([[-chosen, u, v] for u, v in compatible])
            by_side[a].append(chosen)
            by_side[b].append(chosen)
            covered.add((a, b))
            if (a, b) in prices.required:
                result.append([chosen])
        for options in by_side.values():
            result.append(options)
            result.extend([-a, -b] for a, b in combinations(options, 2))
    for bond in named_pairs(prices) - covered:
        result.append([losses[bond]])
    if any((a in modeled_sides or b in modeled_sides) and (a, b) not in covered
           for a,b in prices.required):
        result.append([])
    return result
