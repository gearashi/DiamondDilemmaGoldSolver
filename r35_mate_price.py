"""Price disjoint reciprocal matching classes under shared departure obligations."""
from functools import lru_cache
from math import inf

from r32_gold_bonds import inventory
from r35_loss_witness import reciprocal
from r35_signed_mate_support import raw_edges, reciprocal_classes


def _matching_prices(family, choices, named, covers):
    """Map a mask of satisfied groups to a minimum over complete local matchings."""
    index = {side: i for i, side in enumerate(family)}
    neighbors = [0] * len(family)
    for a, b in choices:
        neighbors[index[a]] |= 1 << index[b]
        neighbors[index[b]] |= 1 << index[a]

    # Charging each named bond at its first endpoint gives it a single owner.
    owned = {}
    for a, b in named:
        owned.setdefault(a, []).append((b, covers[a, b]))

    @lru_cache(None)
    def price(mask):
        if not mask:
            return {0: 0}
        unmatched = (i for i in range(len(family)) if mask & (1 << i))
        x = min(unmatched, key=lambda i: (mask & neighbors[i]).bit_count())
        options = mask & neighbors[x]
        best = {}
        while options:
            bit = options & -options
            options ^= bit
            a, b = family[x], family[bit.bit_length() - 1]
            cost = hits = 0
            for side, partner in ((a, b), (b, a)):
                for intended, coverage in owned.get(side, ()):
                    if intended != partner:
                        cost += 1
                        hits |= coverage
            for tag, value in price(mask ^ (1 << x) ^ bit).items():
                key = tag | hits
                best[key] = min(best.get(key, inf), cost + value)
        return best

    return price((1 << len(family)) - 1)


def _relaxed_prices(named, covers):
    """Large families can discard each named bond independently."""
    best = {0: 0}
    for bond in named:
        hits = covers[bond]
        if hits:
            for tag, cost in list(best.items()):
                key = tag | hits
                best[key] = min(best.get(key, inf), cost + 1)
    return best


class ClassPrices:
    def __init__(self, problem, ref, groups, required, *, limit=16):
        self.sides, classes = reciprocal_classes(problem.tiles)
        self.ref = frozenset(ref)
        self.required = frozenset(required)
        graph = inventory(problem).possible
        self.groups = tuple(frozenset(g) for g in groups)
        self.classes = [(group, raw_edges(self.sides, group, graph)) for group in classes]
        self.limit = limit
        self.memo = lru_cache(maxsize=256)(self._price)

    def _price(self, phase):
        labels = [(s, color ^ ((phase >> (i // 3)) & 1)) for i, (s, color) in enumerate(self.sides)]
        fixed = {}
        for a, b in self.required:
            if labels[a] != reciprocal(labels[b]) or fixed.get(a, b) != b or fixed.get(b, a) != a:
                return inf
            fixed[a] = b
            fixed[b] = a
        allowed = []
        for _, edges in self.classes:
            choices = {(a, b) for a, b, _ in edges if labels[b] == reciprocal(labels[a])
                       and fixed.get(a, b) == b and fixed.get(b, a) == a}
            allowed.append(choices)
        all_edges = set().union(*allowed)
        if not self.required <= all_edges:
            return inf
        forced_losses = self.ref - all_edges
        remaining = self.ref - forced_losses - self.required
        unresolved = {g & remaining for g in self.groups if not g & forced_losses}
        obligations = []
        for group in sorted(unresolved, key=lambda g: (len(g), sorted(g))):
            if not group:
                return inf
            if not any(prior <= group for prior in obligations):
                obligations.append(group)
        if len(obligations) > 12:
            # The caller can still use its ordinary exact hitting price.
            return None

        covers = {e: sum(1 << i for i, g in enumerate(obligations) if e in g) for e in remaining}
        states = {0: 0}
        for (family, _), choices in zip(self.classes, allowed):
            named = remaining & choices
            if len(family) <= self.limit:
                frontier = _matching_prices(family, choices, named, covers)
                if not frontier:
                    return inf
            else:
                frontier = _relaxed_prices(named, covers)
            joined = {}
            for a, x in states.items():
                for b, y in frontier.items():
                    mask = a | b
                    joined[mask] = min(joined.get(mask, inf), x + y)
            states = joined
        return len(forced_losses) + states.get((1 << len(obligations)) - 1, inf)
