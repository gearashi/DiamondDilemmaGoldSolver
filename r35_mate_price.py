"""Price disjoint reciprocal matching classes under shared departure obligations."""
from functools import lru_cache
from math import inf
from time import monotonic

from r35_family_frontiers import exact_frontier, relaxed_frontier
from r32_gold_bonds import inventory
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
        self.full_mask = (1 << (len(self.sides) // 3)) - 1
        self.limit = limit
        self.memo = lru_cache(maxsize=256)(self._price)

    def _price(self, phase):
        return self.partial(phase, self.full_mask)

    def partial(self, phase, known, *, allow_broad=True):
        """Lower bound for every completion, relaxing signs at all unpinned endpoints."""
        fixed = {}
        for a, b in self.required:
            if fixed.get(a, b) != b or fixed.get(b, a) != a:
                return inf
            fixed[a] = b
            fixed[b] = a
        tile_phase = [(phase >> t) & 1 if known & (1 << t) else None
                      for t in range(len(self.sides) // 3)]
        allowed = []
        for _, edges in self.classes:
            choices = set()
            for a, b, parity in edges:
                x, y = tile_phase[a // 3], tile_phase[b // 3]
                if x is not None and y is not None and x ^ y != parity:
                    continue
                if fixed.get(a, b) == b and fixed.get(b, a) == a:
                    choices.add((a, b))
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
            if not allow_broad:
                return None
            from r35_family_obligations import family_bound
            return len(forced_losses) + family_bound(self.classes, allowed, remaining, obligations, self.limit)

        covers = {e: sum(1 << i for i, g in enumerate(obligations) if e in g) for e in remaining}
        states = {0: 0}
        for (family, _), choices in zip(self.classes, allowed):
            named = remaining & choices
            if len(family) <= self.limit:
                frontier = exact_frontier(_matching_prices, family, choices, named, covers)
                if not frontier:
                    return inf
            else:
                frontier = relaxed_frontier(_relaxed_prices, named, covers)
            if frontier == ((0, 0),):
                continue
            joined = {}
            for a, x in states.items():
                for b, y in frontier:
                    mask = a | b
                    joined[mask] = min(joined.get(mask, inf), x + y)
            states = joined
        return len(forced_losses) + states.get((1 << len(obligations)) - 1, inf)

    def conflict(self, phase, cap, deadline):
        """Return pins sufficient to refute the cap, including when minimization is interrupted."""
        confirmed = self.memo(phase)
        if confirmed is None or confirmed <= cap:
            raise ValueError("Cannot exclude a phase without a cost above the requested cap.")
        mask = self.full_mask
        for tile in reversed(range(mask.bit_length())):
            if monotonic() >= deadline:
                break
            trial = mask ^ (1 << tile)
            # For a clause, use the exact obligation table. Every completion
            # can only remove already-hit groups; its exact table still fits.
            bound = self.partial(phase, trial, allow_broad=False)
            if bound is not None and bound > cap:
                mask = trial
        return self._one_per_required_component(phase, mask)

    def _one_per_required_component(self, phase, mask):
        # Required pairs fix a signed phase difference. A single retained pin
        # thus determines the rest of its component in any feasible completion.
        parent = list(range(len(self.sides) // 3))

        def root(tile):
            while parent[tile] != tile:
                tile = parent[tile]
            return tile

        for a, b in self.required:
            x, y = a // 3, b // 3
            slots, start = self.sides[a]
            parity = start ^ self.sides[b][1] ^ (len(slots) & 1)
            if ((phase >> x) ^ (phase >> y)) & 1 != parity:
                # The seed itself contradicts this required pair.
                return (1 << x) | (1 << y)
            parent[root(x)] = root(y)
        reduced = 0
        seen = set()
        for tile in range(mask.bit_length()):
            if mask & (1 << tile):
                representative = root(tile)
                if representative not in seen:
                    reduced |= 1 << tile
                    seen.add(representative)
        return reduced
