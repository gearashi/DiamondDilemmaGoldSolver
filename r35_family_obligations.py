"""Bound broad overlapping departure systems by branching on unhit groups."""
from functools import lru_cache
from math import inf


def _disjoint_in_order(groups):
    occupied = count = 0
    for group in groups:
        if not group & occupied:
            occupied |= group
            count += 1
    return count


def family_bound(classes, allowed, remaining, obligations, limit, *, node_limit=4096, work=None):
    # Each profile records exactly which compatible named bonds this matching
    # loses. Larger classes retain their independent voluntary-loss relaxation.
    from r35_family_frontiers import exact_frontier
    from r35_mate_price import _matching_prices
    positions = {e: 1 << i for i, e in enumerate(sorted(remaining))}
    loose = sum(positions.values())
    profiles = []
    for (family, _), choices in zip(classes, allowed):
        if len(family) > limit:
            continue
        named = remaining & choices
        mask = sum(positions[e] for e in named)
        options = tuple(mask for mask, _ in exact_frontier(_matching_prices, family, choices, named, positions))
        if not options:
            return inf
        loose &= ~mask
        if mask:
            profiles.append((mask, options))
    original_groups = {sum(positions[e] for e in g) for g in obligations}
    physical_groups = set()
    for owner, options in profiles:
        if owner.bit_count() > 8 or not all(options):
            continue
        sub = owner
        candidates = []
        while sub:
            if all(sub & opt for opt in options):
                candidates.append(sub)
            sub = (sub - 1) & owner
        small = []
        for sub in sorted(candidates, key=int.bit_count):
            if not any(old & sub == old for old in small):
                small.append(sub)
        physical_groups.update(small)
    groups = []
    for g in sorted(original_groups | physical_groups, key=lambda mask: (mask.bit_count(), mask)):
        if not any(old & g == old for old in groups):
            groups.append(g)
    if work is not None:
        work.update(projected_groups=len(physical_groups), input_groups=len(original_groups))
    visits = 0

    @lru_cache(maxsize=1024)
    def disjoint_lower_bound(pending):
        # A low-degree group leaves more disjoint groups available. Every
        # selected group is certified to need a separately charged bond.
        neighbors = {g: frozenset(v for v in pending if v & g) for g in pending}
        remaining = set(pending)
        count = 0
        while remaining:
            group = min(remaining, key=lambda g: (len(neighbors[g] & remaining), g.bit_count(), g))
            remaining.difference_update(neighbors[group])
            count += 1
        return count

    def propagate(selected, cap):
        while True:
            cost = (selected & loose).bit_count()
            factors = []
            for owner, options in profiles:
                must = selected & owner
                domain = [(v, v.bit_count()) for v in options if v & must == must]
                if not domain:
                    return inf, 0, ()
                cheapest = min(n for _, n in domain)
                cost += cheapest
                factors.append((owner, domain, cheapest))
            if cost > cap:
                return inf, 0, ()
            slack = cap - cost
            forced = selected
            possible = loose if slack else selected & loose
            for owner, domain, cheapest in factors:
                common = owner
                for v, n in domain:
                    if n <= cheapest + slack:
                        common &= v
                        possible |= v
                forced |= common
            pending = [g & possible for g in groups if not g & forced]
            if not all(pending):
                return inf, 0, ()
            must = 0
            for g in pending:
                if g & (g - 1) == 0:
                    must |= g
            if not must:
                return cost, forced, sorted(set(pending), key=lambda g: (g.bit_count(), g))
            selected = forced | must

    def under(selected, cap, seen):
        nonlocal visits
        visits += 1
        if visits > node_limit:
            return None
        price, forced, pending = propagate(selected, cap)
        if price > cap or forced in seen:
            return False
        seen.add(forced)
        if not pending:
            return True
        # Every set in a disjoint subfamily must acquire a separately charged
        # member in addition to the already forced losses.
        spare = cap - forced.bit_count()
        if _disjoint_in_order(pending) > spare:
            return False
        if disjoint_lower_bound(tuple(pending)) > spare:
            return False
        row = pending[0]
        candidates = []
        while row:
            bit = row & -row
            row ^= bit
            candidates.append(bit)
        candidates.sort(key=lambda bit: sum(bool(bit & g) for g in pending), reverse=True)
        for bit in candidates:
            answer = under(forced | bit, cap, seen)
            if answer is not False:
                return answer
        return False

    price, forced, _ = propagate(0, inf)
    if price == inf:
        return inf
    upper = len(positions)
    for cap in range(price, upper + 1):
        answer = under(forced, cap, set())
        if answer is not False:
            # A successful search attains cap. A cutoff only certifies the
            # previous failed caps; returning cap remains a safe lower bound.
            if work is not None:
                work.update(packing_states=disjoint_lower_bound.cache_info().misses,
                            packing_reuses=disjoint_lower_bound.cache_info().hits, nodes=visits,
                            finished=answer is True, outstanding_groups=len(groups), certified=cap)
            return cap
    if work is not None:
        work.update(packing_states=disjoint_lower_bound.cache_info().misses,
                    packing_reuses=disjoint_lower_bound.cache_info().hits, nodes=visits,
                    finished=True, outstanding_groups=len(groups), infeasible=True)
    return inf
