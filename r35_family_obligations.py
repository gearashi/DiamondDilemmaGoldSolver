"""Bound broad overlapping departure systems by branching on unhit groups."""
from math import inf


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
    groups = sorted({sum(positions[e] for e in g) for g in obligations}, key=int.bit_count)
    visits = 0

    def propagate(selected):
        forced = selected
        cost = (selected & loose).bit_count()
        for owner, options in profiles:
            must = selected & owner
            common = owner
            cheapest = inf
            for losses in options:
                if losses & must != must:
                    continue
                common &= losses
                cheapest = min(cheapest, losses.bit_count())
            if cheapest == inf:
                return inf, 0
            forced |= common
            cost += cheapest
        return cost, forced

    def under(selected, cap, seen):
        nonlocal visits
        visits += 1
        if visits > node_limit:
            return None
        price, forced = propagate(selected)
        if price > cap or forced in seen:
            return False
        seen.add(forced)
        pending = [g for g in groups if not g & forced]
        if not pending:
            return True
        # Every set in a disjoint subfamily must acquire a separately charged
        # member in addition to the already forced losses.
        disjoint = count = 0
        for g in pending:
            if not disjoint & g:
                disjoint |= g
                count += 1
        if forced.bit_count() + count > cap:
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

    price, forced = propagate(0)
    if price == inf:
        return inf
    upper = len(positions)
    for cap in range(price, upper + 1):
        answer = under(forced, cap, set())
        if answer is not False:
            # A successful search attains cap. A cutoff only certifies the
            # previous failed caps; returning cap remains a safe lower bound.
            if work is not None:
                work.update(nodes=visits, finished=answer is True, outstanding_groups=len(groups), certified=cap)
            return cap
    if work is not None:
        work.update(nodes=visits, finished=True, outstanding_groups=len(groups), infeasible=True)
    return inf
