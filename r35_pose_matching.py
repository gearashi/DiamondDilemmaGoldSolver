"""Price oriented reference poses under the one-copy-per-tile constraint."""
from functools import lru_cache

from constraint_connectivity_round26 import assignment_upper_bound



def owner_candidate(domains, max_kept):
    # Every fully open cell can rotate its assigned tile away from its reference
    # pose. Even conditioning one pose adds at most one to this maximum.
    n = len(domains)
    full = (1 << (3*n)) - 1
    starts = full // 7
    count = 0
    for mask in domains:
        if mask == full:
            continue
        a, b, c = mask, mask >> 1, mask >> 2
        if (a | b | c) & ~(a & b | b & c | c & a) & starts:
            count += 1
            if count >= max_kept:
                return True
    return count >= max_kept


def indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask -= bit


def groups(rows):
    by_tile = {}
    for cell, tiles in rows.items():
        for tile in tiles:
            by_tile.setdefault(tile, []).append(cell)
    remaining = set(rows)
    while remaining:
        cell = min(remaining)
        remaining.remove(cell)
        cells, tiles, queue = [cell], set(), [cell]
        while queue:
            for tile in rows[queue.pop()]:
                if tile in tiles:
                    continue
                tiles.add(tile)
                for other in by_tile[tile]:
                    if other in remaining:
                        remaining.remove(other)
                        cells.append(other)
                        queue.append(other)
        yield sorted(cells), sorted(tiles)


def tight_support(domains, codes):
    from constraint_advanced import _complete_matching_supports
    starts = ((1 << (3*len(domains))) - 1) // 7
    rows = [(mask | mask>>1 | mask>>2) & starts for mask in domains]
    supported,_ = _complete_matching_supports(rows, hint=[3*(c//3) for c in codes])
    if supported is None:
        return None
    return tuple(mask & (tiles | tiles<<1 | tiles<<2) for tiles,mask in zip(supported,domains))


# Large matrices are temporary. Only small exact numeric problems and their
# witnesses can be shared across cuts or differently numbered components.
@lru_cache(maxsize=64)
def _small_dual(matrix):
    return assignment_upper_bound(matrix)


def clear_owner_cache():
    _price.cache_clear()
    _small_dual.cache_clear()
    _image_codes.cache_clear()


@lru_cache(maxsize=16)
def _price(domains, codes, cap, charged):
    n = len(domains)
    priced = dict(charged)
    full = (1 << (3 * n)) - 1
    tile_starts = full // 7
    rows = {}
    free = {}
    for c, mask in enumerate(domains):
        if mask == full:
            continue
        starts = (mask | (mask >> 1) | (mask >> 2)) & tile_starts
        rows[c] = tuple(first // 3 for first in indices(starts))
        free[c] = len(rows[c]) - int(c in priced)
    # A row with k distinct zero-price tiles can always extend an injection
    # from k-1 other rows. Remove such rows in reverse reinsertion order.
    active = sorted(rows, key=lambda c: (free[c], c))
    while active and free[active[-1]] >= len(active):
        active.pop()
    rows = {c: rows[c] for c in active}

    # Each row avoiding its exact pose must use a different zero-cost tile.
    # A single donor bank is often already too small, with no weighted solve.
    zero_bits = {c: sum(1 << t for t in row if t != priced.get(c)) for c,row in rows.items()}
    for ordered in (active, [c for c in active if c in priced]):
        union = 0
        for count, c in enumerate(ordered, 1):
            union |= zero_bits[c]
            shortage = count - union.bit_count()
            if shortage > cap:
                return True, shortage, (), 0, 0, shortage

    lower = checks = matrices = 0
    offsets, columns = {}, {}
    for cells, tiles in groups(rows):
        if len(tiles) < len(cells):
            return True, n + 1, (), matrices, checks, 0
        cells = sorted(cells, key=lambda c:(len(rows[c]),c))
        # Align columns with their own reference cell first. Congruent star
        # components then have byte-for-byte equal numeric problems.
        local = set(tiles)
        own = [codes[c]//3 for c in cells if codes[c]//3 in local]
        tiles = own + sorted(local-set(own))
        allowed = {c: set(rows[c]) for c in cells}
        weights = tuple(tuple(int(priced.get(c) != tile) if tile in allowed[c] else None
                              for tile in tiles) for c in cells)
        small = len(cells)*len(tiles) <= 256
        misses = _small_dual.cache_info().misses
        proof = _small_dual(weights) if small else assignment_upper_bound(weights)
        if not small or _small_dual.cache_info().misses > misses:
            checks += proof['dual_checks']
            matrices += 1
        if any(weights[i][j] is None for i,j in enumerate(proof['assignment'])):
            return True, n + 1, (), matrices, checks, 0
        lower += len(cells) - proof['bound']
        if lower > cap:
            return True, lower, (), matrices, checks, 0
        offsets.update((c, price - 1) for c, price in zip(cells, proof['rows']))
        columns.update(zip(tiles, proof['columns']))

    # Fixing one tile spends its column price, even in a row outside the dual.
    # Within a priced row also charge its row price minus its maximum score.
    # Keeping the exact reference pose additionally spends one unit.
    forbidden = {}
    removals = []
    for c, (mask, code) in enumerate(zip(domains, codes)):
        base = lower + offsets.get(c, 0)
        if base not in forbidden:
            forbidden[base] = sum(7 << (3*t) for t, price in columns.items() if base + price > cap)
        remove = mask if base > cap else mask & forbidden[base]
        if base + columns.get(code // 3, 0) + 1 > cap:
            remove |= mask & (1 << code)
        if remove:
            removals.append((c, remove))
    # The geographic caller propagates fresh tile matching support after
    # changed domains. Repeating its matching here would use the same graph.
    return False, lower, tuple(removals), matrices, checks, 0


@lru_cache(maxsize=16)
def _image_codes(image, n):
    if len(image) != n:
        raise ValueError('Reference image must cover every cell.')
    codes = [-1]*n
    for cell,pose in image:
        if (type(cell) is not int or not 0 <= cell < n or codes[cell] != -1 or
                type(pose) is not int or pose <= 0 or pose.bit_count() != 1 or pose.bit_length() > 3*n):
            raise ValueError('Invalid reference cell or pose.')
        codes[cell] = pose.bit_length() - 1
    if len({code//3 for code in codes}) != n:
        raise ValueError('Each reference image must use every tile exactly once.')
    return tuple(codes)


def owner_bounds(domains, images, max_kept):
    narrowed = list(domains)
    stats = dict(over=False, removed=0, weighted_queries=0, dual_checks=0, cache_hits=0, lower=0,
                 repeated_matrices=0, zero_pool_conflicts=0)
    if not all(narrowed):
        stats.update(over=True, reason='empty_domain')
        return narrowed, stats
    if not owner_candidate(narrowed, max_kept):
        return narrowed, stats
    for image in images:
        codes = _image_codes(tuple(image),len(narrowed))
        # Under any tile assignment a charged row may cost one; every other
        # row can choose a free orientation. Conditioning a pose adds at most
        # one. Rejecting a query or removing a pose is impossible below cap.
        charged = tuple((c, code // 3) for c, code in enumerate(codes)
                        if narrowed[c] & (7 << (3 * (code // 3))) == 1 << code)
        if len(charged) < max_kept:
            continue
        before = _price.cache_info().hits
        reused_matrices = _small_dual.cache_info().hits
        over, lower, removals, count, checks, cheap = _price(tuple(narrowed), tuple(codes), max_kept, charged)
        if _price.cache_info().hits > before:
            stats['cache_hits'] += 1
        else:
            stats['weighted_queries'] += count
            stats['dual_checks'] += checks
            stats['repeated_matrices'] += _small_dual.cache_info().hits - reused_matrices
        stats['zero_pool_conflicts'] += int(cheap>0)
        if lower <= len(domains):
            stats['lower'] = max(stats['lower'], lower)
        if over:
            stats.update(over=True, reason='tile_owner_hall' if lower>len(domains) else 'pose_owner_distance')
            return narrowed, stats
        for cell, mask in removals:
            stats['removed'] += (narrowed[cell] & mask).bit_count()
            narrowed[cell] &= ~mask
            if not narrowed[cell]:
                stats.update(over=True, reason='pose_owner_domain')
                return narrowed, stats
    return narrowed, stats
