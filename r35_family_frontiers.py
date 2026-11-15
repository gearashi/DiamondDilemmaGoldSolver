"""Reuse exact matching costs across identical local graphs and loss obligations."""
from functools import lru_cache


def _columns(tags):
    # A column identifies the named bonds that satisfy one global obligation.
    # Equal columns are always satisfied together within this family.
    columns = {}
    for row, tag in enumerate(tags):
        while tag:
            bit = tag & -tag
            columns[bit] = columns.get(bit, 0) | (1 << row)
            tag ^= bit
    group = {}
    for bit, column in columns.items():
        group[column] = group.get(column, 0) | bit
    ordered = sorted(group)
    local = tuple(sum(1 << j for j, col in enumerate(ordered) if col & (1 << row))
                  for row in range(len(tags)))
    return local, tuple(group[col] for col in ordered)


def _lift(frontier, columns):
    if all(mask == 1 << i for i, mask in enumerate(columns)):
        return frontier
    rows = []
    for hits, cost in frontier:
        original = 0
        while hits:
            bit = hits & -hits
            original |= columns[bit.bit_length() - 1]
            hits ^= bit
        rows.append((original, cost))
    return tuple(rows)


@lru_cache(maxsize=128)
def _local_exact(solver, size, choices, charges):
    covers = {(a, b): tag for a, b, tag in charges}
    return tuple(solver(range(size), choices, covers, covers).items())


@lru_cache(maxsize=256)
def _physical_exact(solver, family, choices, charges):
    index = {side: i for i, side in enumerate(family)}
    edges = frozenset(tuple(sorted((index[a], index[b]))) for a, b in choices)
    ranked = sorted((*sorted((index[a], index[b])), tag) for a, b, tag in charges)
    tags, columns = _columns([tag for _, _, tag in ranked])
    local = tuple((a, b, tag) for (a, b, _), tag in zip(ranked, tags))
    return _lift(_local_exact(solver, len(family), edges, local), columns)


def _small_table(tags):
    if len(tags) <= 8:
        return True
    active = 0
    for tag in tags:
        active |= tag
    return active.bit_count() <= 8


def exact_frontier(solver, family, choices, named, covers):
    # Up to eight sides have at most 105 perfect matchings. Otherwise limit
    # the possible retained cost-table size to 256 outcome masks per key.
    if len(family)>16 or (len(family)>8 and not _small_table({covers[e] for e in named if covers[e]})):
        return tuple(solver(family, choices, named, covers).items())
    charges = frozenset((a, b, covers[a, b]) for a, b in named)
    return _physical_exact(solver, tuple(family), frozenset(choices), charges)


@lru_cache(maxsize=128)
def _local_relaxed(tags):
    best = {0: 0}
    for tag in tags:
        for old, price in tuple(best.items()):
            result = old | tag
            if result not in best or price + 1 < best[result]:
                best[result] = price + 1
    return tuple(best.items())


@lru_cache(maxsize=128)
def _original_relaxed(tags):
    compressed, columns = _columns(tags)
    return _lift(_local_relaxed(tuple(sorted(compressed))), columns)


def relaxed_frontier(solver, named, covers):
    # A second departure with the same coverage cannot improve its cost.
    tags = tuple(sorted({covers[e] for e in named if covers[e]}))
    if not _small_table(tags):
        return tuple(solver(named, covers).items())
    return _original_relaxed(tags)
