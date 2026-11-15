"""Exact raw signed-inventory census for a partially pinned physical phase."""
from collections import Counter
from functools import reduce
from math import gcd


def gap_profiles(tiles):
    """Paint the twelve open intervals on each physical side from the raw endpoints."""
    sides = []
    for tile in tiles:
        endpoints = {point for segment in tile["segments"] for point in map(tuple, segment)}
        bit = 0
        for side in range(3):
            pattern = 0
            for interval in range(12):
                if (side, interval) in endpoints:
                    bit ^= 1
                pattern |= bit << interval
            sides.append(pattern)
        if bit:
            raise ValueError("Odd endpoint count on tile.")
    return tuple(sides)


def reverse_side(pattern):
    return int(f"{pattern:012b}"[::-1], 2)


def count_equations(tiles):
    raw = gap_profiles(tiles)
    full = (1 << 12) - 1
    keys = {min(mask, reverse_side(mask)) for side in raw for mask in (side, side ^ full)}
    result = []
    for k in sorted(keys):
        opposite = reverse_side(k)
        terms = []
        constant = 0
        for t in range(len(tiles)):
            masks = raw[3*t:3*t+3]
            low = sum(mask == k for mask in masks)
            high = sum((mask ^ full) == k for mask in masks)
            if k != opposite:
                low -= sum(mask == opposite for mask in masks)
                high -= sum((mask ^ full) == opposite for mask in masks)
            constant += low
            if high != low:
                terms.append((t,high-low))
        result.append((tuple(terms), -constant, k == opposite))
    return tuple(result)

def census(terms, n, pins, budget=10000):
    """Enumerate exact inventory phases; budget exhaustion is explicitly incomplete."""
    if type(n) is not int or n < 1 or type(budget) is not int or budget < 0:
        raise ValueError("Invalid problem size or node budget.")
    if any(type(t) is not int or not 0 <= t < n or type(v) is not int or v not in (0,1)
           for t,v in pins.items()):
        raise ValueError("Invalid tile phase pin.")
    work = Counter()
    solutions = []

    def propagate(values):
        while True:
            before = len(values)
            equations = []
            for row, target, parity in terms:
                rest = []
                for tile, coeff in row:
                    if tile in values:
                        target -= coeff * values[tile]
                    else:
                        rest.append((tile, coeff))
                if parity:
                    mask = sum(1 << t for t,c in rest if c % 2)
                    equations.append((mask, target % 2))
                    continue
                low = sum(min(0,c) for _,c in rest)
                high = sum(max(0,c) for _,c in rest)
                if not low <= target <= high:
                    return False
                if not rest:
                    continue
                div = reduce(gcd, (abs(c) for _,c in rest))
                if target % div:
                    return False
                mask = sum(1 << t for t,c in rest if (c // div) % 2)
                equations.append((mask,(target // div) % 2))
                for t,c in rest:
                    lo, hi = low - min(0,c), high - max(0,c)
                    allow0 = lo <= target <= hi
                    allow1 = lo <= target - c <= hi
                    if not (allow0 or allow1):
                        return False
                    if allow0 != allow1:
                        values[t] = int(allow1)
            pivots = {}
            for mask, value in equations:
                for i in sorted(pivots):
                    if mask >> i & 1:
                        row, bit = pivots[i]
                        mask ^= row
                        value ^= bit
                if not mask:
                    if value:
                        return False
                    continue
                pivot = (mask & -mask).bit_length() - 1
                pivots[pivot] = mask, value
                for i in pivots:
                    row, bit = pivots[i]
                    if i != pivot and row >> pivot & 1:
                        pivots[i] = row ^ mask, bit ^ value
            for t,(mask,bit) in pivots.items():
                if mask == 1 << t:
                    if t in values and values[t] != bit:
                        return False
                    values[t] = bit
            if len(values) == before:
                return True

    def visit(values):
        if work["nodes"] >= budget:
            work["limited"] += 1
            return
        work["nodes"] += 1
        if not propagate(values):
            work["empty_leaves"] += 1
            return
        if len(values) == n:
            solutions.append(sum(v << t for t,v in values.items()))
            return
        degree = Counter()
        for row,_,_ in terms:
            available = [t for t,c in row if t not in values]
            if available:
                weight = 256 // len(available)
                for t in available:
                    degree[t] += weight
        tile = max((t for t in range(n) if t not in values), key=lambda t:(degree[t],-t))
        visit(values | {tile:0})
        visit(values | {tile:1})

    visit(dict(pins))
    return dict(complete=not work["limited"], phases=solutions, work=dict(work))
