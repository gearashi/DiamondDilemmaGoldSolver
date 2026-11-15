"""Exact cost of disjoint reference-bond loss groups with a small shared phase image."""

from math import inf
import time


class GroupImages:
    """Enumerate attainable mismatch vectors after eliminating outside phase equations."""

    @classmethod
    def compile(cls, groups, required, equations, outside_count, max_rank=14):
        phase_columns = {}
        symbols = [0] * (outside_count + 1)
        group_masks = []
        separators = 0
        required_mask = 0
        position = 1
        for group in groups:
            group_mask = 0
            for bond in sorted(group):
                row, symbol = equations[bond]
                group_mask |= position
                if bond in required:
                    required_mask |= position
                free = row >> 1
                while free:
                    bit = free & -free
                    phase_columns[bit] = phase_columns.get(bit, 0) | position
                    free ^= bit
                while symbol:
                    bit = symbol & -symbol
                    symbols[bit.bit_length() - 1] |= position
                    symbol ^= bit
                position <<= 1
            group_masks.append(group_mask)
            separators |= position
            position <<= 1

        pivots = {}
        for column in phase_columns.values():
            while column:
                pivot = column.bit_length() - 1
                if pivot not in pivots:
                    pivots[pivot] = column
                    if len(pivots) > max_rank:
                        return None
                    break
                column ^= pivots[pivot]

        # Reducing every RHS column makes equivalent outside patterns exact
        # cache hits. Reduction modulo the common phase image is linear.
        for pivot, vector in sorted(pivots.items(), reverse=True):
            for i, symbol in enumerate(symbols):
                if symbol >> pivot & 1:
                    symbols[i] = symbol ^ vector
        images = [0]
        for vector in pivots.values():
            images.extend([image ^ vector for image in images])
        return cls(tuple(group_masks), separators, required_mask, tuple(symbols), tuple(images))

    def __init__(self, groups, separators, required, symbols, images):
        self.group_count = len(groups)
        self.separators = separators
        self.bond_bits = sum(groups)
        self.required = required
        self.symbols = symbols
        self.images = images
        self.impossible = any(not (group & ~required) for group in groups)
        self.minima = {}
        self.cache_hits = 0
        self.images_priced = 0

    def minimum(self, flipped, deadline):
        """Return the minimum grouped cost; outside-error cardinality is charged by the caller."""
        if time.monotonic() >= deadline:
            raise TimeoutError
        if self.impossible:
            return inf
        key = self.symbols[0]
        for index in flipped:
            key ^= self.symbols[index + 1]
        if key in self.minima:
            self.cache_hits += 1
            return self.minima[key]

        best = inf
        for ordinal, image in enumerate(self.images):
            if ordinal % 16 == 0 and time.monotonic() >= deadline:
                raise TimeoutError
            mismatches = key ^ image
            self.images_priced += 1
            if mismatches & self.required:
                continue
            # Count each incompatible bond once and pay one loss in every
            # conflict group with zero phase mismatches.
            # Each group has a zero padding bit above its bond field. Adding
            # all the field bits sets that padding bit exactly when a bond in
            # the group mismatches; the carry cannot reach the next group.
            nonempty = ((mismatches + self.bond_bits) & self.separators).bit_count()
            cost = mismatches.bit_count() + self.group_count - nonempty
            best = min(best, cost)
            if best == self.group_count:
                break
        if len(self.minima) >= 16384:
            self.minima.clear()
        self.minima[key] = best
        return best
