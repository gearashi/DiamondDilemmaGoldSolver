"""Prove whole intervals of colex support ranks using fixed high cells and free low slots."""
from itertools import combinations
from math import comb

from r28_exchange_cuda import unrank
from r30_bounds import can_hit_cores
from r31_route_geometry import connected_mask


class PrefixRanges:
    def __init__(self, free, close, cores, requirements, geometries, families, caps, target):
        self.free = tuple(map(int,free))
        free = self.free
        self.close = close
        self.centers = tuple(zip((tuple(map(int,row)) for row in cores), requirements, geometries, caps))
        self.families = families
        self.target = target
        self.low = [0]
        for c in free:
            self.low.append(self.low[-1] | (1 << c))
        self.contacts = {c: sum(1 << j for j in free if j != c and close[c, j]) for c in free}

    def possible(self, fixed, width, limit):
        """Each wildcard can use any lower cell. A retained completion is only optimistic."""
        low = self.low[limit]
        fixed_mask = sum(1 << c for c in fixed)
        k = width + len(fixed)
        pairs = tuple(combinations(range(k), 2))
        fixed_count = sum(bool(self.close[c, d]) for c, d in combinations(fixed, 2))
        # Partition free cells by their exact contact pattern with the fixed ones.
        # For one remaining slot, distinct bits give exact geometry without visiting each cell.
        if width == 1:
            patterns = [(low, 0)]
            for i, c in enumerate(fixed):
                neighbors = self.contacts[c]
                patterns = [(part, mask) for group, old in patterns
                            for part, mask in ((group & neighbors, old | (1 << i)),
                                               (group & ~neighbors, old)) if part]
        else:
            bits = sum(1 << i for i, c in enumerate(fixed) if self.contacts[c] & low)
            patterns = [(low, bits)]

        for allowed, pattern in patterns:
            contacts = fixed_count + width * pattern.bit_count() + comb(width, 2)
            edges = None
            for cores, required, geometry, cap in self.centers:
                if cap < self.target or contacts < required:
                    continue
                if cores and not can_hit_cores(cores, fixed_mask, allowed, width):
                    continue
                if any(f not in geometry for f in self.families):
                    return True
                if edges is None:
                    def adjacent(i, j):
                        if j < width:
                            return True
                        if i < width:
                            return bool(pattern & (1 << (j-width)))
                        return self.close[fixed[i-width], fixed[j-width]]
                    edges = tuple(e for e,(i,j) in enumerate(pairs) if adjacent(i,j))
                if any(connected_mask(f, edges) for f in self.families):
                    return True
        return False

    def spans(self, k, start, stop, work):
        """Yield open rank intervals, including slices starting inside colex blocks."""
        current = start
        active = None
        while current < stop:
            indices = unrank(current, len(self.free), k)
            end = current + 1
            rejected = False
            # Leave at least one fixed cell. Large blocks are cheap when fully excluded,
            # while testing three free slots at most keeps partial-interval work bounded.
            for width in range(min(k-1, 3), 0, -1):
                limit = indices[width]
                block = comb(limit, width)
                inner = sum(comb(x, i+1) for i,x in enumerate(indices[:width]))
                boundary = min(stop, current + block - inner)
                if boundary <= current + 1:
                    continue
                fixed = tuple(self.free[j] for j in indices[width:])
                work['interval_probes'] += 1
                if not self.possible(fixed, width, limit):
                    end = boundary
                    rejected = True
                    work['supports_before_generation'] += end-current
                    break
                if width == 1:
                    end = boundary
            if rejected:
                if active is not None:
                    yield active, current
                    active = None
            elif active is None:
                active = current
            current = end
        if active is not None:
            yield active, stop
