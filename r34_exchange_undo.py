"""Discard a five-swap endpoint only when a checked predecessor dominates it."""
from functools import lru_cache
from itertools import product

import numpy as np


class CoveredUndo:
    def __init__(self, bound, covered):
        self.bound = bound
        self.covered = covered
        self.gain = lru_cache(maxsize=8192)(self._gain)
        self.near = [set(map(int,row))-{bound.n} for row in bound.neighbors[:bound.n]]

    def _gain(self, cell, donor):
        """Largest incident-score gain with the same exterior rotations before and after."""
        bound = self.bound
        options = []
        for source in (cell, donor):
            tile = bound.tiles[source]
            options.append([3*tile+r for r in range(3) if bound.model.legal[cell,3*tile+r]])
        if not options[1]:
            # The endpoint itself is illegal. Leave its accounting to the ordinary filter.
            return 3
        neighbors = [(s, n, bound.model.problem.sides[cell,s])
                     for s,n in enumerate(bound.model.problem.neighbors[cell]) if n >= 0]
        rotations = list(product(range(3), repeat=len(neighbors)))
        values = []
        for codes in options:
            scores = np.zeros((len(rotations),len(codes)), dtype=np.int8)
            for i,(s,n,t) in enumerate(neighbors):
                other = 3*bound.tiles[n] + np.array([row[i] for row in rotations])
                scores += bound.model.masks[codes,s] == bound.model.reverse[bound.model.masks[other,t,None]]
            values.append(scores.max(axis=1))
        return int((values[1]-values[0]).max())

    def rejects(self, cycles):
        if len(cycles) < 2:
            return False
        moved = {c for cycle in cycles for c in cycle}
        for cycle in cycles:
            if any(self.near[c] & moved for c in cycle):
                continue
            donors = (*cycle[1:],cycle[0])
            if sum(self.gain(c,d) for c,d in zip(cycle,donors)) > 0:
                continue
            # With the other donors and all exterior rotations fixed, independently
            # undoing these nonadjacent cells cannot lower the score. Require
            # a complete negative for the remaining physical permutation.
            remaining = tuple(row for row in cycles if row != cycle)
            if self.covered(remaining):
                return True
        return False
