"""Disjoint subproblems for nonadjacent donor cycles, including legal in-place rotations."""
import numpy as np


class CycleBranches:
    def __init__(self, model, cells, board, grid, prior, policy):
        if policy not in ('disjoint','neighbors','warm'):
            raise ValueError('Unknown nonadjacent circulation branching policy.')
        self.policy=policy
        self.cells=cells
        n=len(cells)
        diagonal=grid[np.arange(n),np.arange(n)].copy()
        diagonal[np.arange(n),board[cells]%3]=-10
        self.rotation=diagonal.argmax(axis=1)
        self.diagonal_legal=diagonal.max(axis=1)>=0
        self.diagonal_gain=diagonal.max(axis=1).astype(int)-prior
        slots={int(c):i for i,c in enumerate(cells)}
        self.neighbors=[frozenset(slots[int(v)] for v in model.problem.neighbors[c] if v in slots)
                        for c in cells]

    def require_changes(self, forced, allowed, weights, rotation):
        if not forced:
            return rotation
        forced=np.asarray(forced,dtype=int)
        rotation=rotation.copy()
        allowed[forced,forced]=self.diagonal_legal[forced]
        weights[forced,forced]=self.diagonal_gain[forced]
        rotation[forced,forced]=self.rotation[forced]
        return rotation

    def children(self, conflict, pinned, forced):
        # For a physical conflict (i,j): i stays, or i changes and j stays.
        first,other=conflict
        siblings=[(set(pinned)|{first},set(forced)),(set(pinned)|{other},set(forced)|{first})]
        for stay,change in siblings:
            if self.policy in ('neighbors','warm'):
                for active in change:
                    stay.update(self.neighbors[active])
            if not stay & change:
                yield tuple(sorted(stay)),tuple(sorted(change))
