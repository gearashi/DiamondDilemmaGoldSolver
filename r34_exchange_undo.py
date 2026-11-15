"""Discard a five-swap endpoint only when a checked predecessor dominates it."""
from functools import lru_cache
from itertools import combinations, product

import numpy as np


class CoveredUndo:
    def __init__(self, bound, covered):
        self.bound = bound
        self.covered = covered
        self.gain = lru_cache(maxsize=8192)(self._gain)
        self.region_gain = lru_cache(maxsize=4096)(self._region_gain)
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

    def _region_gain(self, cells, donors, changed_ports=()):
        b=self.bound
        outside=sorted(set().union(*(self.near[c] for c in cells))-set(cells))
        # Bound the conditioning table. A skipped component must stay open.
        if len(outside)+len(cells)>12:
            return None
        exterior=np.array(list(product(range(3),repeat=len(outside))),dtype=int)
        other=dict(changed_ports)
        ports={c:(3*b.tiles[other.get(c,c)]+exterior[:,i,None]) for i,c in enumerate(outside)}
        edges=[(a,s,n,t) for a,s,n,t in b.model.problem.edges if a in cells or n in cells]
        best=[]
        for sources in (cells,donors):
            domains=[[3*b.tiles[d]+r for r in range(3) if b.model.legal[c,3*b.tiles[d]+r]]
                     for c,d in zip(cells,sources)]
            if not all(domains):
                return None
            rows=np.array(list(product(*domains)))
            placed=ports | {c:rows[:,i] for i,c in enumerate(cells)}
            score=np.zeros((len(exterior),len(rows)),dtype=np.int8)
            for a,s,n,t in edges:
                score += b.model.masks[placed[a],s] == b.model.reverse[b.model.masks[placed[n],t]]
            best.append(score.max(axis=1))
        return int((best[1]-best[0]).max())

    def _group_gain(self,moves,exterior):
        left=set(moves)
        total=0
        while left:
            group={min(left)}
            left-=group
            while True:
                more=set().union(*(self.near[c] for c in group))&left
                if not more:
                    break
                group.update(more)
                left-=more
            cells=tuple(sorted(group))
            donors=tuple(moves[c] for c in cells)
            boundary=set().union(*(self.near[c] for c in cells))-group
            ports=tuple((c,exterior[c]) for c in sorted(boundary & exterior.keys()))
            if len(cells)==1 and not ports:
                gain=self.gain(cells[0],donors[0])
            else:
                gain=self.region_gain(cells,donors,ports)
            if gain is None:
                return None
            total+=gain
        return total

    def _cycle_gain(self,cycle):
        return self._group_gain(dict(zip(cycle,(*cycle[1:],cycle[0]))),{})

    def rejects(self, cycles):
        # The complement must name a complete, checked predecessor. Fix its
        # actual physical donors on exterior seams common to both arrangements.
        moves=[dict(zip(cycle,(*cycle[1:],cycle[0]))) for cycle in cycles]
        for size in range(1,len(cycles)):
            for selected in combinations(range(len(cycles)),size):
                remaining=tuple(row for i,row in enumerate(cycles) if i not in selected)
                if not self.covered(remaining):
                    continue
                removed={c:d for i in selected for c,d in moves[i].items()}
                exterior={c:d for i,move in enumerate(moves) if i not in selected for c,d in move.items()}
                gain=self._group_gain(removed,exterior)
                if gain is not None and gain<=0:
                    return True
        return False
