"""Exact rotation elimination, strict dominance, and factor reuse between physical exchanges."""
from collections import OrderedDict
import itertools
import math

import numpy as np


class RotationOracle:
    def __init__(self, model, capacity=4096, *, observe=None, cache_bytes=None):
        self.model = model
        self.capacity = capacity
        self.observe = observe
        if cache_bytes is not None and (type(cache_bytes) is not int or cache_bytes < 1):
            raise ValueError("Positive factor-cache byte budget required")
        self.cache_bytes = cache_bytes
        self.priority = {}
        self.cache_size = 0
        self.age = 0.
        self.cache = OrderedDict()
        self.edge_cache = {}
        self.work = dict(dominated=0, messages=0, avoided=0, hits=0, threshold_removed=0, threshold_rejected=0)

    def _edge(self, left, ls, right, rs):
        key = (left, ls, right, rs)
        cached = self.edge_cache.get(key)
        if cached is None:
            lm = self.model.masks[3*left:3*left+3, ls, None]
            rm = self.model.reverse[self.model.masks[3*right:3*right+3, rs]]
            cached = (lm == rm).astype(np.int16)
            if len(self.edge_cache) >= 16384:
                self.edge_cache.clear()
            self.edge_cache[key] = cached
        return cached

    def reduced(self, values):
        board = self.model.legal_board(values)
        tiles = board // 3
        domains = [set(np.flatnonzero(self.model.legal[c, 3*t:3*t+3]).tolist())
                   for c,t in enumerate(tiles)]
        edges = []
        incident = [[] for _ in domains]
        for a,sa,b,sb in self.model.problem.edges:
            table = self._edge(tiles[a], sa, tiles[b], sb)
            edges.append((a,b,table))
            incident[a].append((b,table))
            incident[b].append((a,table.T))
        pending = set(range(len(domains)))
        while pending:
            c = pending.pop()
            for r in sorted(domains[c]):
                for alt in sorted(domains[c]-{r}):
                    # Goldstein's minimum difference: the *same* alternative wins
                    # for every joint neighbor choice, even when choices correlate.
                    difference = sum(min(int(tab[alt,t])-int(tab[r,t]) for t in domains[n])
                                     for n,tab in incident[c])
                    if difference > 0:
                        domains[c].remove(r)
                        self.work["dominated"] += 1
                        pending.update(n for n,_ in incident[c])
                        break
        return tiles, [tuple(sorted(d)) for d in domains], edges

    def threshold_reduce(self, tiles, domains, edges, floor):
        """An unsupported rotation cannot attain the requested seam threshold.

        Fixing one rotation affects only the center and at most three neighbors.
        A second sweep uses removals from the previous one. Group same-neighbor
        seams when evaluating the upper bound (separate support is safe).
        """
        incident = [[] for _ in domains]
        for a,b,table in edges:
            incident[a].append((b,table))
            incident[b].append((a,table.T))
        masks = [sum(1<<r for r in d) for d in domains]
        lookup = {}
        tables = [[] for _ in domains]
        for c,neighbors in enumerate(incident):
            for nb,table in neighbors:
                key = table.tobytes()
                if key not in lookup:
                    lookup[key] = [[int(any(table[r,t] for t in range(3) if m>>t&1)) for r in range(3)]
                                   for m in range(8)]
                tables[c].append((nb,lookup[key]))
        def star(c):
            price = [0,0,0]
            for nb,support in tables[c]:
                a,b,d = support[masks[nb]]
                price[0] += a
                price[1] += b
                price[2] += d
            return max(price[r] for r in range(3) if masks[c]>>r&1)
        stars = [star(c) for c in range(len(masks))]
        total = sum(stars)
        if total < 2*floor:
            return None
        changed = True
        while changed:
            changed = False
            for c,mask in enumerate(masks):
                if mask.bit_count()<2:
                    continue
                touched = sorted({c,*(nb for nb,_ in tables[c])})
                without = total-sum(stars[i] for i in touched)
                keep = mask
                for r in range(3):
                    if mask>>r&1:
                        masks[c] = 1<<r
                        if without+sum(star(i) for i in touched) < 2*floor:
                            keep ^= 1<<r
                masks[c] = keep
                if not keep:
                    return None
                if keep!=mask:
                    changed = True
                    self.work["threshold_removed"] += mask.bit_count()-keep.bit_count()
                    for i in touched:
                        stars[i] = star(i)
                    total = without+sum(stars[i] for i in touched)
                    if total < 2*floor:
                        return None
        return [tuple(r for r in range(3) if mask>>r&1) for mask in masks]

    def optimum(self, values, *, max_entries=59049, ties=False, limit=1):
        """All tied optima up to limit; None means the next factor exceeds budget."""
        tiles, domains, edges = self.reduced(values)
        # Start at an admissible upper bound. Reject thresholds without expanding
        # the entire large rotation graph. Strict dominance preserves all optima.
        local = [[] for _ in domains]
        for a,b,table in edges:
            local[a].append((b,table))
            local[b].append((a,table.T))
        upper = sum(max(sum(any(tab[r,t] for t in domains[b]) for b,tab in local[c]) for r in ds)
                    for c,ds in enumerate(domains))//2
        lower = int(self.model.matches(np.asarray(values)).sum())
        for target in range(upper,lower-1,-1):
            restricted = self.threshold_reduce(tiles,domains,edges,target)
            if restricted is None:
                self.work["threshold_rejected"] += 1
                continue
            result = self._eliminate(tiles,restricted,edges,max_entries,ties,limit)
            if result is None:
                # Fall back to an unrestricted exact optimizer at the caller.
                return None
            if result[0]>=target:
                return result
            self.work["threshold_rejected"] += 1
        raise RuntimeError("Rotation threshold reduction lost its feasible input.")

    def _eliminate(self, tiles, domains, edges, max_entries, ties, limit):
        n = len(tiles)
        widths = [len(d) for d in domains]
        unary = [np.zeros(w,dtype=np.int32) for w in widths]
        factors = []
        for a,b,raw in edges:
            grid = raw[np.ix_(domains[a], domains[b])].astype(np.int32)
            # Extract additive row and column contributions. A zero residual severs
            # the dependency, including cases with a single remaining rotation.
            unary[a] += grid[:,0]
            unary[b] += grid[0,:]-grid[0,0]
            remainder = grid-grid[:,0,None]-grid[0,:]+grid[0,0]
            if np.any(remainder):
                factors.append(((a,b),remainder) if a<b else ((b,a),remainder.T))
        factors.extend(((c,),u) for c,u in enumerate(unary))
        trace = []
        active = set(range(n))
        score = 0
        while active:
            adj = {c:{c} for c in active}
            for scope,_ in factors:
                for c in scope:
                    adj[c].update(scope)
            cell = min(active, key=lambda c:(math.prod(widths[v] for v in adj[c]), c))
            scope = tuple(sorted(adj[cell]))
            product = math.prod(widths[v] for v in scope)
            if product > max_entries:
                return None
            local = [(s,v) for s,v in factors if cell in s]
            factors = [(s,v) for s,v in factors if cell not in s]
            axis = scope.index(cell)
            # Relative axes and exact factor bytes capture the entire message
            # input; unrelated assignments can safely reuse this table.
            key = (axis,tuple(widths[c] for c in scope),
                   tuple((tuple(scope.index(v) for v in s),f.tobytes()) for s,f in local))
            entry = self.cache.get(key)
            if entry is None:
                grid = np.zeros(tuple(widths[v] for v in scope),dtype=np.int32)
                for s,f in local:
                    grid += f.reshape([widths[v] if v in s else 1 for v in scope])
                best = grid.max(axis=axis)
                rows = np.moveaxis(grid,axis,-1)
                alternatives = ((rows == best[...,None]) * (1<<np.arange(widths[cell]))).sum(axis=-1)
                entry = best,alternatives
                self.work["messages"] += product
                if self.cache_bytes is not None:
                    # Match the trace metric: serialized operand representation
                    # and numeric answers. Python object headers are additional.
                    size = len(repr(key).encode("ascii")) + best.nbytes + alternatives.nbytes
                    if size <= self.cache_bytes:
                        while self.cache_size + size > self.cache_bytes:
                            victim = min(self.priority, key=lambda k: self.priority[k][0])
                            self.age, removed = self.priority.pop(victim)
                            self.cache_size -= removed
                            del self.cache[victim]
                        self.cache[key] = entry
                        self.priority[key] = self.age + product/size, size
                        self.cache_size += size
                elif self.capacity:
                    self.cache[key] = entry
                    if len(self.cache)>self.capacity:
                        self.cache.popitem(last=False)
            else:
                self.cache.move_to_end(key)
                if self.cache_bytes is not None:
                    size = self.priority[key][1]
                    self.priority[key] = self.age + product/size, size
                self.work["avoided"] += product
                self.work["hits"] += 1
            best,alternatives = entry
            if self.observe is not None:
                self.observe(key, best.nbytes + alternatives.nbytes, product)
            rest = tuple(v for v in scope if v!=cell)
            trace.append((cell,rest,alternatives))
            if rest:
                factors.append((rest,best))
            else:
                score += int(best)
            active.remove(cell)
        def reconstruct(index):
            if index < 0:
                yield (3*tiles+np.array([domains[c][index_by_cell[c]] for c in range(n)])).tolist()
                return
            cell,rest,alternatives = trace[index]
            mask = int(alternatives[tuple(index_by_cell[v] for v in rest)])
            for r in range(widths[cell]):
                if mask & (1<<r):
                    index_by_cell[cell] = r
                    yield from reconstruct(index-1)
                    if not ties:
                        return
        index_by_cell = [0]*n
        return score, itertools.islice(reconstruct(n-1), limit)
