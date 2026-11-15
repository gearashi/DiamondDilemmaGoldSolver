"""Exact rotation elimination, strict dominance, and factor reuse between physical exchanges."""
from collections import OrderedDict
import itertools
import sys

from r35_rotation_graph import RotationThreshold, elimination_plan

import numpy as np


class RotationOracle:
    def __init__(self, model, capacity=4096, *, observe=None, cache_bytes=None, components=True, poll=None):
        self.model = model
        self.capacity = capacity
        self.observe = observe
        self.poll=poll
        if cache_bytes is not None and (type(cache_bytes) is not int or cache_bytes < 1):
            raise ValueError("Positive factor-cache byte budget required")
        self.cache_bytes = cache_bytes
        self.component_cache=None
        self.factor_budget=cache_bytes
        if components and observe is None and (capacity or cache_bytes):
            from r35_rotation_components import ComponentCache
            limit=min(512*1024,128*capacity) if cache_bytes is None else min(512*1024,cache_bytes//4)
            self.component_cache=ComponentCache(limit)
            if cache_bytes is not None:self.factor_budget-=limit
        self.priority = {}
        self.cache_size = 0
        self.age = 0.
        self.cache = OrderedDict()
        self.edge_cache = {}
        self.rotation_supports = {}
        self.plans = OrderedDict()
        self.plan_bytes = 0
        # Numeric factors take priority within a user-supplied byte budget.
        # A cold oracle retains neither numeric answers nor structural plans.
        self.plan_limit = min(512*1024, cache_bytes) if cache_bytes is not None else (512*1024 if capacity else 0)
        self.work = dict(dominated=0, messages=0, avoided=0, hits=0, threshold_removed=0, threshold_rejected=0,
                         threshold_stars=0, threshold_supports=0, threshold_argmax_checks=0,
                         threshold_remote_events=0, threshold_condition_hits=0,
                         singleton_responses=0, component_keys=0, component_hits=0, component_factors_reused=0, component_frame_hits=0, component_reindex=0, component_vertex_hits=0,
                         component_tuples_encoded=0, component_tuples_decoded=0, numeric_keys=0,
                         plan_hits=0, plan_builds=0, plan_extensions=0, plan_neighbor_products=0)

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
        """Remove rotations that cannot attain the requested score."""
        return RotationThreshold(domains, edges, self.rotation_supports, self.work).reduce(floor)

    def optimum(self, values, *, max_entries=59049, ties=False, limit=1):
        """All tied optima up to limit; None means the next factor exceeds budget."""
        tiles, domains, edges = self.reduced(values)
        # Start at an admissible upper bound. Reject thresholds without expanding
        # the entire large rotation graph. Strict dominance preserves all optima.
        threshold = RotationThreshold(domains, edges, self.rotation_supports, self.work)
        upper = threshold.total // 2
        lower = int(self.model.matches(np.asarray(values)).sum())
        for target in range(upper,lower-1,-1):
            restricted = threshold.reduce(target)
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

    def _trim_plans(self, budget):
        while self.plans and self.plan_bytes > budget:
            _, (_, _, removed) = self.plans.popitem(last=False)
            self.plan_bytes -= removed

    def _plan(self, widths, pairs, cap):
        key = tuple(widths), tuple(sorted(set(pairs)))
        prefix, residual = (), None
        found = self.plans.get(key)
        if found is not None:
            prefix,residual,size = found
            if residual is None or prefix[-1][2] > cap:
                self.work['plan_hits'] += 1
                self.plans.move_to_end(key)
                return prefix
            self.work['plan_extensions'] += 1
            self.plan_bytes -= size
            del self.plans[key]
        else:
            self.work['plan_builds'] += 1
        steps,remainder = elimination_plan(*key,self.work,cap,prefix,residual,retain=bool(self.plan_limit))
        if self.plan_limit:
            # Include record and scope containers, integer products and the
            # suspended structural state. Physical cell integers are shared.
            size = (sys.getsizeof(key) + sum(sys.getsizeof(part) for part in key)
                    + sum(sys.getsizeof(pair) for pair in key[1]) + sys.getsizeof(steps)
                    + sum(sys.getsizeof(step) + sys.getsizeof(step[1]) + sys.getsizeof(step[2]) for step in steps))
            if remainder is not None:
                size += sys.getsizeof(remainder) + sum(sys.getsizeof(row) + sys.getsizeof(row[1])
                                                       + sys.getsizeof(row[2]) for row in remainder)
            budget = self.plan_limit if self.cache_bytes is None else min(self.plan_limit, self.factor_budget-self.cache_size)
            if size <= budget:
                self._trim_plans(budget-size)
                self.plans[key] = steps,remainder,size
                self.plan_bytes += size
        return steps

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
        plan = self._plan(widths, [scope for scope,_ in factors], max_entries)
        if (self.component_cache is not None and self.observe is None and len(plan)==n
                and all(product<=max_entries for _,_,product in plan)):
            from r35_rotation_components import solve_components
            score,trace=solve_components(self,widths,unary,factors,plan,max_entries)
        else:
            factors.extend(((c,),u) for c,u in enumerate(unary))
            solved=self._factor_trace(widths,factors,plan,max_entries)
            if solved is None:return None
            score,trace=solved
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

    def _factor_trace(self,widths,factors,plan,max_entries):
        trace = []
        score = 0
        for cell,scope,product in plan:
            if self.poll is not None:self.poll()
            if product > max_entries:
                return None
            local = [(s,v) for s,v in factors if cell in s]
            factors = [(s,v) for s,v in factors if cell not in s]
            axis = scope.index(cell)
            # Relative axes and exact factor bytes capture the entire message
            # input; unrelated assignments can safely reuse this table.
            key = (axis,tuple(widths[c] for c in scope),
                   tuple((tuple(scope.index(v) for v in s),f.tobytes()) for s,f in local))
            self.work['numeric_keys']+=1
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
                    if size <= self.factor_budget:
                        self._trim_plans(self.factor_budget-self.cache_size-size)
                        while self.cache_size + size > self.factor_budget:
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
        return score,trace
