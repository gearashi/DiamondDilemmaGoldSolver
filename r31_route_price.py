"""Scoped bounds for typed exchanges with free rotations, plus exact exterior elimination."""
from collections import Counter
from itertools import combinations, permutations, product

import numpy as np

from r30_donor_dd import donor_factors
from r31_route_geometry import cycle_mapping


class DefectPrices:
    """Track physically impossible edges and rotation conflicts with the proposed donors."""
    def __init__(self,bound):
        self.b = bound
        self.edges = tuple(bound.model.problem.edges)
        masks = bound.masks[:,0]
        self.fit = (masks[:,None,:,None] == bound.model.reverse[masks[None,:,None,:]]).any((2,3))
        self.incident = [[] for _ in range(bound.n)]
        for e,(a,s,b,t) in enumerate(self.edges):
            self.incident[a].append((e,s,b))
            self.incident[b].append((e,t,a))
        self.original = bound.tiles

    def actor_upper(self, groups, remaining, pool=None):
        """Upper with forced actors and at most remaining further moved destinations."""
        held = cycle_mapping(groups)
        ids = self.original.copy()
        for c,j in held.items():
            ids[c] = self.original[j]
        free = set(self.b.model.free if pool is None else pool) - held.keys()
        deficits = Counter()
        bad = 0
        repairable = 0
        for a,_,b,_ in self.edges:
            if self.fit[ids[a],ids[b]]:
                continue
            bad += 1
            if a in free or b in free:
                repairable += 1
                if a in free:
                    deficits[a] += 1
                if b in free:
                    deficits[b] += 1
        best = min(repairable, sum(sorted(deficits.values(),reverse=True)[:remaining]))
        return len(self.edges) - bad + best

    def endpoint_upper(self,cells,order):
        """Pack rotation-inconsistent incident seams using the actual incoming pieces."""
        ids = self.original.copy()
        ids[list(cells)] = ids[[cells[j] for j in order]]
        broken = {e for e,(a,_,b,_) in enumerate(self.edges) if not self.fit[ids[a],ids[b]]}
        candidates = []
        for c,adj in enumerate(self.incident):
            good = [(e,self.b.support[self.b.masks[ids[c],:,s],ids[b]].astype(bool))
                    for e,s,b in adj if e not in broken]
            for width in range(2,len(good)+1):
                for subset in combinations(good,width):
                    if not np.logical_and.reduce([allowed for e,allowed in subset]).any():
                        candidates.append(tuple(e for e,mask in subset))
        extra = 0
        for ordered in (sorted(candidates,key=lambda s:(len(s),s)),sorted(candidates,key=lambda s:(len(s),s[::-1]))):
            occupied = set()
            count = 0
            for row in ordered:
                if occupied.isdisjoint(row):
                    occupied.update(row)
                    count += 1
            extra = max(extra,count)
        return len(self.edges) - len(broken) - extra


class SharedStarPrice:
    """A per-star factor uses all changed neighboring donors simultaneously."""
    def __init__(self,bound,cells):
        self.base,self.factors = donor_factors(bound,cells)

    def upper(self, orders):
        """Price a class in batches; correlations retained within each shared star."""
        if not orders:
            return -1
        rows = np.asarray(orders)
        total = np.full(len(rows),self.base)
        for axes,table in self.factors:
            total += table[tuple(rows[:,a] for a in axes)]
        return int(total.max())


def six_comparisons(group):
    family = tuple(sorted(map(len,group)))
    found = set()
    for keep in group:
        length = 6 - len(keep)
        if length not in (3,4):
            continue
        other = sorted({c for g in group if g != keep for c in g})
        for chosen in combinations(other,length):
            for tail in permutations(chosen[1:]):
                old = (keep,(chosen[0],*tail))
                found.add(tuple(sorted(cycle_mapping(old).items())))
    return tuple(sorted(found)) if family in ((2,2,3),(2,2,2,2)) else ()


def comparison_relations(bound, group, max_entries=65536, max_comparisons=32):
    """Retain endpoint rotations that beat every selected proved six-cell comparator.

    Exact comparison optimizes changed cells under each assignment to their
    common exterior. Rows compare the same exterior and the same rotations on
    unchanged cells. Domain restrictions in either alternative are observed.
    An unprocessed comparison, or an over-budget join, cannot reject.
    """
    movement = cycle_mapping(group)
    endpoint = bound.tiles.copy()
    for c,j in movement.items():
        endpoint[c] = bound.tiles[j]
    rows = []
    work = Counter()
    comps = sorted(six_comparisons(group),key=lambda m:sum(dict(m).get(c,c)!=j for c,j in movement.items()))
    for pairs in comps[:max_comparisons]:
        other = bound.tiles.copy()
        for c,j in pairs:
            other[c] = bound.tiles[j]
        inside = tuple(map(int,np.flatnonzero(endpoint!=other)))
        if any(not bound.model.legal[c,3*other[c]:3*other[c]+3].any() for c in inside):
            continue
        edges = [(a,s,b,t) for a,s,b,t in bound.model.problem.edges if a in inside or b in inside]
        outside = tuple(sorted({x for a,_,b,_ in edges for x in (a,b)}-set(inside)))
        scope = outside+inside
        if 3**len(scope) > max_entries:
            work['tables_skipped'] += 1
            continue
        orientations = np.asarray(list(product(range(3),repeat=len(scope))),dtype=np.int8)
        scores = []
        for pieces in (endpoint,other):
            score = np.zeros(len(orientations),dtype=np.int16)
            valid = np.ones(len(score),dtype=bool)
            def oriented(c):
                return 3*int(pieces[c])+orientations[:,scope.index(c)].astype(int)
            for a,s,b,t in edges:
                score += bound.model.masks[oriented(a),s] == bound.model.reverse[bound.model.masks[oriented(b),t]]
            for c in scope:
                valid &= bound.model.legal[c,oriented(c)]
            score[~valid] = -1000
            scores.append(score)
        work['tables'] += 1
        width = 3**len(inside)
        winners = scores[0].reshape(-1,width) > scores[1].reshape(-1,width).max(axis=1)[:,None]
        allowed = orientations[winners.ravel()]
        if not len(allowed):
            return False,dict(work)
        if len(allowed) < len(orientations):
            rows.append((scope,allowed))
    # Exact bounded join enforces the same rotation at every shared cell.
    joined = [{}]
    for scope,allowed in sorted(rows,key=lambda r:len(r[1])):
        if len(joined)*len(allowed)>max_entries:
            break
        next_rows = []
        for old in joined:
            for row in allowed:
                if all(c not in old or old[c]==v for c,v in zip(scope,row)):
                    next_rows.append(old | dict(zip(scope,row)))
        joined = next_rows
        if not joined:
            return False,dict(work)
    return True,dict(work)


def eliminate(factors, cells, cap, work):
    factors = list(factors)
    cells = set(cells)
    while cells:
        incident = {c:{c} for c in cells}
        for scope,_ in factors:
            for c in scope:
                if c in cells:
                    incident[c].update(scope)
        c = min(cells,key=lambda j:(len(incident[j]),j))
        scope = tuple(sorted(incident[c]))
        if 3**len(scope) > cap:
            return None
        grid = np.zeros((3,)*len(scope),dtype=np.int32)
        retained = []
        for keys,f in factors:
            if c in keys:
                grid += np.asarray(f).reshape(tuple(3 if x in keys else 1 for x in scope))
            else:
                retained.append((keys,f))
        retained.append((tuple(x for x in scope if x!=c),grid.max(axis=scope.index(c))))
        work['eliminated_variables'] += 1
        work['table_entries'] += grid.size
        factors = retained
        cells.remove(c)
    return factors


class Exterior:
    """Eliminate fixed exterior physical identities once; rotations remain free."""
    def __init__(self,bound,cells,cap=65536):
        self.b = bound
        self.cells = tuple(cells)
        self.cap = cap
        self.work = Counter()
        self.dynamic = []
        moved = set(cells)
        boundary = set()
        factors = []
        for edge in bound.model.problem.edges:
            a,s,b,t = edge
            if a in moved or b in moved:
                self.dynamic.append(edge)
                boundary.update({a,b}-moved)
            else:
                factors.append(self.edge(edge,bound.tiles))
        outside = set(range(bound.n))-moved
        factors.extend(self.domains(outside,bound.tiles))
        self.fixed = eliminate(factors,outside-boundary,cap,self.work)
        self.active = moved | boundary

    def domains(self,cells,tiles):
        for c in cells:
            ok = self.b.model.legal[c,3*tiles[c]:3*tiles[c]+3]
            # One impossible rotation costs more than every physical seam combined.
            yield (c,),np.where(ok,0,-(len(self.b.model.problem.edges)+1))

    def edge(self,edge,tiles):
        a,s,b,t = edge
        m = self.b.model
        grid = m.masks[3*tiles[a]:3*tiles[a]+3,s,None] == m.reverse[m.masks[3*tiles[b]:3*tiles[b]+3,t]]
        return ((a,b),grid) if a<b else ((b,a),grid.T)

    def score(self,order,*,shared=True):
        if self.fixed is None:
            return None
        tiles = self.b.tiles.copy()
        tiles[list(self.cells)] = tiles[[self.cells[j] for j in order]]
        if any(not self.b.model.legal[c,3*tiles[c]:3*tiles[c]+3].any() for c in self.cells):
            return -1
        if shared:
            factors = self.fixed + list(self.domains(self.cells,tiles)) + [self.edge(e,tiles) for e in self.dynamic]
            active = self.active
        else:
            active = set(range(self.b.n))
            factors = list(self.domains(active,tiles)) + [self.edge(e,tiles) for e in self.b.model.problem.edges]
        final = eliminate(factors,active,self.cap,self.work)
        return None if final is None else sum(int(f) for _,f in final)
