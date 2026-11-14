"""Exact rotation responses with domain slicing, additive factors and bounded conditioning."""
from collections import Counter, OrderedDict
from math import prod
import numpy as np


def factor(keys,values):
    """Drop invariant axes; split a two-variable additive matrix exactly."""
    v = np.asarray(values,dtype=np.int32)
    keys = list(keys)
    for axis in reversed(range(len(keys))):
        first = v.take(0,axis=axis)
        if np.all(v==np.expand_dims(first,axis)):
            keys.pop(axis)
            v = first
    if len(keys)==2 and np.array_equal(v,v[:,[0]]+v[[0],:]-v[0,0]):
        return [((keys[0],),v[:,0]), ((keys[1],),v[0,:]-v[0,0])]
    return [(tuple(keys),v)]


class ExteriorCache:
    """Shared immutable leaf tables and exact separator responses for one center."""
    def __init__(self,bound,max_bytes=8*1024*1024):
        self.bound = bound
        self.edges = {}
        self.responses = OrderedDict()
        self.bytes = 0
        self.max_bytes = max_bytes

    def key(self,cell,scope,dims,rows):
        # Preserve multiplicity; row order has no effect on an integer sum.
        if sum(t.nbytes for _,t in rows)>self.max_bytes//4:
            return None
        return (cell,tuple(scope),dims,tuple(sorted((k,t.shape,t.tobytes()) for k,t in rows)))

    def put(self,key,result):
        size = 96+sum(96+len(v) for _,_,v in key[-1])+sum(96+t.nbytes for _,t in result)
        if size<=self.max_bytes:
            while self.bytes+size>self.max_bytes:
                _,(_,evicted) = self.responses.popitem(last=False)
                self.bytes -= evicted
            self.responses[key] = result,size
            self.bytes += size


def eliminate(rows, variables, widths, cap, work, cache=None):
    rows = list(rows)
    pending = set(variables)
    while pending:
        neighbors = {c:{c} for c in pending}
        for scope,_ in rows:
            for c in scope:
                if c in pending:
                    neighbors[c].update(scope)
        c = min(pending,key=lambda c:(prod(widths[x] for x in neighbors[c]),c))
        scope = sorted(neighbors[c])
        dims = tuple(widths[j] for j in scope)
        if prod(dims)>cap:
            return None,c
        used,keep = [],[]
        for keys,table in rows:
            (used if c in keys else keep).append((keys,table))
        key = cache.key(c,scope,dims,used) if cache is not None else None
        cached = cache.responses.get(key) if key is not None else None
        if cached is None:
            total = np.zeros(dims,dtype=np.int32)
            for keys,table in used:
                total += table.reshape(tuple(widths[j] if j in keys else 1 for j in scope))
            result = factor([j for j in scope if j!=c],total.max(axis=scope.index(c)))
            work['table_entries'] += prod(dims)
            if key is not None:
                cache.put(key,result)
        else:
            result = cached[0]
            cache.responses.move_to_end(key)
            work['reused_responses'] += 1
            work['avoided_table_entries'] += prod(dims)
        rows = keep+result
        work['eliminated_variables'] += 1
        pending.remove(c)
    return rows,None


class CompactExterior:
    def __init__(self,bound,cells,cap=65536,condition=1,cache=None):
        if cap<1 or condition<0:
            raise ValueError("Invalid exterior budget.")
        self.b = bound
        if cache is not None and cache.bound is not bound:
            raise ValueError("Exterior cache belongs to another center.")
        self.cache = cache
        self.cells = tuple(cells)
        self.cap = cap
        self.work = Counter()
        self.dynamic = []
        fixed = []
        moved = set(cells)
        boundary = set()
        domains = self.domains(bound.tiles)
        self.dom = domains
        for e in bound.model.problem.edges:
            a,_,b,_ = e
            if {a,b}&moved:
                self.dynamic.append(e)
                boundary.update({a,b}-moved)
            else:
                if cache is None:
                    fixed.extend(self.edge(e,bound.tiles,domains))
                else:
                    if e not in cache.edges:
                        cache.edges[e] = self.edge(e,bound.tiles,domains)
                    else:
                        self.work['reused_fixed_edges'] += 1
                    fixed.extend(cache.edges[e])
        self.active = moved|boundary
        self.leaves = []
        outside = set(range(bound.n))-self.active
        self.fixed = self.compile(fixed,outside,domains,condition)

    def domains(self,tiles):
        return {j:np.flatnonzero(self.b.model.legal[j,3*t:3*t+3]) for j,t in enumerate(tiles)}

    def edge(self,e,tiles,domains):
        a,s,b,t = e
        m = self.b.model
        left = m.masks[3*tiles[a]+domains[a],s]
        right = m.reverse[m.masks[3*tiles[b]+domains[b],t]]
        grid = left[:,None]==right
        return factor((a,b),grid) if a<b else factor((b,a),grid.T)

    def compile(self,rows,variables,domains,budget):
        widths = {k:len(v) for k,v in domains.items()}
        reduced,pivot = eliminate(rows,variables,widths,self.cap,self.work,self.cache)
        if reduced is not None:
            self.leaves.append(reduced)
            return True
        if budget==0:
            return False
        # Only fixed exterior identities are conditioned. Each branch retains its
        # exact response on every boundary port for all donor queries.
        for r in range(widths[pivot]):
            chosen = []
            for keys,table in rows:
                if pivot in keys:
                    chosen.extend(factor(tuple(k for k in keys if k!=pivot),table.take(r,axis=keys.index(pivot))))
                else:
                    chosen.append((keys,table))
            if not self.compile(chosen,variables-{pivot},domains,budget-1):
                return False
        return True

    def score(self,order):
        if not self.fixed:
            return None
        tiles = self.b.tiles.copy()
        tiles[list(self.cells)] = tiles[[self.cells[j] for j in order]]
        dom = self.dom.copy()
        for c in self.cells:
            t = tiles[c]
            dom[c] = np.flatnonzero(self.b.model.legal[c,3*t:3*t+3])
            if not len(dom[c]):
                return -1
        dynamic = [f for e in self.dynamic for f in self.edge(e,tiles,dom)]
        widths = {c:len(v) for c,v in dom.items()}
        best = -1
        for rows in self.leaves:
            final,_ = eliminate(rows+dynamic,self.active,widths,self.cap,self.work)
            if final is None:
                return None
            best = max(best,sum(int(v) for _,v in final))
        return best
