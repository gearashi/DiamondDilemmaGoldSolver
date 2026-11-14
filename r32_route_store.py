"""Exact ownership dependencies and cross-traversal coverage for four-swap search."""
from collections import defaultdict
from functools import lru_cache
from itertools import permutations
import json
from math import comb, factorial

import numpy as np

from r29_exchange_route import legal_cell_rotations
from r30_cycles import canonical_cycles


def schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS r32_page_proof(
        scope TEXT, lo INTEGER, hi INTEGER, owner TEXT, deps TEXT, PRIMARY KEY(scope,lo,owner))""")
    db.execute("""CREATE TABLE IF NOT EXISTS r32_root_binding(
        scope TEXT PRIMARY KEY, domain TEXT, owner TEXT, family TEXT, target INTEGER, pool TEXT)""")


def publish_page(db, scope, start, end, owners, dependencies):
    db.executemany('INSERT INTO r32_page_proof VALUES(?,?,?,?,?)', (
        (scope, start, end, owner, json.dumps(sorted(dependencies.get(owner, ())))) for owner in owners))


def inherited(db, scope, binding, families, target, owners, start, stop):
    """Each deferred endpoint names an earlier center's own support in this source union."""
    @lru_cache(maxsize=4096)
    def proven(source, owner, rank, depth):
        # Depth guard leaves unexpectedly long dependency chains open.
        if depth > 64:
            return rank
        row = db.execute("""SELECT c.hi,p.best,c.deps FROM r32_page_proof c JOIN r30_pages p
            ON p.scope=c.scope AND p.lo=c.lo WHERE c.scope=? AND c.owner=? AND c.lo<=?
            AND c.hi>? ORDER BY c.lo DESC LIMIT 1""", (source,owner,rank,rank)).fetchone()
        if row is None or (row[1]!='null' and json.loads(row[1])['score']>=target):
            return rank
        for donor, other_rank in json.loads(row[2]):
            if proven(source, donor, other_rank, depth+1)<=other_rank:
                return rank
        return row[0]

    end = stop
    for family in families:
        for owner in owners:
            sources = db.execute("""SELECT scope FROM r31_members WHERE domain=? AND family=?
                AND owner=? AND target<=? AND scope!=?""", (binding,json.dumps(family),owner,target,scope))
            hi = max((proven(source,owner,start,0) for source, in sources),default=start)
            end = min(end,hi)
            if end<=start:
                return start
    return end


def colex(values, indices):
    return sum(comb(i,j) for j,i in enumerate(sorted(indices[v] for v in values),1))


class Dependencies:
    """Map rotated endpoint cells back to the coordinates of their earlier owner."""
    def __init__(self, model, bounds, owners, families):
        self.bounds = bounds
        self.owners = owners
        self.families = set(families)
        self.indices = {c:i for i,c in enumerate(sorted(model.free))}
        self.transforms = [np.asarray(p,dtype=np.intp) for p in legal_cell_rotations(model)]
        self.deferred = defaultdict(set)

    def record(self,index,cells,order):
        endpoint = self.bounds[index].tiles.copy()
        endpoint[list(cells)] = endpoint[[cells[j] for j in order]]
        for earlier, bound in enumerate(self.bounds[:index]):
            inverse = np.argsort(bound.tiles)
            for transform in self.transforms:
                rel = inverse[endpoint[transform]]
                moved = np.flatnonzero(rel!=np.arange(len(rel))).tolist()
                if not all(c in self.indices for c in moved) or len(moved)!=len(cells):
                    continue
                local = {c:j for j,c in enumerate(moved)}
                perm = tuple(local[int(rel[c])] for c in moved)
                family = tuple(sorted(map(len,canonical_cycles(moved,perm))))
                if family in self.families:
                    self.deferred[self.owners[index]].add((self.owners[earlier],colex(moved,self.indices)))
                    return
        raise RuntimeError("Deferred endpoint did not map back to an eligible owner.")


class RootCoverage:
    """Compute the represented order mask on demand from completed, negative roots."""
    def __init__(self,db,binding,owner,family,target):
        self.db = db
        self.family = tuple(family)
        self.target = target
        self.scopes = []
        for scope,raw in db.execute("""SELECT scope,pool FROM r32_root_binding WHERE domain=?
            AND owner=? AND family=? AND target<=?""",(binding,owner,json.dumps(family),target)):
            pool = json.loads(raw)
            if not db.execute('SELECT 1 FROM r31_roots WHERE scope=? LIMIT 1',(scope,)).fetchone():
                continue
            self.scopes.append((scope,{c:i for i,c in enumerate(pool)}))
        self.finished = lru_cache(maxsize=8192)(self._finished)

    def _finished(self,scope,idx):
        row = self.db.execute('SELECT best FROM r31_roots WHERE scope=? AND idx=?',(scope,idx)).fetchone()
        return row is not None and (row[0]=='null' or json.loads(row[0])['score']<self.target)

    def covered(self,cells,order,close):
        from r31_route_geometry import parent,connected
        group = tuple(sorted(canonical_cycles(cells,order)))
        if not connected(group,close):
            return False
        while len(group)>1:
            group = parent(group,close)
        cycle, = group
        for scope,indices in self.scopes:
            if not all(c in indices for c in cells):
                continue
            size = len(cycle)
            offset = sum(comb(len(indices),k)*factorial(k-1) for k in set(self.family) if k<size)
            orient = list(permutations(sorted(cycle)[1:])).index(cycle[1:])
            idx = offset+colex(cycle,indices)*factorial(size-1)+orient
            if self.finished(scope,idx):
                return True
        return False
