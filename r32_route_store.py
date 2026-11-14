"""Exact ownership dependencies and cross-traversal coverage for four-swap search."""
from collections import OrderedDict
from functools import lru_cache
from itertools import permutations
import json
from math import comb, factorial

from r30_cycles import canonical_cycles


def schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS r32_page_proof(
        scope TEXT, lo INTEGER, hi INTEGER, owner TEXT, deps TEXT, PRIMARY KEY(scope,lo,owner))""")
    db.execute("""CREATE TABLE IF NOT EXISTS r34_page_scores(
        scope TEXT, lo INTEGER, owner TEXT, family TEXT, score INTEGER NOT NULL,
        PRIMARY KEY(scope,lo,owner,family))""")
    db.execute("""CREATE TABLE IF NOT EXISTS r32_root_binding(
        scope TEXT PRIMARY KEY, domain TEXT, owner TEXT, family TEXT, target INTEGER, pool TEXT)""")


def publish_scores(db, scope, start, owners, families, scores):
    """Save local maxima only on a positive page. A deferred endpoint still requires its owner."""
    db.executemany("INSERT INTO r34_page_scores VALUES(?,?,?,?,?)", (
        (scope,start,owner,json.dumps(family),scores.get((index,family),-1))
        for index,owner in enumerate(owners) for family in families))


def publish_page(db, scope, start, end, owners, dependencies):
    db.executemany('INSERT INTO r32_page_proof VALUES(?,?,?,?,?)', (
        (scope, start, end, owner, json.dumps(sorted(dependencies.get(owner, ())))) for owner in owners))


class PageProofs:
    """Reuse completed dependency checks for one target and one database connection."""
    def __init__(self, db, target):
        self.db = db
        self.target = target
        self.resolved = OrderedDict()

    def proven_end(self, source, owner, rank, depth=0, family=None):
        key = (source, owner, rank, family)
        if key in self.resolved:
            self.resolved.move_to_end(key)
            return self.resolved[key]
        if depth > 64:
            return rank
        row = self.db.execute("""SELECT c.hi,p.best,c.deps,c.lo FROM r32_page_proof c JOIN r30_pages p
            ON p.scope=c.scope AND p.lo=c.lo WHERE c.scope=? AND c.owner=? AND c.lo<=?
            AND c.hi>? ORDER BY c.lo DESC LIMIT 1""", (source, owner, rank, rank)).fetchone()
        if row is None:
            return rank
        if row[1] != 'null' and json.loads(row[1])['score'] >= self.target:
            sql = 'SELECT max(score) FROM r34_page_scores WHERE scope=? AND lo=? AND owner=?'
            args = (source, row[3], owner)
            if family is not None:
                sql += ' AND family=?'
                args += (json.dumps(family),)
            local = self.db.execute(sql, args).fetchone()[0]
            if local is None or local >= self.target:
                return rank
        for donor, other_rank in json.loads(row[2]):
            if self.proven_end(source, donor, other_rank, depth + 1) <= other_rank:
                return rank
        # Missing dependencies can arrive on the next page. Only successes are immutable.
        self.resolved[key] = row[0]
        if len(self.resolved) > 4096:
            self.resolved.popitem(last=False)
        return row[0]

    def owner_end(self, scope, binding, family, owner, start, stop):
        sources = self.db.execute("""SELECT scope FROM r31_members WHERE domain=? AND family=?
            AND owner=? AND target<=? AND scope!=?""", (binding, json.dumps(family), owner, self.target, scope))
        hi = start
        for source, in sources:
            hi = max(hi, self.proven_end(source, owner, start, family=family))
            if hi >= stop:
                break
        return hi


def colex(values, indices):
    return sum(comb(i,j) for j,i in enumerate(sorted(indices[v] for v in values),1))


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
