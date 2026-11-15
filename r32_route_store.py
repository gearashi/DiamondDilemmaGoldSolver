"""Exact ownership dependencies and cross-traversal coverage for four-swap search."""
from collections import Counter
from r35_route_dependencies import CheckedRanges,selected
from functools import lru_cache
from itertools import permutations
import json
from math import comb, factorial

from r30_cycles import canonical_cycles


def schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS r32_page_proof(
        scope TEXT, lo INTEGER, hi INTEGER, owner TEXT, deps TEXT, PRIMARY KEY(scope,lo,owner))""")
    db.execute("""CREATE TABLE IF NOT EXISTS r35_page_dependencies(
        scope TEXT,lo INTEGER,owner TEXT,family TEXT,payload TEXT,PRIMARY KEY(scope,lo,owner,family))""")
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
        self.resolved=CheckedRanges()
        self.work=Counter()
        self.has_detail=db.execute("SELECT 1 FROM sqlite_master WHERE name='r35_page_dependencies'").fetchone() is not None

    def _row(self,source,owner,rank,family):
        self.work['page_reads']+=1
        row=self.db.execute("""SELECT c.hi,p.best,c.deps,c.lo,p.hi FROM r32_page_proof c JOIN r30_pages p
            ON p.scope=c.scope AND p.lo=c.lo WHERE c.scope=? AND c.owner=? AND c.lo<=?
            AND c.hi>? ORDER BY c.lo DESC LIMIT 1""",(source,owner,rank,rank)).fetchone()
        if row is None or row[0]>row[4]:return None
        hi,best,deps,lo,_=row
        if best!='null' and json.loads(best)['score']>=self.target:
            sql='SELECT max(score) FROM r34_page_scores WHERE scope=? AND lo=? AND owner=?'
            args=(source,lo,owner)
            if family is not None:
                sql+=' AND family=?';args+=(json.dumps(family),)
            local=self.db.execute(sql,args).fetchone()[0]
            if local is None or local>=self.target:return None
        detail=self.db.execute('SELECT payload FROM r35_page_dependencies WHERE scope=? AND lo=? AND owner=? AND family=?',
                               (source,lo,owner,'*' if family is None else json.dumps(family))).fetchone() if self.has_detail else None
        if detail:
            lo,hi,links=selected(detail[0],rank,lo,hi)
            return lo,hi,[(donor,r,tuple(f)) for donor,r,f in links]
        return lo,hi,[(donor,r,None) for donor,r in json.loads(deps)]

    def proven_end(self,source,owner,rank,depth=0,family=None):
        request=(owner,rank,None if family is None else tuple(family))
        # Exit frames publish a range only after every dependency on its path
        # has been checked. An intervening missing page or cycle is unresolved.
        pending=[(request,None)]
        active=set()
        while pending:
            item,exit_range=pending.pop()
            o,r,f=item;key=source,o,f
            if exit_range is not None:
                active.remove(item)
                self.resolved.add(key,*exit_range)
                continue
            if self.resolved.get(key,r)>r:
                self.work['interval_hits']+=1
                continue
            if item in active:return rank
            info=self._row(source,o,r,f)
            if info is None:return rank
            lo,hi,links=info
            active.add(item);pending.append((item,(lo,hi)))
            pending.extend((child,None) for child in reversed(links))
        return self.resolved.get((source,owner,request[2]),rank)

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
