"""Subsumption and checked persistence for two-budget seam/mover obstructions."""
from collections import Counter,defaultdict
from hashlib import sha256
import json


def canonical(rows):
    # With the same failed-edge choices, a smaller set of movers gives a
    # stronger disjunction. Distinct incomparable choices are both necessary.
    grouped=defaultdict(set)
    for edges,mask in rows:
        edges=tuple(sorted(set(edges)))
        if any(type(e) is not int or e<0 for e in edges) or type(mask) is not int or mask<0:
            raise ValueError('Invalid seam/mover clause.')
        masks=grouped[edges]
        if any(old&~mask==0 for old in masks):continue
        masks.difference_update([old for old in masks if mask&~old==0])
        masks.add(mask)
    return tuple(sorted((e,m) for e,choices in grouped.items() for m in choices))


def implies(current,old):
    """Each saved disjunction has a current clause contained in it."""
    from itertools import combinations
    for edges,mask in old:
        if not any(s&~mask==0 for k in range(len(edges)+1) for subset in combinations(edges,k)
                   for s in current.get(subset,())):
            return False
    return True


def by_edges(rows):
    out=defaultdict(list)
    for e,m in rows:out[e].append(m)
    return out


def native_negative(rows,movers,failures,seconds):
    """Cold external rows are hints until a native solver verifies their own CNF."""
    from ortools.sat.python import cp_model
    model=cp_model.CpModel();edge={};moved={}
    for edges,mask in rows:
        options=[]
        for e in edges:
            if e not in edge:edge[e]=model.new_bool_var('seam')
            options.append(edge[e])
        while mask:
            bit=mask&-mask;mask^=bit
            if bit not in moved:moved[bit]=model.new_bool_var('mover')
            options.append(moved[bit])
        model.add_bool_or(options)
    model.add(sum(edge.values())<=failures);model.add(sum(moved.values())<=movers)
    solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=seconds
    return solver.solve(model)==cp_model.INFEASIBLE


class CoverCores:
    def __init__(self,db=None,binding=None,*,verify_seconds=2):
        self.db=db;self.binding=binding;self.verify_seconds=verify_seconds
        self.cores={};self.anchors=defaultdict(set);self.trusted=set();self.work=Counter()
        if db is not None:
            if not isinstance(binding,str) or not binding:raise ValueError('Durable cover rows require a domain binding.')
            db.execute('CREATE TABLE IF NOT EXISTS r35_defect_cores(binding TEXT,key TEXT,payload TEXT,PRIMARY KEY(binding,key))')
            for key,raw in db.execute('SELECT key,payload FROM r35_defect_cores WHERE binding=?',(binding,)):
                try:
                    if self._key(raw)!=key:continue
                    movers,failures,clauses=json.loads(raw)
                    if type(movers) is not int or type(failures) is not int:continue
                    rows=canonical((e,int(m,16)) for e,m in clauses)
                    if any(len(e)>3 for e,_ in rows):continue
                    self._index(key,movers,failures,rows)
                    self.work['disk_candidates']+=1
                except (ValueError,TypeError,OverflowError):continue

    def _key(self,raw):
        return sha256(('seam-mover-clauses-v1/'+(self.binding or '')+'/'+raw).encode()).hexdigest()

    def _index(self,key,movers,failures,rows):
        self.cores[key]=(movers,failures,rows)
        if not rows or any(not e for e,_ in rows):
            self.anchors[None].add(key)
            return
        # Any predicate proving a saved clause must share a seam with its
        # anchor (unless it uses a mover-only clause).
        anchor=min((e for e,_ in rows),key=lambda e:(sum(len(self.anchors[d]) for d in e),len(e),e))
        for edge in anchor:self.anchors[edge].add(key)

    def _remove(self,key):
        self.cores.pop(key,None);self.trusted.discard(key)
        for keys in self.anchors.values():keys.discard(key)

    def covered(self,rows,remaining,failures):
        current=by_edges(rows)
        options=set(self.anchors.get(None,()))
        if () in current:options.update(self.cores)
        else:
            for edges in current:
                if edges:options.update(self.anchors.get(edges[0],()))
        for key in sorted(options):
            limit,cap,old=self.cores[key]
            self.work['candidate_checks']+=1
            if remaining>limit or failures>cap or not implies(current,old):continue
            if key not in self.trusted:
                self.work['cold_verifications']+=1
                if not native_negative(old,limit,cap,self.verify_seconds):
                    self.work['unverified']+=1
                    continue
                self.trusted.add(key)
            return True
        return False

    def add(self,rows,remaining,failures,*,certified=False):
        rows=canonical(rows);current=by_edges(rows)
        if any(remaining<=limit and failures<=cap and implies(current,old)
               for key,(limit,cap,old) in self.cores.items() if key in self.trusted):
            self.work['dominated_new']+=1
            return
        obsolete=[key for key,(limit,cap,old) in self.cores.items()
                  if limit<=remaining and cap<=failures and implies(by_edges(old),rows)]
        raw=json.dumps([remaining,failures,[(e,hex(m)) for e,m in rows]],separators=(',',':'))
        key=self._key(raw)
        for old in obsolete:self._remove(old)
        self._index(key,remaining,failures,rows);self.trusted.add(key)
        self.work['saved_clauses']+=len(rows);self.work['absorbed_cores']+=len(obsolete)
        if self.db is not None and certified:
            self.db.execute('INSERT OR IGNORE INTO r35_defect_cores VALUES(?,?,?)',(self.binding,key,raw))
            self.db.executemany('DELETE FROM r35_defect_cores WHERE binding=? AND key=?',
                                ((self.binding,old) for old in obsolete if old!=key))
