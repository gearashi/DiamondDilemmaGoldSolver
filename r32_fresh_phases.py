"""Signed-phase construction, conditional exact lifts and reusable Gold repairs."""
from collections import Counter
from contextlib import closing
import hashlib
import json
import sqlite3

import numpy as np

from r27_incumbent import assignment
from r28_geo_color import signed_phase_model
from r29_gold_phase import EndpointCuts, checked_pins
from r30_gold import _groups, endpoint_constraint, search
from r31_fresh import physical_bonds
from validator import reverse_mask
from functools import lru_cache


def phase_cube(problem, width, index):
    """Low numbered physical tile phases partition assignments modulo global color reversal."""
    if not 0 <= width < problem.n or not 0 <= index < 2**width:
        raise ValueError('Phase cube index or width is out of range.')
    return {0: 0, **{i+1: index >> i & 1 for i in range(width)}}


def fit(problem, required, pins, bonds=(), *, seed=0, effort=.15, hint=None):
    """Return only integer-inventory feasible, connected endpoint types. The ceiling is relaxed."""
    from ortools.sat.python import cp_model
    if effort <= 0:
        raise ValueError('Expected a positive deterministic phase budget.')
    data = EndpointCuts(problem)
    m, phases = signed_phase_model(problem, pins=checked_pins(problem.n, pins), bonds=sorted(required))
    terms = []
    for a, b in sorted(bonds):
        ma, ca = data.sides[a]
        mb, cb = data.sides[b]
        if reverse_mask(ma) != mb:
            raise ValueError('Nonreciprocal scored bond.')
        equal = (ca ^ cb ^ (ma.bit_count() % 2)) == 0
        x, y = phases[a//3], phases[b//3]
        ok = m.new_bool_var('compatible_reference_bond')
        m.add(x == y if equal else x != y).only_enforce_if(ok)
        m.add(x != y if equal else x == y).only_enforce_if(~ok)
        terms.append(ok)
    if hint is not None:
        if len(hint)!=problem.n or any(x not in (0,1) for x in hint):
            raise ValueError('Invalid phase hint.')
        for var, value in zip(phases,hint):
            m.add_hint(var,value)
    if terms:
        m.maximize(sum(terms))
    solver = cp_model.CpSolver()
    solver.parameters.random_seed = int(seed % 2147483647)
    solver.parameters.num_search_workers = 1
    remaining = effort
    calls = branches = 0
    ceiling = len(terms)
    seen = set()
    while remaining > 0 and calls < 16:
        solver.parameters.max_deterministic_time = remaining
        status = solver.solve(m)
        calls += 1
        branches += solver.num_branches
        remaining -= max(.000001, solver.response_proto.deterministic_time)
        info = dict(calls=calls, branches=branches, cuts=len(seen), ceiling=ceiling)
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            return dict(status=solver.status_name(status), **info)
        if terms:
            # Integer optimum cannot exceed the rounded-up relaxation upper bound.
            ceiling = min(ceiling, int(np.ceil(solver.best_objective_bound)))
        values = [int(solver.value(v)) for v in phases]
        groups = _groups(edge for t,p in enumerate(values) for edge in data.strands[t][p])
        if len(groups) == 1:
            return dict(status=solver.status_name(status), phases=values, kept=sum(solver.value(v) for v in terms),
                        calls=calls, branches=branches, cuts=len(seen), ceiling=ceiling)
        for g in groups:
            if g in seen:
                raise ArithmeticError('Phase cut repeated.')
            endpoint_constraint(m, phases, data, g)
            seen.add(g)
    return dict(status='UNKNOWN', calls=calls, branches=branches, cuts=len(seen), ceiling=ceiling)


def lift_cube(problem, required, pins, domains=None, seconds=1, cache=None):
    """Record a failed cube only after a checked full one-loop geographic decision."""
    from r29_gold_geography import checked_domains
    from r30_gold import restriction_binding
    domains = checked_domains(problem, domains)
    scope = restriction_binding(problem, domains) + repr(sorted(required))
    pins = checked_pins(problem.n, pins)
    def solve(p):
        return search(problem, domains=domains, pins=p, bonds=required, geography=True,
                      channel=True, seconds=seconds)
    if cache:
        with closing(sqlite3.connect(cache, timeout=60)) as conn, conn:
            conn.execute('CREATE TABLE IF NOT EXISTS r32_phase_cuts (scope TEXT, pins TEXT, PRIMARY KEY(scope,pins))')
            for (raw,) in conn.execute('SELECT pins FROM r32_phase_cuts WHERE scope=?', (scope,)):
                core = json.loads(raw)
                if all(pins.get(tile) == phase for tile, phase in core):
                    return dict(status='INFEASIBLE', reused=True, pins=core, lift_calls=0)
    row = solve(pins)
    row['lift_calls'] = 1
    if row['status'] == 'INFEASIBLE' and cache:
        kept = dict(pins)
        for tile in sorted(pins):
            weaker = {k:v for k,v in kept.items() if k != tile}
            row['lift_calls'] += 1
            if solve(weaker)['status'] == 'INFEASIBLE':
                kept = weaker
        core = sorted(kept.items())
        with closing(sqlite3.connect(cache, timeout=60)) as conn, conn:
            conn.execute('INSERT OR IGNORE INTO r32_phase_cuts VALUES (?,?)', (scope,json.dumps(core)))
        row['proved_pins'] = core
    return row


class SignedRepair:
    """Optimize the count of signed-compatible seams on nonadjacent cells with distinct donors."""
    def __init__(self, model, phases, forbidden=(), database=None):
        self.model = model
        p = model.problem
        data = EndpointCuts(p)
        if len(phases) != p.n or any(v not in (0,1) for v in phases):
            raise ValueError('Expected one phase bit per physical tile.')
        self.phases = tuple(phases)
        self.sides = np.array([[3*(v//3)+(s-v%3)%3 for s in range(3)] for v in range(3*p.n)])
        base = [2*m+(c^phases[i//3]) for i,(m,c) in enumerate(data.sides)]
        self.labels = np.array(base)[self.sides]
        self.reverse = np.array([2*int(model.reverse[v//2]) + ((v%2)^((v//2).bit_count()%2)) for v in range(4096)])
        self.forbid = frozenset(tuple(sorted(x)) for x in forbidden)
        self.blocked = {}
        for a,b in self.forbid:
            self.blocked.setdefault(a, []).append(b)
            self.blocked.setdefault(b, []).append(a)
        self.stats = Counter()
        self.last_hit_keys = set()
        self.last_new_keys = set()
        self.database = database
        self._allowed = lru_cache(maxsize=4096)(self._allowed)
        self.scope = hashlib.sha256(repr((p.fingerprint,self.phases,sorted(self.forbid),'signed-v1')).encode()).hexdigest()
        if database:
            with closing(sqlite3.connect(database,timeout=60)) as db,db:
                db.execute('CREATE TABLE IF NOT EXISTS r32_price_rows (scope TEXT, signature TEXT, value BLOB, PRIMARY KEY(scope,signature))')

    def score(self, board):
        m = self.model
        ok = self.labels[board[m.a],m.sa] == self.reverse[self.labels[board[m.b],m.sb]]
        present = physical_bonds(m, board)
        for i in np.flatnonzero(ok):
            a = int(self.sides[board[m.a[i]],m.sa[i]])
            b = int(self.sides[board[m.b[i]],m.sb[i]])
            if (min(a,b),max(a,b)) in self.forbid:
                ok[i] = False
        return int(ok.sum()), len(self.forbid & present)

    def _allowed(self, sides):
        allowed = np.ones(3*self.model.n,dtype=bool)
        for s,dart in enumerate(sides):
            if dart in self.blocked:
                allowed &= ~np.isin(self.sides[:,s],self.blocked[dart])
        return allowed

    def block(self, values, cells):
        m = self.model
        board = m.legal_board(values)
        cells = np.asarray(cells, dtype=int)
        if not len(cells):
            return board
        if (np.any(cells<0) or np.any(cells>=m.n) or len(set(cells))!=len(cells)
                or set(cells)&set(m.problem.neighbors[cells].ravel()) or set(cells)&set(m.problem.fixed)):
            raise ValueError('Expected distinct movable independent cells.')
        codes = 3*np.sort(board[cells]//3)[:,None] + np.arange(3)
        ds = m.legal[cells[:,None,None],codes]
        prices = np.zeros((len(cells),len(cells),3),dtype=np.int16)
        # The row response to every donor depends only on the three oriented outside sides.
        # Domain checks are applied after reuse. Exchange and tie reconstruction stay local.
        self.last_hit_keys.clear()
        self.last_new_keys.clear()
        def compute(db):
            for i, cell in enumerate(cells):
                neighbors = m.problem.neighbors[cell]
                sides = tuple(int(self.sides[board[c],s]) if c>=0 else -1 for c,s in zip(neighbors,m.problem.sides[cell]))
                key = ','.join(map(str,sides))
                row = None
                if db is not None:
                    found=db.execute('SELECT value FROM r32_price_rows WHERE scope=? AND signature=?',(self.scope,key)).fetchone()
                    if found:
                        row=np.frombuffer(found[0],dtype=np.int8)
                        if row.size != 3*m.n:
                            raise ValueError('Invalid cached signed repair row.')
                        self.stats['row_hits']+=1
                        self.last_hit_keys.add(key)
                if row is None:
                    row=np.zeros(3*m.n,dtype=np.int8)
                    for s,dart in enumerate(sides):
                        if dart < 0:
                            continue
                        t,u=divmod(dart,3)
                        ok=self.labels[:,s]==self.reverse[self.labels[3*t,u]]
                        if dart in self.blocked:
                            ok &= ~np.isin(self.sides[:,s], self.blocked[dart])
                        row+=ok
                    self.stats['rows_computed']+=1
                    self.last_new_keys.add(key)
                    if db is not None:
                        db.execute('INSERT OR IGNORE INTO r32_price_rows VALUES (?,?,?)',(self.scope,key,row.tobytes()))
                ds[i] &= self._allowed(sides)[codes]
                prices[i] = row[codes]
        if self.database:
            with closing(sqlite3.connect(self.database,timeout=60)) as db,db:
                compute(db)
        else:
            compute(None)
        # Prefer the current pose on equal prices so neutral cycles cannot recreate work.
        weight = prices * (len(cells)+1) + (codes[None,:,:]==board[cells,None,None])
        weight = np.where(ds,weight,-1)
        rot = weight.argmax(axis=2)
        donors = assignment(weight.max(axis=2),ds.any(axis=2))
        board[cells] = codes[donors,rot[np.arange(len(cells)),donors]]
        self.stats['assignments']+=1
        return board
