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


def independent_phases(problem):
    """Order independent tile colors by how many other phases a choice fixes."""
    from r29_gold_phase import SignedParity
    from r32_gold_bonds import inventory
    saved = getattr(problem,'_independent_phase_order',None)
    if saved is not None:
        return saved
    parity = SignedParity(problem)
    parity.pin(0,0)
    for a,b in sorted(inventory(problem).required):
        parity.bond(a,b)
    if parity.base.conflict is not None:
        raise ValueError('No signed parity assignment for the mandatory Gold mates.')
    expressions = []
    free = []
    for t in range(problem.n):
        pivot = parity.base.pivots.get(t)
        if pivot is None:
            free.append(t)
            expressions.append(1<<t)
        else:
            dependencies = pivot[0]^(1<<t)
            mask = 0
            while dependencies:
                bit = dependencies & -dependencies
                mask ^= expressions[bit.bit_length()-1]
                dependencies ^= bit
            expressions.append(mask)
    order = []
    while free:
        # Prefer immediate contractions first, then shorten the fewest-variable
        # residual equations. Tie-breaking by physical tile ID is reproducible.
        def benefit(tile):
            counts = [0]*(problem.n+1)
            for mask in expressions:
                if mask >> tile & 1:
                    counts[mask.bit_count()] += 1
            return counts
        t = max(free,key=benefit)
        free.remove(t)
        order.append(t)
        expressions = [mask & ~(1<<t) for mask in expressions]
    problem._independent_phase_order = tuple(order)
    return problem._independent_phase_order


def phase_cube(problem, width, index, *, independent=False):
    """Low numbered physical tile phases partition assignments modulo global color reversal."""
    if not 0 <= width < problem.n or not 0 <= index < 2**width:
        raise ValueError('Phase cube index or width is out of range.')
    if independent and width:
        coordinates = independent_phases(problem)
        if width>len(coordinates):
            raise ValueError('Cube width exceeds the remaining independent phase dimension.')
    else:
        coordinates = range(1,problem.n)
    return {0: 0, **{tile: index >> i & 1 for i,tile in enumerate(coordinates[:width])}}


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
    from r34_phase_cache import checked_witness, negative, schema, witness
    domains = checked_domains(problem, domains)
    required = tuple(sorted(tuple(sorted(pair)) for pair in required))
    scope = restriction_binding(problem, domains) + repr(sorted(required))
    pins = checked_pins(problem.n, pins)
    def solve(p):
        return search(problem, domains=domains, pins=p, bonds=required, geography=True,
                      channel=True, seconds=seconds)
    if cache:
        with closing(sqlite3.connect(cache, timeout=60)) as conn, conn:
            schema(conn)
            core = negative(conn,scope,pins)
            if core is not None:
                return dict(status='INFEASIBLE',reused=True,pins=core,lift_calls=0)
            saved = witness(conn,problem,domains,pins,required)
            if saved is not None:
                return saved
    row = solve(pins)
    row['lift_calls'] = 1
    if row['status'] == 'GOLD' and cache:
        checked = checked_witness(problem,row['codes'],row['phases'],domains,pins,required)
        if checked is None:
            raise ValueError('Geographic phase lift failed independent witness validation.')
        with closing(sqlite3.connect(cache,timeout=60)) as conn, conn:
            conn.execute('INSERT OR IGNORE INTO r34_phase_witnesses VALUES(?,?,?)',
                         (problem.fingerprint,json.dumps(row['codes']),json.dumps(row['phases'])))
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


class _DartPrices:
    """One phase-independent compatibility vector per physical exterior side."""
    def __init__(self, model, forbidden):
        p = model.problem
        self.fingerprint = p.fingerprint
        self.forbidden = forbidden
        self.sides = np.array([[3*(v//3)+(s-v%3)%3 for s in range(3)] for v in range(3*p.n)])
        data = EndpointCuts(p)
        self.masks = np.asarray([mask for mask,_ in data.sides],dtype=np.int16)
        self.colors = np.asarray([color for _,color in data.sides],dtype=np.int8)
        self.reverse = np.asarray([2*int(model.reverse[v//2]) + ((v%2)^((v//2).bit_count()%2)) for v in range(4096)])
        self.blocked = {}
        for a,b in forbidden:
            self.blocked.setdefault(a,[]).append(b)
            self.blocked.setdefault(b,[]).append(a)
        self.templates = {}

    def price(self, dart, phases, stats):
        if dart not in self.templates:
            mask = int(self.masks[dart])
            ok = self.masks==reverse_mask(mask)
            required = np.where(ok,self.colors^self.colors[dart]^(mask.bit_count()%2),-1).astype(np.int8)
            required[self.blocked.get(dart,[])] = -1
            self.templates[dart] = required
            stats['dart_templates_built'] += 1
        else:
            stats['dart_templates_reused'] += 1
        stats['signed_dart_prices'] += 1
        return self.templates[dart] == (phases^phases[dart])


class SignedRepair:
    """Optimize the count of signed-compatible seams on nonadjacent cells with distinct donors."""
    def __init__(self, model, phases, forbidden=(), database=None):
        self.model = model
        p = model.problem
        if len(phases) != p.n or any(v not in (0,1) for v in phases):
            raise ValueError('Expected one phase bit per physical tile.')
        self.phases = tuple(phases)
        self.forbid = frozenset(tuple(sorted(x)) for x in forbidden)
        geometry = getattr(p,'_signed_dart_prices',None)
        if geometry is None or geometry.fingerprint != p.fingerprint or geometry.forbidden != self.forbid:
            geometry = _DartPrices(model,self.forbid)
            p._signed_dart_prices = geometry
        self.geometry = geometry
        self.sides = geometry.sides
        self.blocked = geometry.blocked
        self.reverse = geometry.reverse
        self.dart_phases = np.repeat(np.asarray(phases,dtype=np.int8),3)
        self.labels = (2*geometry.masks+(geometry.colors^self.dart_phases))[self.sides]
        self.prices = {}
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
                        if dart not in self.prices:
                            self.prices[dart] = self.geometry.price(dart,self.dart_phases,self.stats)
                        else:
                            self.stats['signed_dart_reuses'] += 1
                        row += self.prices[dart][self.sides[:,s]]
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
