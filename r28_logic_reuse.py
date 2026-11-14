"""Scoped physical dominance, exact residual CNF reuse and unit reason replay."""
from collections import defaultdict
from itertools import combinations, product
import math
import time

from r27_reason_maxsat import domain_rows
from validator import reverse_mask


def dominance(problem, rows, cells, *, target='gold', max_assignments=4096):
    """Compare extension sets exactly over a small complete physical product.

    Substitutions are scoped to the supplied domains and existential target.
    A retained row covers every physical outside tuple extending a removed row.
    """
    rows = domain_rows(problem,rows)
    if target not in ('gold','edges'):
        raise ValueError('Expected gold or edges.')
    if any(not row for row in rows):
        return dict(status='complete',removed=0,kept=[],replacements=[],checks=0)
    if math.prod(map(len,rows)) > max_assignments:
        return dict(status='unknown',removed=0)
    cells = tuple(sorted(set(cells)))
    if not cells or any(type(c)is not int or not 0<=c<problem.n for c in cells):
        raise ValueError('Invalid local region.')
    outside = tuple(c for c in range(problem.n) if c not in cells)
    local = list(product(*(rows[c] for c in cells)))
    extensions = {x:0 for x in local}
    # Every bit represents one complete labelled exterior. Keeping it is essential
    # when a replacement shares the masks but changes a closed strand component.
    work = 0
    for index, ext in enumerate(product(*(rows[c] for c in outside))):
        base = dict(zip(outside,ext))
        for patch in local:
            values = {**base,**dict(zip(cells,patch))}
            board = [values[c] for c in range(problem.n)]
            if len({v//3 for v in board})!=problem.n:
                continue
            report = problem.validate(board)
            work += 1
            if report['valid'] if target=='gold' else report['matched_edges']==len(problem.edges):
                extensions[patch] |= 1<<index
    keep = []
    replacements = []
    for patch in sorted(local,key=lambda p:(-extensions[p].bit_count(),p)):
        exterior = extensions[patch]
        if not exterior:
            replacements.append([list(patch),None])
            continue
        inventory = sorted(v//3 for v in patch)
        representative = next((r for r in keep if inventory==sorted(v//3 for v in r)
                               and not exterior & ~extensions[r]),None)
        if representative is None:
            keep.append(patch)
        else:
            replacements.append([list(patch),list(representative)])
    return dict(status='complete',removed=len(replacements),kept=keep,replacements=replacements,checks=work)


def direct_cnf(problem,rows,*,factored=False,encoding=None):
    """Represent complete inventory and seam feasibility. Gold cuts are external."""
    from r28_logic_sat import placements
    from pysat.card import CardEnc, EncType
    rows = domain_rows(problem,rows)
    if encoding is None:
        encoding = 'biclique' if factored else 'pair'
    if encoding not in ('pair','biclique','support','bitcode'):
        raise ValueError('Unknown seam CNF encoding.')
    placements_by_cell,clauses,top = placements(rows)
    owned = defaultdict(list)
    for row in placements_by_cell:
        for code,lit in row.items():
            owned[code//3].append(lit)
    for tile in range(problem.n):
        lits = owned[tile]
        clauses.append(lits)
        if len(lits)>1:
            encoded = CardEnc.atmost(lits,1,top_id=top,encoding=EncType.seqcounter)
            top = max(top,encoded.nv)
            clauses.extend(encoded.clauses)
    factorizations = 0
    for a,sa,b,sb in problem.edges:
        left = defaultdict(list)
        right = defaultdict(list)
        for x,lit in placements_by_cell[a].items():
            left[int(problem.masks[x,sa])].append(lit)
        for y,lit in placements_by_cell[b].items():
            right[reverse_mask(problem.masks[y,sb])].append(lit)
        if encoding=='bitcode':
            alphabet = sorted(left.keys() | right.keys())
            symbols = {s:i for i,s in enumerate(alphabet)}
            width = (len(symbols)-1).bit_length() if symbols else 0
            for i in range(width):
                top += 1
                for buckets in (left,right):
                    for symbol,lits in buckets.items():
                        bit = top if symbols[symbol] & (1<<i) else -top
                        clauses.extend([-lit,bit] for lit in lits)
            continue
        if encoding=='support':
            for buckets,other in ((left,right),(right,left)):
                for symbol,lits in buckets.items():
                    allowed = other.get(symbol,())
                    clauses.extend([-lit,*allowed] for lit in lits)
            continue
        for mask,lits in left.items():
            forbidden = [lit for other,values in right.items() if mask!=other for lit in values]
            if encoding=='biclique' and len(lits)+len(forbidden)<len(lits)*len(forbidden):
                # Existentially eliminating z recovers exactly this complete biclique.
                top += 1
                clauses.extend([-lit,top] for lit in lits)
                clauses.extend([-top,-lit] for lit in forbidden)
                factorizations += 1
            else:
                clauses.extend([-x,-y] for x in lits for y in forbidden)
    return placements_by_cell,clauses,factorizations


class ClauseCache:
    """SAT decisions on exact simplified connected components plus their witnesses."""
    def __init__(self,limit=512):
        self.cache = {}
        self.calls = self.hits = 0
        self.limit = limit

    def solve(self,clauses,assumptions=()):
        from pysat.solvers import Glucose4
        known = set(assumptions)
        if any(-v in known for v in known):
            return None
        remaining = set()
        for clause in clauses:
            if any(x in known for x in clause):
                continue
            row = tuple(sorted(x for x in set(clause) if -x not in known))
            if not row:
                return None
            if not any(-x in row for x in row):
                remaining.add(row)
        by_var = defaultdict(list)
        for row in remaining:
            for x in row:
                by_var[abs(x)].append(row)
        while remaining:
            start = min(remaining)
            work = [start]
            remaining.remove(start)
            group = []
            seen_vars = set()
            while work:
                row = work.pop()
                group.append(row)
                for x in row:
                    v = abs(x)
                    if v not in seen_vars:
                        seen_vars.add(v)
                        for adjacent in by_var[v]:
                            if adjacent in remaining:
                                remaining.remove(adjacent)
                                work.append(adjacent)
            key = tuple(sorted(group))  # includes every surviving clause, including inventory
            if key in self.cache:
                self.hits += 1
                witness = self.cache[key]
            else:
                self.calls += 1
                with Glucose4(bootstrap_with=key) as solver:
                    witness = (tuple(x for x in solver.get_model() if abs(x) in seen_vars)
                               if solver.solve() else None)
                if self.limit>0:
                    if len(self.cache)>=self.limit:
                        del self.cache[next(iter(self.cache))]
                    self.cache[key] = witness
            if witness is None:
                return None
            known.update(witness)
        return known


class TrailReplay:
    """A saved reason is just a hint until its clause is unit in the new trail."""
    def __init__(self,clauses):
        self.clauses = tuple(tuple(dict.fromkeys(cl)) for cl in clauses)
        self.trail = []
        self.reused = self.scans = self.checks = 0

    def propagate(self,assumptions,*,reuse=True):
        known = set(assumptions)
        if any(-v in known for v in known):
            return None
        next_trail = []
        if reuse:
            for index,lit in self.trail:
                if lit in known:
                    continue
                self.checks += 1
                if all(x==lit or -x in known for x in self.clauses[index]):
                    if -lit in known:
                        return None
                    known.add(lit)
                    next_trail.append((index,lit))
                    self.reused += 1
        while True:
            old = len(known)
            for index,clause in enumerate(self.clauses):
                self.scans += 1
                if any(x in known for x in clause):
                    continue
                pending = [x for x in clause if -x not in known]
                if not pending:
                    return None
                if len(pending)==1:
                    known.add(pending[0])
                    next_trail.append((index,pending[0]))
            if len(known)==old:
                self.trail = next_trail
                return known


def live_session(clauses,*,slice_conflicts=8,slices=128,seconds=10,slice_seconds=.04):
    """Use the same native solver at each quiescent boundary; compare to cold replay.

    The tested Python binding has no serialized internal CDCL-state restore API.
    Keep the older durable frontier for restarting a process.
    """
    from pysat.solvers import Glucose4
    if slice_conflicts<1 or slices<1 or seconds<=0 or slice_seconds<=0:
        raise ValueError('Expected positive slice and time budgets.')
    from threading import Timer
    start = time.monotonic()
    counts = {}
    for mode in ('warm','cold'):
        deadline = time.monotonic()+seconds/2
        solver = None
        previous = decisions = conflicts = calls = 0
        try:
            answer = None
            while calls<slices and time.monotonic()<deadline:
                if solver is None:
                    solver = Glucose4(bootstrap_with=clauses)
                    previous = 0
                solver.conf_budget(slice_conflicts)
                timer = Timer(min(slice_seconds,max(.001,deadline-time.monotonic())),solver.interrupt)
                timer.start()
                try:
                    answer = solver.solve_limited(expect_interrupt=True)
                finally:
                    timer.cancel()
                    timer.join()
                    solver.clear_interrupt()
                stats = solver.accum_stats()
                decisions += stats['decisions']-previous
                previous = stats['decisions']
                if mode=='cold':
                    conflicts += stats['conflicts']
                    solver.delete()
                    solver = None
                else:
                    conflicts = stats['conflicts']
                calls += 1
                if answer is not None:
                    break
            counts[mode] = dict(answer=answer,calls=calls,decisions=decisions,conflicts=conflicts)
        finally:
            if solver is not None:
                solver.delete()
    if counts['warm']['answer'] is not None and counts['cold']['answer'] is not None and counts['warm']['answer']!=counts['cold']['answer']:
        raise RuntimeError('Warm and cold native SAT decisions disagree.')
    return dict(runs=counts, seconds=time.monotonic()-start, full_native_state_serializable=False)
