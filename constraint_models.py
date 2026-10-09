"""Exact SAT, CP-SAT and traditional CP models for the encoded Gold puzzle.

Integer endpoint masks match after eleven-bit reversal. Model solutions always
pass the independent problem validator. Gold adds sound closed-component cuts;
no elapsed-time limit is reported as an infeasibility proof. Native solver state
is not portable: checkpoints retain rejected-board witnesses and restart with
revalidated Gold cuts, so resumed runs may repeat earlier internal work.

API references: pysathq.github.io/docs/html/api/solvers.html and the official
OR-Tools cp_model / constraint_solver.pywrapcp documentation. SAT uses optional
python-sat==1.9.dev15 (Glucose4); both CP engines use ortools==9.15.6755.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import math
import threading
import time

from validator import reverse_mask

SCHEMA = 'diamond-constraint-models-v1'


class _Interrupted(Exception):
    pass


class _Run:
    def __init__(self, problem, engine, seconds, seed, target, stop, progress):
        self.problem, self.engine, self.seed, self.target = problem, engine, seed, target
        self.started = time.monotonic()
        self.deadline = self.started + seconds
        self.stop, self.progress = stop, progress
        self.reason = None
        self.error = None
        self.nodes = 0
        self.last_progress = self.started
        self.last_checkpoint = self.started
        self.rejected = []
        self.cuts = []
        self.last_codes = None
        self.stats = {'engine': engine, 'seed': seed, 'target': target,
                      'candidates_checked': 0, 'gold_cuts': 0,
                      'resumed': False, 'resumption_mode': 'restart_with_revalidated_gold_cuts'}

    def cancelled(self):
        if self.reason:
            return True
        requested = self.stop() if callable(self.stop) else self.stop.is_set() if self.stop is not None else False
        if requested:
            self.reason = 'stopped'
        elif time.monotonic() >= self.deadline:
            self.reason = 'timeout'
        if self.reason is None and time.monotonic()-self.last_progress >= 1:
            self.emit('searching' if 'model_seconds' in self.stats else 'building_model')
        return self.reason is not None

    def check(self):
        if self.error is not None:
            raise self.error
        if self.cancelled():
            raise _Interrupted

    def checkpoint(self):
        return {'schema': SCHEMA, 'fingerprint': self.problem.fingerprint,
                'target': self.target, 'rejected_boards': [list(codes) for codes in self.rejected],
                'hints': list(self.last_codes) if self.last_codes is not None else None,
                'resume_note': 'Native learned clauses and search stacks restart; revalidated Gold cuts are retained.'}

    def emit(self, phase, **extra):
        now = time.monotonic()
        self.last_progress = now
        if phase == 'model_ready':
            self.stats['model_seconds'] = now-self.started
        if self.progress is not None:
            event = {'phase': phase, 'engine': self.engine, 'nodes': self.nodes,
                     'elapsed_seconds': now-self.started, 'stats': dict(self.stats),
                     'candidates_checked': self.stats['candidates_checked'], **extra}
            if phase in ('model_ready', 'finished') or now-self.last_checkpoint >= 5:
                event['checkpoint'] = self.checkpoint()
                self.last_checkpoint = now
            self.progress(event)

    def finish(self, status, codes=None, validation=None):
        result = {'status': status, 'codes': codes, 'validation': validation,
                  'elapsed_seconds': time.monotonic()-self.started, 'nodes': int(self.nodes),
                  'complete': status in ('solved', 'edge_perfect', 'infeasible'),
                  'stats': dict(self.stats),
                  'checkpoint': self.checkpoint()}
        self.emit('finished', status=status)
        return result

    def validate_edge_candidate(self, codes):
        if len(codes) != self.problem.n or any(code not in self.problem.domains[cell] for cell, code in enumerate(codes)):
            raise RuntimeError('Constraint solver returned a code outside the problem domains')
        if len({code//3 for code in codes}) != self.problem.n:
            raise RuntimeError('Constraint solver repeated a physical tile')
        report = self.problem.validate(codes)
        if report.get('matched_edges') != len(self.problem.edges):
            raise RuntimeError('Constraint solver returned a mismatched seam')
        # Open-board fixtures must not leak gold through an unpaired boundary.
        for cell, code in enumerate(codes):
            for side in range(3):
                if self.problem.neighbors[cell][side] < 0 and int(self.problem.masks[code][side]):
                    raise RuntimeError('Constraint solver returned a gold endpoint on an open boundary')
        return report

    def candidate(self, codes):
        self.check()
        report = self.validate_edge_candidate(codes)
        self.stats['candidates_checked'] += 1
        self.last_codes = list(codes)
        if report.get('valid'):
            return self.finish('solved', list(codes), report)
        if self.target == 'edges':
            return self.finish('edge_perfect', list(codes), report)
        cut = self.make_cut(codes)
        self.rejected.append(list(codes))
        self.cuts.append(cut)
        self.stats['gold_cuts'] += 1
        self.emit('gold_cut', codes=list(codes), validation=report, cut_cells=len(cut))
        self.check()
        return None

    def make_cut(self, codes):
        cut = [(int(cell), int(code)) for cell, code in self.problem.gold_nogood(codes)]
        if not cut or len({cell for cell, _ in cut}) != len(cut):
            raise RuntimeError('Gold cut must name a nonempty set of distinct cells')
        if any(not 0 <= cell < self.problem.n or codes[cell] != code for cell, code in cut):
            raise RuntimeError('Gold cut does not match its independently checked witness')
        return cut


@contextmanager
def _watch(run, interrupt):
    """Interrupt a native call; lifetime ends before its native solver is freed."""
    done = threading.Event()
    def monitor():
        while not done.wait(.01):
            try:
                cancelled = run.cancelled()
            except BaseException as exc:
                run.error = exc
                cancelled = True
            if cancelled:
                interrupt()
                return
    thread = threading.Thread(target=monitor, name='diamond-constraint-limit', daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join()


def _hints(problem, hints):
    if hints is None:
        return {}
    values = hints.items() if isinstance(hints, dict) else enumerate(hints)
    result = {}
    for cell, code in values:
        if type(cell) is int and 0 <= cell < problem.n and code is not None:
            code = int(code)
            if code in problem.domains[cell]:
                result[cell] = code
    return result


def _resume(run, resume):
    if resume is None:
        return
    if not isinstance(resume, dict) or resume.get('schema') != SCHEMA:
        raise ValueError('Unsupported constraint checkpoint schema')
    if resume.get('fingerprint') != run.problem.fingerprint or resume.get('target') != run.target:
        raise ValueError('Constraint checkpoint belongs to different inputs or objective')
    witnesses = resume.get('rejected_boards', [])
    if not isinstance(witnesses, list):
        raise ValueError('Malformed rejected-board checkpoint witnesses')
    for codes in witnesses:
        run.check()
        if not isinstance(codes, list) or any(type(code) is not int for code in codes):
            raise ValueError('Malformed rejected-board checkpoint witness')
        report = run.validate_edge_candidate(codes)
        if report.get('valid') or run.target != 'gold':
            raise ValueError('Checkpoint may not exclude a valid Gold solution')
        run.rejected.append(list(codes))
        run.cuts.append(run.make_cut(codes))
    run.stats['resumed'] = True
    run.stats['restored_gold_cuts'] = len(run.cuts)


def _domains(run):
    problem = run.problem
    domains = []
    for cell in range(problem.n):
        run.check()
        row = sorted(set(int(code) for code in problem.domains[cell]))
        if any(not 0 <= code < 3*problem.n for code in row):
            raise ValueError('Invalid orientation code in a cell domain')
        if cell in problem.fixed:
            row = [code for code in row if code == problem.fixed[cell]]
        row = [code for code in row if all(problem.neighbors[cell][side] >= 0 or int(problem.masks[code][side]) == 0 for side in range(3))]
        domains.append(row)
    return domains


def _edge_groups(problem, domains, edge):
    a, sa, b, sb = map(int, edge)
    left, right = defaultdict(list), defaultdict(list)
    for code in domains[a]:
        left[int(problem.masks[code][sa])].append(code)
    for code in domains[b]:
        right[reverse_mask(problem.masks[code][sb])].append(code)
    return a, b, left, right


def _sat(run, domains, hints):
    try:
        from pysat.solvers import Glucose4
    except ImportError as exc:
        raise RuntimeError('SAT requires optional python-sat==1.9.dev15') from exc
    run.check()
    with Glucose4() as solver:
        variables, next_id, clause_count = [], 0, 0
        def fresh():
            nonlocal next_id
            next_id += 1
            return next_id
        def clause(literals):
            nonlocal clause_count
            solver.add_clause(literals)
            clause_count += 1
            if clause_count % 256 == 0:
                run.check()
        def exactly_one(literals):
            clause(literals)
            if len(literals) < 2:
                return
            previous = fresh()
            clause([-literals[0], previous])
            for literal in literals[1:-1]:
                current = fresh()
                clause([-literal, current])
                clause([-previous, current])
                clause([-literal, -previous])
                previous = current
            clause([-literals[-1], -previous])
        for row in domains:
            run.check()
            variables.append({code: fresh() for code in row})
        for row in variables:
            exactly_one(list(row.values()))
        by_tile = [[] for _ in range(run.problem.n)]
        for row in variables:
            for code, literal in row.items():
                by_tile[code//3].append(literal)
        for literals in by_tile:
            exactly_one(literals)
        # Shared seam-signature literals avoid a quadratic table of forbidden
        # orientation pairs. Each signature has support on both incident cells.
        for edge in run.problem.edges:
            run.check()
            a, b, left, right = _edge_groups(run.problem, domains, edge)
            common = left.keys() & right.keys()
            for mask in left.keys() - common:
                for code in left[mask]: clause([-variables[a][code]])
            for mask in right.keys() - common:
                for code in right[mask]: clause([-variables[b][code]])
            for mask in sorted(common):
                signature = fresh()
                for cell, codes in ((a, left[mask]), (b, right[mask])):
                    support = [variables[cell][code] for code in codes]
                    clause([-signature, *support])
                    for literal in support: clause([-literal, signature])
        def cut_clause(cut):
            # A cut containing an unavailable value is already satisfied.
            if all(code in variables[cell] for cell, code in cut):
                clause([-variables[cell][code] for cell, code in cut])
        for cut in run.cuts: cut_clause(cut)
        phases = [variables[cell][code] for cell, code in hints.items() if code in variables[cell]]
        if phases: solver.set_phases(phases)
        run.stats.update(sat_solver='Glucose4', sat_variables=next_id, sat_clauses=clause_count,
                         seed_effect='Glucose deterministic; seed does not randomize this encoding')
        run.emit('model_ready')
        while True:
            run.check()
            solver.clear_interrupt()
            solver.conf_budget(2000)
            with _watch(run, solver.interrupt):
                answer = solver.solve_limited(expect_interrupt=True)
            statistics = solver.accum_stats()
            run.nodes = int(statistics.get('decisions', 0))
            run.stats.update(sat_statistics=statistics, node_metric='SAT decisions')
            run.check()
            if answer is False:
                return run.finish('infeasible')
            if answer is None:
                continue
            selected = set(literal for literal in solver.get_model() if literal > 0)
            codes = [next(code for code, literal in row.items() if literal in selected) for row in variables]
            result = run.candidate(codes)
            if result is not None: return result
            cut_clause(run.cuts[-1])


def _cpsat(run, domains, hints):
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError('CP-SAT requires ortools==9.15.6755; run setup or install the common requirements') from exc
    run.check()
    model = cp_model.CpModel()
    codes, tiles = [], []
    for cell, domain in enumerate(domains):
        run.check()
        codes.append(model.new_int_var_from_domain(cp_model.Domain.from_values(domain), f'code_{cell}'))
        tiles.append(model.new_int_var(0, run.problem.n-1, f'tile_{cell}'))
        model.add_allowed_assignments([codes[-1], tiles[-1]], [(code, code//3) for code in domain])
    model.add_all_different(tiles)
    for edge_index, edge in enumerate(run.problem.edges):
        run.check()
        a, b, left, right = _edge_groups(run.problem, domains, edge)
        common = sorted(left.keys() & right.keys())
        if not common: return run.finish('infeasible')
        seam = model.new_int_var_from_domain(cp_model.Domain.from_values(common), f'seam_{edge_index}')
        model.add_allowed_assignments([codes[a], seam], [(code, mask) for mask in common for code in left[mask]])
        model.add_allowed_assignments([codes[b], seam], [(code, mask) for mask in common for code in right[mask]])
    def add_cut(cut):
        model.add_forbidden_assignments([codes[cell] for cell, _ in cut], [tuple(code for _, code in cut)])
    for cut in run.cuts: add_cut(cut)
    for cell, code in hints.items(): model.add_hint(codes[cell], code)
    model.add_decision_strategy(codes, cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MIN_VALUE)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = run.seed % 2147483647
    run.stats.update(node_metric='CP-SAT branches', threads=1)
    run.emit('model_ready')
    while True:
        run.check()
        solver.parameters.max_time_in_seconds = max(.000001, run.deadline-time.monotonic())
        with _watch(run, solver.stop_search):
            status = solver.solve(model)
        run.nodes += int(solver.num_branches)
        run.stats['last_native_status'] = solver.status_name(status)
        run.check()
        if status == cp_model.INFEASIBLE:
            return run.finish('infeasible')
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            result = run.candidate([int(solver.value(code)) for code in codes])
            if result is not None: return result
            add_cut(run.cuts[-1])
        elif status == cp_model.MODEL_INVALID:
            raise RuntimeError('CP-SAT rejected the constraint model: '+model.validate())
        else:
            return run.finish('timeout')


def _cp(run, domains, hints):
    try:
        from ortools.constraint_solver import pywrapcp
    except ImportError as exc:
        raise RuntimeError('CP requires ortools==9.15.6755; run setup or install the common requirements') from exc
    run.check()
    solver = pywrapcp.Solver('Diamond Gold CP')
    codes, tiles = [], []
    for cell, domain in enumerate(domains):
        run.check()
        codes.append(solver.IntVar(domain, f'code_{cell}'))
        tiles.append(solver.IntVar(0, run.problem.n-1, f'tile_{cell}'))
        solver.Add(solver.AllowedAssignments([codes[-1], tiles[-1]], [(code, code//3) for code in domain]))
    solver.Add(solver.AllDifferent(tiles))
    for edge_index, edge in enumerate(run.problem.edges):
        run.check()
        a, b, left, right = _edge_groups(run.problem, domains, edge)
        common = sorted(left.keys() & right.keys())
        if not common: return run.finish('infeasible')
        seam = solver.IntVar(common, f'seam_{edge_index}')
        solver.Add(solver.AllowedAssignments([codes[a], seam], [(code, mask) for mask in common for code in left[mask]]))
        solver.Add(solver.AllowedAssignments([codes[b], seam], [(code, mask) for mask in common for code in right[mask]]))
    def add_cut(cut):
        solver.Add(solver.Sum([solver.IsDifferentCstVar(codes[cell], code) for cell, code in cut]) >= 1)
    for cut in run.cuts: add_cut(cut)
    phase = solver.Phase(codes, solver.CHOOSE_MIN_SIZE_LOWEST_MIN, solver.ASSIGN_MIN_VALUE)
    if hints:
        assignment = solver.Assignment()
        for cell, code in hints.items():
            assignment.Add(codes[cell])
            assignment.SetValue(codes[cell], code)
        phase = solver.DecisionBuilderFromAssignment(assignment, phase, codes)
    run.stats.update(node_metric='CP branches', threads=1, seed_effect='Deterministic minimum-domain/minimum-value phase')
    run.emit('model_ready')
    while True:
        run.check()
        # CustomLimit records why it fired, unlike the opaque TimeLimit SWIG
        # handle, so a bounded false return can never be mistaken for UNSAT.
        custom = solver.CustomLimit(run.cancelled)
        solver.NewSearch(phase, [custom])
        try:
            found = solver.NextSolution()
            candidate = [int(code.Value()) for code in codes] if found else None
            run.nodes = int(solver.Branches())
        finally:
            solver.EndSearch()
        run.check()
        if not found:
            return run.finish('infeasible')
        result = run.candidate(candidate)
        if result is not None: return result
        add_cut(run.cuts[-1])


def solve(problem, *, engine='sat', seconds=10, seed=0, target='gold', hints=None,
          stop=None, progress=None, resume=None):
    """Find a validated witness or exhaust the encoded finite problem.

    ``seconds`` is a finite total budget, including model construction and cut
    validation; zero returns timeout immediately. Native calls use their own
    limits plus interruption/callback checks (not a process-kill deadline).
    Progress includes one-second heartbeats and five-second cut/hint snapshots;
    native branch counters update only after each native call returns.
    Hints never restrict domains. ``complete`` means a target witness or a native
    infeasibility proof with sound Gold cuts, never a timeout. Nodes have an
    engine-specific meaning named in stats and must not be compared as rates.
    """
    if engine not in ('sat', 'cp-sat', 'cp') or target not in ('gold', 'edges'):
        raise ValueError('Choose sat/cp-sat/cp and gold/edges')
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be a finite nonnegative budget')
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    if stop is not None and not callable(stop) and not callable(getattr(stop, 'is_set', None)):
        raise ValueError('stop must be a callback or an Event')
    run = _Run(problem, engine, seconds, seed, target, stop, progress)
    try:
        run.check()
        _resume(run, resume)
        if hints is None and resume is not None:
            hints = resume.get('hints')
        domains = _domains(run)
        if any(not row for row in domains): return run.finish('infeasible')
        run.emit('building_model')
        return {'sat': _sat, 'cp-sat': _cpsat, 'cp': _cp}[engine](run, domains, _hints(problem, hints))
    except _Interrupted:
        if run.error is not None: raise run.error
        return run.finish(run.reason or 'timeout')
