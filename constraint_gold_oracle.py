"""Bounded direct-Gold CP-SAT assistance for narrowed exact DFS branches.

A conclusive INFEASIBLE answer rejects the submitted domain tuple and its subsets.
The model contains physical-tile, seam, boundary, and single-loop constraints;
its witnesses are independently validated. Unknown, construction limits, missing
OR-Tools, cancellation and local deadline expiry never prove infeasibility.
Caches are private, bounded and bound to exact problem structure and domains.
Subsumption checks only narrowing, never widening or a timed-out native query.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
import threading
import time

from constraint_circuit import build_circuit_model, CircuitLimit, CircuitModel

STAT_KEYS = ('gold_oracle_calls', 'gold_oracle_solved', 'gold_oracle_infeasible',
             'gold_oracle_unknown', 'gold_oracle_skips', 'gold_oracle_cache_hits',
             'gold_oracle_build_ns', 'gold_oracle_solve_ns', 'gold_oracle_native_branches',
             'gold_oracle_subsumption_hits', 'gold_oracle_contracted_ports',
             'gold_oracle_contracted_paths', 'gold_oracle_residual_placements',
             'gold_oracle_template_hits', 'gold_oracle_template_misses',
             'gold_oracle_template_evictions', 'gold_oracle_template_restrictions',
             'gold_oracle_template_bytes', 'gold_oracle_template_entries',
             'gold_oracle_template_clone_ns', 'gold_oracle_template_admission_ns',
             'gold_oracle_template_oversize', 'gold_oracle_proofs_subsumed')
GAUGE_KEYS = ('gold_oracle_template_bytes', 'gold_oracle_template_entries')


@dataclass(frozen=True)
class OracleResult:
    status: str
    codes: tuple | None = None
    validation: dict | None = None
    native_status: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class _CircuitTemplate:
    # Private model never passed to a solver or edited after admission.
    model: object
    code_indices: tuple
    stats: dict
    charge: int


class _LocalDeadline(Exception):
    pass


def _load_cp_model():
    from ortools.sat.python import cp_model
    return cp_model


def _codes(bits):
    while bits:
        bit = bits & -bits
        yield bit.bit_length()-1
        bits ^= bit


class GoldOracle:
    """Private helper reused only for exactly the same immutable puzzle.

    ``initial`` and queried domains are Python integer orientation bitsets.
    ``begin_slice`` resets the call/cooldown budget at the start of one bounded
    DFS solve invocation; it preserves conclusive proofs and immutable model
    templates. Templates are encodings only, not proofs or native solver state.
    ``should_query`` is optional scheduling guidance. ``query`` enforces the
    same gates itself, so callers cannot accidentally issue unbounded probes.
    """
    def __init__(self, problem, initial, target='gold', *, max_attempts=4,
                 max_open=24, max_placements=1024, cache_limit=64, contract_fixed=True,
                 template_limit=4, template_byte_limit=8*1024*1024):
        if type(contract_fixed) is not bool:
            raise ValueError('contract_fixed must be a boolean.')
        self.contract_fixed = contract_fixed
        if target not in ('gold', 'edges'):
            raise ValueError('Gold oracle target must be gold or edges.')
        for name, value in (('max_attempts', max_attempts), ('max_open', max_open),
                            ('max_placements', max_placements), ('cache_limit', cache_limit)):
            if type(value) is not int or value < 1:
                raise ValueError(name+' must be a positive integer.')
        for name, value in (('template_limit', template_limit),
                            ('template_byte_limit', template_byte_limit)):
            if type(value) is not int or value < 0:
                raise ValueError(name+' must be a nonnegative integer.')
        self.template_limit, self.template_byte_limit = template_limit, template_byte_limit
        self.templates = OrderedDict()
        self.template_bytes = 0
        self.problem, self.initial, self.target = problem, tuple(initial), target
        self.max_attempts, self.max_open = max_attempts, max_open
        self.max_placements, self.cache_limit = max_placements, cache_limit
        allowed = tuple(sum(1 << code for code in set(row)) for row in problem.domains)
        if (len(self.initial) != problem.n or
            any(type(value) is not int or value < 0 or value & ~mask
                for value, mask in zip(self.initial, allowed))):
            raise ValueError('Gold oracle initial domains must be legal orientation bitsets.')
        self.binding = self._binding()
        self.cache = OrderedDict()
        self.stats = dict.fromkeys(STAT_KEYS, 0)
        self.unavailable = False
        self.begin_slice()

    def _binding(self):
        problem = self.problem
        # Compare actual contents, not a saved fingerprint alone: a caller may
        # have mutated a SearchProblem without recomputing its fingerprint.
        return (self.target, self.initial, self.contract_fixed, problem.n,
                tuple(tuple(tuple(tuple(point) for point in segment)
                            for segment in tile['segments']) for tile in problem.tiles),
                tuple(tuple(edge) for edge in problem.edges),
                len(problem.board.cells), tuple(tuple(edge) for edge in problem.board.edges),
                tuple(tuple(int(value) for value in row) for row in problem.neighbors),
                tuple(tuple(int(value) for value in row) for row in problem.masks),
                tuple(tuple(row) for row in problem.domains),
                tuple(sorted(problem.fixed.items())))

    def begin_slice(self):
        self.attempts = 0
        self.next_node = 0
        self.misses = 0

    def _domain_key(self, domains):
        key = tuple(domains)
        if (len(key) != self.problem.n or
            any(type(value) is not int or value < 0 or value & ~initial
                for value, initial in zip(key, self.initial))):
            raise ValueError('Gold oracle domains must be subsets of its initial bitsets.')
        return key

    def _eligible(self, key, node):
        return (self.target == 'gold' and not self.unavailable and
                self.attempts < self.max_attempts and node >= self.next_node and
                sum(value.bit_count() > 1 for value in key) <= self.max_open and
                sum(value.bit_count() for value in key) <= self.max_placements)

    def _cached_proof(self, key, check=lambda: None):
        if self.target != 'gold':
            return None
        if key in self.cache:
            return key
        # Every query is already a subset of initial. A proof only needs to
        # inspect coordinates that were narrower than that immutable root.
        for proven, restricted in reversed(self.cache.items()):
            check()
            if all(not key[cell] & ~mask for cell, mask in restricted):
                return proven
        return None

    def _template_model(self, key, check):
        """Clone an exact ancestor encoding; only the private clone is narrowed.

        Residual contraction is an equivalence for all ancestor assignments.
        Keeping that topology while forbidding more code values therefore
        retains exactly the requested subset, even when more tiles are fixed.
        No native learned clauses or solver search state are retained.
        """
        if not self.template_limit or not self.template_byte_limit:
            return None
        candidates = []
        for source, template in self.templates.items():
            check()
            if all(not child & ~parent for child, parent in zip(key, source)):
                candidates.append((template.charge, source, template))
        if not candidates:
            self.stats['gold_oracle_template_misses'] += 1
            return None
        _, source, template = min(candidates, key=lambda item: item[0])
        started = time.perf_counter_ns()
        try:
            check()
            model = template.model.clone()
            check()
            codes = [model.get_int_var_from_proto_index(index) for index in template.code_indices]
            cp_model = _load_cp_model()
            restrictions = 0
            for cell, (child, parent) in enumerate(zip(key, source)):
                check()
                if child != parent:
                    # A single code-domain constraint replaces one constraint
                    # per forbidden placement, including contracted constants.
                    model.add_linear_expression_in_domain(codes[cell], cp_model.Domain.from_values(tuple(_codes(child))))
                    restrictions += 1
            check()
        finally:
            self.stats['gold_oracle_template_clone_ns'] += max(0, time.perf_counter_ns()-started)
        self.templates.move_to_end(source)
        self.stats['gold_oracle_template_hits'] += 1
        self.stats['gold_oracle_template_restrictions'] += restrictions
        stats = dict(template.stats)
        stats['circuit_constraints'] = len(model.proto.constraints)
        return CircuitModel(model, codes, [], stats)

    def _remember_template(self, key, built, check):
        if not self.template_limit or not self.template_byte_limit:
            return
        started = time.perf_counter_ns()
        try:
            check()
            # OR-Tools 9.15's Python model proto has no binary ByteSize API.
            # Budget its UTF-8 proto representation plus domain/index metadata;
            # this is a reproducible representation cap, not a process-RSS cap.
            charge = (len(str(built.model.proto).encode('utf-8')) +
                      sum((mask.bit_length()+7)//8+32 for mask in key) + 8*len(built.codes))
            check()
            if charge > self.template_byte_limit:
                self.stats['gold_oracle_template_oversize'] += 1
                return
            model = built.model.clone()
            check()
            if self._binding() != self.binding:
                raise ValueError('Gold oracle problem structure changed during template construction.')
            template = _CircuitTemplate(model, tuple(code.index for code in built.codes),
                                        dict(built.stats), charge)
            # Nothing is admitted until all fallible/cancellable work finished.
            while (len(self.templates) >= self.template_limit or
                   self.template_bytes+charge > self.template_byte_limit):
                _, removed = self.templates.popitem(last=False)
                self.template_bytes -= removed.charge
                self.stats['gold_oracle_template_evictions'] += 1
            self.templates[key] = template
            self.template_bytes += charge
            self.stats['gold_oracle_template_bytes'] = self.template_bytes
            self.stats['gold_oracle_template_entries'] = len(self.templates)
        finally:
            self.stats['gold_oracle_template_admission_ns'] += max(0, time.perf_counter_ns()-started)

    def _remember_proof(self, key, check):
        # A broader conclusive proof makes its narrower cached proofs redundant.
        # Compute removals before mutation so interruption cannot partially alter
        # the cache or turn unfinished work into evidence of infeasibility.
        covered = []
        for proven in self.cache:
            check()
            if all(not old & ~new for old, new in zip(proven, key)):
                covered.append(proven)
        check()
        if self._binding() != self.binding:
            raise ValueError('Gold oracle problem changed during proof admission.')
        for proven in covered:
            del self.cache[proven]
        self.stats['gold_oracle_proofs_subsumed'] += len(covered)
        self.cache[key] = tuple((cell, mask) for cell, mask in enumerate(key)
                                if mask != self.initial[cell])
        while len(self.cache) > self.cache_limit:
            self.cache.popitem(last=False)

    def should_query(self, domains, *, node=0):
        if type(node) is not int or node < 0:
            raise ValueError('Gold oracle node count must be a nonnegative integer.')
        key = self._domain_key(domains)
        return self._eligible(key, node) or self._cached_proof(key) is not None

    def query(self, domains, check=lambda: None, *, seconds=.05, node=0):
        if (type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0):
            raise ValueError('Gold oracle seconds must be finite and nonnegative.')
        if type(node) is not int or node < 0:
            raise ValueError('Gold oracle node count must be a nonnegative integer.')
        started = time.monotonic()
        deadline = started + seconds
        key = self._domain_key(domains)
        check()
        if self._binding() != self.binding:
            raise ValueError('Gold oracle problem structure changed; create a fresh oracle.')
        proven = self._cached_proof(key, check)
        if proven is not None:
            if self._binding() != self.binding:
                raise ValueError('Gold oracle problem structure changed; create a fresh oracle.')
            self.cache.move_to_end(proven)
            self.stats['gold_oracle_cache_hits'] += 1
            if proven != key:
                self.stats['gold_oracle_subsumption_hits'] += 1
            return OracleResult('infeasible', reason='exact_domain_cache' if proven == key else 'narrower_domain_cache')
        if not self._eligible(key, node):
            self.stats['gold_oracle_skips'] += 1
            return OracleResult('skip', reason='trigger_or_attempt_limit')
        self.attempts += 1
        self.stats['gold_oracle_calls'] += 1

        def bounded_check():
            check()
            if time.monotonic() >= deadline:
                raise _LocalDeadline

        def finish(status, **extra):
            if status == 'unknown' or status == 'skip':
                self.misses = min(5, self.misses + 1)
            else:
                self.misses = 0
            self.next_node = node + 32 * (4 ** self.misses)
            self.stats['gold_oracle_'+('skips' if status == 'skip' else status)] += 1
            return OracleResult(status, **extra)

        try:
            bounded_check()
            try:
                cp_model = _load_cp_model()
            except ImportError:
                self.unavailable = True
                return finish('skip', reason='ortools_unavailable')
            bounded_check()
            began_build = time.perf_counter_ns()
            try:
                built = self._template_model(key, bounded_check)
                if built is None:
                    built = build_circuit_model(self.problem, [tuple(_codes(value)) for value in key],
                                                check=bounded_check,
                                                max_placements=self.max_placements,
                                                contract_fixed=self.contract_fixed)
                    self._remember_template(key, built, bounded_check)
            except CircuitLimit as exc:
                return finish('skip', reason=str(exc))
            finally:
                self.stats['gold_oracle_build_ns'] += max(0, time.perf_counter_ns()-began_build)
            for field in ('contracted_ports', 'contracted_paths', 'residual_placements'):
                self.stats['gold_oracle_'+field] += int(built.stats.get('circuit_'+field, 0))
            bounded_check()
            solver = cp_model.CpSolver()
            solver.parameters.num_search_workers = 1
            solver.parameters.random_seed = 0
            solver.parameters.max_time_in_seconds = max(.000001, deadline-time.monotonic())
            done, interrupted = threading.Event(), []

            def monitor():
                while not done.wait(.002):
                    try:
                        bounded_check()
                    except BaseException as exc:
                        interrupted.append(exc)
                        solver.stop_search()
                        return

            watcher = threading.Thread(target=monitor, name='diamond-gold-oracle-stop', daemon=True)
            began_solve = time.perf_counter_ns()
            watcher.start()
            try:
                status = solver.solve(built.model)
            finally:
                done.set()
                watcher.join()
                self.stats['gold_oracle_solve_ns'] += max(0, time.perf_counter_ns()-began_solve)
            if interrupted:
                raise interrupted[0]
            bounded_check()
            if self._binding() != self.binding:
                raise ValueError('Gold oracle problem changed during its native query.')
            self.stats['gold_oracle_native_branches'] += int(solver.num_branches)
            native = solver.status_name(status)
            if status == cp_model.MODEL_INVALID:
                raise RuntimeError('Gold subtree circuit model is invalid: '+built.model.validate())
            if status == cp_model.INFEASIBLE:
                self._remember_proof(key, bounded_check)
                return finish('infeasible', native_status=native)
            if status in (cp_model.FEASIBLE, cp_model.OPTIMAL):
                codes = tuple(int(solver.value(variable)) for variable in built.codes)
                if (len(codes) != self.problem.n or
                    any(not 0 <= code < 3*self.problem.n or not key[cell] & (1 << code)
                        for cell, code in enumerate(codes)) or
                    len({code//3 for code in codes}) != self.problem.n):
                    raise RuntimeError('Gold oracle witness violates submitted domains or tile inventory.')
                validation = self.problem.validate(list(codes))
                if not validation['valid']:
                    raise RuntimeError('Gold oracle witness failed independent single-loop validation.')
                bounded_check()
                return finish('solved', codes=codes, validation=validation, native_status=native)
            return finish('unknown', native_status=native, reason='native_inconclusive')
        except _LocalDeadline:
            return finish('unknown', reason='local_deadline')
