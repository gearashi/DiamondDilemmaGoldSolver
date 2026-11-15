"""Integer signed-inventory refinement for an outside phase-error pattern."""
from collections import Counter
import json
from math import isnan
from pathlib import Path
from time import monotonic

from r28_geo_color import signed_phase_model
from r29_gold_phase import EndpointCuts
from validator import reverse_mask


def _published_core(problem, outside):
    """Read an optional candidate. Only independent model replay can validate it."""
    try:
        path = Path(__file__).parent/"docs/R33_HIT_CORE.json"
        record = json.loads(path.read_text(encoding="utf8"))
        if record["fingerprint"] != problem.fingerprint:
            return None
        agreements = set(map(tuple,record["agree"]))
        disagreements = set(map(tuple,record["disagree"]))
    except (OSError, KeyError, ValueError, TypeError):
        return None
    bonds = agreements | disagreements
    if agreements & disagreements or not bonds <= set(outside):
        return None
    positions = {pair:1<<i for i,pair in enumerate(outside)}
    return sum(positions[b] for b in bonds), sum(positions[b] for b in disagreements)


class IntegerCoreStore:
    """Store candidate cores in the same transaction as their loss page."""
    def __init__(self, db, scope):
        self.db = db
        self.scope = scope
        db.execute("CREATE TABLE IF NOT EXISTS r34_integer_cores(scope TEXT, mask TEXT, value TEXT, "
                   "PRIMARY KEY(scope,mask,value))")

    def read(self):
        for mask, value in self.db.execute("SELECT mask,value FROM r34_integer_cores WHERE scope=?",
                                           (self.scope,)):
            try:
                yield int(mask,16), int(value,16)
            except (TypeError, ValueError):
                continue

    def remember(self, mask, value):
        self.db.execute("INSERT OR IGNORE INTO r34_integer_cores VALUES (?,?,?)",
                        (self.scope,format(mask,'x'),format(value,'x')))


def _validate_partition(problem, reference, packed_groups, required, outside, cap):
    reference = set(reference)
    required = set(required)
    groups = [set(group) for group in packed_groups]
    grouped = set().union(*groups)
    if sum(map(len, groups)) != len(grouped):
        raise ValueError("Reference loss groups must be disjoint.")
    outside = tuple(outside)
    if type(cap) is not int or cap < 0 or not grouped <= reference:
        raise ValueError("Invalid reference loss budget or bond groups.")
    if len(set(outside)) != len(outside) or set(outside) != reference - grouped - required:
        raise ValueError("Outside bonds must cover exactly the ungrouped, nonmandatory reference bonds.")

    sides = EndpointCuts(problem).sides
    for a, b in reference | required:
        if not (0 <= a < b < len(sides) and a//3 != b//3 and reverse_mask(sides[a][0]) == sides[b][0]):
            raise ValueError("Expected a reciprocal between-tile physical bond.")

    return required, groups, grouped, outside, sides


class IntegerPatternFilter:
    """Test an exact outside error set under a physical-loss cap. Euler is omitted."""

    def __init__(self, problem, reference, packed_groups, required, outside, cap, *, store=None, work=None):
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError("Integer loss refinement requires OR-Tools in the full Python profile.") from exc

        required, groups, grouped, outside, sides = _validate_partition(
            problem, reference, packed_groups, required, outside, cap)

        self.model, phases = signed_phase_model(problem, euler=False, pins={}, bonds=sorted(required))
        self.phases = phases
        self.last_witness = None
        self.solver = cp_model.CpSolver()
        self.solver.parameters.num_search_workers = 1
        self.cp = cp_model

        def phase_requirement(pair):
            a, b = pair
            mask, color = sides[a]
            return phases[a//3], phases[b//3], color ^ sides[b][1] ^ (mask.bit_count() & 1)

        errors = []
        for pair in outside:
            u, v, flip = phase_requirement(pair)
            error = self.model.new_bool_var("outside_phase_error")
            self.model.add_bool_xor([u, v, error if flip else ~error])
            errors.append(error)
        self.errors = tuple(errors)
        self._var_to_bit = {v.index:1<<i for i,v in enumerate(errors)}
        self.store = store
        self.work = Counter() if work is None else work
        self.verified = set()
        self.pending = set()
        self.published = _published_core(problem,outside)
        self._public_problem = problem
        self._public_outside = outside
        self._public_sides = sides
        full = (1 << len(errors))-1
        if store is not None:
            self.pending.update((mask,value) for mask,value in store.read()
                                if 0 <= mask <= full and value >= 0 and (value & mask) == value)
        if self.published is not None:
            self.pending.add(self.published)

        # An outside mismatch costs one physical loss. A compatible outside bond
        # can still be lost; choosing that extra loss can never help a group.
        # Inside a group, a loss can satisfy its required departure even when
        # the tile phases agree. Only bonds kept there imply phase compatibility.
        possible_loss = {}
        for pair in sorted(grouped - required):
            u, v, flip = phase_requirement(pair)
            loss = self.model.new_bool_var("possible_reference_loss")
            self.model.add(u != v if flip else u == v).only_enforce_if(~loss)
            possible_loss[pair] = loss
        for group in groups:
            self.model.add(sum(possible_loss[b] for b in group - required) >= 1)
        self.model.add(sum(errors) + sum(possible_loss.values()) <= cap)

    def _count(self, name):
        self.work[name] = self.work.get(name,0)+1

    def _decide(self, mask, value, deadline):
        assumptions = [literal if value & (1<<i) else ~literal
                       for i,literal in enumerate(self.errors) if mask & (1<<i)]
        self.model.add_assumptions(assumptions)
        try:
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError("Integer pattern search reached its deadline.")
            self.solver.parameters.max_time_in_seconds = remaining
            self._count("integer_native_calls")
            return self.solver.solve(self.model)
        finally:
            self.model.clear_assumptions()

    def _learn(self, mask, value):
        # CP-SAT returns a sufficient UNSAT subset of the supplied assumptions.
        keep = flips = 0
        for literal in self.solver.sufficient_assumptions_for_infeasibility():
            bit = self._var_to_bit[literal if literal>=0 else -literal-1]
            if not bit & mask or bool(bit & value) != (literal>=0):
                raise ArithmeticError("The integer solver returned a foreign core literal.")
            keep |= bit
            if literal>=0:
                flips |= bit
        if any((a & keep)==a and (flips & a)==b for a,b in self.verified):
            return
        self.verified = {(a,b) for a,b in self.verified if (a & keep)!=keep or (b & keep)!=flips}
        self.verified.add((keep,flips))
        self._count("integer_cores_learned")
        if self.store is not None:
            self.store.remember(keep,flips)

    def _check_hints(self, value, deadline):
        for mask, flips in sorted(self.pending):
            if value & mask != flips:
                continue
            self.pending.discard((mask,flips))
            self._count("integer_core_replays")
            if (mask,flips) == self.published:
                # This particular core was proved without any loss-budget
                # variables. Replaying the original small model costs less.
                model, phases = signed_phase_model(self._public_problem,euler=False,pins={})
                for i,(a,b) in enumerate(self._public_outside):
                    if not (mask >> i) & 1:
                        continue
                    side, color = self._public_sides[a]
                    different = color ^ self._public_sides[b][1] ^ (side.bit_count() & 1) ^ ((flips>>i)&1)
                    model.add(phases[a//3] != phases[b//3] if different else phases[a//3] == phases[b//3])
                left = min(1.0,deadline-monotonic())
                if left <= 0:
                    raise TimeoutError
                self.solver.parameters.max_time_in_seconds = left
                self._count("integer_native_calls")
                outcome = self.solver.solve(model)
                if outcome == self.cp.INFEASIBLE:
                    self.verified.add((mask,flips))
                    return True
            else:
                outcome = self._decide(mask,flips,min(deadline,monotonic()+1.0))
                if outcome == self.cp.INFEASIBLE:
                    self._learn(mask,flips)
                    return True
            if outcome not in (self.cp.FEASIBLE,self.cp.OPTIMAL,self.cp.UNKNOWN):
                raise RuntimeError(f"Integer core replay failed with status {outcome}.")
        return False

    def excludes(self, indices, deadline=float("inf")):
        """Use verified cores, checking a matching dormant hint only once."""
        if not self.verified and not self.pending:
            return False
        value = sum(1 << i for i in indices)
        return any((value & mask)==flips for mask,flips in self.verified) or self._check_hints(value,deadline)

    def __call__(self, error_indices_set, deadline_monotonic):
        errors = frozenset(error_indices_set)
        if any(type(i) is not int or not 0 <= i < len(self.errors) for i in errors):
            raise ValueError("Outside error index out of range.")
        if isnan(deadline_monotonic):
            raise ValueError("Invalid monotonic deadline.")
        if deadline_monotonic <= monotonic():
            raise TimeoutError("Integer pattern search reached its deadline.")
        value = sum(1 << i for i in errors)
        if any((value & mask)==flips for mask,flips in self.verified):
            self._count("integer_native_calls_avoided")
            return False
        if self._check_hints(value,deadline_monotonic):
            return False
        mask = (1 << len(self.errors))-1
        outcome = self._decide(mask,value,deadline_monotonic)
        if outcome == self.cp.INFEASIBLE:
            self._learn(mask,value)
            return False
        if outcome in (self.cp.FEASIBLE, self.cp.OPTIMAL):
            self.last_witness = None, sum(self.solver.value(x)<<i for i,x in enumerate(self.phases))
            return True
        if outcome == self.cp.UNKNOWN:
            raise TimeoutError("Integer pattern search returned UNKNOWN.")
        raise RuntimeError(f"Integer pattern search failed with status {outcome}.")


def make_integer_filter(problem, reference, packed_groups, required, outside, cap, *, store=None, work=None):
    """Select the exact finite phase model where the outside equations leave a small rank."""
    from r35_integer_phases import EnumeratedPatternFilter, IntegerPhaseImages
    _validate_partition(problem, reference, packed_groups, required, outside, cap)
    image = IntegerPhaseImages.compile(problem, packed_groups, required, outside)
    if image is not None:
        return EnumeratedPatternFilter(image, outside, cap, store, work, _published_core(problem,outside))
    return IntegerPatternFilter(problem, reference, packed_groups, required, outside, cap, store=store, work=work)
