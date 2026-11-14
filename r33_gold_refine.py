"""Integer signed-inventory refinement for an outside phase-error pattern."""
from math import isnan
from time import monotonic

from r28_geo_color import signed_phase_model
from r29_gold_phase import EndpointCuts
from validator import reverse_mask


class IntegerPatternFilter:
    """Test an exact outside error set under a physical-loss cap. Euler is omitted."""

    def __init__(self, problem, reference, packed_groups, required, outside, cap):
        try:
            from ortools.sat.python import cp_model
        except ImportError as exc:
            raise RuntimeError("Integer loss refinement requires OR-Tools in the full Python profile.") from exc

        reference = set(reference)
        required = set(required)
        groups = [set(group) for group in packed_groups]
        grouped = set().union(*groups)
        outside = tuple(outside)
        if type(cap) is not int or cap < 0 or not grouped <= reference:
            raise ValueError("Invalid reference loss budget or bond groups.")
        if len(set(outside)) != len(outside) or set(outside) != reference - grouped - required:
            raise ValueError("Outside bonds must cover exactly the ungrouped, nonmandatory reference bonds.")

        sides = EndpointCuts(problem).sides
        for a, b in reference | required:
            if not (0 <= a < b < len(sides) and a//3 != b//3 and reverse_mask(sides[a][0]) == sides[b][0]):
                raise ValueError("Expected a reciprocal between-tile physical bond.")

        self.model, phases = signed_phase_model(problem, euler=False, pins={}, bonds=sorted(required))
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

    def __call__(self, error_indices_set, deadline_monotonic):
        errors = frozenset(error_indices_set)
        if any(type(i) is not int or not 0 <= i < len(self.errors) for i in errors):
            raise ValueError("Outside error index out of range.")
        if isnan(deadline_monotonic):
            raise ValueError("Invalid monotonic deadline.")
        assumptions = [literal if i in errors else ~literal for i, literal in enumerate(self.errors)]
        self.model.add_assumptions(assumptions)
        try:
            remaining = deadline_monotonic - monotonic()
            if remaining <= 0:
                raise TimeoutError("Integer pattern search reached its deadline.")
            self.solver.parameters.max_time_in_seconds = remaining
            outcome = self.solver.solve(self.model)
        finally:
            self.model.clear_assumptions()
        if outcome == self.cp.INFEASIBLE:
            return False
        if outcome in (self.cp.FEASIBLE, self.cp.OPTIMAL):
            return True
        if outcome == self.cp.UNKNOWN:
            raise TimeoutError("Integer pattern search returned UNKNOWN.")
        raise RuntimeError(f"Integer pattern search failed with status {outcome}.")
