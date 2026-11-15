"""SAT encoding of signed side inventories and disjoint physical reference-loss charges."""
from math import inf, isnan
from threading import Timer

from r29_gold_phase import EndpointCuts
from r31_gold_connectivity import PhaseInterface
from validator import reverse_mask


class InventorySAT:
    """The first n variables are physical tile colors; its global complement is equivalent."""

    def __init__(self, problem, groups, required, outside, cap, *, native='g4', excess=False, additional_groups=(), projected=False, parity_only=False, mate_cost_pairs=None):
        from pysat.card import CardEnc, ITotalizer
        from pysat.formula import IDPool
        from pysat.solvers import Solver

        groups = [frozenset(g) for g in groups]
        additional_groups = [frozenset(g) for g in additional_groups]
        required = set(required)
        self.mate_cost_pairs = None if mate_cost_pairs is None else frozenset(mate_cost_pairs)
        priced_pairs = self.mate_cost_pairs or frozenset()
        covered = set()
        for group in groups:
            if covered.intersection(group):
                raise ValueError('Reference-loss groups must be pairwise disjoint.')
            covered.update(group)
        if (type(cap) is not int or cap < 0 or len(set(outside)) != len(outside)
                or set(outside).intersection(covered | required)):
            raise ValueError('Invalid loss cap or outside-error partition.')
        extra_bonds = set().union(*additional_groups)
        if not extra_bonds <= covered | set(outside) | required:
            raise ValueError('An additional departure group uses a bond outside the reference.')
        sides = EndpointCuts(problem).sides
        for a, b in set(outside) | covered | required:
            if not (0 <= a < b < len(sides) and a//3 != b//3 and reverse_mask(sides[a][0]) == sides[b][0]):
                raise ValueError('Expected a reciprocal physical seam on two different tiles.')

        self.native_backend = native
        self.n = problem.n
        ids = IDPool(start_from=problem.n+1)
        clauses = [[-1]]

        def exactly(literals, count):
            if not 0 <= count <= len(literals):
                clauses.append([])
            elif literals:
                clauses.extend(CardEnc.equals(lits=literals, bound=count, vpool=ids).clauses)

        def parity(literals, bit):
            if not literals:
                if bit:
                    clauses.append([])
                return
            aggregate = literals[0]
            for literal in literals[1:]:
                out = ids.id()
                clauses.extend([[aggregate,literal,-out], [-aggregate,-literal,-out],
                                [aggregate,-literal,out], [-aggregate,literal,out]])
                aggregate = out
            clauses.append([aggregate if bit else -aggregate])

        data = PhaseInterface(problem)
        # The final column describes Euler characteristic; this relaxation uses inventories only.
        for j, modulo_two in enumerate(data.mod[:-1]):
            constant = sum(low[j] for low, _ in data.rows)
            changes = [high[j]-low[j] for low, high in data.rows]
            if modulo_two or parity_only:
                parity([i+1 for i, weight in enumerate(changes) if weight % 2], constant % 2)
                continue
            count = -constant
            literals = []
            for i, weight in enumerate(changes):
                if not weight:
                    continue
                if weight < 0:
                    count -= weight
                lit = i+1 if weight > 0 else -i-1
                literals.append(lit)
                # Cardinality encoders take distinct variables. Duplicate weights use aliases.
                for _ in range(1, abs(weight)):
                    alias = ids.id()
                    clauses.extend([[-alias,lit], [alias,-lit]])
                    literals.append(alias)
            exactly(literals, count)

        def compatible(pair):
            a, b = pair
            x, y = a//3+1, b//3+1
            flip = sides[a][1] ^ sides[b][1] ^ (sides[a][0].bit_count() % 2)
            return ([x,y],[-x,-y]) if flip else ([x,-y],[-x,y])

        for pair in sorted(required):
            clauses.extend(compatible(pair))
        phase_errors = {}
        voluntary = set()
        if projected:
            from r29_gold_phase import SignedParity
            signed = SignedParity(problem)
            for pair in sorted(required):
                signed.bond(*pair)
            mark = len(signed.base.trail)
            for group in [*groups,*additional_groups]:
                signed.base.rollback(mark)
                for pair in sorted(group):
                    if not signed.bond(*pair):
                        break
                else:
                    voluntary.update(group)
            signed.base.rollback(mark)

        def error_id(pair):
            first, second = compatible(pair)
            error = ids.id()
            clauses.extend([first+[error], second+[error],
                            [first[0],-first[1],-error], [second[0],-second[1],-error]])
            phase_errors[pair] = error
            return error

        actual_losses = {}
        outside_charges = []
        errors = []
        for pair in outside:
            error = error_id(pair)
            errors.append(error)
            if pair in extra_bonds or pair in priced_pairs:
                # A compatible outside reference seam may still be dropped to
                # satisfy an overlapping certified group. Count it just once.
                charge = ids.id()
                clauses.append([-error,charge])
                actual_losses[pair] = charge
            else:
                charge = error
            outside_charges.append(charge)
            actual_losses[pair]=charge
        losses = []
        group_losses = []
        for group in groups:
            local = []
            for pair in sorted(group-required):
                if projected:
                    error = error_id(pair)
                    if pair in voluntary or pair in priced_pairs:
                        loss = ids.id()
                        clauses.append([-error,loss])
                    else:
                        loss = error
                else:
                    loss = ids.id()
                    for clause in compatible(pair):
                        clauses.append(clause+[loss])
                local.append(loss)
                actual_losses[pair] = loss
            clauses.append(local)
            losses.extend(local)
            group_losses.append(local)
        for group in additional_groups:
            clauses.append([actual_losses[pair] for pair in sorted(group-required)])
        if excess:
            spare = cap-len(groups)
            if spare < 0:
                clauses.append([])
            elif spare < len(errors)+len(losses)-len(groups):
                charges = list(outside_charges)
                for local in group_losses:
                    if len(local) < 2:
                        continue
                    # Every group already needs one missing bond. The outputs
                    # for at least 2, 3, ... losses charge each additional bond.
                    with ITotalizer(lits=local, ubound=spare+1, top_id=ids.top) as tree:
                        clauses.extend(tree.cnf.clauses)
                        ids.top = tree.top_id
                        charges.extend(tree.rhs[1:])
                if spare < len(charges):
                    clauses.extend(CardEnc.atmost(lits=charges, bound=spare, vpool=ids).clauses)
        elif cap < len(errors)+len(losses):
            clauses.extend(CardEnc.atmost(lits=outside_charges+losses, bound=cap, vpool=ids).clauses)
        if projected:
            from r32_gold_syndrome import eliminate_phases
            base={i:(row,rhs) for i,(row,rhs,_) in signed.base.pivots.items()}
            if signed.base.conflict is not None:
                clauses.append([])
            pairs=sorted(phase_errors)
            eq=[((1<<(a//3))^(1<<(b//3)),sides[a][1]^sides[b][1]^(sides[a][0].bit_count()%2)) for a,b in pairs]
            _,checks=eliminate_phases(base,eq)
            for row in checks:
                if (row>>1).bit_count()<=16:
                    parity([phase_errors[pair] for i,pair in enumerate(pairs) if row>>(i+1)&1],row&1)
        self.physical_losses = actual_losses
        self.charge_literals = tuple(outside_charges+losses)
        self.clauses = clauses
        self.solver = Solver(name=native, bootstrap_with=clauses) if native else None
        self.errors = errors
        self.clause_count = len(clauses)

    def solve(self, mask, value, budget=100000):
        """Conflict-budget comparator. Native solvers may overshoot this soft limit."""
        self.solver.conf_budget(budget)
        return self.solver.solve_limited(assumptions=[
            v if value>>i & 1 else -v for i, v in enumerate(self.errors) if mask>>i & 1])

    def decide(self, seconds, assumptions=()):
        """Spend a bounded native search on this exact scope. An interruption returns None."""
        if isnan(seconds) or seconds < 0:
            raise ValueError('Expected nonnegative search duration.')
        if not seconds or seconds == inf:
            return self.solver.solve(assumptions=assumptions)
        if self.native_backend in ('cadical153','cadical195'):
            # These bindings honor conflict budgets and do not implement
            # asynchronous interruption. Retain learned clauses between slices.
            from time import monotonic
            deadline=monotonic()+seconds
            while monotonic()<deadline:
                self.solver.conf_budget(128)
                answer=self.solver.solve_limited(assumptions=assumptions)
                if answer is not None:return answer
            return None
        timer = Timer(seconds, self.solver.interrupt)
        timer.daemon = True
        timer.start()
        try:
            return self.solver.solve_limited(assumptions=assumptions, expect_interrupt=True)
        finally:
            timer.cancel()
            timer.join()
            self.solver.clear_interrupt()

    def phase(self):
        return sum(1 << (x-1) for x in self.solver.get_model() if 0 < x <= self.n)

    def close(self):
        if self.solver is not None:
            self.solver.delete()
