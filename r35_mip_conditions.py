"""Apply necessary spherical and physical side conditions to signed integer search."""
from collections import Counter
from functools import reduce
from math import gcd


def euler_row(engine, phase, interface):
    """Enforce exactly the reachable spherical totals in native linear rows."""
    base = sum(low[-1] for low, _ in interface.rows)
    differences = [high[-1] - low[-1] for low, high in interface.rows]
    scale = reduce(gcd, differences, 0) or 1
    allowed = sorted((value - base) // scale for value in interface.euler if (value - base) % scale == 0)
    weight = sum((difference // scale) * flag for difference, flag in zip(differences, phase) if difference)
    if not allowed:
        engine.Add(False)
    elif len(allowed) == 1:
        engine.Add(weight == allowed[0])
    elif allowed[-1] - allowed[0] + 1 == len(allowed):
        engine.Add(weight >= allowed[0])
        engine.Add(weight <= allowed[-1])
    else:
        selected = [engine.BoolVar('') for _ in allowed]
        engine.Add(sum(selected) == 1)
        engine.Add(weight == sum(value * flag for value, flag in zip(allowed, selected)))


def _violates(clause, value):
    return all(((value >> (abs(literal) - 1)) & 1) != (literal > 0) for literal in clause)


class Conditions:
    def __init__(self, rules, engine, phase, losses, *, startup='warm'):
        self.rules = rules
        self.engine = engine
        self.phase = phase
        self.posted = set()
        self.work = Counter()
        rules._prepare()
        if rules.euler is not None:
            euler_row(engine, phase, rules.euler)
            self.work['native_euler'] = 1
        initial = rules.initial
        if startup in ('short', 'warm'):
            initial = {clause for clause in initial if len(clause) <= 5}
        elif startup == 'lazy':
            initial = set()
        if startup == 'warm' and rules.mates:
            # Seeds only nominate assignments. Recheck the actual reciprocal
            # family before using either polarity of a partial matching conflict.
            from r35_loss_bound import REFERENCE_PHASES
            for candidate in REFERENCE_PHASES:
                if candidate < 1 << len(phase):
                    for index, (_, _, _, mask) in enumerate(rules.families):
                        rules.cached_match(index, candidate & mask)
        self.post(initial | rules.rejecting)

    def post(self, clauses):
        count = 0
        for clause in sorted(clauses, key=lambda row: (len(row), row)):
            # Tile zero is fixed clear in the physical integer model.
            if -1 in clause:
                continue
            reduced = frozenset(literal for literal in clause if literal != 1)
            if reduced in self.posted or any(previous < reduced for previous in self.posted):
                continue
            self.engine.Add(sum(self.phase[v - 1] if v > 0 else 1 - self.phase[-v - 1]
                                for v in sorted(reduced)) >= 1)
            self.posted.add(reduced)
            count += 1
        self.work['native_signed_clauses'] += count
        return count

    def reject(self, value, cost, deadline):
        rules = self.rules
        if rules.accepts(value):
            if not rules.mate_cost or rules.price(value, cost) <= rules._scope[3]:
                return False
        known = rules.rejecting | rules.initial
        violated = [clause for clause in known if _violates(clause, value)]
        if not violated and rules.mate_cost and rules.price(value, cost) > rules._scope[3]:
            violated = rules._price_conflict(value, deadline)
        if not violated or not self.post(violated):
            raise ArithmeticError('Signed integer witness violates an already installed condition.')
        self.work['native_partial_rejections'] += 1
        return True
