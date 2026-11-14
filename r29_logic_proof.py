"""Checked resolution DAGs and Craig boundary circuits for one or many partitions."""
from collections import Counter


class ProofLimit(Exception):
    pass


class Resolution:
    def __init__(self, groups, limit):
        self.rows = []
        self.sources = []
        self.limit = limit
        self.steps = 0
        for group_id, clauses in enumerate(groups):
            for clause in clauses:
                row = frozenset(clause)
                if any(type(x) is not int or x == 0 for x in row):
                    raise ValueError("Clause variables must be positive integer IDs with optional sign.")
                if not any(-x in row for x in row):
                    self.rows.append(row)
                    self.sources.append((group_id,))
        self.original = len(self.rows)
        self.known = {row: i for i, row in enumerate(self.rows)}

    def tick(self, n=1):
        self.steps += n
        if self.steps > self.limit:
            raise ProofLimit

    def resolve(self, a, b, pivot):
        if -pivot not in self.rows[b] or pivot not in self.rows[a]:
            raise ArithmeticError("Resolution pivot missing.")
        row = (self.rows[a] - {pivot}) | (self.rows[b] - {-pivot})
        if any(-x in row for x in row):
            return None
        if row not in self.known:
            self.known[row] = len(self.rows)
            self.rows.append(row)
            self.sources.append((a, b, abs(pivot)))
        return self.known[row]

    def rup(self, clause):
        trail = []
        chosen = {-x for x in clause}
        while True:
            progress = False
            for idx, row in enumerate(self.rows):
                self.tick()
                if row & chosen:
                    continue
                remaining = [x for x in row if -x not in chosen]
                if not remaining:
                    current = idx
                    for value, reason in reversed(trail):
                        if -value in self.rows[current]:
                            current = self.resolve(current, reason, -value)
                    if current is None or not self.rows[current] <= frozenset(clause):
                        raise ArithmeticError("Invalid unit-conflict reconstruction.")
                    return current
                if len(remaining) == 1:
                    value = remaining[0]
                    trail.append((value, idx))
                    chosen.add(value)
                    progress = True
            if not progress:
                return None

    def eliminate(self):
        pool = set(self.known.values())
        while pool and frozenset() not in self.known:
            positives, negatives = Counter(), Counter()
            for i in pool:
                self.tick()
                for x in self.rows[i]:
                    (positives if x > 0 else negatives)[abs(x)] += 1
            pivot = min(positives.keys() | negatives.keys(),
                        key=lambda v: (positives[v] * negatives[v], positives[v] + negatives[v], v))
            pos = [i for i in pool if pivot in self.rows[i]]
            neg = [i for i in pool if -pivot in self.rows[i]]
            pool.difference_update(pos, neg)
            for a in pos:
                for b in neg:
                    self.tick()
                    child = self.resolve(a, b, pivot)
                    if child is not None:
                        if not self.rows[child]:
                            return child
                        pool.add(child)
        return self.known.get(frozenset())

    def check(self, groups, root):
        if self.rows[root]:
            return False
        for i, (row, source) in enumerate(zip(self.rows, self.sources)):
            if len(source) == 1:
                if row not in groups[source[0]]:
                    return False
            else:
                a, b, pivot = source
                if max(a,b) >= i:
                    return False
                positive = pivot if pivot in self.rows[a] else -pivot
                if positive not in self.rows[a] or -positive not in self.rows[b]:
                    return False
                if row != (self.rows[a] - {positive}) | (self.rows[b] - {-positive}):
                    return False
        return True


def _interpolate(proof, root, groups, left):
    from pysat.solvers import Glucose4
    a, b = [], []
    for g, clauses in enumerate(groups):
        (a if g in left else b).extend(clauses)
    av = {abs(lit) for c in a for lit in c}
    bv = {abs(lit) for c in b for lit in c}
    shared = av & bv
    # Gates are private to the boundary circuit; signed input literals retain
    # physical IDs. True and False constants have dedicated Tseitin variables.
    truth = max(av | bv, default=0) + 1
    clauses = [[truth]]
    gates = {}
    definitions = []
    def gate(op, inputs):
        values = set(inputs)
        neutral = -truth if op == 'or' else truth
        if -neutral in values or any(-v in values for v in values):
            return -neutral
        values.discard(neutral)
        key = op, tuple(sorted(values))
        if not values:
            return neutral
        if len(values) == 1:
            return next(iter(values))
        if key not in gates:
            g = truth + 1 + len(gates)
            gates[key] = g
            definitions.append((g,op,key[1]))
            if op == 'or':
                clauses.extend([g,-v] for v in key[1])
                clauses.append([-g,*key[1]])
            else:
                clauses.extend([-g,v] for v in key[1])
                clauses.append([g,*(-v for v in key[1])])
        return gates[key]

    needed = {root}
    for i in range(root, -1, -1):
        if i in needed and len(proof.sources[i]) == 3:
            needed.update(proof.sources[i][:2])
    label = {}
    for i in sorted(needed):
        source = proof.sources[i]
        if len(source) == 1:
            label[i] = (gate('or', [x for x in proof.rows[i] if abs(x) in shared])
                        if source[0] in left else truth)
        else:
            l,r,pivot = source
            label[i] = gate('or' if pivot in av - bv else 'and', (label[l], label[r]))
    out = label[root]
    with Glucose4(bootstrap_with=[*a,*clauses]) as lhs, Glucose4(bootstrap_with=[*b,*clauses]) as rhs:
        if lhs.solve(assumptions=[-out]) or rhs.solve(assumptions=[out]):
            raise ArithmeticError("Resolution interpolant failed independent SAT checks.")
    return dict(shared=sorted(shared), circuit=clauses, output=out, truth=truth,
                definitions=definitions, gates=len(gates), proof_nodes=len(needed))


def proof_interpolants(groups, partitions, *, max_work=100000):
    """No boundary is returned without both semantic checks and a checked refutation."""
    from pysat.solvers import Glucose4
    groups = [[frozenset(row) for row in group] for group in groups]
    parts = [set(part) for part in partitions]
    if not groups or any(not s or not s < set(range(len(groups))) for s in parts):
        raise ValueError("Each requested boundary must separate two nonempty groups.")
    proof = Resolution(groups, max_work)
    with Glucose4() as solver:
        solver.append_formula(proof.rows)
        if solver.solve():
            return dict(status="joined", boundaries=[])
    method = "unit proof"
    try:
        root = proof.rup([])
        if root is None:
            from constraint_isolated_proof_round26 import provider
            method = "native RUP"
            trace = provider().produce_rup(proof.rows,seconds=.5) or []
            for line in trace:
                fields = line.split()
                if not fields or fields[0] == 'd':
                    continue
                if fields[-1] != '0':
                    break
                row = list(map(int,fields[:-1]))
                if any(-x in row for x in row):
                    continue
                reason = proof.rup(row)
                if reason is None:
                    break
                if not proof.rows[reason]:
                    root = reason
                    break
        if root is None:
            method = "bounded resolution elimination"
            root = proof.eliminate()
    except ProofLimit:
        return dict(status="unknown", work=proof.steps, boundaries=[])
    if root is None or not proof.check([set(group) for group in groups], root):
        raise ArithmeticError("Invalid resolution refutation.")
    boundaries = [_interpolate(proof,root,groups,side) for side in parts]
    return dict(status="separated", method=method, work=proof.steps,
                proof_builds=1, proof_rows=len(proof.rows), boundaries=boundaries)


def reuse_boundary(left, right, boundary):
    """Rebuild a total shared Boolean circuit, then check both sides under fresh gate IDs.

    A changed formula can remap the meaning of a placement variable. Both independent
    SAT checks are repeated on its current encoding before any join can be skipped.
    """
    from pysat.solvers import Glucose4
    lv = {abs(x) for clause in left for x in clause}
    rv = {abs(x) for clause in right for x in clause}
    common = lv & rv
    ids = {x:x for x in boundary['shared'] if type(x) is int and x > 0 and x in common}
    physical = set(boundary['shared'])
    truth = boundary['truth']
    if type(truth) is not int or truth<1 or truth in physical:
        raise ValueError("Invalid boundary constant.")
    free = max(lv | rv, default=0)+1
    ids[truth] = free
    clauses = [[free]]
    def translate(lit):
        if type(lit) is not int or not lit or abs(lit) not in ids:
            raise KeyError(lit)
        return ids[abs(lit)] * (1 if lit>0 else -1)
    try:
        for gate, op, operands in boundary['definitions']:
            if type(gate) is not int or gate in physical or gate in ids or gate < 1:
                raise ValueError("Invalid boundary gate.")
            args = [translate(x) for x in operands]
            free += 1
            ids[gate] = free
            if op == 'or':
                clauses.extend([free,-x] for x in args)
                clauses.append([-free,*args])
            elif op == 'and':
                clauses.extend([-free,x] for x in args)
                clauses.append([free,*(-x for x in args)])
            else:
                raise ValueError("Unknown boundary gate operation.")
        out = translate(boundary['output'])
    except KeyError:
        return False
    with Glucose4(bootstrap_with=[*left,*clauses]) as lhs, Glucose4(bootstrap_with=[*right,*clauses]) as rhs:
        return not lhs.solve(assumptions=[-out]) and not rhs.solve(assumptions=[out])
