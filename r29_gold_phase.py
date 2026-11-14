"""One-loop endpoint-type cuts and reversible physical signed-inventory parity."""
from collections import defaultdict
from itertools import combinations

from r27_structure_filters import Union
from r28_geo_color import _sphere
from validator import reverse_mask


def _reciprocal(mask, color):
    return reverse_mask(mask), color ^ (mask.bit_count() & 1)


class EndpointCuts:
    """A tile strand joins two endpoint classes. Any global tour projects to one connected graph."""

    def __init__(self, problem):
        _, local = _sphere(problem)
        self.problem = problem
        self.strands = []
        self.endpoints = {}
        self.sides = []
        self.graph = defaultdict(set)
        for t, (tile, intervals) in enumerate(zip(problem.tiles, local)):
            masks = [int(m) for m in problem.masks[3*t]]
            self.sides.extend((m, intervals[12*s].bit_count() & 1) for s, m in enumerate(masks))
            phased = [[], []]
            for idx, strand in enumerate(tile["segments"]):
                for phase, row in enumerate(phased):
                    keys = []
                    for side, slot in strand:
                        mask, start = self.sides[3*t+side]
                        color = start ^ phase
                        key = min((mask, color, slot), (*_reciprocal(mask, color), 12-slot))
                        keys.append(key)
                        self.endpoints[3*t+side, slot] = (t, idx)
                    a, b = keys
                    row.append((a, b))
                    self.graph[a].add(b)
                    self.graph[b].add(a)
            self.strands.append(phased)

    def _subsets(self, width):
        layer = {frozenset((node,)) for node in self.graph}
        for k in range(1,width+1):
            if not layer:
                break
            if k == 1:
                yield from sorted(layer,key=lambda s:tuple(sorted(s)))
                continue
            next_layer = set()
            for part in sorted(layer,key=lambda s:tuple(sorted(s))):
                for n in sorted(set.union(*(self.graph[a] for a in part))-part):
                    candidate = part | {n}
                    if candidate not in next_layer:
                        next_layer.add(candidate)
                        yield candidate
            layer = next_layer

    def clauses(self, width=2, max_subsets=10000):
        """Forbid isolating a nonempty proper strand group; saturation only affects discovery."""
        if width < 1 or max_subsets < 1:
            raise ValueError("Expected positive cut width and subset budget.")
        found = set()
        subsets = 0
        complete = True
        for part in self._subsets(width):
            if subsets == max_subsets:
                complete = False
                break
            subsets += 1
            fixed = []
            inside, outside = [], []
            # None denotes an unconditional internal/external strand under the fixed literals.
            for t, phases in enumerate(self.strands):
                allowed = []
                for value, row in enumerate(phases):
                    if all((a in part) == (b in part) for a, b in row):
                        allowed.append((value, any(a in part for a, b in row),
                                         any(a not in part for a, b in row)))
                if not allowed:
                    break
                if len(allowed) == 1:
                    fixed.append((t, allowed[0][0]))
                for witnesses, index in ((inside, 1), (outside, 2)):
                    if witnesses == [None]:
                        continue
                    if all(x[index] for x in allowed):
                        witnesses[:] = [None]
                    else:
                        witnesses.extend((t, x[0]) for x in allowed if x[index])
            else:
                base = frozenset(fixed)
                for i in inside:
                    for o in outside:
                        clause = base | {v for v in (i, o) if v is not None}
                        if len(dict(clause)) == len(clause):
                            found.add(clause)
        minimal = []
        by_size = defaultdict(list)
        for rule in sorted(found, key=lambda x:(len(x), sorted(x))):
            if any(old <= rule for size, rules in by_size.items() if size < len(rule)
                   for old in rules):
                continue
            by_size[len(rule)].append(rule)
            minimal.append(tuple(sorted(rule)))
        return minimal, dict(subsets=subsets, complete=complete, distinct_rules=len(found),
                             kept_rules=len(minimal))

    def possible_tour(self, pins=None, bonds=()):
        """Union over every possible signed seam, retaining actual physical bundles and slots."""
        pins = checked_pins(self.problem.n, pins or {})
        mate = {}
        for a, b in bonds:
            if type(a) is not int or type(b) is not int or min(a,b) < 0 or max(a,b) >= len(self.sides):
                raise ValueError("Invalid physical side.")
            if a//3 == b//3 or mate.get(a,b) != b or mate.get(b,a) != a:
                return False
            mate[a], mate[b] = b, a
        groups = defaultdict(list)
        for a, (mask, start) in enumerate(self.sides):
            for p in ((pins[a//3],) if a//3 in pins else (0,1)):
                groups[mask, start ^ p].append(a)
        offset = [0]
        for row in self.strands:
            offset.append(offset[-1]+len(row[0]))
        graph = Union(offset[-1])
        for label, sides in groups.items():
            mask, color = label
            partners = groups.get(_reciprocal(mask,color), ())
            for a in sides:
                for b in partners:
                    if a >= b or a//3 == b//3 or mate.get(a,b) != b or mate.get(b,a) != a:
                        continue
                    for k in range(1,12):
                        if (mask >> (k-1)) & 1:
                            t, i = self.endpoints[a,k]
                            u, j = self.endpoints[b,12-k]
                            graph.join(offset[t]+i, offset[u]+j)
        return len({graph.find(i) for i in range(offset[-1])}) == 1


def checked_pins(n, pins):
    if any(type(t) is not int or not 0 <= t < n or type(p) is not int or p not in (0,1)
           for t,p in pins.items()):
        raise ValueError("Invalid phase pin.")
    return dict(pins)


def propagate_rules(rules, pins):
    """Unit propagation of forbidden conjunctions, including their color complements when generated."""
    values = dict(pins)
    while True:
        count = len(values)
        for clause in rules:
            if any(t in values and values[t] != v for t,v in clause):
                continue
            unknown = [(t,v) for t,v in clause if t not in values]
            if not unknown:
                return None
            if len(unknown) == 1:
                t,v = unknown[0]
                values[t] = 1-v
        if len(values) == count:
            return values


class PhaseBasis:
    """Persistent echelon rows with XOR explanations and LIFO branch rollback."""
    def __init__(self):
        self.pivots = {}
        self.projections = {}
        self.projection_hits = 0
        self.trail = []
        self.conflict = None

    def add(self, mask, rhs, labels):
        if self.conflict is not None:
            return False
        labels = frozenset(labels)
        while mask:
            p = mask.bit_length()-1
            if p not in self.pivots:
                self.pivots[p] = (mask, rhs, labels)
                self.trail.append(p)
                return True
            row, value, support = self.pivots[p]
            mask ^= row
            rhs ^= value
            labels ^= support
        if rhs:
            self.conflict = labels
            self.trail.append(None)
        return not rhs

    def forced(self):
        """Back-substitution exposes every variable fixed by these linear equations."""
        expanded = {}
        fixed = {}
        for pivot in sorted(self.pivots):
            row,rhs,_ = self.pivots[pivot]
            for i in sorted(expanded,reverse=True):
                if row >> i & 1:
                    mask,value = expanded[i]
                    row ^= mask
                    rhs ^= value
            expanded[pivot] = row,rhs
            if row == 1<<pivot:
                fixed[pivot] = rhs
        return fixed

    def rollback(self, mark):
        if type(mark) is not int or not 0 <= mark <= len(self.trail):
            raise ValueError("Invalid parity rollback mark.")
        while len(self.trail) > mark:
            p = self.trail.pop()
            if p is None:
                self.conflict = None
            else:
                del self.pivots[p]

    def project(self, keep):
        """Eliminate all other variables and return consequences on this boundary only."""
        if self.conflict is not None:
            return [(0,1,self.conflict)]
        signature = keep,tuple(self.pivots.values())
        if signature in self.projections:
            self.projection_hits += 1
            return list(self.projections[signature])
        inner, outer = {}, PhaseBasis()
        for row, rhs, why in self.pivots.values():
            while row & ~keep:
                pivot = (row & ~keep).bit_length()-1
                if pivot not in inner:
                    inner[pivot] = row,rhs,why
                    break
                other,value,support = inner[pivot]
                row ^= other
                rhs ^= value
                why ^= support
            else:
                outer.add(row,rhs,why)
        result = tuple(outer.pivots.values())
        if len(self.projections) >= 128:
            self.projections.pop(next(iter(self.projections)))
        self.projections[signature] = result
        return list(result)


class SignedParity:
    """Modulo-two projection of the model's reciprocal signed class equations."""
    def __init__(self, problem, sides=None):
        if sides is None:
            sides = EndpointCuts(problem).sides
        self.sides = sides
        self.n = problem.n
        self.base = PhaseBasis()
        parity = {}
        for a,(mask,start) in enumerate(sides):
            for color in (0,1):
                label = (mask,color)
                pair = min(label, _reciprocal(mask,color))
                row, rhs = parity.get(pair,(0,0))
                parity[pair] = row ^ (1 << (a//3)), rhs ^ int(color == start)
        for label,(row,rhs) in sorted(parity.items()):
            self.base.add(row, rhs, [('class', *label)])
        self.equations = len(parity)

    def bond(self, a, b):
        if type(a) is not int or type(b) is not int or not 0 <= a < 3*self.n or not 0 <= b < 3*self.n:
            raise ValueError("Invalid physical side.")
        ma,ca = self.sides[a]
        mb,cb = self.sides[b]
        why = [('bond', *sorted((a,b)))]
        if a//3 == b//3 or mb != reverse_mask(ma):
            return self.base.add(0,1,why)
        return self.base.add(1 << (a//3) ^ 1 << (b//3), ca ^ cb ^ (ma.bit_count() & 1), why)

    def pin(self, t, value):
        checked_pins(self.n, {t:value})
        return self.base.add(1<<t,value,[('pin',t,value)])
