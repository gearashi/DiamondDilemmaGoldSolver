"""Check the VIPR starting axioms against raw physical side labels and reference departures."""
from collections import Counter, defaultdict
from fractions import Fraction

from r35_contact_certificate import xor_conflict
from r35_loss_witness import raw_sides, reciprocal


def signed_axioms(problem, reference, required, groups, cycles, cap):
    expected = Counter()
    def axiom(sense, rhs, terms):
        coef = Counter()
        for name, value in terms:
            coef[name] += value
        if sense == 'L':
            sense, rhs = 'G', -rhs
            coef = {k:-v for k,v in coef.items()}
        expected[(sense, rhs, tuple(sorted((k,v) for k,v in coef.items() if v)))] += 1
    def tile(i):
        return 'tile_'+str(i)
    def lost(b):
        return f'lost_{b[0]}_{b[1]}'

    sides = raw_sides(problem.tiles)
    def integer_label(label):
        slots, color = label
        return sum(1<<(slot-1) for slot in slots), color

    balance = defaultdict(lambda: [Counter(), Counter()])
    self_paired = set()
    for side, (slots, start) in enumerate(sides):
        for choice in (0,1):
            record = (slots, start ^ choice)
            reverse = reciprocal(record)
            a, b = integer_label(record), integer_label(reverse)
            key = min(a, b)
            balance[key][choice][side//3] += 1 if a == key else -1
            if a == b:
                self_paired.add(key)

    domains = {tile(i):(0,1) for i in range(problem.n)}
    axiom('E', 0, [(tile(0), 1)])
    for j, key in enumerate(sorted(balance)):
        low, high = balance[key]
        terms = [(tile(i), high[i]-low[i]) for i in low.keys() | high.keys()]
        if key in self_paired:
            name = 'parity_'+str(j)
            domains[name] = (-3*problem.n, 3*problem.n)
            terms.append((name,-2))
        axiom('E', -sum(low.values()), terms)

    losses = reference-required
    domains.update((lost(b), (0,1)) for b in losses)
    for a,b in sorted(reference | required):
        left, right = sides[a], sides[b]
        if left[0] != reciprocal(right)[0]:
            raise ValueError('A named reference bond joins nonreciprocal raw side labels.')
        x, y = tile(a//3), tile(b//3)
        extra = [(lost((a,b)),1)] if (a,b) in losses else []
        if left[1] ^ reciprocal(right)[1]:
            axiom('G', 1, [(x,1),(y,1),*extra])
            axiom('G', -1, [(x,-1),(y,-1),*extra])
        else:
            axiom('G', 0, [(x,1),(y,-1),*extra])
            axiom('G', 0, [(x,-1),(y,1),*extra])
    allowed = reference|required
    for row in cycles:
        if not row <= allowed or not xor_conflict(problem, row|required):
            raise ValueError('A redundant signed cut fails the raw parity equations.')
    for row in [*groups,*cycles]:
        axiom('G', 1, [(lost(b),1) for b in row-required])
    axiom('L', cap, [(lost(b),1) for b in losses])
    for name,(low,high) in domains.items():
        axiom('G', low, [(name,1)])
        axiom('L', high, [(name,1)])
    return set(domains), {lost(b):1 for b in losses}, expected


def replay_header(path, problem, reference, required, groups, cycles, cap, *, additional_axioms=()):
    domains, cost, expected = signed_axioms(problem, reference, required, groups, cycles, cap)
    expected.update(additional_axioms)
    with path.open(encoding='ascii') as source:
        def fields():
            return source.readline().split()
        def header(kind):
            row = fields()
            if not row or row[0] != kind:
                raise ValueError('Missing VIPR input section '+kind)
            return row[1:]
        if header('VER') != ['1.0']:
            raise ValueError('Unexpected VIPR certificate format.')
        names = [fields()[0].removeprefix('t_') for _ in range(int(header('VAR')[0]))]
        if len(names) != len(domains) or set(names) != domains:
            raise ValueError('VIPR variables differ from the physical puzzle scope.')
        integer = [int(fields()[0]) for _ in range(int(header('INT')[0]))]
        if sorted(integer) != list(range(len(names))):
            raise ValueError('The certificate must declare every physical variable integral.')
        def unpack(tokens):
            n = int(tokens[0])
            if len(tokens) != 1+2*n:
                raise ValueError('Malformed VIPR linear expression.')
            result = {}
            for i in range(n):
                index, coefficient = int(tokens[1+2*i]), Fraction(tokens[2+2*i])
                if not 0 <= index < len(names) or names[index] in result:
                    raise ValueError('Malformed VIPR coefficient index.')
                result[names[index]] = coefficient
            return result
        if header('OBJ') != ['min'] or unpack(fields()) != cost:
            raise ValueError('The VIPR objective differs from reference loss.')
        constraints, bounds = map(int,header('CON'))
        if bounds != 2*len(names):
            raise ValueError('Unexpected number of bound axioms.')
        actual = Counter()
        for _ in range(constraints):
            name,sense,rhs,*expression = fields()
            row, rhs = unpack(expression), Fraction(rhs)
            if sense == 'L':
                sense, rhs = 'G', -rhs
                row = {k:-v for k,v in row.items()}
            actual[(sense, rhs, tuple(sorted((k,v) for k,v in row.items() if v)))] += 1
        if actual != expected:
            difference = (actual-expected) or (expected-actual)
            raise ValueError('A starting inequality differs from the independent physical model: '+str(next(iter(difference)))[:220])
        if header('RTP') != ['infeas']:
            raise ValueError('The certificate does not claim infeasibility of the supplied cutoff.')
    return dict(variables=len(names), starting_axioms=constraints, bounds=bounds)
