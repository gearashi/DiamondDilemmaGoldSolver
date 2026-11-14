"""Small exact-replay lifted LP and Boolean polynomial contradiction experiments."""
from collections import Counter
from fractions import Fraction
from itertools import combinations

from r27_reason_maxsat import domain_rows
from validator import reverse_mask


def placement_relations(problem, rows):
    ids = [{v:i for v,i in row} for row in _indices(rows)]
    groups = [tuple(row.values()) for row in ids]
    tiles = [[] for _ in rows]
    for row in ids:
        for code,v in row.items():
            tiles[code//3].append(v)
    groups += [tuple(group) for group in tiles]
    excluded = {tuple(sorted(pair)) for group in groups for pair in combinations(group,2)}
    for c,s,d,t in problem.edges:
        for x,i in ids[c].items():
            for y,j in ids[d].items():
                if int(problem.masks[x,s]) != reverse_mask(int(problem.masks[y,t])):
                    excluded.add(tuple(sorted((i,j))))
    return groups,sorted(excluded)


def _indices(rows):
    offset = 0
    for row in rows:
        yield list(zip(row,range(offset,offset+len(row))))
        offset += len(row)


def products(row, variable, complement):
    """Squarefree coefficient map for f*x or f*(1-x)."""
    multiplied = Counter()
    for monomial,value in row.items():
        multiplied[tuple(sorted(set(monomial) | {variable}))] += value
    if complement:
        original = Counter(row)
        original.subtract(multiplied)
        multiplied = original
    return {m:v for m,v in multiplied.items() if v}


def gf2_certificate(groups, excluded, n, degree):
    equations = [{():1, **{(i,):1 for i in group}} for group in groups]
    equations += [{pair:1} for pair in excluded]
    basis = {}
    generated = []
    terms = {}
    for source,equation in enumerate(equations):
        if max(map(len,equation)) > degree:
            continue
        variants = [(None,equation)]
        if degree == 2 and max(map(len,equation)) <= 1:
            variants.extend((var,products(equation,var,False)) for var in range(n))
        for var,variant in variants:
            packed = 0
            for m,v in variant.items():
                if v%2:
                    packed ^= 1 << terms.setdefault(m,len(terms))
            provenance = 1 << len(generated)
            generated.append((source,var))
            # The constant is pivot zero. A singleton constant is an explicit
            # polynomial identity 1 = sum h_i f_i in the Boolean quotient.
            while packed:
                pivot = packed.bit_length()-1
                if packed == 1 and terms.get(()) == 0:
                    replay = Counter()
                    cert = []
                    while provenance:
                        bit = provenance & -provenance
                        src,v = generated[bit.bit_length()-1]
                        cert.append((src,v))
                        row = equations[src]
                        replay.update(row if v is None else products(row,v,False))
                        provenance ^= bit
                    if {m for m,v in replay.items() if v%2} != {()}:
                        raise RuntimeError('GF(2) contradiction failed replay.')
                    return cert,len(generated)
                if pivot not in basis:
                    basis[pivot] = packed,provenance
                    break
                other,which = basis[pivot]
                packed ^= other
                provenance ^= which
    return None,len(generated)


def lifted_rows(groups, excluded, n, degree):
    rows = []
    for group in groups:
        row = {():-1, **{(i,):1 for i in group}}
        rows.extend((row,{m:-v for m,v in row.items()}))
    for i,j in excluded:
        rows.append({():1,(i,):-1,(j,):-1})
    for i in range(n):
        rows.extend(({(i,):1},{():1,(i,):-1}))
    unique = {}
    def add(row):
        key = tuple(sorted(row.items()))
        if key:
            unique[key] = row
    for row in rows:
        add(row)
        if degree == 2:
            for i in range(n):
                add(products(row,i,False))
                add(products(row,i,True))
    return list(unique.values())


def lp_certificate(rows, seconds):
    """Seek nonnegative row combination equal to -1, then replay over Q."""
    from ortools.linear_solver import pywraplp
    model = pywraplp.Solver.CreateSolver('GLOP')
    if model is None:
        return None
    model.SetTimeLimit(max(1,int(seconds*1000)))
    constraints = {}
    for i,row in enumerate(rows):
        weight = model.NumVar(0,model.infinity(),'w'+str(i))
        for term,coeff in row.items():
            if term not in constraints:
                rhs = -1 if not term else 0
                constraints[term] = model.Constraint(rhs,rhs)
            constraints[term].SetCoefficient(weight,coeff)
    if () not in constraints or model.Solve() not in (model.OPTIMAL,model.FEASIBLE):
        return None
    certificate = []
    replay = Counter()
    for i,var in enumerate(model.variables()):
        value = Fraction(var.solution_value()).limit_denominator(100000)
        if value < 0:
            return None
        if value:
            certificate.append((i,str(value)))
            for term,coeff in rows[i].items():
                replay[term] += coeff*value
    if {m:v for m,v in replay.items() if v} != {():-1}:
        return None
    return certificate


def algebra(problem, *, domains=None, method='gf2', degree=1, max_vars=28, seconds=3):
    if degree not in (1,2) or method not in ('gf2','lp'):
        raise ValueError('Supported methods are gf2/lp at degrees 1 and 2.')
    rows = domain_rows(problem,domains)
    n = sum(map(len,rows))
    if n > max_vars:
        return {'status':'skipped','variables':n}
    groups,excluded = placement_relations(problem,rows)
    if method == 'gf2':
        cert,count = gf2_certificate(groups,excluded,n,degree)
    else:
        inequalities = lifted_rows(groups,excluded,n,degree)
        count = len(inequalities)
        cert = lp_certificate(inequalities,seconds)
    return {'status':'certified_infeasible' if cert is not None else 'unknown',
            'variables':n,'rows':count,'certificate':cert}

