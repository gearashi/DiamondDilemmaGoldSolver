"""Scoped SAT experiments with independently checked terminal placements."""
from collections import defaultdict
from itertools import combinations
import time

from r27_reason_maxsat import Budget, SeamFormula, domain_rows
from validator import reverse_mask


def seam_hitting_set(problem, rows, *, previous=(), seconds=20, threshold=None):
    """An implicit minimum defect cover. Recheck previous cores against this scope."""
    from ortools.sat.python import cp_model
    from pysat.solvers import Glucose4
    rows = domain_rows(problem, rows)
    end = time.monotonic()+seconds
    result = dict(status='unknown', upper=len(problem.edges), cores=[], calls=0, codes=None)
    def check():
        if time.monotonic() >= end:
            raise Budget
    def decide(sat, assumptions):
        from threading import Timer
        check()
        timer = Timer(max(.001,end-time.monotonic()),sat.interrupt)
        timer.start()
        try:
            result['calls'] += 1
            answer = sat.solve_limited(assumptions=list(assumptions), expect_interrupt=True)
        finally:
            timer.cancel()
            timer.join()
            sat.clear_interrupt()
        if answer is None:
            raise Budget
        return answer
    try:
        with Glucose4() as sat:
            formula = SeamFormula(problem,rows,sat,check)
            defects = formula.defects
            ids = {v:i for i,v in enumerate(defects)}
            cover = cp_model.CpModel()
            lost = [cover.new_bool_var(f'defect_{i}') for i in range(len(defects))]
            cover.minimize(sum(lost))
            optimizer = cp_model.CpSolver()
            optimizer.parameters.num_search_workers = 1
            if not decide(sat,[]):
                result.update(status='infeasible',upper=None)
                return result
            def add(indices):
                row = sorted(set(indices))
                if not row:
                    raise RuntimeError('Empty core after feasible inventory check.')
                cover.add_bool_or([lost[i] for i in row])
                result['cores'].append(row)
            for row in previous:
                if any(type(i) is not int or not 0<=i<len(defects) for i in row):
                    raise ValueError('A core contains an invalid seam index.')
                if decide(sat,[-defects[i] for i in row]):
                    continue
                add(row)
            while True:
                check()
                optimizer.parameters.max_time_in_seconds = max(.001,end-time.monotonic())
                if optimizer.solve(cover) != cp_model.OPTIMAL:
                    raise Budget
                released = {i for i,x in enumerate(lost) if optimizer.value(x)}
                result['upper'] = len(defects)-len(released)
                if threshold is not None and result['upper'] < threshold:
                    result['status'] = 'threshold_impossible'
                    return result
                if decide(sat,[-v for i,v in enumerate(defects) if i not in released]):
                    codes = formula.board(sat.get_model())
                    report = problem.validate(codes)
                    if not report['unique_tiles'] or report['matched_edges']!=result['upper']:
                        raise RuntimeError('Minimum cover score disagrees with independent validation.')
                    result.update(status='optimal',codes=codes)
                    return result
                add(ids[-v] for v in sat.get_core())
    except Budget:
        return result


def placements(rows):
    ids = []
    top = 0
    clauses = []
    for row in rows:
        choices = {}
        for code in row:
            top += 1
            choices[code] = top
        ids.append(choices)
        values = list(choices.values())
        clauses.append(values)
        # A sequential AMO avoids quadratic cost on the full 480-wide data.
        if len(values)>5:
            from pysat.card import CardEnc, EncType
            encoded = CardEnc.atmost(values,1,top_id=top,encoding=EncType.seqcounter)
            top = max(top,encoded.nv)
            clauses.extend(encoded.clauses)
        else:
            clauses.extend([-u,-v] for u,v in combinations(values,2))
    return ids,clauses,top


def cegar(problem, rows, target='gold', *, seconds=20):
    """Lazily refine side bits, tile ownership, and exact Gold cycle failures."""
    from pysat.solvers import Glucose4
    from threading import Timer
    if target not in ('gold','edges'):
        raise ValueError('Expected gold or edges.')
    rows = domain_rows(problem,rows)
    deadline = time.monotonic()+seconds
    ids, clauses, top = placements(rows)
    owned = defaultdict(list)
    for row in ids:
        for code,lit in row.items():
            owned[code//3].append(lit)
    result = dict(candidates=0,mask_refinements=0,inventory_refinements=0,gold_cuts=0,
                  status='unknown',codes=None)
    used_owners = set()
    used_bits = set()

    with Glucose4(bootstrap_with=clauses) as sat:

        while time.monotonic()<deadline:
            timer = Timer(max(.001,deadline-time.monotonic()),sat.interrupt)
            timer.start()
            try:
                answer = sat.solve_limited(expect_interrupt=True)
            finally:
                timer.cancel()
                timer.join()
                sat.clear_interrupt()
            if answer is None:
                return result
            if not answer:
                result['status'] = 'infeasible'
                return result
            selected = set(sat.get_model())
            codes = [next(code for code,lit in row.items() if lit in selected) for row in ids]
            result['candidates'] += 1
            duplicate = False
            seen = set()
            for code in codes:
                tile = code//3
                if tile in seen and tile not in used_owners:
                    from pysat.card import CardEnc, EncType
                    encoding = CardEnc.atmost(owned[tile],1,top_id=top,encoding=EncType.seqcounter)
                    sat.append_formula(encoding.clauses)
                    top = max(top,encoding.nv)
                    used_owners.add(tile)
                    result['inventory_refinements'] += 1
                    duplicate = True
                seen.add(tile)
            refined = False
            for k,(a,sa,b,sb) in enumerate(problem.edges):
                difference = int(problem.masks[codes[a],sa]) ^ reverse_mask(problem.masks[codes[b],sb])
                if not difference:
                    continue
                bit = difference & -difference
                if (k,bit) in used_bits:
                    raise RuntimeError('A refined seam bit was violated again.')
                used_bits.add((k,bit))
                top += 1
                for c,s in ((a,sa),(b,sb)):
                    for code,lit in ids[c].items():
                        mask = int(problem.masks[code,s])
                        if c==b:
                            mask = reverse_mask(mask)
                        sat.add_clause([-lit,top if mask & bit else -top])
                result['mask_refinements'] += 1
                refined = True
            if duplicate or refined:
                continue
            report = problem.validate(codes)
            if not report['unique_tiles'] or report['matched_edges'] != len(problem.edges):
                raise RuntimeError('Refined SAT witness violates exact constraints.')
            if target=='edges' or report['valid']:
                result.update(status='witness',codes=codes)
                return result
            sat.add_clause([-ids[c][code] for c,code in problem.gold_nogood(codes)])
            result['gold_cuts'] += 1
    return result


def checked_interpolant(left, right, interface, *, max_cubes=4096):
    """Produce a CNF I over the shared variables with left => I and right & I unsat.

    Enumerate right boundary models; a left UNSAT core generalizes each cube.
    A satisfying join or interrupted coverage never yields a certificate.
    """
    from pysat.solvers import Glucose4
    interface = tuple(interface)
    if len(set(interface))!=len(interface) or any(type(v)is not int or v<=0 for v in interface):
        raise ValueError('Interface variables must be distinct positive integers.')
    shared = {abs(x) for cl in left for x in cl} & {abs(x) for cl in right for x in cl}
    if not shared<=set(interface):
        raise ValueError('Every shared variable must be visible at the interface.')
    cuts = []
    with Glucose4(bootstrap_with=left) as lhs, Glucose4(bootstrap_with=right) as rhs:
        for _ in range(max_cubes+1):
            if not rhs.solve():
                if not verify_interpolant(left,right,interface,cuts):
                    raise RuntimeError('Interpolant did not pass independent verification.')
                return dict(status='separated',clauses=cuts)
            model = set(rhs.get_model())
            cube = [v if v in model else -v for v in interface]
            if lhs.solve(assumptions=cube):
                return dict(status='joined',clauses=[])
            if len(cuts)==max_cubes:
                break
            cut = [-v for v in (lhs.get_core() or ())]
            cuts.append(cut)
            rhs.add_clause(cut)
    return dict(status='unknown',clauses=[])


def verify_interpolant(left,right,interface,cuts):
    from pysat.solvers import Glucose4
    shared = set(interface)
    if any(abs(v) not in shared for cl in cuts for v in cl):
        return False
    with Glucose4(bootstrap_with=left) as lhs, Glucose4(bootstrap_with=right) as rhs:
        for cut in cuts:
            if lhs.solve(assumptions=[-v for v in cut]):
                return False
        rhs.append_formula(cuts)
        return not rhs.solve()


def determining_support(clauses, variables):
    """Greedily remove fixed/functional physical variables by a two-copy SAT miter."""
    from pysat.solvers import Glucose4
    physical = tuple(dict.fromkeys(variables))
    if any(type(v)is not int or v<=0 for v in physical):
        raise ValueError('Expected positive physical variable IDs.')
    n = max([0,*physical,*(abs(x) for cl in clauses for x in cl)])
    twin = lambda x: x+n if x>0 else x-n
    reduced = set(physical)
    with Glucose4(bootstrap_with=clauses) as solver:
        solver.append_formula([[twin(x) for x in cl] for cl in clauses])
        for k,v in enumerate(physical,1):
            gate = 2*n+k
            solver.add_clause([-gate,-v,twin(v)])
            solver.add_clause([-gate,v,-twin(v)])
        guards = {v:2*n+k for k,v in enumerate(physical,1)}
        for v in physical:
            # Two copies are symmetric, so one discrepancy direction suffices.
            equal = [guards[w] for w in reduced if w!=v]
            if not solver.solve(assumptions=[*equal,v,-twin(v)]):
                reduced.remove(v)
    return sorted(reduced)


def project_models(clauses,support,variables,limit=128):
    """Reconstruct the full physical projection, enumerate only support assignments."""
    from pysat.solvers import Glucose4
    projected = []
    with Glucose4(bootstrap_with=clauses) as solver:
        for v in variables:
            solver.add_clause([v,-v])
        while len(projected)<limit and solver.solve():
            model = set(solver.get_model())
            projected.append(tuple(v if v in model else -v for v in variables))
            solver.add_clause([-v if v in model else v for v in support])
    return projected
