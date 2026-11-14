"""Checked Craig separators for exact regional placement and Gold conflicts."""
from r27_reason_maxsat import domain_rows
from r28_logic_reuse import direct_cnf
from r28_logic_sat import checked_interpolant


def regional_interpolant(problem, domains, left, *, target='gold', cap=128, proof=False, subregions=(), proof_work=100000, previous=None):
    """Build an actual two-sided CNF and independently check both implications.

    Crossing seam constraints and ownership clauses remain in the formula.
    A closed Gold-loop cut from a validated full candidate is global and can
    expose additional physical variables on the common interface.
    """
    from pysat.solvers import Glucose4
    rows = domain_rows(problem,domains)
    cells = set(left)
    if not cells or len(cells)>=problem.n or any(type(c)is not int or not 0<=c<problem.n for c in cells):
        raise ValueError('Expected a nonempty proper region.')
    if target not in ('gold','edges'):
        raise ValueError('Expected gold or edges.')
    ids,clauses,_ = direct_cnf(problem,rows,factored=True)
    inside = {v for c,row in enumerate(ids) if c in cells for v in row.values()}
    outside = {v for c,row in enumerate(ids) if c not in cells for v in row.values()}
    lhs,rhs = [],[]
    for cl in clauses:
        variables = {abs(x) for x in cl}
        (lhs if variables & inside and not variables & outside else rhs).append(cl)
    if proof and previous:
        from r29_logic_proof import reuse_boundary
        for boundary in previous:
            if reuse_boundary(lhs,rhs,boundary):
                return dict(status="separated",verified=True,method="rechecked boundary",proof_builds=0,
                            boundaries=[boundary],gold_cuts=0,
                            shared=len({abs(x) for cl in lhs for x in cl}&{abs(x) for cl in rhs for x in cl}))
    gold_cuts = 0
    with Glucose4(bootstrap_with=clauses) as combined:
        while combined.solve():
            chosen = set(combined.get_model())
            codes = [next(code for code,lit in row.items() if lit in chosen) for row in ids]
            report = problem.validate(codes)
            if not report['unique_tiles'] or report['matched_edges']!=len(problem.edges):
                raise RuntimeError('Regional formula admitted an invalid edge candidate.')
            if target=='edges' or report['valid']:
                return dict(status='joined',codes=codes,gold_cuts=gold_cuts)
            if gold_cuts>=cap:
                return dict(status='unknown',gold_cuts=gold_cuts)
            cut = [-ids[c][code] for c,code in problem.gold_nogood(codes)]
            rhs.append(cut)
            combined.add_clause(cut)
            gold_cuts += 1
    variables = sorted({abs(x) for cl in lhs for x in cl} & {abs(x) for cl in rhs for x in cl})
    if proof:
        from r29_logic_proof import proof_interpolants
        # Split the left side into disjoint clause groups. A single checked DAG
        # supplies the full boundary and each requested nested boundary.
        nested = [set(s) for s in subregions]
        if any(not s or not s < cells for s in nested) or any(
                a & b and not (a<=b or b<=a) for i,a in enumerate(nested) for b in nested[:i]):
            raise ValueError("Nested regions must form a laminar family inside the left region.")
        nested.sort(key=len)
        supports = [{v for c in s for v in ids[c].values()} for s in nested]
        groups = [[] for _ in range(len(nested)+2)]
        for cl in lhs:
            local = {abs(x) for x in cl} & inside
            idx = next((i for i,ss in enumerate(supports) if local and local<=ss),len(nested))
            groups[idx].append(cl)
        groups[-1] = rhs
        partitions = [list(range(len(groups)-1))]
        partitions.extend([i for i,s in enumerate(nested) if s<=container] for container in nested)
        result = proof_interpolants(groups,partitions,max_work=proof_work)
        result.update(shared=len(variables),gold_cuts=gold_cuts,verified=result['status']=='separated')
        return result
    result = checked_interpolant(lhs,rhs,variables,max_cubes=cap)
    result.update(shared=len(variables), gold_cuts=gold_cuts, verified=result['status']=='separated')
    return result
