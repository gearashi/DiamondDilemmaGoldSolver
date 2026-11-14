"""Checked Craig separators for exact regional placement and Gold conflicts."""
from r27_reason_maxsat import domain_rows
from r28_logic_reuse import direct_cnf
from r28_logic_sat import checked_interpolant


def regional_interpolant(problem, domains, left, *, target='gold', cap=128):
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
    result = checked_interpolant(lhs,rhs,variables,max_cubes=cap)
    result.update(shared=len(variables), gold_cuts=gold_cuts, verified=result['status']=='separated')
    return result
