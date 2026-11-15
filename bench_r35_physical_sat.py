"""Fixed-phase SAT cross-check for the sparse physical circuit and signed mate model."""
from collections import defaultdict


def solve_fixed(check,phases,required,*,arc_limit=80000,limit=1):
    from ortools.sat.python import cp_model
    from pysat.card import CardEnc,EncType
    from pysat.formula import IDPool
    from pysat.solvers import Solver
    from r28_geo_color import signed_phase_model
    from r35_physical_tour import tour_structure
    if check.blank_projection is not None or len(phases)!=check.problem.n:
        raise ValueError('Fixed-phase SAT requires a full-side physical scope.')
    m,_=signed_phase_model(check.problem,pins=dict(enumerate(phases)),bonds=required)
    phase_solver=cp_model.CpSolver()
    if phase_solver.solve(m)!=cp_model.OPTIMAL:
        return dict(status='INFEASIBLE',models=[],complete=True,phase_feasible=False)
    n,arcs,banned,impossible=tour_structure(check,required,True)
    if impossible or not set(required)<=check.mate.keys():
        return dict(status='INFEASIBLE',models=[],complete=True)
    if max(n*n,len(arcs)*n)>arc_limit:
        return dict(status='SIZE_LIMIT',models=[],complete=False)
    ids=IDPool()
    clauses=[]
    def exactly(lits):
        lits=list(lits)
        if not lits:
            clauses.append([])
        else:
            clauses.extend(CardEnc.equals(lits=lits,bound=1,vpool=ids,encoding=EncType.seqcounter).clauses)
    choose={pair:ids.id(('mate',pair)) for pair in sorted(check.mate)}
    owners=defaultdict(list)
    for (a,b),lit in choose.items():
        mask,left=check.inputs.sides[a]
        _,right=check.inputs.sides[b]
        incompatible=left^right^phases[a//3]^phases[b//3]^(mask.bit_count()&1)
        if incompatible or (a,b) in banned:
            clauses.append([-lit])
        owners[a].append(lit);owners[b].append(lit)
    for dart in range(3*check.problem.n):
        exactly(owners[dart])
    for pair in required:
        clauses.append([choose[pair]])
    if n>1:
        position=[[ids.id(('position',v,k)) for k in range(n)] for v in range(n)]
        clauses.append([position[0][0]])
        for row in position:exactly(row)
        for col in zip(*position):exactly(col)
        for (u,v),supports in arcs.items():
            literals={choose[pair] for pair,value in supports if phases[pair[0]//3]==value}
            if not literals:continue
            arc=ids.id(('arc',u,v))
            clauses.extend([[-lit,arc] for lit in sorted(literals)])
            clauses.append([-arc,*sorted(literals)])
            for k in range(n):
                clauses.append([-arc,-position[u][k],position[v][(k+1)%n]])
    witnesses=[]
    with Solver(name='g4',bootstrap_with=clauses) as solver:
        while len(witnesses)<limit and solver.solve():
            model=set(solver.get_model())
            selected=sorted(pair for pair,lit in choose.items() if lit in model)
            witnesses.append(selected)
            solver.add_clause([-choose[p] for p in selected])
        done=len(witnesses)<limit
        stats=solver.accum_stats()
    return dict(status='CONNECTED_SEWING' if witnesses else 'INFEASIBLE',models=witnesses,complete=done,
                clauses=len(clauses),variables=ids.top,decisions=stats['decisions'])
