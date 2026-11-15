"""Translate checked physical Gold mates directly to mesh seam placement implications."""
from collections import Counter
from functools import lru_cache

from r32_gold_bonds import forbidden_codes
from r35_physical_adjacency import geographic_inventory


def _post(problem, domains, emit, check):
    rules = geographic_inventory(problem)
    rows = [set(row) for row in domains]
    count = Counter()
    claims = getattr(problem,'physical_conflicts',())
    if claims:
        from r27_structure_filters import bits
        from r32_gold_watch import BondWatcher
        masks = [sum(1<<v for v in row) for row in rows]
        checked = masks.copy()
        watcher = BondWatcher(problem,claims,check)
        if watcher.prune(checked):
            for c,(old,new) in enumerate(zip(masks,checked)):
                for code in bits(old^new):
                    emit(c,code,c,None,True)
                    count['physical_core_units'] += 1

    @lru_cache(maxsize=9)
    def templates(s,t):
        return (tuple(forbidden_codes(sorted(rules.forbidden),s,t)),
                tuple(forbidden_codes(sorted(rules.required),s,t)))

    for a,s,b,t in problem.edges:
        check()
        forbidden,required = templates(s,t)
        for i,j in forbidden:
            if i in rows[a] and j in rows[b]:
                emit(a,i,b,j,False)
                count['physical_forbidden_clauses'] += 1
        for i,j in required:
            for c,v,n,w in ((a,i,b,j),(b,j,a,i)):
                if v in rows[c] and rows[n] != {w}:
                    emit(c,v,n,w if w in rows[n] else None,True)
                    count['physical_required_clauses'] += 1
    return dict(count)


def post_sat(problem, variables, clause, check):
    def emit(a,i,b,j,required):
        row = [-variables[a][i]]
        if j is not None:
            right = variables[b][j]
            row.append(right if required else -right)
        clause(row)
    return _post(problem,variables,emit,check)


def post_cpsat(problem, model, codes, domains, check, placements=None):
    @lru_cache(None)
    def literal(c,v):
        if placements is not None:
            return placements[c][v]
        lit = model.new_bool_var('physical_owner')
        model.add(codes[c]==v).only_enforce_if(lit)
        model.add(codes[c]!=v).only_enforce_if(~lit)
        return lit

    def emit(a,i,b,j,required):
        if j is None:
            model.add(codes[a]!=i)
        elif placements is not None:
            right = literal(b,j)
            model.add_implication(literal(a,i),right if required else ~right)
        else:
            model.add(codes[b]==j if required else codes[b]!=j).only_enforce_if(literal(a,i))
    return _post(problem,domains,emit,check)


def post_cp(problem, solver, codes, domains, check):
    @lru_cache(None)
    def literal(c,v):
        return solver.IsEqualCstVar(codes[c],v)

    def emit(a,i,b,j,required):
        if j is None:
            solver.Add(codes[a]!=i)
        else:
            left,right = literal(a,i),literal(b,j)
            solver.Add(left<=right if required else left+right<=1)
    return _post(problem,domains,emit,check)
