"""Optional optimal fixed-code swap selection under the physical seam objective."""

def exact(model,board,pairs,codes,old,effort):
    from ortools.sat.python import cp_model
    cp = cp_model.CpModel()
    selected = [cp.new_bool_var('') for _ in pairs]
    at = [[] for _ in range(model.n)]
    for i,((a,b),(new_a,new_b)) in enumerate(zip(pairs,codes)):
        at[a].append((i,new_a))
        at[b].append((i,new_b))
    for options in at:
        if len(options)>1:
            cp.add_at_most_one(selected[i] for i,_ in options)
    terms = []
    def side(c,s,reverse):
        masks = model.reverse[model.masks[:,s]] if reverse else model.masks[:,s]
        offset = int(masks[board[c]])
        return offset+sum((int(masks[code])-offset)*selected[i] for i,code in at[c] if masks[code]!=offset)
    initial = board.copy()
    for i in old:
        a,b = pairs[i]
        initial[a],initial[b] = codes[i]
    prior = model.matches(initial)
    old = set(old)
    for i,x in enumerate(selected):
        cp.add_hint(x,int(i in old))
    for edge,(a,s,b,t) in enumerate(model.problem.edges):
        l,r = side(a,s,False),side(b,t,True)
        if type(l) is int and type(r) is int:
            terms.append(int(l==r))
        else:
            v = cp.new_bool_var('')
            cp.add(l==r).only_enforce_if(v)
            cp.add(l!=r).only_enforce_if(~v)
            cp.add_hint(v,int(prior[edge]))
            terms.append(v)
    cp.add(sum(terms)>=int(prior.sum()))
    cp.maximize(sum(terms))
    solver=cp_model.CpSolver()
    solver.parameters.max_deterministic_time=effort
    solver.parameters.num_search_workers=1
    decision=solver.solve(cp)
    data=dict(status=solver.status_name(decision),branches=int(solver.num_branches),
              prior=int(prior.sum()),count=len(pairs),optimal=decision==cp_model.OPTIMAL)
    chosen = sorted(old)
    if decision in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        proposed = []
        result=board.copy()
        for i,x in enumerate(selected):
            if solver.value(x):
                proposed.append(i)
                a,b=pairs[i]
                result[a],result[b]=codes[i]
        report=model.problem.validate(model.legal_board(result).tolist())
        if report['matched_edges']!=round(solver.objective_value):
            raise ArithmeticError('Fixed-code CP result failed direct board verification.')
        data.update(score=report['matched_edges'],upper=int(solver.best_objective_bound))
        if data['score'] > data['prior']:
            chosen = proposed
    return chosen,data
