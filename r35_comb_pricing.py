"""Find candidate comb handles and select an odd matching of disjoint crossing teeth."""
from collections import defaultdict


def candidate_handles(n,weights,flow_handles=(),maximum=160):
    universe=set(range(n))
    handles=set()
    def add(part):
        part=frozenset(part)
        if not 3<=len(part)<=n-3:
            return
        other=frozenset(universe-part)
        handles.add(min((part,other),key=lambda h:(len(h),sorted(h))))
    for part in flow_handles:add(part)
    for threshold in (.125,.375,.5,.625,.875):
        graph=[set() for _ in range(n)]
        for (u,v),weight in weights.items():
            if weight>=threshold:
                graph[u].add(v);graph[v].add(u)
        unseen=set(universe)
        while unseen:
            root=unseen.pop()
            component={root}
            stack=[root]
            while stack:
                for v in graph[stack.pop()]&unseen:
                    component.add(v);stack.append(v);unseen.remove(v)
            add(component)
        for v,row in enumerate(graph):
            group=row|{v};add(group)
            group=group.union(*(graph[u] for u in row));add(group)
    def price(part):
        return sum(w for (u,v),w in weights.items() if (u in part)!=(v in part))
    return sorted(handles,key=lambda h:(price(h),len(h),sorted(h)))[:maximum]


def price_teeth(handle,weights,boundary,stats,scale=1000000):
    from ortools.graph.python.min_cost_flow import SimpleMinCostFlow
    crossing={(u,v) if u in handle else (v,u):2*w-1 for (u,v),w in weights.items()
              if w>0 and (u in handle)!=(v in handle)}
    left=sorted({u for u,_ in crossing});right=sorted({v for _,v in crossing})
    if min(len(left),len(right))<3:
        return
    scores=sorted(crossing.values(),reverse=True)
    # At boundary >= 2 a zero-weight tooth cannot improve on an odd shorter family.
    # Floating prices only discover rows. The caller reprices the exact integer formula.
    upper=0.0
    for t,score in enumerate(scores[:min(len(left),len(right))],1):
        upper+=score
        if t<3 or t%2==0:continue
        if upper+1<=boundary+1e-7:
            if score<=0:break
            continue
        flow=SimpleMinCostFlow()
        src=max(left+right)+1;dst=src+1
        for u in left:flow.add_arc_with_capacity_and_unit_cost(src,u,1,0)
        indices={flow.add_arc_with_capacity_and_unit_cost(u,v,1,-round(value*scale)):(u,v)
                 for (u,v),value in crossing.items()}
        for v in right:flow.add_arc_with_capacity_and_unit_cost(v,dst,1,0)
        flow.set_node_supply(src,t);flow.set_node_supply(dst,-t)
        stats['cardinality_flows']+=1
        if flow.solve()!=flow.OPTIMAL:
            break
        teeth=tuple(sorted(tuple(sorted(edge)) for i,edge in indices.items() if flow.flow(i)))
        if sum(2*weights[e]-1 for e in teeth)+1>boundary+1e-7:
            yield teeth


def rows(n,weights,flow_handles,stats,maximum=160):
    for part in candidate_handles(n,weights,flow_handles,maximum):
        stats['handles']+=1
        boundary=sum(w for (u,v),w in weights.items() if (u in part)!=(v in part))
        for teeth in price_teeth(part,weights,boundary,stats):
            yield part,teeth

def larger_rows(n,weights,flow_handles,stats,maximum=160,max_size=4):
    """Price connected teeth up to a bounded size and verify every proposed comb independently."""
    from ortools.sat.python import cp_model
    adj=[set() for _ in range(n)]
    for (u,v),w in weights.items():
        if w>1e-8:
            adj[u].add(v);adj[v].add(u)
    frontier={frozenset((v,)) for v in range(n)}
    teeth={}
    for _ in range(2,max_size+1):
        grown=set()
        for subset in frontier:
            for u in set.union(*(adj[v] for v in subset))-subset:
                grown.add(subset|{u})
        frontier=grown
        for subset in frontier:
            # Use actual degrees, including possible parallel crossing strands.
            boundary=sum(weights[tuple(sorted((u,v)))] for u in subset for v in adj[u] if v not in subset)
            teeth[subset]=3.-boundary
    stats['connected_tooth_sets']=len(teeth)
    for part in candidate_handles(n,weights,flow_handles,maximum):
        boundary=sum(w for (u,v),w in weights.items() if (u in part)!=(v in part))
        cross={t:w for t,w in teeth.items() if t&part and t-part}
        positive=sum(max(0.,w) for w in cross.values())
        if positive<=boundary-1+1e-7:continue
        # Extremely negative teeth cannot participate even alongside every positive tooth.
        cross={t:w for t,w in cross.items() if w+positive>boundary-1-1e-7}
        m=cp_model.CpModel()
        x={t:m.new_bool_var('tooth') for t in cross}
        through=defaultdict(list)
        for t,v in x.items():
            for u in t:through[u].append(v)
        for vals in through.values():m.add_at_most_one(vals)
        half=m.new_int_var(1,len(cross)//2,'half')
        m.add(sum(x.values())==2*half+1)
        m.maximize(sum(round(w*1000000)*x[t] for t,w in cross.items()))
        solver=cp_model.CpSolver()
        solver.parameters.num_search_workers=1
        stats['larger_packing_prices']+=1
        for effort in (.025,.25,2.5):
            solver.parameters.max_deterministic_time=effort
            status=solver.solve(m)
            if status in (cp_model.OPTIMAL,cp_model.INFEASIBLE):break
            if status==cp_model.FEASIBLE:
                if solver.best_objective_bound/1000000+.5e-6*len(x)<=boundary-1+1e-7:
                    break
                pick=[t for t,v in x.items() if solver.value(v)]
                if sum(cross[t] for t in pick)>boundary-1+1e-7:
                    break
            stats['larger_packing_retries']+=1
        if status==cp_model.UNKNOWN or (status==cp_model.FEASIBLE and
                solver.best_objective_bound/1000000+.5e-6*len(x)>boundary-1+1e-7):
            stats['unsettled_or_positive_price']+=1
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            pick=tuple(sorted(tuple(sorted(t)) for t,v in x.items() if solver.value(v)))
            if sum(cross[frozenset(t)] for t in pick)>boundary-1+1e-7:
                yield part,pick
