"""Gold connectivity encodings and conditional integer inventory interfaces."""
from collections import Counter, defaultdict

from r27_structure_filters import Union, _regions, _vertex_links, bits, oriented_segments
from r28_geo_color import _features, _sphere
from r29_gold_geography import checked_domains
from r29_gold_phase import _reciprocal, checked_pins
from validator import reverse_mask


def connected_colors(problem, domains, method='parents', seconds=5, max_rows=100000):
    """At most two connected complementary components of a complete noncrossing placement."""
    from ortools.sat.python import cp_model
    from r28_geo_color import dual_cuts
    domains=checked_domains(problem,domains)
    if method=='lazy':
        return dual_cuts(problem,domains,deterministic_seconds=seconds)
    if method not in ('parents','flow') or seconds<=0:
        raise ValueError("Unknown color connectivity method or budget.")
    corner,_=_sphere(problem)
    if not problem.total_segments:
        return dict(status='INFEASIBLE',boards=0)
    if sum(d.bit_count()*36 for d in domains)>max_rows:
        return dict(status='SIZE_LIMIT',boards=0)
    m=cp_model.CpModel()
    choices,nodes,activations=_post_regions(problem,domains,method,corner,m)
    solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=seconds
    status=solver.solve(m)
    result=dict(status=solver.status_name(status),branches=solver.num_branches,nodes=len(nodes),
                possible_links=len(activations),boards=0)
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        codes=[next(v for v,x in row.items() if solver.value(x)) for row in choices]
        if not problem.validate(codes)['valid']:raise ArithmeticError("Invalid geographic color witness.")
        result.update(witness=codes,boards=1)
    return result


def _post_regions(problem,domains,method,corner,m):
    choices=[{v:m.new_bool_var('placement') for v in bits(ds)} for ds in domains]
    for row in choices:m.add_exactly_one(row.values())
    for t in range(problem.n):
        m.add_exactly_one(x for row in choices for v,x in row.items() if v//3==t)
    mesh=Union(36*problem.n)
    for a,s,b,t in problem.edges:
        masks=defaultdict(list)
        for v,x in choices[b].items():masks[int(problem.masks[v,t])].append(x)
        for v,x in choices[a].items():
            m.add_bool_or([~x,*masks[reverse_mask(int(problem.masks[v,s]))]])
        for k in range(12):
            mesh.join(36*a+12*s+k,36*b+12*t+11-k)
    by_code={}
    def local(v):
        if v not in by_code:
            grouped=defaultdict(list)
            for j,label in enumerate(_regions(oriented_segments(problem,v))):
                grouped[label].append(j)
            by_code[v]=[(g[0],end) for g in grouped.values() for end in g[1:]]
        return by_code[v]
    # Contract links guaranteed by every option in a cell.
    for c,row in enumerate(choices):
        if row:
            sets=iter(row)
            common=set(local(next(sets)))
            for v in sets:
                common.intersection_update(local(v))
                if not common:break
            for a,b in common:
                mesh.join(36*c+a,36*c+b)
    nodes=sorted({mesh.find(i) for i in range(36*problem.n)})
    activations=defaultdict(list)
    for c,row in enumerate(choices):
        for v,x in row.items():
            induced={tuple(sorted((mesh.find(36*c+a),mesh.find(36*c+b)))) for a,b in local(v)}
            for a,b in induced:
                if a!=b:activations[a,b].append(x)
    odd={v for v,d in Counter(corner).items() if d%2}
    apex={mesh.find(36*(j//3)+12*(j%3)) for j,v in enumerate(corner) if v in odd}
    fixed_roots=len(odd)==2 and problem.total_segments%2
    roots={u:m.new_bool_var('root') for u in nodes}
    m.add(sum(roots.values())==2)
    if fixed_roots:
        for u,x in roots.items():m.add(x==int(u in apex))
    size=len(nodes)
    input_arcs=defaultdict(list)
    output=defaultdict(list)
    ranks={u:m.new_int_var(0,size,'rank') for u in nodes} if method=='parents' else {}
    for (u,v),opts in activations.items():
        on=m.new_bool_var('available')
        m.add_max_equality(on,opts)
        for a,b in ((u,v),(v,u)):
            if method=='parents':
                edge=m.new_bool_var('parent_edge')
                m.add_implication(edge,on)
                m.add(ranks[b]>ranks[a]).only_enforce_if(edge)
            else:
                edge=m.new_int_var(0,size,'flow')
                m.add(edge<=size*on)
            input_arcs[b].append(edge)
            output[a].append(edge)
    for u in nodes:
        if method=='parents':
            m.add(sum(input_arcs[u])+roots[u]==1)
        else:
            supply=m.new_int_var(0,size,'root_supply')
            m.add(supply<=size*roots[u])
            m.add(sum(input_arcs[u])+supply-sum(output[u])==1)
    return choices,nodes,activations


class PhaseInterface:
    """Project integer signed inventory onto selected phases, summing equal residual tuples."""
    def __init__(self, problem):
        corner,regions=_sphere(problem)
        raw=[]
        labels=set()
        for t,reg in enumerate(regions):
            pair=[]
            for z in (0,1):
                colors,weight=_features(reg,z)
                counts=Counter()
                for j,color in enumerate(colors):
                    label=(int(problem.masks[3*t,j]),color)
                    mate=_reciprocal(*label)
                    k=min(label,mate)
                    labels.add(k)
                    counts[k]+=1 if label==k else -1
                pair.append((counts,weight))
            raw.append(pair)
        keys=sorted(labels)
        self.mod=[_reciprocal(*k)==k for k in keys]+[False]
        self.rows=[tuple(tuple(c[k] for k in keys)+(w,) for c,w in pair) for pair in raw]
        degree=Counter(corner)
        odd=[v for v,d in degree.items() if d%2]
        sums={0}
        for v,d in degree.items():
            if d==6 or len(odd)==2 and v in odd:continue
            sums |= {x+6-d for x in sums}
        if len(odd)==2:
            a,b=(6-degree[v] for v in odd)
            sums={x+(a*i+b*(i^(problem.total_segments%2))) for x in sums for i in (0,1)}
        self.euler={6-x for x in sums}
        self.cache={}

    def project(self, pins, boundary=(), max_states=10000, *, method='auto'):
        if method not in ('auto','single','split','legacy'):
            raise ValueError('Unknown signed phase projection order.')
        pins=checked_pins(len(self.rows),pins)
        boundary=tuple(boundary)
        if (len(set(boundary))!=len(boundary) or any(type(t)is not int or not 0<=t<len(self.rows) for t in boundary)
                or max_states<1):
            raise ValueError("Invalid phase interface or budget.")
        key=tuple(sorted(pins.items())),boundary
        if key in self.cache:
            return dict(self.cache[key],transitions=0,reused=True)
        for (old_pins,old_boundary),compiled in self.cache.items():
            base=dict(old_pins)
            if not all(pins.get(i)==v for i,v in base.items()):
                continue
            extra=set(pins)-base.keys()
            if not extra<=set(old_boundary) or not set(boundary)<=set(old_boundary):
                continue
            patterns=set()
            for mask in compiled['patterns']:
                if all((mask>>old_boundary.index(t))&1==pins[t] for t in extra):
                    patterns.add(sum(((mask>>old_boundary.index(t))&1)<<j for j,t in enumerate(boundary)))
            return dict(status='COMPLETE',patterns=sorted(patterns),transitions=0,reused=True)
        if method!='legacy':
            from r35_phase_frontier import compile_projection
            result=compile_projection(self,pins,boundary,max_states=max_states,
                                      split=None if method=='auto' else method=='split')
            result['reused']=False
            if result['status']=='COMPLETE':
                self.cache[key]=result
            return result
        start=(0,)*len(self.mod)
        order=sorted(range(len(self.rows)),key=lambda t:(t not in pins,t in boundary,t))
        # Once a signed class has no remaining contributing tile, its final
        # balance is known. Failing rows can be deleted before the next layer.
        retire=defaultdict(list)
        for j in range(len(start)-1):
            live=[i for i,t in enumerate(order) if any(r[j] for r in self.rows[t])]
            if live:retire[max(live)].append(j)
        states={(start,0)}
        work=0
        for i,t in enumerate(order):
            nxt=set()
            for counts,mask in states:
                for z in ((pins[t],) if t in pins else (0,1)):
                    work+=1
                    vals=tuple((a+b)%2 if self.mod[j] else a+b
                               for j,(a,b) in enumerate(zip(counts,self.rows[t][z])))
                    if any(vals[j] for j in retire[i]):continue
                    new=mask | (z<<boundary.index(t)) if t in boundary else mask
                    nxt.add((vals,new))
                    if len(nxt)>max_states:
                        return dict(status='UNKNOWN',transitions=work,reused=False)
            states=nxt
        patterns=sorted({mask for vals,mask in states if vals[-1] in self.euler})
        result=dict(status='COMPLETE',patterns=patterns,transitions=work,reused=False)
        self.cache[key]=result
        return result
