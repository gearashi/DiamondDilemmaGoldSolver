"""Generate physical-tour cuts and oriented corner-ring columns on demand."""
from collections import Counter, defaultdict
from itertools import combinations

from r27_structure_filters import _vertex_links
from validator import reverse_mask


def fractional_cuts(check, bonds=(), pins=None, limit=12, scale=100000):
    """GLOP proposes structural inequalities; only valid integer inequalities enter CP."""
    from ortools.graph.python.max_flow import SimpleMaxFlow
    from ortools.linear_solver import pywraplp
    s = pywraplp.Solver.CreateSolver('GLOP')
    x = {p:s.NumVar(0,1,'') for p in check.mate}
    by_side = defaultdict(list)
    phase = [s.NumVar(0,1,'') for _ in range(check.problem.n)]
    for t,v in (pins if pins is not None else {0:0}).items():
        s.Add(phase[t]==v)
    for pair,var in x.items():
        a,b = pair
        by_side[a].append(var)
        by_side[b].append(var)
        mask,c = check.inputs.sides[a]
        flip = c ^ check.inputs.sides[b][1] ^ (mask.bit_count()&1)
        diff = phase[a//3] - (1-phase[b//3] if flip else phase[b//3])
        s.Add(diff<=1-var)
        s.Add(-diff<=1-var)
    for a in range(3*check.problem.n):
        s.Add(sum(by_side[a])==1)
    for pair in bonds:
        if tuple(sorted(pair)) in x:
            s.Add(x[tuple(sorted(pair))]==1)
    records = []
    seen = set()
    def add_cut(kind, coeff, rhs):
        key = (tuple(sorted(coeff.items())),rhs)
        if key in seen or sum(c*x[p].solution_value() for p,c in coeff.items())>=rhs-1e-6:
            return False
        seen.add(key)
        check.m.add(sum(c*check.mate[p] for p,c in coeff.items())>=rhs)
        s.Add(sum(c*x[p] for p,c in coeff.items())>=rhs)
        records.append(kind)
        return True
    def crossings(part):
        return {p:c for p,links in check.transitions.items()
                if (c:=sum((i in part)!=(j in part) for i,j in links))}
    for _ in range(limit):
        if s.Solve()!=s.OPTIMAL:
            return dict(cuts=records,status='LP_INFEASIBLE_UNVERIFIED')
        weights = Counter()
        for pair,links in check.transitions.items():
            value = x[pair].solution_value()
            if value>1e-9:
                for a,b in links:
                    weights[tuple(sorted((a,b)))] += value
        flow = SimpleMaxFlow()
        for (u,v),w in weights.items():
            cap = int(round(w*scale))
            flow.add_arc_with_capacity(u,v,cap)
            flow.add_arc_with_capacity(v,u,cap)
        changed = False
        for t in range(1,check.strands):
            if flow.solve(0,t)!=flow.OPTIMAL:
                continue
            part = set(flow.get_source_side_min_cut())
            if 0<len(part)<check.strands and flow.optimal_flow()<2*scale+check.strands:
                if add_cut('strand',crossings(part),2):
                    changed = True
                    break
        if changed:
            continue
        # A proper odd set of physical sides needs at least one external mate.
        by_mask = defaultdict(list)
        for a,(mask,_) in enumerate(check.inputs.sides):
            if mask==reverse_mask(mask):
                by_mask[mask].append(a)
        for sides in by_mask.values():
            if len(sides)>14:
                continue
            for width in range(3,len(sides)//2+1,2):
                for subset in combinations(sides,width):
                    part = set(subset)
                    if add_cut('odd-sides',{p:1 for p in x if (p[0] in part)!=(p[1] in part)},1):
                        changed = True
                        break
                if changed:break
            if changed:break
        if changed:continue
        # Test three disjoint two-strand teeth crossing a fractional handle.
        near = sorted((edge for edge,v in weights.items() if 1e-7<v<1-1e-7),
                      key=lambda edge:-weights[edge])[:12]
        for teeth in combinations(near,3):
            if len(set().union(*map(set,teeth)))!=6:continue
            for flip in range(8):
                handle = {p[(flip>>i)&1] for i,p in enumerate(teeth)}
                coeff = Counter(crossings(handle))
                for tooth in teeth:
                    coeff.update(crossings(set(tooth)))
                if add_cut('comb',dict(coeff),10):
                    changed = True
                    break
            if changed:break
        if not changed:
            break
    return dict(cuts=records,status='CANDIDATES_ONLY')


def price_rings(problem, options, *, max_work=100000, limit=32):
    """Phase-I column generation. Each column uses the same mate at both ends of a side."""
    from ortools.linear_solver import pywraplp
    degrees = Counter(Counter(_vertex_links(problem)).values())
    s = pywraplp.Solver.CreateSolver('GLOP')
    coverage = [s.Constraint(1,1) for _ in options]
    census = {k:s.Constraint(v,v) for k,v in degrees.items()}
    consistency = {(a,b):s.Constraint(0,0) for a,opts in enumerate(options) for b in opts}
    mates = {(a,b):s.NumVar(0,1,'') for a,b in consistency if a<b}
    for (a,b),row in consistency.items():
        row.SetCoefficient(mates[min(a,b),max(a,b)],-1)
    for a,opts in enumerate(options):
        s.Add(sum(mates[min(a,b),max(a,b)] for b in opts)==1)
    penalty = []
    for row in [*coverage,*census.values(),*consistency.values()]:
        for sign in (-1,1):
            v = s.NumVar(0,s.infinity(),'')
            row.SetCoefficient(v,sign)
            penalty.append(v)
    s.Minimize(sum(penalty))
    generated = set()
    price_work = 0
    for _ in range(limit):
        if s.Solve()!=s.OPTIMAL:
            return dict(status='LP_INFEASIBLE_UNVERIFIED',columns=len(generated),examined=price_work)
        if s.Objective().Value()<1e-7:
            return dict(status='FRACTIONAL_COVER',columns=len(generated),examined=price_work)
        dual = {k:v.dual_value() for k,v in consistency.items()}
        cost = {(a,b):coverage[a].dual_value()+v for (a,b),v in dual.items()}
        gains = {k:v.dual_value() for k,v in census.items()}
        # An optimistic bound permits repeated physical tiles and ignores the
        # required closure, so it can only skip rows with nonnegative reduced cost.
        max_arc = max(cost.values(),default=0)
        added = []
        for first in range(len(options)):
            stack = [(first,(first,),0.0)]
            while stack:
                a,row,score = stack.pop()
                if max((score+(k-len(row)+1)*max_arc+gains[k]
                        for k in degrees if k>=len(row)),default=0)<=1e-8:
                    continue
                for b in options[a]:
                    price_work += 1
                    if price_work>max_work:
                        return dict(status='UNKNOWN',columns=len(generated),examined=price_work)
                    nxt = 3*(b//3)+(b%3+1)%3
                    new = score+cost[a,b]
                    if nxt==first:
                        if len(row) in degrees and new+gains[len(row)]>1e-8 and row not in generated:
                            added.append(row)
                            generated.add(row)
                    elif len(row)<max(degrees) and nxt>first and all(x//3!=nxt//3 for x in row):
                        stack.append((nxt,(*row,nxt),new))
                if len(added)>=128:break
            if len(added)>=128:break
        if not added:
            # Floating-point separation is never a certificate of impossibility.
            return dict(status='NO_PRICED_COLUMN_UNVERIFIED',columns=len(generated),examined=price_work)
        for row in added:
            v=s.NumVar(0,s.infinity(),'ring')
            census[len(row)].SetCoefficient(v,1)
            for a,nxt in zip(row,row[1:]+row[:1]):
                b=3*(nxt//3)+(nxt%3-1)%3
                consistency[a,b].SetCoefficient(v,1)
                coverage[a].SetCoefficient(v,1)
    return dict(status='UNKNOWN',columns=len(generated),examined=price_work)
