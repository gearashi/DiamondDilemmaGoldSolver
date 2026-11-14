"""Signed inventories and conditional planar complementary-region cuts on the Gold sphere."""
from collections import Counter, defaultdict
from functools import lru_cache

from r27_structure_filters import Union, _regions, _vertex_links, bits, domain_bits, oriented_segments
from validator import reverse_mask


def _sphere(problem):
    corners = _vertex_links(problem)
    if corners is None:
        raise ValueError("Color constraints require an oriented spherical mesh.")
    regions = [_regions(oriented_segments(problem,3*t)) for t in range(problem.n)]
    if any(r is None for r in regions):
        raise ValueError("Crossing gold arcs invalidate the complementary-region model.")
    return corners,regions


def _features(labels, phase):
    color = [(x.bit_count() & 1) ^ phase for x in labels]
    corners = [color[12*s] for s in range(3)]
    runs = sum(color[12*s+k] for s in range(3) for k in range(12)
               if k == 0 or labels[12*s+k] != labels[12*s+k-1])
    faces = len({x for k,x in enumerate(labels) if color[k]})
    return corners,sum(corners)-3*runs+6*faces


def signed_phase_model(problem, *, euler=True, pins=None, bonds=()):
    """Physical necessary conditions, independent of geographic placements.

    For each physical side its signed label is (endpoint mask, starting color).
    Reversing a side reverses the mask and flips its starting color iff its
    endpoint count is odd. Every signed class needs its reciprocal inventory.
    """
    from ortools.sat.python import cp_model
    corners,regions = _sphere(problem)
    model = cp_model.CpModel()
    xs = [model.new_bool_var("phase_%d"%t) for t in range(problem.n)]
    if pins is None:model.add(xs[0] == 0)  # global color complement symmetry
    else:
        for t,phase in pins.items():
            if not 0 <= t < problem.n or phase not in (0,1):raise ValueError("Invalid tile phase.")
            model.add(xs[t] == phase)
    counts = defaultdict(list)
    corner_colors,weights = [],[]
    for t,(x,r) in enumerate(zip(xs,regions)):
        color,w0 = _features(r,0)
        _,w1 = _features(r,1)
        weights.append(w0+(w1-w0)*x)
        corner_colors.append(color)
        for s in range(3):
            m = int(problem.masks[t*3,s])
            counts[m,color[s]].append(1-x)
            counts[m,color[s]^1].append(x)
    done = set()
    for (m,c),items in list(counts.items()):
        if (m,c) in done:continue
        mate = reverse_mask(m),c^(m.bit_count() & 1)
        if mate == (m,c):
            model.add(sum(items) == 2*model.new_int_var(0,len(items)//2,"pairs"))
        else:model.add(sum(items) == sum(counts.get(mate,())))
        done.update(((m,c),mate))
    for a,b in bonds:
        t,s = divmod(a,3)
        u,j = divmod(b,3)
        ma,mb = int(problem.masks[3*t,s]),int(problem.masks[3*u,j])
        if t == u or reverse_mask(ma) != mb:
            model.add_bool_or([])
            continue
        flip = corner_colors[t][s] ^ corner_colors[u][j] ^ (ma.bit_count() & 1)
        model.add(xs[t] != xs[u] if flip else xs[t] == xs[u])
    if euler:
        degree = Counter(corners)
        exceptional = {v:model.new_bool_var("corner_%d"%v) for v,d in degree.items() if d != 6}
        odd = [v for v,d in degree.items() if d % 2]
        if len(odd) == 2:
            a,b = (exceptional[v] for v in odd)
            model.add(a != b if problem.total_segments % 2 else a == b)
        # Six times the Euler characteristic of the dark disk. Every mesh
        # vertex contributes once for each of its incident tile corners.
        model.add(sum(weights) + sum((6-degree[v])*x for v,x in exceptional.items()) == 6)
    return model,xs


def phase_probe(problem, *, euler=True, pins=None, bonds=(), probe_tiles=(), potential_bonds=(), seconds=3):
    """Certify unsupported phase choices and individual physical bonds; UNKNOWN never prunes."""
    from ortools.sat.python import cp_model
    if seconds <= 0:raise ValueError("Expected a positive per-query timeout.")
    model,phase = signed_phase_model(problem,euler=euler,pins=pins,bonds=bonds)
    s = cp_model.CpSolver()
    s.parameters.num_search_workers = 1
    s.parameters.max_time_in_seconds = seconds
    seen = set()
    r = s.solve(model)
    calls = 1
    if r in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        seen.update((t,s.value(x)) for t,x in enumerate(phase))
    unsupported = []
    if seen:
        for t in probe_tiles:
            for value in (0,1):
                if (t,value) in seen:continue
                model.clear_assumptions()
                model.add_assumption(phase[t] if value else ~phase[t])
                rr = s.solve(model)
                calls += 1
                if rr == cp_model.INFEASIBLE:unsupported.append((t,value))
                if rr in (cp_model.FEASIBLE,cp_model.OPTIMAL):
                    seen.update((u,s.value(x)) for u,x in enumerate(phase))
        model.clear_assumptions()
    rejected = []
    for a,b in potential_bonds:
        trial,_ = signed_phase_model(problem,euler=euler,pins=pins,bonds=(*bonds,(a,b)))
        if s.solve(trial) == cp_model.INFEASIBLE:rejected.append((a,b))
        calls += 1
    return dict(status=s.status_name(r),unsupported_phases=unsupported,unsupported_bonds=rejected,calls=calls)


def dual_cuts(problem, domains, *, max_boards=10000, deterministic_seconds=1):
    """Exact residual placement model with lazy cuts for sealed complementary pockets."""
    from ortools.sat.python import cp_model
    _sphere(problem)
    ds = tuple(domains)
    base = domain_bits(problem)
    if len(ds) != problem.n or any(type(d) is not int or d < 0 or d & ~a for d,a in zip(ds,base)):
        raise ValueError("Invalid domain restriction.")
    pairs = sum(a.bit_count() for a in ds)
    if pairs+sum(ds[a].bit_count()*ds[b].bit_count() for a,s,b,t in problem.edges) > 2000000:
        return dict(complete=False,exists=None,boards=0,cuts=0,reason="residual-model-size")
    m = cp_model.CpModel()
    choices = [{v:m.new_bool_var("p%d_%d"%(c,v)) for v in bits(mask)} for c,mask in enumerate(ds)]
    for row in choices:m.add_exactly_one(row.values())
    for t in range(problem.n):
        m.add_exactly_one(x for row in choices for code,x in row.items() if code//3 == t)
    for a,s,b,t in problem.edges:
        by = defaultdict(list)
        for v,x in choices[b].items():by[int(problem.masks[v,t])].append(x)
        for v,x in choices[a].items():m.add_bool_or([~x,*by[reverse_mask(int(problem.masks[v,s]))]])
    # Contract the permanent seam intervals once. Each local successor graph
    # is guarded by its placement. A component survives if all its incident
    # placements survive, regardless of everything outside it.
    seams = Union(36*problem.n)
    for a,s,b,t in problem.edges:
        for k in range(12):seams.join(36*a+12*s+k,36*b+12*t+11-k)
    base_links = seams.p.copy()
    degrees = Counter(_vertex_links(problem))
    apex = {v for v,d in degrees.items() if d%2}
    corner = _vertex_links(problem)
    dual_cut_count = boards = 0
    @lru_cache(maxsize=512)
    def intra(v):
        labels = _regions(oriented_segments(problem,v))
        first = {}
        return tuple((k,first.setdefault(label,k)) for k,label in enumerate(labels))
    s = cp_model.CpSolver()
    s.parameters.num_search_workers = 1
    s.parameters.max_deterministic_time = deterministic_seconds
    while boards < max_boards:
        result = s.solve(m)
        if result == cp_model.INFEASIBLE:
            return dict(complete=True,exists=False,boards=boards,cuts=dual_cut_count)
        if result not in (cp_model.FEASIBLE,cp_model.OPTIMAL):break
        codes = [next(v for v,x in row.items() if s.value(x)) for row in choices]
        boards += 1
        uf = Union(0)
        uf.p = base_links.copy()
        for c,v in enumerate(codes):
            for i,j in intra(v):uf.join(36*c+i,36*c+j)
        regions = defaultdict(set)
        for i in range(36*problem.n):regions[uf.find(i)].add(i//36)
        if len(regions) == 2:
            if not problem.validate(codes)["valid"]:raise ArithmeticError("Invalid two-region witness.")
            return dict(complete=True,exists=True,boards=boards,cuts=dual_cut_count,witness=codes)
        if len(apex) == 2 and problem.total_segments%2:
            roots = {uf.find(36*(i//3)+12*(i%3)) for i,v in enumerate(corner) if v in apex}
            # Gold separates the two odd-degree apices for odd segment count.
            # An entire region with no such apex is forbidden.
            pockets = [cells for root,cells in regions.items() if root not in roots]
        else:
            # Three preserved distinct regions contradict one simple loop.
            pockets = [set.union(*sorted(regions.values(),key=len)[:3])]
        if not pockets:raise ArithmeticError("Disconnected complements without a cut.")
        for cells in pockets:
            m.add_bool_or([~choices[c][codes[c]] for c in cells])
            dual_cut_count += 1
    return dict(complete=False,exists=None,boards=boards,cuts=dual_cut_count)



def phase_bond_core(problem, bonds, *, euler=True, seconds=3):
    """Shrink an incompatible physical seam set; every reported core is independently UNSAT."""
    from ortools.sat.python import cp_model
    model,xs = signed_phase_model(problem,euler=euler)
    _,regs = _sphere(problem)
    guards = {}
    for a,b in bonds:
        t,s = divmod(a,3)
        u,j = divmod(b,3)
        if t == u or reverse_mask(int(problem.masks[3*t,s])) != int(problem.masks[3*u,j]):
            return dict(status="INFEASIBLE",bonds=[(a,b)],minimal=True,queries=0)
        ca = regs[t][12*s].bit_count() & 1
        cb = regs[u][12*j].bit_count() & 1
        flip = ca ^ cb ^ (int(problem.masks[3*t,s]).bit_count()%2)
        guard = model.new_bool_var("bond_%d_%d"%(a,b))
        model.add(xs[t] != xs[u] if flip else xs[t] == xs[u]).only_enforce_if(guard)
        guards[guard.index] = (guard,(a,b))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = seconds
    calls = 0
    def check(subset):
        nonlocal calls
        calls += 1
        model.clear_assumptions()
        model.add_assumptions([guards[i][0] for i in subset])
        return solver.solve(model)
    result = check(list(guards))
    if result != cp_model.INFEASIBLE:return dict(status=solver.status_name(result),bonds=None,queries=calls)
    core = list(solver.sufficient_assumptions_for_infeasibility())
    minimal = True
    for x in list(core):
        if x not in core:continue
        remaining = [v for v in core if x != v]
        r = check(remaining)
        if r == cp_model.INFEASIBLE:
            core = list(solver.sufficient_assumptions_for_infeasibility())
        elif r not in (cp_model.FEASIBLE,cp_model.OPTIMAL):minimal = False
    # Rebuild using only the returned physical seams, without guard variables.
    reduced = [guards[i][1] for i in core]
    independent,_ = signed_phase_model(problem,euler=euler,bonds=reduced)
    if solver.solve(independent) != cp_model.INFEASIBLE:
        return dict(status="UNKNOWN",bonds=None,queries=calls+1)
    return dict(status="INFEASIBLE",bonds=reduced,minimal=minimal,queries=calls+1)



def prune_signed_core(problem, domains, bonds, *, seconds=3):
    """Remove residual placements that would force every seam of a proved physical core."""
    base = domain_bits(problem)
    ds = tuple(domains)
    if len(ds) != len(base) or any(type(v) is not int or v<0 or v&~b for v,b in zip(ds,base)):
        raise ValueError("Invalid domain restriction.")
    proof = phase_bond_core(problem,bonds,seconds=seconds)
    if proof["status"] != "INFEASIBLE":return ds,dict(proof=proof,removed=0)
    core = frozenset(tuple(sorted(pair)) for pair in proof["bonds"])
    def dart(v,s):return 3*(v//3)+(s-v%3)%3
    def bond(a,s,b,t):return tuple(sorted((dart(a,s),dart(b,t))))
    fixed = {c:row.bit_length()-1 for c,row in enumerate(ds) if row.bit_count()==1}
    known = {bond(fixed[a],s,fixed[b],t) for a,s,b,t in problem.edges if a in fixed and b in fixed}
    missing = core-known
    if not missing:
        return (0,)*problem.n,dict(proof=proof,removed=sum(x.bit_count() for x in ds),infeasible=True)
    if len(missing)>3:return ds,dict(proof=proof,removed=0)
    out = list(ds)
    for c,row in enumerate(ds):
        if c in fixed:continue
        contacts = [(s,fixed[b],t) for s,(b,t) in enumerate(problem.board.neighbors[c]) if b in fixed]
        if len(contacts)<len(missing):continue
        for v in bits(row):
            if missing <= {bond(v,s,other,t) for s,other,t in contacts}:
                out[c] ^= 1<<v
    return tuple(out),dict(proof=proof,removed=sum((a^b).bit_count() for a,b in zip(ds,out)),infeasible=not all(out))
