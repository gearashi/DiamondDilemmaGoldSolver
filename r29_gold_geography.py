"""Geographic signed/unsigned seam inventories and partial complementary-region capacity."""
from collections import Counter, defaultdict, deque
from functools import lru_cache
from itertools import combinations
from math import ceil

from r27_structure_filters import Union, bits, domain_bits
from r28_geo_color import _features, _sphere
from validator import reverse_mask


def checked_domains(problem, ds):
    base = domain_bits(problem)
    if ds is None:
        return base
    ds = tuple(ds)
    if len(ds) != problem.n or any(type(v) is not int or v < 0 or v & ~old for v,old in zip(ds,base)):
        raise ValueError("Invalid geographic domains.")
    return ds


def geography(problem, detail):
    board = problem.board
    if detail not in ("rings","sectors","landmark","vertices"):
        raise ValueError("Invalid geographic refinement.")
    graph = defaultdict(set)
    for cell in board.cells:
        for s, v in enumerate(cell.vertices):
            graph[v].add(cell.vertices[s-1])
            graph[v].add(cell.vertices[(s+1)%3])
    def distances(root):
        queue = deque([root])
        dist = {root:0}
        while queue:
            v = queue.popleft()
            for w in graph[v]:
                if w not in dist:
                    dist[w] = dist[v]+1
                    queue.append(w)
        return dist
    def macro_vertex(k):
        return next(v for v,x in enumerate(board.vertices) if x == ((k,board.subdivision),))
    ring = distances(macro_vertex(0))
    second = distances(macro_vertex(2))
    level = [(ring[v], second[v] if detail == "landmark" else -1) for v in range(len(graph))]
    def cell_class(cell):
        return (tuple(sorted(level[v] for v in cell.vertices)),
                cell.face if detail == "sectors" else -1)
    classes = [cell_class(c) for c in board.cells]
    halves = {}
    for a,s,b,t in problem.edges:
        cell = board.cells[a]
        start,end = level[cell.vertices[s]],level[cell.vertices[(s+1)%3]]
        halves[a,s] = (classes[a], classes[b], start, end)
        halves[b,t] = (classes[b], classes[a], end, start)
    return ring, halves, {v for v,adj in graph.items() if len(adj) != 6 or detail == "vertices"}


def _reverse_side(half):
    a,b,x,y = half
    return b,a,y,x


def layer_bound(problem, ds=None, *, gold=False, detail="rings", seconds=3, max_atoms=50000, pins=None):
    """Maximize possible pairings of reciprocal half-sides with physical tile and group capacities."""
    from ortools.sat.python import cp_model
    if seconds <= 0 or max_atoms < 1:
        raise ValueError("Expected positive solver limits.")
    ds = checked_domains(problem, ds)
    ring,half,tracked = geography(problem, detail)
    pins = {} if pins is None else dict(pins)
    if any(v not in ring or p not in (0,1) for v,p in pins.items()) or (pins and not gold):
        raise ValueError("Expected mesh vertex color pins in Gold mode.")
    tracked.update(pins)
    regs = _sphere(problem)[1] if gold else None
    model = cp_model.CpModel()
    vertex = {v:model.new_bool_var("corner") for v in sorted(tracked)} if gold else {}
    for v,col in pins.items():
        model.add(vertex[v] == col)
    rows = defaultdict(list)
    for cell,d in zip(problem.board.cells, ds):
        labels = tuple(half[cell.index,s] for s in range(3))
        corners = tuple(v if v in vertex else -1 for v in cell.vertices)
        rows[d,labels,corners].append(cell.index)
    stock, incidence, euler = defaultdict(list), defaultdict(list), []
    atom_count = 0
    weights = [(_features(r,0)[1],_features(r,1)[1]) for r in regs] if gold else []
    for (d, labels, corners), cells in rows.items():
        atoms = {}
        for code in bits(d):
            t, r = divmod(code,3)
            for phase in (range(2) if gold else (0,)):
                terms = []
                channels = []
                for s,h in enumerate(labels):
                    mask = int(problem.masks[code,s])
                    start = (regs[t][12*((s-r)%3)].bit_count() & 1) ^ phase if gold else 0
                    terms.append((h,mask,start))
                    if corners[s] >= 0:
                        channels.append((corners[s],start))
                sig = (t, tuple(sorted(terms)), tuple(sorted(channels)), weights[t][phase] if gold else 0)
                if sig in atoms:
                    continue
                atom_count += 1
                if atom_count > max_atoms:
                    return dict(status="UNKNOWN", reason="atom-limit", atoms=atom_count, upper=None)
                var = atoms[sig] = model.new_bool_var("placement")
                stock[t].append(var)
                for key,num in Counter(terms).items():
                    incidence[key].append(num*var)
                for v,col in channels:
                    model.add(vertex[v] == col).only_enforce_if(var)
                if gold:
                    euler.append(weights[t][phase]*var)
        model.add(sum(atoms.values()) == len(cells))
    for t in range(problem.n):
        model.add_exactly_one(stock[t])
    if gold:
        degree = Counter(v for c in problem.board.cells for v in c.vertices)
        model.add(sum(euler)+sum((6-degree[v])*x for v,x in vertex.items()) == 6)
    total = []
    done = set()
    for key, terms in list(incidence.items()):
        if key in done:
            continue
        h,mask,start = key
        mate = _reverse_side(h), reverse_mask(mask), start ^ ((mask.bit_count() & 1) if gold else 0)
        done.update((key,mate))
        a = sum(terms)
        if mate == key:
            count = model.new_int_var(0,len(problem.edges),"pairs")
            if gold:
                model.add(a == 2*count)
            else:
                model.add(2*count <= a)
                total.append(count)
        else:
            b = sum(incidence.get(mate,()))
            if gold:
                model.add(a == b)
            else:
                count = model.new_int_var(0,len(problem.edges),"pairs")
                model.add(count <= a)
                model.add(count <= b)
                total.append(count)
    if not gold:
        model.maximize(sum(total))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = seconds
    result = solver.solve(model)
    upper = None
    if not gold and result in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        upper = min(len(problem.edges),ceil(solver.best_objective_bound))
    return dict(status=solver.status_name(result), upper=upper, atoms=atom_count,
                cell_groups=len(rows),branches=solver.num_branches,
                rings=[count for level,count in sorted(Counter(ring.values()).items())], seam_classes=len({min(h,_reverse_side(h)) for h in half.values()}))


def partition_capacity(problem, ds=None, *, bags=None, width=2, max_bags=100, pins=None):
    """Check weighted partition inequalities after contracting all optimistic links outside a cell bag."""
    from r29_gold_phase import checked_pins
    ds = checked_domains(problem,ds)
    pins = checked_pins(problem.n,pins or {})
    if max_bags < 1 or width < 1:
        raise ValueError("Expected positive separator size and budget.")
    if not all(ds):
        return dict(feasible=False,reason="empty-domain",bags=0)
    _,regs = _sphere(problem)
    n = 36*problem.n
    seams = Union(n)
    for a,s,b,t in problem.edges:
        for k in range(12):
            seams.join(36*a+12*s+k,36*b+12*t+11-k)
    @lru_cache(maxsize=None)
    def local(code):
        t,r = divmod(code,3)
        groups = defaultdict(list)
        for j in range(36):
            original = (j-12*r)%36
            groups[regs[t][original]].append(j)
        return [(label.bit_count() & 1,positions) for label,positions in groups.items()]
    candidates = [c for c,d in enumerate(ds) if d.bit_count() > 1]
    if bags is None:
        bags = (bag for size in range(min(len(candidates),width)+1)
                for bag in combinations(candidates,size))
    colors = [set(),set()]
    for c,d in enumerate(ds):
        for k in range(36):
            observed = set()
            for code in bits(d):
                t,r = divmod(code,3)
                if t not in pins:
                    observed.update((0,1))
                    break
                observed.add((regs[t][(k-12*r)%36].bit_count() & 1) ^ pins[t])
                if len(observed) == 2:
                    break
            if len(observed) == 1:
                colors[observed.pop()].add(36*c+k)
    checks = 0
    weights = [(1,0),(0,1),(1,1),(1,2),(2,1)]
    for bag in bags:
        if checks >= max_bags:
            break
        bag = set(bag)
        if any(type(c) is not int or not 0 <= c < problem.n for c in bag):
            raise ValueError("Invalid capacity bag.")
        checks += 1
        uf = Union(0)
        uf.p = seams.p.copy()
        for c,d in enumerate(ds):
            if c in bag:
                continue
            for code in bits(d):
                for _,positions in local(code):
                    for k in positions[1:]:
                        uf.join(36*c+positions[0],36*c+k)
        roots = {uf.find(i) for i in range(n)}
        if len(roots) <= 2:
            continue
        targets = [max(0,len({uf.find(i) for i in known})-1) for known in colors]
        need = [len(roots)-2]+[a*targets[0]+b*targets[1] for a,b in weights]
        cap = [0]*len(need)
        for c in bag:
            options = []
            for code in bits(ds[c]):
                t = code//3
                gains = [0,0]
                # A region of r distinct contracted bins can add at most r-1 independent links.
                for color, positions in local(code):
                    gains[color] += len({uf.find(36*c+k) for k in positions})-1
                for phase in ((pins[t],) if t in pins else (0,1)):
                    x,y = gains[phase], gains[phase^1]
                    options.append([x+y]+[a*x+b*y for a,b in weights])
            cap = [v+max(opt[j] for opt in options) for j,v in enumerate(cap)]
        for i,(available,required) in enumerate(zip(cap,need)):
            if required > available:
                return dict(feasible=False,bags=checks,bag=sorted(bag),required=required,
                            capacity=available,weights="both-regions" if i==0 else weights[i-1])
    return dict(feasible=None,bags=checks)
