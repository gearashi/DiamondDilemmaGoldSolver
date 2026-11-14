"""Physical sewing deductions for the spherical Gold puzzle."""
from collections import Counter, defaultdict
from itertools import combinations
from random import Random

from r27_structure_filters import Union, bits
from r28_geo_sewing import PosePacking, _bonds
from r29_gold_phase import SignedParity
from validator import reverse_mask


def ribbon_genus(problem, bonds):
    """Genus of the oriented ribbon graph of sewn tile sides (isolates omitted)."""
    mates = _bonds(problem, bonds)
    if not mates:
        return 0
    vertices = {s // 3 for s in mates}
    neighbors = {v: [3*v + i for i in range(3) if 3*v+i in mates] for v in vertices}
    union = Union(problem.n)
    for a, b in mates.items():
        union.join(a // 3, b // 3)
    successor = {}
    for a, b in mates.items():
        row = neighbors[b // 3]
        successor[a] = row[(row.index(b) + 1) % len(row)]
    remaining = set(mates)
    boundaries = 0
    while remaining:
        v = next(iter(remaining))
        boundaries += 1
        while v in remaining:
            remaining.remove(v)
            v = successor[v]
    components = len({union.find(v) for v in vertices})
    return (2*components - len(vertices) + len(mates)//2 - boundaries)//2


class GeographicConflicts:
    """Explain failure with every unchosen physical bond free under fixed domains."""
    def __init__(self, problem, domains):
        self.problem = problem
        self.packing = PosePacking(problem, domains)
        self.cores = []
        self.queries = 0

    def impossible(self, rows):
        self.queries += 1
        return (ribbon_genus(self.problem, rows) > 0
                or self.packing.check(rows, max_steps=2000)['feasible'] is False)

    def learn(self, bonds, limit=48):
        rows = frozenset(tuple(sorted(pair)) for pair in bonds)
        for core in self.cores:
            if core <= rows:
                return core
        if not self.impossible(rows):
            return None
        # Delete chunks first. A budget only limits generalization.
        remaining = sorted(rows)
        attempts = 0
        width = max(1, len(remaining)//2)
        while width and attempts < limit:
            pos = 0
            while pos < len(remaining) and attempts < limit:
                trial = remaining[:pos] + remaining[pos+width:]
                attempts += 1
                if self.impossible(trial):
                    remaining = trial
                else:
                    pos += width
            width //= 2
        core = frozenset(remaining)
        if not self.impossible(core):
            raise ArithmeticError("Geographic conflict failed independent replay.")
        self.cores = [old for old in self.cores if not core <= old]
        self.cores.append(core)
        return core


def _partner_channel(check):
    """Connect geographic dart IDs through one shared partner variable per physical side."""
    from ortools.sat.python import cp_model
    p, m = check.problem, check.m
    n = 3*p.n
    possible = [[] for _ in range(n)]
    for a,b in check.mate:
        possible[a].append(b)
        possible[b].append(a)
    if any(not row for row in possible) or any(not row for row in check.domains):
        m.add_bool_or([])
        return dict(status='CHANNELLED',encoding='partner-elements',entries=0,geographic_seams=len(p.edges))
    partner = [m.new_int_var_from_domain(cp_model.Domain.from_values(row),'partner') for row in possible]
    for (a,b),chosen in check.mate.items():
        m.add(partner[a]==b).only_enforce_if(chosen)
        m.add(partner[b]==a).only_enforce_if(chosen)
    tiles = []
    darts = []
    for domain in check.domains:
        pose = m.new_int_var_from_domain(cp_model.Domain.from_values(list(bits(domain))),'placement')
        tile = m.new_int_var(0,p.n-1,'tile')
        rot = m.new_int_var(0,2,'rotation')
        tiles.append(tile)
        m.add(pose==3*tile+rot)
        row = []
        for side in range(3):
            local = m.new_int_var(0,2,'physical_side')
            m.add_element(rot,[(side-i)%3 for i in range(3)],local)
            dart = m.new_int_var(0,n-1,'physical_dart')
            m.add(dart==3*tile+local)
            row.append(dart)
        darts.append(row)
    m.add_all_different(tiles)
    for a,s,b,t in p.edges:
        m.add_element(darts[a][s],partner,darts[b][t])
    return dict(status='CHANNELLED',encoding='partner-elements',entries=len(p.edges)*n,
                geographic_seams=len(p.edges))


def channel_geographic(check, max_rows=120000):
    """Channel each mesh seam to its physical bond and both signed placements."""
    p, ds = check.problem, check.domains
    if sum(ds[a].bit_count()*ds[b].bit_count() for a, _, b, _ in p.edges) > max_rows:
        if len(p.edges)*3*p.n <= max_rows:
            return _partner_channel(check)
        return {'status':'SIZE_LIMIT'}
    from ortools.sat.python import cp_model
    m = check.m
    board = [m.new_int_var_from_domain(cp_model.Domain.from_values(list(bits(row)) or [-1]), 'pose') for row in ds]
    tile = [m.new_int_var(0,p.n-1,'tile') for _ in ds]
    for row, value, t in zip(ds, board, tile):
        if not row:
            m.add_bool_or([])
        m.add_allowed_assignments([value,t], [(v,v//3) for v in bits(row)])
    m.add_all_different(tile)
    pairs = {p:i for i,p in enumerate(check.mate)}
    locations = defaultdict(list)
    count = 0
    for a,s,b,t in p.edges:
        rows = []
        for v in bits(ds[a]):
            first = 3*(v//3)+(s-v%3)%3
            for w in bits(ds[b]):
                second = 3*(w//3)+(t-w%3)%3
                pair = tuple(sorted((first, second)))
                if pair in pairs:
                    rows.append((v,w,pairs[pair]))
        choices = m.new_int_var(0,max(0,len(pairs)-1),'physical_bond_at_mesh_seam')
        m.add_allowed_assignments([board[a],board[b],choices], rows)
        count += len(rows)
        for k in {r[2] for r in rows}:
            used = m.new_bool_var('bond_occupies')
            m.add(choices==k).only_enforce_if(used)
            m.add(choices!=k).only_enforce_if(~used)
            locations[k].append(used)
    for pair, index in pairs.items():
        m.add(sum(locations[index])==check.mate[pair])
    return dict(status='CHANNELLED',rows=count, geographic_seams=len(p.edges))


def physical_bonds(problem, codes):
    if (len(codes) != problem.n or sorted(c//3 for c in codes)!=list(range(problem.n))
            or any(c not in problem.domains[i] for i,c in enumerate(codes))):
        raise ValueError("Expected one legal placement of every tile.")
    pairs = []
    for a,s,b,t in problem.edges:
        v,w = codes[a],codes[b]
        if int(problem.masks[v,s]) == reverse_mask(int(problem.masks[w,t])):
            pairs.append(tuple(sorted((3*(v//3)+(s-v%3)%3,3*(w//3)+(t-w%3)%3))))
    return sorted(pairs)


def departure_cores(problem, bonds, trials=96, seed=31, seconds=2):
    """Mine location-independent physical XOR obstructions then hit all of them."""
    from ortools.sat.python import cp_model
    rng = Random(seed)
    parity = SignedParity(problem)
    mark = len(parity.base.trail)
    cores = set()
    for _ in range(trials):
        ordered = list(bonds)
        rng.shuffle(ordered)
        parity.base.rollback(mark)
        for a,b in ordered:
            if parity.bond(a,b):
                continue
            core = frozenset((a,b) for typ,a,b in parity.base.conflict if typ=='bond')
            if not core:
                raise ValueError("Intrinsic signed inventory contradiction.")
            parity.base.rollback(mark)
            if all(parity.bond(a,b) for a,b in sorted(core)):
                raise ArithmeticError("Physical XOR certificate failed replay.")
            if not any(old <= core for old in cores):
                cores = {old for old in cores if not core < old}
                cores.add(core)
            break
    ordered = sorted(cores,key=lambda c:(len(c),sorted(c)))
    disjoint=[]
    used=set()
    for core in ordered:
        if not used.intersection(core):
            disjoint.append(sorted(core))
            used.update(core)
    pack=cp_model.CpModel()
    chosen=[pack.new_bool_var('disjoint_core') for core in ordered]
    by_bond=defaultdict(list)
    for core,var in zip(ordered,chosen):
        for pair in core:by_bond[pair].append(var)
    for occurrences in by_bond.values():
        pack.add_at_most_one(occurrences)
    pack.maximize(sum(chosen))
    picker=cp_model.CpSolver()
    picker.parameters.num_search_workers=1
    picker.parameters.max_time_in_seconds=seconds
    selected=picker.solve(pack)
    if selected in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        packed=[sorted(core) for core,var in zip(ordered,chosen) if picker.value(var)]
        if len(packed)>len(disjoint):
            disjoint=packed
    if sum(map(len,disjoint))!=len({p for witness in disjoint for p in witness}):
        raise ArithmeticError("Physical departure proof reuses a bond.")
    m = cp_model.CpModel()
    breaks = {p:m.new_bool_var('broken') for p in sorted(set().union(*cores))}
    for core in ordered:
        m.add_bool_or(breaks[p] for p in core)
    m.minimize(sum(breaks.values()))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = seconds
    status = solver.solve(m)
    result = dict(cores=[sorted(c) for c in ordered], trials=trials, status=solver.status_name(status),
                  disjoint_witnesses=disjoint, lower_bound=len(disjoint))
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        result['repair_bonds']=[p for p,v in breaks.items() if solver.value(v)]
    if status == cp_model.OPTIMAL:
        result.update(minimum_breaks=int(round(solver.objective_value)),
                      repair_bonds=[p for p,v in breaks.items() if solver.value(v)])
    if ordered:
        # One subproblem for the first broken bond in the shortest core.
        # Together these disjoint branches cover every possible Gold board.
        first=sorted(ordered[0])
        result['branches']=[dict(bonds=first[:i],forbidden=[pair]) for i,pair in enumerate(first)]
    return result


def add_departure_rules(check, cores):
    """Physical parity certificates are independently recomputed under the active inventory."""
    parity=SignedParity(check.problem)
    mark=len(parity.base.trail)
    retained=0
    for row in cores:
        core=tuple(sorted({tuple(sorted(pair)) for pair in row}))
        if not core or any(p not in check.mate for p in core):
            continue
        parity.base.rollback(mark)
        if all(parity.bond(a,b) for a,b in core):
            continue
        check.m.add_bool_or([~check.mate[p] for p in core])
        retained+=1
    return retained


def tighten_bridge(check, group, seconds=2, max_cases=256):
    """For an exact two-crossing cut, test only compatible pairs and path connectivity."""
    group = frozenset(group)
    cert = check.crossing_capacity(group, seconds)
    if cert.get('upper')!=2:
        return dict(status='NO_TIGHT_BOUND',capacity=cert)
    check.strand_cut(group)
    weights = {p:sum((u in group)!=(v in group) for u,v in links)
               for p,links in check.transitions.items()}
    contributors = [p for p,n in weights.items() if n]
    ones = [p for p in contributors if weights[p]==1]
    if len(ones)*(len(ones)-1)//2 + sum(weights[p]==2 for p in contributors)>max_cases:
        return dict(status='CASE_LIMIT',capacity=cert)
    bridge_paths(check, group)
    candidates = [(p,) for p in contributors if weights[p]==2]
    candidates.extend(pair for pair in combinations(ones,2) if len(set(pair[0]+pair[1]))==4)
    rejected = 0
    for choice in candidates:
        fixed = set(choice)
        occupied = {i for p in fixed for i in p}
        components = Union(check.strands)
        for pair, links in check.transitions.items():
            if pair not in fixed and (weights[pair] or occupied.intersection(pair)):
                continue
            for u,v in links:
                if (u in group)==(v in group):
                    components.join(u,v)
        if any(len({components.find(i) for i in part})!=1
               for part in (group,set(range(check.strands))-group)):
            check.m.add_bool_or([~check.mate[p] for p in choice])
            rejected += 1
    return dict(status='CHECKED',capacity=cert,cases=len(candidates),rejected=rejected)


def salvage(problem,domains,required,bonds,seconds=5):
    """Find a complete legal seed retaining many compatible optional physical bonds."""
    from ortools.sat.python import cp_model
    _bonds(problem,required)
    _bonds(problem,bonds)
    m=cp_model.CpModel()
    pose=[]
    cells=[]
    for t in range(problem.n):
        opts=[3*c+r for c,d in enumerate(domains) for r in range(3) if d>>(3*t+r)&1]
        if not opts:return dict(status='INFEASIBLE')
        v=m.new_int_var_from_domain(cp_model.Domain.from_values(opts),'tile_pose')
        cell=m.new_int_var(0,problem.n-1,'physical_cell')
        m.add_division_equality(cell,v,3)
        pose.append(v)
        cells.append(cell)
    m.add_all_different(cells)
    required={tuple(sorted(p)) for p in required}
    all_bonds=sorted(required|{tuple(sorted(p)) for p in bonds})
    rows={}
    retained=[]
    for a,b in all_bonds:
        t,i=divmod(a,3)
        u,j=divmod(b,3)
        if int(problem.masks[3*t,i])!=reverse_mask(int(problem.masks[3*u,j])):
            raise ValueError("Salvage uses compatible side masks only.")
        if (i,j) not in rows:
            options=[]
            for c in range(problem.n):
                for r in range(3):
                    other,side=problem.board.neighbors[c][(i+r)%3]
                    options.append((3*c+r,3*other+(side-j)%3))
            rows[i,j]=options
        rule=m.add_allowed_assignments([pose[t],pose[u]],rows[i,j])
        if (a,b) not in required:
            keep=m.new_bool_var('retain_bond')
            rule.only_enforce_if(keep)
            retained.append(keep)
    m.maximize(sum(retained))
    solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=seconds
    status=solver.solve(m)
    out=dict(status=solver.status_name(status))
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        board=[-1]*problem.n
        for t,p in enumerate(pose):
            c,r=divmod(solver.value(p),3)
            board[c]=3*t+r
        actual=set(physical_bonds(problem,board))
        if not required<=actual:raise ArithmeticError("Salvage lost a mandatory bond.")
        out.update(codes=board,retained=len(actual.intersection(all_bonds)),offered=len(all_bonds))
    return out


def bridge_paths(check, group):
    """Require the strands on each side of a certified two-crossing cut to form one path."""
    for part in (group,set(range(check.strands))-group):
        ids={v:i+1 for i,v in enumerate(sorted(part))}
        arcs=[]
        for pair,links in check.transitions.items():
            for u,v in links:
                a,b=ids.get(u,0),ids.get(v,0)
                if a==b:
                    if a:check.m.add(check.mate[pair]==0)
                    continue
                ab=check.m.new_bool_var('path_step')
                ba=check.m.new_bool_var('reverse_path_step')
                check.m.add(ab+ba==check.mate[pair])
                arcs.extend(((a,b,ab),(b,a,ba)))
        # An absent vertex cannot be implicitly omitted by AddCircuit.
        for v in range(len(ids)+1):
            if not any(a==v for a,b,x in arcs):
                check.m.add_bool_or([])
                break
        else:
            check.m.add_circuit(arcs)
