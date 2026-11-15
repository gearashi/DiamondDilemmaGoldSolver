"""Exact outer odd-seam skeletons and reciprocal slot-projected geographic relaxations."""
from collections import Counter, defaultdict
from r27_structure_filters import bits
from r29_gold_geography import checked_domains


def slot_layers(mask):
    """Six parity bits, one for each reversal orbit in slots 1..11."""
    values = []
    for i in range(6):
        slots = (1 << i) | (1 << (10-i))
        values.append((int(mask) & slots).bit_count() % 2)
    return tuple(values)


def _matching(domains):
    """Deterministic Hall test with a physical-tile witness."""
    owner = {}
    def augment(cell, visited):
        for tile in bits(domains[cell]):
            if tile in visited:
                continue
            visited.add(tile)
            if tile not in owner or augment(owner[tile], visited):
                owner[tile] = cell
                return True
        return False
    return all(augment(c,set()) for c in sorted(range(len(domains)),key=lambda c:domains[c].bit_count()))


class _SkeletonMatching:
    def __init__(self, rows):
        self.rows = list(rows)
        self.owner = [-1]*len(rows)
        self.tile = [-1]*len(rows)
        self.feasible = all(self.augment(c,set()) for c in sorted(range(len(rows)),key=lambda c:rows[c].bit_count()))

    def augment(self, cell, seen):
        for t in bits(self.rows[cell]):
            if t in seen:
                continue
            seen.add(t)
            if self.owner[t] < 0 or self.augment(self.owner[t],seen):
                self.owner[t] = cell
                self.tile[cell] = t
                return True
        return False

    def restrict(self, c, row):
        self.rows[c] = row
        old = self.tile[c]
        if row >> old & 1:
            return True
        self.owner[old] = -1
        if self.augment(c,set()):
            return True
        self.owner[old] = c
        return False


def _apex_path(problem):
    """For an odd strand count on a noncrossing sphere, the two odd-degree apices differ in color."""
    if problem.total_segments % 2 == 0:
        return set()
    from r28_geo_color import _sphere
    try:
        _sphere(problem)
    except ValueError:
        return set()
    degrees = Counter(v for cell in problem.board.cells for v in cell.vertices)
    apex = [v for v in degrees if degrees[v]%2]
    if len(apex)!=2:
        return set()
    adj = defaultdict(list)
    for j,(a,s,_,_) in enumerate(problem.edges):
        vertices=problem.board.cells[a].vertices
        u,v=vertices[s],vertices[(s+1)%3]
        adj[u].append((v,j))
        adj[v].append((u,j))
    stack=[(apex[0],())]
    seen={apex[0]}
    while stack:
        v,path=stack.pop()
        if v==apex[1]:
            return set(path)
        for u,e in adj[v]:
            if u not in seen:
                stack.append((u,(*path,e)))
                seen.add(u)
    raise ValueError('Disconnected geographic mesh.')


def _skeleton_pose_masks(problem, ds, patterns):
    """Index the forbidden local parities of each complete mandatory component pose."""
    from constraint_pruning import Pruner
    rules = Pruner(problem,regions=[],parity=False,gold_bonds=True)
    alive = []
    blocked = [defaultdict(list) for _ in ds]
    for _,_,embeddings in rules.clusters:
        poses = [row for row in embeddings if all(ds[c] >> int(v) & 1 for c,v in row)]
        group = len(alive)
        alive.append((1<<len(poses))-1)
        bad = defaultdict(int)
        for i,row in enumerate(poses):
            for c,v in row:
                for pattern in (0,3,5,6):
                    if pattern != patterns[v]:
                        bad[c,pattern] |= 1<<i
        for (c,pattern),mask in bad.items():
            blocked[c][pattern].append((group,mask))
    return tuple(alive),blocked


def skeletons(problem, domains=None, *, max_nodes=25000, limit=1, apex=True, merge=True, early_hall=True, mandatory_poses=False):
    """Enumerate geographic odd-edge patterns. Only completely failed continuations are cached."""
    if max_nodes<1 or limit<1:
        raise ValueError('Positive skeleton budget and result limit required.')
    ds=checked_domains(problem,domains)
    n=problem.n
    patterns=[sum((int(m).bit_count()%2)<<s for s,m in enumerate(row)) for row in problem.masks]
    active=sum(bool(patterns[3*t]) for t in range(n))
    if any(v.bit_count() not in (0,2) for v in patterns):
        raise ValueError('Odd endpoint count on a tile.')
    by_pattern=[]
    for domain in ds:
        groups=defaultdict(int)
        for v in bits(domain):
            groups[patterns[v]] |= 1 << (v//3)
        by_pattern.append(groups)
    side_edges=[[-1]*3 for _ in range(n)]
    for j,(a,s,b,t) in enumerate(problem.edges):
        side_edges[a][s]=side_edges[b][t]=j
    if any(-1 in row for row in side_edges):
        raise ValueError('A skeleton requires a closed mesh.')
    apex_path=_apex_path(problem) if apex else set()
    path_mask=sum(1<<j for j in apex_path)
    # Domains allowing every rotation distinguish active tiles only by zero/two odd sides.
    full=all(d == (1 << (3*n))-1 for d in ds)
    prefix=[0]*n
    answers=[]
    negative=set()
    work=Counter()
    matcher = None
    if early_hall and not full:
        tile_rows = []
        for groups in by_pattern:
            row = 0
            for tiles in groups.values():
                row |= tiles
            tile_rows.append(row)
        matcher = _SkeletonMatching(tile_rows)
        if not matcher.feasible:
            return dict(patterns=[],active=active,complete=True,work={'early_hall_rejections':1})
    pose_root,blocked = _skeleton_pose_masks(problem,ds,patterns) if mandatory_poses else ((),None)
    if any(not group for group in pose_root):
        return dict(patterns=[],active=active,complete=True,work={'mandatory_pose_rejections':1})
    class Capped(Exception):
        pass
    def visit(c,frontier,path,r,available,poses):
        if r<0 or r>n-c:
            return False
        if c==n:
            if apex_path and not path:
                return False
            if not full and matcher is None:
                work['hall_checks']+=1
                if not _matching(available):
                    return False
            answers.append(tuple(prefix))
            return True
        # Earlier regional interiors can merge if their remaining inventory obligations match.
        key=(c,frontier,path,r,() if full else tuple(sorted(available)),poses)
        if merge and key in negative:
            work['merged']+=1
            return False
        if work['states']>=max_nodes:
            raise Capped
        work['states']+=1
        past=sum(1<<s for s,b in enumerate(problem.neighbors[c]) if b<c)
        actual=sum(((frontier>>j)&1)<<s for s,j in enumerate(side_edges[c]))
        found=False
        for pattern,tiles in sorted(by_pattern[c].items(),key=lambda row:(-int(r>=(n-c)/2)*bool(row[0]), row[0])):
            if (pattern^actual)&past:
                continue
            opened=closed=0
            for s,j in enumerate(side_edges[c]):
                if not(pattern>>s&1):
                    continue
                if past>>s&1:
                    closed |= 1<<j
                else:
                    opened |= 1<<j
            child_poses = poses
            if poses and pattern in blocked[c]:
                remaining = list(poses)
                for g,mask in blocked[c][pattern]:
                    remaining[g] &= ~mask
                    if not remaining[g]:
                        break
                if not all(remaining):
                    work['mandatory_pose_rejections'] += 1
                    continue
                child_poses = tuple(remaining)
            if matcher is not None:
                work['early_hall_checks'] += 1
                if not matcher.restrict(c,tiles):
                    matcher.rows[c] = tile_rows[c]
                    work['early_hall_rejections'] += 1
                    continue
            prefix[c]=pattern
            found |= visit(c+1,(frontier & ~closed)|opened,path^((opened&path_mask).bit_count()%2),
                           r-bool(pattern),() if full else (*available,tiles),child_poses)
            if matcher is not None:
                # The matching from a child also satisfies its widened parent row.
                matcher.rows[c] = tile_rows[c]
            if len(answers)==limit:
                raise Capped
        if merge and not found:
            negative.add(key)
        return found
    complete=True
    try:
        visit(0,0,0,active,(),pose_root)
    except Capped:
        complete=False
    return dict(patterns=answers, active=active, complete=complete, work=dict(work))


def skeleton_domains(problem, domains, pattern):
    ds=checked_domains(problem,domains)
    if len(pattern)!=problem.n:
        raise ValueError('One geographic pattern per cell required.')
    return tuple(sum(1<<v for v in bits(d)
                     if sum((int(m).bit_count()%2)<<s for s,m in enumerate(problem.masks[v]))==p)
                 for p,d in zip(pattern,ds))


def projected(problem, domains=None, *, layers=(0,), target=None, regions=(), effort=.1):
    """Complete physical bijection with projected seam bits and shared mismatch flags across layers."""
    from ortools.sat.python import cp_model
    ds=checked_domains(problem,domains)
    layers=tuple(layers)
    target=len(problem.edges) if target is None else target
    if (not layers or len(set(layers))!=len(layers) or any(type(k)!=int or k not in range(6) for k in layers)
            or not 0<=target<=len(problem.edges) or effort<=0):
        raise ValueError('Invalid slot layers, soft threshold or budget.')
    if not all(ds):
        return dict(status='INFEASIBLE',branches=0)
    m=cp_model.CpModel()
    tiles=[]
    labels=[]
    layerbits=[[slot_layers(int(mask)) for mask in row] for row in problem.masks]
    for d in ds:
        tile=m.new_int_var(0,problem.n-1,'tile')
        sides=[[m.new_bool_var('slot') for _ in layers] for _ in range(3)]
        variables=[tile,*[v for side in sides for v in side]]
        rows=sorted({(v//3,*(layerbits[v][s][k] for s in range(3) for k in layers)) for v in bits(d)})
        m.add_allowed_assignments(variables,rows)
        labels.append(sides)
        tiles.append(tile)
    m.add_all_different(tiles)
    defects=[]
    differences=[]
    for a,s,b,t in problem.edges:
        diffs=[]
        for j in range(len(layers)):
            flag=m.new_bool_var('layer_defect')
            m.add_bool_xor([labels[a][s][j],labels[b][t][j],~flag])
            diffs.append(flag)
        differences.append(diffs)
        defect=m.new_bool_var('projected_seam_defect')
        m.add_max_equality(defect,diffs)
        defects.append(defect)
    m.add(sum(defects)<=len(defects)-target)
    for region in regions:
        inner=set(region)
        if not inner or not inner<=set(range(problem.n)):
            raise ValueError('Expected a nonempty set of geographic cells.')
        for j,k in enumerate(layers):
            charges=[sum(layerbits[3*t][s][k] for s in range(3))%2 for t in range(problem.n)]
            terms=[]
            for c in sorted(inner):
                bit=m.new_bool_var('tile_charge')
                m.add_element(tiles[c],charges,bit)
                terms.append(bit)
            for e,(a,s,b,t) in enumerate(problem.edges):
                if a in inner and b in inner:
                    terms.append(differences[e][j])
                elif a in inner:
                    terms.append(labels[a][s][j])
                elif b in inner:
                    terms.append(labels[b][t][j])
            m.add_bool_xor([*terms,m.new_constant(1)])
    solver=cp_model.CpSolver()
    solver.parameters.max_deterministic_time=effort
    solver.parameters.num_search_workers=1
    status=solver.solve(m)
    row=dict(status=solver.status_name(status),branches=int(solver.num_branches),layers=layers,
             shared_defects=len(defects),regions=len(regions))
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        codes=[]
        for cell,(var,sides) in enumerate(zip(tiles,labels)):
            tile=solver.value(var)
            values=tuple(solver.value(bit) for col in sides for bit in col)
            codes.append(next(v for v in bits(ds[cell]) if v//3==tile and
                              tuple(layerbits[v][s][k] for s in range(3) for k in layers)==values))
        row.update(codes=codes,projected_defects=sum(solver.value(x) for x in defects),
                   actual_matches=int(problem.validate(codes)['matched_edges']))
    return row
