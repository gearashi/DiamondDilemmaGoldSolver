"""Necessary physical sewing filters and exact transition-family checks for the Gold sphere."""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations

from r27_structure_filters import _vertex_links, bits, domain_bits
from validator import reverse_mask


def _bonds(problem,bonds):
    forced = {}
    for a,b in bonds:
        if (not all(type(x) is int and 0 <= x < 3*problem.n for x in (a,b)) or
                a//3 == b//3 or a in forced or b in forced):
            raise ValueError("Invalid or duplicate sewn physical dart.")
        forced[a],forced[b] = b,a
    return forced


def _ring_columns(options, degrees, limit):
    columns = []
    visits = 0
    for first in range(len(options)):
        stack = [(first,(first,))]
        while stack:
            d,row = stack.pop()
            for mate in options[d]:
                visits += 1
                if visits > limit:return None,visits
                nxt = 3*(mate//3)+(mate%3+1)%3
                if nxt == first:
                    if len(row) in degrees:columns.append(row)
                elif len(row) < max(degrees) and nxt > first and all(x//3 != nxt//3 for x in row):
                    stack.append((nxt,(*row,nxt)))
    return columns,visits


def ring_census(problem, options, *, max_work=500000, seconds=3):
    """Pack all physical corners into oriented vertex rings; one shared mate per side."""
    from ortools.sat.python import cp_model
    corner = _vertex_links(problem)
    if corner is None:raise ValueError("Ring census requires a spherical mesh.")
    degrees = Counter(Counter(corner).values())
    if len(options) != 3*problem.n:raise ValueError("Invalid physical pairing options.")
    for a,opts in enumerate(options):
        if any(not 0<=b<len(options) or a//3 == b//3 or a not in options[b] for b in opts):
            raise ValueError("Physical candidate relation must be symmetric and exclude self tiles.")
    cols,visits = _ring_columns(options,degrees,max_work)
    if cols is None:return dict(status="UNKNOWN",examined=visits,complete_columns=False)
    m = cp_model.CpModel()
    rows = [m.new_bool_var("r%d"%j) for j in range(len(cols))]
    mates = {(a,b):m.new_bool_var("s%d_%d"%(a,b)) for a,row in enumerate(options) for b in row if a < b}
    for a,opts in enumerate(options):m.add_exactly_one(mates[tuple(sorted((a,b)))] for b in opts)
    coverage = defaultdict(list)
    size = defaultdict(list)
    for row,active in zip(cols,rows):
        size[len(row)].append(active)
        for a,nxt in zip(row,row[1:]+row[:1]):
            b = 3*(nxt//3)+(nxt%3-1)%3
            m.add_implication(active,mates[tuple(sorted((a,b)))])
            coverage[a].append(active)
    for a in range(len(options)):m.add_exactly_one(coverage[a])
    for k,n in degrees.items():m.add(sum(size[k]) == n)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = seconds
    r = solver.solve(m)
    return dict(status=solver.status_name(r),columns=len(cols),examined=visits,complete_columns=True)


class PosePacking:
    """Cache all poses by physical component; test simultaneous disjoint occupancy."""
    def __init__(self, problem, domains=None):
        self.problem = problem
        self.ds = domain_bits(problem) if domains is None else tuple(domains)
        if len(self.ds) != problem.n or any(d<0 or d & ~b for d,b in zip(self.ds,domain_bits(problem))):
            raise ValueError("Invalid domain restriction.")
        self.poses = lru_cache(maxsize=256)(self._component)

    def _component(self,tiles,edges):
        problem = self.problem
        root = min(tiles)
        bonds = _bonds(problem,edges)
        poses = set()
        for start,domain in enumerate(self.ds):
            for rot in range(3):
                if not domain >> (3*root+rot) & 1:continue
                seen,occupied,stack = {root:(start,rot)},1<<start,[root]
                while stack:
                    u = stack.pop()
                    c,r = seen[u]
                    for side in range(3):
                        mate = bonds.get(3*u+side)
                        if mate is None:continue
                        t,s = divmod(mate,3)
                        b,j = problem.board.neighbors[c][(side+r)%3]
                        pose = b,(j-s)%3
                        if t in seen:
                            if seen[t] != pose:break
                        elif occupied >> b & 1 or not self.ds[b]>>(3*t+pose[1]) & 1:break
                        else:
                            seen[t] = pose
                            occupied |= 1<<b
                            stack.append(t)
                    else:continue
                    break
                else:poses.add(occupied)
        return tuple(sorted(poses))


    def check(self, bonds, max_steps=200000):
        forced = _bonds(self.problem,bonds)
        adj = defaultdict(set)
        for a,b in forced.items():adj[a//3].add(b//3)
        groups = []
        if any(int(self.problem.masks[3*(a//3),a%3]) != reverse_mask(int(self.problem.masks[3*(b//3),b%3]))
               for a,b in forced.items()):
            return dict(feasible=False,independent=False,nodes=0)
        unseen = set(range(self.problem.n))
        while unseen:
            root = min(unseen)
            tiles,stack = {root},[root]
            unseen.remove(root)
            while stack:
                tile = stack.pop()
                for t in adj[tile] & unseen:
                    unseen.remove(t)
                    tiles.add(t)
                    stack.append(t)
            edges = tuple(sorted((a,b) for a,b in forced.items() if a<b and a//3 in tiles))
            options = self.poses(tuple(sorted(tiles)),edges)
            if not options:return dict(feasible=False,independent=False,nodes=0)
            if len(tiles) == 1 and len(options) == self.problem.n:continue
            groups.append(options)
        groups.sort(key=len)
        memo = set()
        nodes = 0
        def visit(i,occupied):
            nonlocal nodes
            if i == len(groups):return True
            if (i,occupied) in memo:return False
            nodes += 1
            if nodes > max_steps:return None
            for pose in groups[i]:
                if pose & occupied:continue
                result = visit(i+1,occupied|pose)
                if result is not False:return result
            memo.add((i,occupied))
            return False
        result = visit(0,0)
        return dict(feasible=result,independent=True,nodes=nodes)


def boundary_lower(problem, max_k, *, max_shapes=1000000):
    """Lower bounds for arbitrary k-cell sets, including disconnected ones.

    Enumerate connected shapes; minimization over component size partitions
    drops only the simultaneous packing requirement. If capped, no bound for
    that layer is published except the universal nonnegative cut bound.
    """
    if not 0 <= max_k <= problem.n:raise ValueError("Invalid subset size.")
    neighbor = [sum(1<<int(b) for b in row if b >= 0) for row in problem.neighbors]
    level = {1<<i for i in range(problem.n)}
    lb = [0]
    sizes = []
    for k in range(1,max_k+1):
        if not level:break
        minimum = min(sum((neighbor[i]&~mask).bit_count() for i in bits(mask)) for mask in level)
        # Keep a relaxation for arbitrary sets even if different optimal
        # connected components cannot coexist in the given geometry.
        lb.append(min([minimum]+[lb[j]+lb[k-j] for j in range(1,k)]))
        sizes.append(len(level))
        if k == max_k:break
        nxt = set()
        for mask in level:
            boundary = 0
            for i in bits(mask):boundary |= neighbor[i]
            for i in bits(boundary & ~mask):nxt.add(mask | 1<<i)
            if len(nxt)>max_shapes:break
        if len(nxt)>max_shapes:break
        level = nxt
    return lb,sizes


def supply(problem, inside, bonds=()):
    """Optimistic maximum number of seams across an inventory-defined cut."""
    inside = set(inside)
    if not inside <= set(range(problem.n)):raise ValueError("Unknown physical tile.")
    assigned = _bonds(problem,bonds)
    def mask(d):
        t,s = divmod(d,3)
        return int(problem.masks[3*t,s])
    if any(mask(a) != reverse_mask(mask(b)) for a,b in assigned.items()):return -1
    fixed = sum((a//3 in inside) != (b//3 in inside) for a,b in assigned.items() if a < b)
    left,right = Counter(),Counter()
    for d in range(3*problem.n):
        if d not in assigned:
            if d//3 in inside:left[mask(d)] += 1
            else:right[reverse_mask(mask(d))] += 1
    return fixed+sum((left&right).values())


def sign_filter(problem, phases, bonds=()):
    """Test the directed strand-permutation sign for a family of unsigned sewings.

    Every planar physical tile has exactly one outgoing endpoint per strand
    after fixing its phase. Swapping two equal signed-side partners exchanges
    one pair per endpoint. A self-reciprocal (even-width) class changes sign
    by an even number of transpositions under the four-dart pairing flip.
    """
    from r28_geo_color import _sphere, _features
    _,regions = _sphere(problem)
    if len(phases) != problem.n or any(p not in (0,1) for p in phases):raise ValueError("Invalid physical phases.")
    forced = _bonds(problem,bonds)
    groups = defaultdict(list)
    ends = {}
    labels = {}
    offset = 0
    for t,reg in enumerate(regions):
        colors,_ = _features(reg,phases[t])
        for s in range(3):
            d = 3*t+s
            m = int(problem.masks[3*t,s])
            labels[d] = (m,colors[s])
            if d not in forced:groups[labels[d]].append(d)
        for j,seg in enumerate(problem.tiles[t]["segments"]):
            directions = []
            for s,p in seg:
                outgoing = ((reg[12*s+p].bit_count() & 1) ^ phases[t]) == 1
                ends[3*t+s,p] = offset+j,outgoing
                directions.append(outgoing)
            if directions[0] == directions[1]:raise ArithmeticError("Arc orientation inconsistent.")
        offset += len(problem.tiles[t]["segments"])
    def reciprocal(label):
        m,c = label
        return reverse_mask(m),c ^ (m.bit_count() & 1)
    if any(labels[b] != reciprocal(labels[a]) for a,b in forced.items()):
        return dict(feasible=False,reason="signed-seam")
    pairs = [(a,b) for a,b in forced.items() if a<b]
    flexible = []
    visited = set()
    for label,row in list(groups.items()):
        if label in visited:continue
        mate = reciprocal(label)
        other = groups[mate]
        if mate == label:
            if len(row)%2:return dict(feasible=False,reason="signed-inventory")
            pairs.extend(zip(row[::2],row[1::2]))
        else:
            if len(row) != len(other):return dict(feasible=False,reason="signed-inventory")
            pairs.extend(zip(row,other))
            if label[0].bit_count()%2 and len(row)>=2:flexible.append((row,other))
        visited.update((label,mate))
    successor = [-1]*offset
    for a,b in pairs:
        for p in range(1,12):
            if (a,p) not in ends:continue
            (i,d),(j,f) = ends[a,p],ends[b,12-p]
            if d == f:raise ArithmeticError("Non-directed seam.")
            if d:successor[i] = j
            else:successor[j] = i
    cycles = _cycles(successor)
    parity_ok = cycles%2 == 1
    result = dict(feasible=bool(flexible) or parity_ok,reference_cycles=cycles,odd_groups=len(flexible))
    if len(flexible)==1 and len(flexible[0][0])==2:
        aa,bb = flexible[0]
        result["required_pairs"] = list(zip(aa,bb if parity_ok else reversed(bb)))
    return result


def _cycles(successor):
    if sorted(successor) != list(range(len(successor))):raise ValueError("Expected a permutation.")
    seen = set()
    count = 0
    for x in range(len(successor)):
        if x in seen:continue
        count += 1
        while x not in seen:
            seen.add(x)
            x = successor[x]
    return count


def _nullity(rows):
    pivots = {}
    for v in rows:
        while v:
            i = v.bit_length()-1
            if i not in pivots:
                pivots[i] = v
                break
            v ^= pivots[i]
    return len(rows)-len(pivots)


def switch_family(tour, switches, mandatory=(), *, max_nodes=100000):
    """Count single tours from disjoint directed two-successor switches.

    The exact identity counts cycles as one plus interlace-matrix nullity.
    Adding one more row and column can reduce nullity by at most one.
    """
    if _cycles(tour) != 1:raise ValueError("Expected a one-circuit reference.")
    flat = [v for pair in switches for v in pair]
    if (any(len(p)!=2 for p in switches) or len(set(flat)) != len(flat) or
            any(type(v)is not int or not 0<=v<len(tour) for v in flat)):
        raise ValueError("Expected disjoint two-successor switches.")
    must = set(mandatory)
    if not must <= set(range(len(switches))):raise ValueError("Invalid mandatory switch.")
    order = {}
    x = 0
    for i in range(len(tour)):
        order[x] = i
        x = tour[x]
    chords = [tuple(sorted((order[a],order[b]))) for a,b in switches]
    adj = [sum(1<<j for j,(c,d) in enumerate(chords) if a<c<b<d or c<a<d<b) for a,b in chords]
    free = sorted(set(range(len(switches)))-must)
    nodes = accepted = skipped = 0
    def walk(i,mask):
        nonlocal nodes,accepted,skipped
        nodes += 1
        if nodes > max_nodes:raise OverflowError
        defect = _nullity([adj[x]&mask for x in bits(mask)])
        if defect > len(free)-i:
            skipped += 1<<(len(free)-i)
            return
        if i == len(free):
            accepted += 1
            return
        walk(i+1,mask)
        walk(i+1,mask|1<<free[i])
    try:
        walk(0,sum(1<<v for v in must))
        complete = True
    except OverflowError:complete = False
    return dict(complete=complete,single_tours=accepted,nodes=nodes,rejected_switch_sets=skipped)



def side_options(problem, domains=None):
    """All candidate physical partners induced by the exact geographic domains."""
    from r27_structure_filters import Work,physical_graph
    ds = domain_bits(problem) if domains is None else domains
    return physical_graph(problem,ds,Work(10000000))


def supply_supports(problem, inside, cut_lb, *, bonds=(), options=None):
    """Check boundary saturation and delete individual bonds making it impossible."""
    maximum = supply(problem,inside,bonds)
    if maximum < cut_lb:return dict(feasible=False,maximum=maximum,unsupported=[])
    if maximum > cut_lb+2:return dict(feasible=True,maximum=maximum,unsupported=[])
    options = side_options(problem) if options is None else options
    used = set(_bonds(problem,bonds))
    left = set(inside)
    # For an extra bond not touching the chosen subset, at most two outside
    # candidates disappear. Cross bonds can also leave the maximum unchanged.
    unsupported = []
    for a,row in enumerate(options):
        if a in used:continue
        for b in row:
            if b <= a or b in used:continue
            if supply(problem,left,(*bonds,(a,b))) < cut_lb:unsupported.append((a,b))
    return dict(feasible=True,maximum=maximum,unsupported=unsupported)



def switch_boards(problem, codes, alternatives, *, max_nodes=100000):
    """Bind two-successor switches to real, independently validated geographic placements.

    Alternatives may permute and rotate pieces within a disjoint patch. Each
    replaces its own physical inventory and leaves every exterior seam mask
    fixed. The derived successor switches must be disjoint. A tile's local
    phase orients its physical strands; labels are the physical segment IDs.
    """
    from r28_geo_color import _sphere
    _,regions = _sphere(problem)
    if not problem.validate(list(codes))["valid"]:raise ValueError("Reference board must be one Gold tour.")
    offsets = []
    offset = 0
    for tile in problem.tiles:
        offsets.append(offset)
        offset += len(tile["segments"])
    def directed(layout):
        phase = {0:0}
        todo = [0]
        while todo:
            a = todo.pop()
            t,r = divmod(layout[a],3)
            for s,(b,j) in enumerate(problem.board.neighbors[a]):
                u,q = divmod(layout[b],3)
                dark = (regions[t][12*((s-r)%3)].bit_count() ^
                        regions[u][12*((j-q)%3)+11].bit_count() ^ phase[a]) & 1
                if b not in phase:
                    phase[b] = dark
                    todo.append(b)
                elif phase[b] != dark:raise ValueError("Edge-perfect assembly has inconsistent color phases.")
        # Fix phase of physical tile zero even when it moves in an alternative.
        flip = next(phase[c] for c,x in enumerate(layout) if x//3==0)
        ends = {}
        for c,v in enumerate(layout):
            t,r = divmod(v,3)
            for k,seg in enumerate(problem.tiles[t]["segments"]):
                for s,p in seg:
                    dark = (regions[t][12*s+p].bit_count() ^ phase[c] ^ flip) & 1
                    ends[c,(s+r)%3,p] = offsets[t]+k,dark
        successor = [-1]*offset
        for a,s,b,t in problem.edges:
            for k in bits(int(problem.masks[layout[a],s])):
                i,d = ends[a,s,k+1]
                j,f = ends[b,t,11-k]
                if d == f:raise ValueError("Orientation inconsistent on a chosen seam.")
                if d:successor[i] = j
                else:successor[j] = i
        _cycles(successor)
        return successor
    base = directed(codes)
    used_cells = set()
    affected = set()
    lookup = []
    for alt in alternatives:
        alt = {int(c):v for c,v in alt.items()}
        if (not alt or any(not 0<=c<problem.n for c in alt) or not used_cells.isdisjoint(alt) or
                sorted(codes[c]//3 for c in alt) != sorted(v//3 for v in alt.values())):
            raise ValueError("Overlapping or non-inventory-preserving switch patch.")
        board = list(codes)
        for c,v in alt.items():board[c] = v
        report = problem.validate(board)
        if report["matched_edges"] != len(problem.edges):raise ValueError("Switch breaks seam matching.")
        for c,v in alt.items():
            for s,(b,j) in enumerate(problem.board.neighbors[c]):
                if b not in alt and problem.masks[v,s] != problem.masks[codes[c],s]:
                    raise ValueError("Patch changes its exterior seam mask.")
        after = directed(board)
        diff = [i for i,(x,y) in enumerate(zip(base,after)) if x != y]
        if len(diff)!=2 or not affected.isdisjoint(diff) or after[diff[0]] != base[diff[1]] or after[diff[1]] != base[diff[0]]:
            raise ValueError("Patch does not induce one disjoint two-successor switch.")
        # The compositional guarantee also requires untouched geography between
        # patches. Touching edits could alter an interface's strand labels twice.
        if any(b in used_cells for c in alt for b,j in problem.board.neighbors[c]):
            raise ValueError("Switch patches must be separated by unchanged cells.")
        affected.update(diff)
        used_cells.update(alt)
        lookup.append(diff)
    return switch_family(base,lookup,max_nodes=max_nodes)
