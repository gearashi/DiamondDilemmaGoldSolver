"""Conservative deductions over physical darts, mandatory strands, and sphere regions.

Public filter_domains receives one integer bitset per cell. An exhausted work
allowance returns the original domains. Every published reduction is scoped to
the supplied domains and the Gold objective.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations

from validator import reverse_mask


class WorkLimit(Exception):
    pass


class Work:
    def __init__(self, limit, check=None):
        if type(limit) is not int or limit <= 0:
            raise ValueError('Expected a positive integer work budget.')
        self.limit = limit
        self.check = check
        self.used = 0
        self.counters = Counter()

    def tick(self, name, cost=1):
        if self.used + cost > self.limit:
            raise WorkLimit
        if self.check is not None and (self.used == 0 or self.used // 128 != (self.used + cost) // 128):
            self.check()
        self.used += cost
        self.counters[name] += cost


class Union:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def join(self, a, b):
        self.p[self.find(a)] = self.find(b)


def bits(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length()-1
        mask ^= bit


def domain_bits(problem):
    return tuple(sum(1 << code for code in cell) for cell in problem.domains)


def _vertex_links(problem):
    """Compute cyclic vertex links and orientable-sphere applicability from sewings."""
    if any(x < 0 for row in problem.neighbors for x in row):
        return None
    tiles = Union(problem.n)
    corners = Union(3*problem.n)
    for a, s, b, t in problem.edges:
        tiles.join(a, b)
        corners.join(3*a+s, 3*b+(t+1)%3)
        corners.join(3*a+(s+1)%3, 3*b+t)
    v = tuple(corners.find(i) for i in range(3*problem.n))
    if len({tiles.find(c) for c in range(problem.n)}) != 1:
        return None
    if len(set(v)) - len(problem.edges) + problem.n != 2:
        return None
    return v


@lru_cache(maxsize=8192)
def _regions(segments):
    """Partition 36 boundary intervals by membership on either side of every arc.

    Bit i is one between arc i's two endpoints in the positive boundary
    direction. In a disk with disjoint chords these vectors precisely identify
    faces; same-side chords remain separate faces at the appropriate intervals.
    """
    chords = sorted(tuple(sorted((12*s+p, 12*t+q))) for (s,p),(t,q) in segments)
    for (a,b),(c,d) in combinations(chords, 2):
        if a < c < b < d or c < a < d < b:
            return None
    intervals = []
    for i in range(36):
        intervals.append(sum(1 << j for j,(a,b) in enumerate(chords) if a <= i < b))
    return tuple(intervals)


def oriented_segments(problem, code):
    return tuple(tuple(((s+code%3)%3, p) for s,p in seg)
                 for seg in problem.tiles[code//3]['segments'])


def _dual(problem, domains, work, rules):
    """Compare the mandatory complement components and feasible signed Euler sum.

    Every completed coloring has one of two local phases per tile. Vertices
    and seam ends cancel in V-E+F except original mesh vertices and one
    contribution for each subdivided edge interval. At crossings use the
    common boundary of the two closed regions when counting Euler terms.
    """
    corners = _vertex_links(problem)
    if corners is None:
        return True
    regions = {}
    for mask in domains:
        for code in bits(mask):
            if code not in regions:
                work.tick('planarity_checks')
                regions[code] = _regions(oriented_segments(problem, code))
                if regions[code] is None:
                    # A crossing alternative makes the planar restriction inapplicable.
                    return True
    n = problem.n
    orbit = Union(36*n)
    for a,s,b,t in problem.edges:
        for k in range(12):
            orbit.join(36*a+12*s+k, 36*b+12*t+11-k)
    local = []
    union_cache={}
    for c, mask in enumerate(domains):
        # Candidates that share a boundary connectivity vector only need one entry.
        choices = {regions[v] for v in bits(mask)}
        if not choices:
            return False
        local.append(choices)
        if 'regions' in rules:
            if mask not in union_cache:
                intra=Union(36)
                for intervals in choices:
                    rep={}
                    for k,label in enumerate(intervals):
                        work.tick('dual_interval_visits')
                        intra.join(k,rep.setdefault(label,k))
                union_cache[mask]=tuple(intra.find(i) for i in range(36))
            for i,j in enumerate(union_cache[mask]):
                orbit.join(36*c+i,36*c+j)
    # Every interval is present; >2 components even after union of *all*
    # possible intra-cell connections is impossible on a Gold board.
    if 'regions' in rules and len({orbit.find(i) for i in range(36*n)}) > 2:
        work.counters['dual_pocket_rejections'] += 1
        return False
    if 'euler' not in rules:
        return True
    # In the closed dark region: V includes each gold endpoint-on-seam, and
    # E includes every strand. They cancel (one seam-endpoint per strand).
    # Thus 2*chi_dark = 2*dark_vertices - sum_dark_intervals_over_sides +
    # 2*dark_tile_patches. Each actual subdivided seam interval is counted once,
    # not once per unit grid slot. Test attainable sums with an overapproximation
    # of corner colors; on fixed parities, DSU shares colors across all corners.
    vertex_ids = sorted(set(corners))
    index = {v:i for i,v in enumerate(vertex_ids)}
    parity = Union(2*len(index))
    def relation(a,b,odd):
        parity.join(2*a,2*b+odd)
        parity.join(2*a+1,2*b+(odd^1))
    odd_vertices=[v for v,degree in Counter(corners).items() if degree%2]
    if len(odd_vertices)==2:
        a,b=(index[v] for v in odd_vertices)
        relation(a,b,problem.total_segments%2)
    for c, choices in enumerate(local):
        for side in range(3):
            a=index[corners[3*c+side]]; b=index[corners[3*c+(side+1)%3]]
            possibilities = {(x[12*side].bit_count() ^ x[12*((side+1)%3)].bit_count()) & 1 for x in choices}
            if len(possibilities) == 1:
                relation(a,b,possibilities.pop())
    if any(parity.find(2*i)==parity.find(2*i+1) for i in range(len(index))):
        return False
    pin=parity.find(0)
    dark = set([parity.find(1)])
    # Pin one original vertex light without assuming that it is an apex.
    vmin = sum(parity.find(2*i) in dark for i in range(len(index)))
    vmax = len(index) - sum(parity.find(2*i)==pin for i in range(len(index)))
    sums={0}
    for c, choices in enumerate(local):
        contributions=set()
        for x in choices:
            for phase in (0,1):
                def color(i):
                    return (x[i].bit_count() & 1) ^ phase
                if any((parity.find(2*index[corners[3*c+s]])==pin and color(12*s)) or
                       (parity.find(2*index[corners[3*c+s]]) in dark and not color(12*s)) for s in range(3)):
                    continue
                patches={x[i] for i in range(36) if color(i)}
                ed=sum(color(12*s+k) for s in range(3) for k in range(12)
                       if k==0 or x[12*s+k]!=x[12*s+k-1])
                contributions.add(2*len(patches)-ed)
        work.tick('euler_sums', len(sums)*max(1,len(contributions)))
        sums={a+b for a in sums for b in contributions}
    ok=any(2-2*i in sums for i in range(vmin,vmax+1))
    if not ok:
        work.counters['euler_rejections']+=1
    return ok


def physical_graph(problem, domains, work):
    """Overapproximate possible pairings of physical sides with permitted neighbors."""
    edges = [set() for _ in range(3*problem.n)]
    if any(x < 0 for row in problem.neighbors for x in row):
        return None
    side_choices = {}
    by_mask = {}
    for c, mask in enumerate(domains):
        for side in range(3):
            key=(mask,side)
            if key in by_mask:
                side_choices[c,side]=by_mask[key]
                continue
            values=defaultdict(set)
            for code in bits(mask):
                work.tick('physical_occurrences')
                dart=3*(code//3)+(side-code%3)%3
                values[int(problem.masks[code,side])].add(dart)
            by_mask[key]=side_choices[c,side]=values
    processed=set()
    for a,s,b,t in problem.edges:
        key=tuple(sorted(((domains[a],s),(domains[b],t))))
        if key in processed:continue
        processed.add(key)
        for sig, left in side_choices[a,s].items():
            right=side_choices[b,t].get(reverse_mask(sig), ())
            for i in left:
                for j in right:
                    work.tick('physical_pair_tests')
                    if i//3!=j//3:
                        edges[i].add(j)
                        edges[j].add(i)
    return edges


def _components(graph, excluded=frozenset()):
    remaining=set(range(len(graph)))-set(excluded)
    result=[]
    while remaining:
        start=remaining.pop()
        component={start}
        stack=[start]
        while stack:
            v=stack.pop()
            new=graph[v]&remaining
            component.update(new)
            remaining.difference_update(new)
            stack.extend(new)
        result.append(component)
    return result


def _bipartite_saturated(edges,nodes,work):
    """Exact support from one perfect matching and alternating reachability."""
    color={nodes[0]:0}
    stack=[nodes[0]]
    while stack:
        v=stack.pop()
        for u in edges[v]:
            if u not in color:color[u]=1-color[v];stack.append(u)
            elif color[u]==color[v]:return False,None
    left=[v for v in nodes if color[v]==0]
    if len(left)*2!=len(nodes):return True,None
    mates={}
    def augment(v,seen):
        for u in sorted(edges[v]):
            work.tick('partner_augment_edges')
            if u in seen:continue
            seen.add(u)
            if u not in mates or augment(mates[u],seen):
                mates[u]=v
                return True
        return False
    if any(not augment(v,set()) for v in left):return True,None
    directed={v:set() for v in nodes}
    for v in left:
        for u in edges[v]:
            if mates[u]==v:directed[u].add(v)
            else:directed[v].add(u)
    supports={v:set() for v in nodes}
    for u,v in mates.items():
        seen={u};pending=[u]
        while pending:
            q=pending.pop()
            work.tick('partner_alternating_edges',1+len(directed[q]))
            new=directed[q]-seen
            pending.extend(new);seen.update(new)
        for x in edges[u]&seen:
            supports[u].add(x);supports[x].add(u)
    return True,supports


def _perfect_supports(edges, work):
    """Subset DP on connected reciprocal families, including self-reversing masks."""
    allowed=[set(x) for x in edges]
    for comp in _components(edges):
        if len(comp)%2:
            return None
        nodes=sorted(comp)
        if len(nodes)>18:
            applicable,supports=_bipartite_saturated(edges,nodes,work)
            if applicable:
                if supports is None:return None
                for v in nodes:allowed[v]=supports[v]
                continue
            # After pinning any edge, deleting its endpoints loses at most
            # two neighbors. Minimum degree >= n/2+1 yields a Hamiltonian
            # cycle on the n-2 vertices by Dirac, hence a perfect matching.
            if min(len(edges[v]) for v in nodes)>=len(nodes)//2+1:
                work.counters['dense_partner_classes_certified']+=1
                continue
        index={v:i for i,v in enumerate(nodes)}
        masks=[sum(1<<index[w] for w in edges[v]) for v in nodes]
        large = len(nodes)>18
        local_states = 0
        @lru_cache(None)
        def feasible(rem):
            nonlocal local_states
            # Keep an unresolved edge if its search allowance runs out. Reserve
            # the outer WorkLimit for the caller's whole-operation budget.
            if large and local_states >= 8192:
                return None
            work.tick('partner_matching_states')
            local_states += 1
            if not rem:return True
            i=min(bits(rem),key=lambda k:(masks[k]&rem).bit_count())
            after=rem^(1<<i)
            for j in bits(after&masks[i]):
                result = feasible(after^(1<<j))
                if result is not False:
                    return result
            return False
        full=(1<<len(nodes))-1
        if feasible(full) is False:
            return None
        for i,v in enumerate(nodes):
            allowed[v]={nodes[j] for j in bits(masks[i]) if feasible(full^(1<<i)^(1<<j)) is not False}
        if large:
            key = 'large_partner_classes_incomplete' if local_states>=8192 else 'large_partner_classes_checked'
            work.counters[key] += 1
    return allowed


def _filter_partners(problem, domains, graph, work):
    from r32_gold_bonds import gold_matching_supports
    allowed=gold_matching_supports(problem,graph,work)
    if allowed is None:
        return (0,)*problem.n
    # The same opposite domain and side recur across the 240 seams. Cache
    # physical dart membership separately from possible support per tested dart.
    opposite_cache={}
    support_cache={}
    updated=[]
    for c,mask in enumerate(domains):
        out=0
        for code in bits(mask):
            tile,rot=divmod(code,3)
            for side in range(3):
                d=3*tile+(side-rot)%3
                other,t=int(problem.neighbors[c,side]),int(problem.sides[c,side])
                key=(domains[other],t)
                if key not in opposite_cache:
                    opposite_cache[key]={3*(v//3)+(t-v%3)%3 for v in bits(key[0])}
                query=(d,key)
                if query not in support_cache:
                    work.tick('partner_placement_checks')
                    support_cache[query]=not allowed[d].isdisjoint(opposite_cache[key])
                if not support_cache[query]:break
            else:out|=1<<code
        updated.append(out)
    return tuple(updated)


def strand_graph(problem, partners):
    ends={}
    i=0
    for tile,p in enumerate(problem.tiles):
        for segment in p['segments']:
            for s,k in segment:
                ends[3*tile+s,k]=i
            i+=1
    graph=[set() for _ in range(i)]
    for (d,k),v in ends.items():
        for other in partners[d]:
            w=ends.get((other,12-k))
            if w is not None:
                graph[v].add(w)
                graph[w].add(v)
    return graph


def separator_ok(graph,work,width=2):
    """A spanning cycle leaves at most |X| components after deleting nonempty X."""
    if len(graph)<2:return True
    cost=len(graph)+sum(map(len,graph))
    # All two-vertex deletions need just one low-link DFS per first
    # vertex. Removing an internal v adds one component for each child
    # subtree with low[child] >= discovery[v]; a root uses child count.
    if width==0:
        work.tick('separator_graph_visits',cost)
        return len(_components(graph))<=1
    skips=[None,*range(len(graph))] if width>=2 else [None]
    for skip in skips:
        work.tick('separator_graph_visits',cost)
        disc={};low={};parent={};children=Counter();separated=Counter()
        cc=0
        for root in range(len(graph)):
            if root==skip or root in disc:continue
            cc+=1
            disc[root]=low[root]=len(disc)
            parent[root]=None
            stack=[(root,iter(graph[root]))]
            while stack:
                a,it=stack[-1]
                b=next(it,None)
                if b is None:
                    stack.pop()
                    v=parent[a]
                    if v is not None:
                        low[v]=min(low[v],low[a])
                        if low[a]>=disc[v]:separated[v]+=1
                    continue
                if b==skip or b==parent[a]:continue
                if b in disc:low[a]=min(low[a],disc[b]);continue
                parent[b]=a;children[a]+=1
                disc[b]=low[b]=len(disc)
                stack.append((b,iter(graph[b])))
        if cc>1:return False
        for v in disc:
            fragments=children[v] if parent[v] is None else 1+separated[v]
            if cc-1+fragments>(1 if skip is None else 2):return False
    return True


def filter_domains(problem, domains, mechanisms=('partners',), *, max_work=200000, check=None):
    """Return (domain bitsets, counters). On a resource limit preserve input exactly."""
    domains=tuple(domains)
    rules=frozenset(mechanisms)
    if rules-{'partners','regions','euler','separator','passages'}:
        raise ValueError('Unknown structural mechanism.')
    permitted=domain_bits(problem)
    if len(domains)!=problem.n or any(type(v)is not int or v<0 or v&~base for v,base in zip(domains,permitted)):
        raise ValueError('Domains must restrict original choices.')
    work=Work(max_work,check)
    try:
        result=domains
        if any(not v for v in domains):
            return domains,dict(work.counters)
        if rules&{'partners','separator'}:
            graph=physical_graph(problem,domains,work)
            if graph is not None:
                if 'partners' in rules:
                    result=_filter_partners(problem,domains,graph,work)
                if 'separator' in rules and not separator_ok(strand_graph(problem,graph),work):
                    result=(0,)*problem.n
        if all(result) and rules&{'regions','euler','passages'}:
            dual_rules=rules|{'regions'} if 'passages' in rules else rules
            if not _dual(problem,result,work,dual_rules):
                return (0,)*problem.n,dict(work.counters)
            if 'passages' in rules:
                # Query distinct local strand wirings once per cell. A rejected
                # class remains impossible in every subset of this invocation.
                result=list(result)
                for c,mask in enumerate(result):
                    groups=defaultdict(int)
                    for v in bits(mask):
                        groups[_regions(oriented_segments(problem,v))]|=1<<v
                    if len(groups)<2:continue
                    for g in groups.values():
                        probe=result.copy()
                        probe[c]=g
                        if not _dual(problem,probe,work,dual_rules):
                            result[c]&=~g
                            work.counters['passage_values_removed']+=g.bit_count()
                            if not result[c]:return (0,)*problem.n,dict(work.counters)
                result=tuple(result)
    except WorkLimit:
        work.counters['limit_hits']+=1
        result=domains
    return result,dict(work.counters)

