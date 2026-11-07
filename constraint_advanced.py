"""Bounded exact local filters and optimistic Gold strand connectivity.

Each deletion is a necessary condition, never a heuristic exclusion. Local
searches cache only completed support unions. The regional cache deliberately
assumes no outside loop pairing: outside cells may be completed independently,
an optimistic relaxation of both targets. Its entries are local subproblems,
not exhausted whole-board positions. Caches belong to one immutable problem.
"""
from __future__ import annotations

from array import array
from itertools import chain
from collections import deque

from constraint_deep import HallCertificateCache
from constraint_pruning import values, _tile_choices, _orientation_choices
from search_options import normalize_options
from validator import reverse_mask
from constraint_native import select_graph_helpers
from constraint_gold_graph import PossibleGoldGraph, STAT_KEYS as GOLD_GRAPH_STATS
from constraint_separator import SeparatorPruner, STAT_KEYS as SEPARATOR_STAT_KEYS
from constraint_regions import HierarchicalRegionPruner, STAT_KEYS as HIERARCHICAL_STAT_KEYS
from constraint_region_network import STAT_KEYS as NETWORK_STAT_KEYS
from constraint_region_cuts import MultiPortCutPruner, EndpointSupplyPruner, STAT_KEYS as REGION_CUT_STAT_KEYS

STAT_KEYS = ('strand_checks', 'strand_prunes', 'strand_failures',
             'cut_checks', 'cut_prunes', 'cut_failures',
             'cut_matching_checks', 'cut_matching_failures', 'cut_matching_support_prunes',
             'cut_global_matching_checks', 'cut_global_matching_failures', 'cut_global_matching_support_prunes',
             'cut_global_matching_steps', 'cut_global_matching_cache_hits', 'cut_global_matching_budget_skips',
             'cut_global_support_steps', 'cut_global_support_budget_skips',
             'conditional_hall_hits', 'conditional_hall_certificates', 'wide_cut_states', 'wide_cut_reuses',
             'dynamic_cut_checks', 'dynamic_cut_prunes', 'dynamic_cut_failures',
             'dynamic_cut_topologies', 'dynamic_cut_cache_hits',
             'regional_checks', 'regional_prunes', 'regional_failures',
             'regional_cache_hits', 'regional_cache_entries',
             'regional_assignments', 'regional_budget_skips',
             'biconnected_checks', 'biconnected_prunes', 'biconnected_failures') + SEPARATOR_STAT_KEYS + HIERARCHICAL_STAT_KEYS + NETWORK_STAT_KEYS + REGION_CUT_STAT_KEYS + (
    'region_plan_hits','region_plan_alternatives','regional_disk_hits','regional_disk_writes','regional_disk_rejections') + GOLD_GRAPH_STATS


class _Budget(Exception):
    pass


def _put(cache, key, value, limit):
    if key not in cache and len(cache) >= limit:
        del cache[next(iter(cache))]
    cache[key] = value


def _port_supports(segments):
    """Build exact orientation supports for all ports in one segment pass."""
    ports = [0]*33
    for code, pairs in enumerate(segments):
        bit = 1 << code
        for a, b in pairs:
            # Repeated endpoints/segments contribute the same orientation once.
            ports[a] |= bit
            ports[b] |= bit
    return tuple(ports)


class _DSU:
    def __init__(self, n):
        self.parents = list(range(n))
        self.ranks = [0]*n

    def find(self, x):
        parents = self.parents
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.ranks[a] < self.ranks[b]:
            a, b = b, a
        self.parents[b] = a
        if self.ranks[a] == self.ranks[b]:
            self.ranks[a] += 1
        return True


def cyclic_edge_blocks(vertex_count, edges, check=lambda: None):
    """Tarjan edge blocks; bridges receive -1, and each cyclic block a fresh ID.

    Iterative DFS avoids recursion limits on the full 5280-port board. Edge IDs,
    rather than parent vertices, also give correct behavior for parallel edges.
    A simple cycle belongs to one such vertex-biconnected block.
    """
    adjacency=[[] for _ in range(vertex_count)]
    for eid,(a,b) in enumerate(edges):
        adjacency[a].append((b,eid));adjacency[b].append((a,eid))
    discovered=[-1]*vertex_count;low=[-1]*vertex_count
    result=[-1]*len(edges);clock=0;block_id=0;pending=[];steps=0
    for root in range(vertex_count):
        if discovered[root]>=0 or not adjacency[root]:
            continue
        discovered[root]=low[root]=clock;clock+=1
        stack=[(root,-1,0)]
        while stack:
            steps+=1
            if steps%128==0:
                check()
            cell,parent_edge,index=stack[-1]
            if index<len(adjacency[cell]):
                other,eid=adjacency[cell][index];stack[-1]=(cell,parent_edge,index+1)
                if eid==parent_edge:
                    continue
                if discovered[other]<0:
                    pending.append(eid);discovered[other]=low[other]=clock;clock+=1
                    stack.append((other,eid,0))
                elif discovered[other]<discovered[cell]:
                    pending.append(eid);low[cell]=min(low[cell],discovered[other])
                continue
            stack.pop()
            if parent_edge<0:
                continue
            a,b=edges[parent_edge];parent=b if a==cell else a
            low[parent]=min(low[parent],low[cell])
            if low[cell]>=discovered[parent]:
                block=[]
                while True:
                    eid=pending.pop();block.append(eid)
                    if eid==parent_edge:
                        break
                vertices={v for eid in block for v in edges[eid]}
                if len(block)>=len(vertices):
                    for eid in block:
                        result[eid]=block_id
                    block_id+=1
    check()
    return tuple(result)


def small_cut_boundaries(vertex_count, edges, possible, check=lambda: None, limit=128):
    """Discover proved two/three-seam boundaries in a bounded set of DFS trees.

    XOR of the incident-edge sets of a subtree is its exact boundary. Repeating
    a spanning forest with each live edge removed exposes arbitrary two-edge
    cuts (a bridge after one deletion), and additional three-edge cuts. Parallel
    seams have distinct bits. The limit bounds discovery, not a proof search:
    every returned boundary is complete even when further regions are skipped.
    """
    adjacency = [[] for _ in range(vertex_count)]
    incident = [0]*vertex_count
    live = []
    for eid, ((a, _, b, _), enabled) in enumerate(zip(edges, possible)):
        if enabled:
            live.append(eid)
            adjacency[a].append((b, eid)); adjacency[b].append((a, eid))
            incident[a] ^= 1 << eid; incident[b] ^= 1 << eid
    two, three = set(), set()
    for removed in (-1, *live):
        check()
        seen = [False]*vertex_count
        boundary = incident.copy()
        for root in range(vertex_count):
            if seen[root]:
                continue
            seen[root] = True
            stack = [(root, -1, 0)]
            while stack:
                cell, parent, index = stack[-1]
                if index < len(adjacency[cell]):
                    other, eid = adjacency[cell][index]
                    stack[-1] = (cell, parent, index+1)
                    if eid == removed or seen[other]:
                        continue
                    seen[other] = True
                    stack.append((other, cell, 0))
                    continue
                stack.pop()
                cut = boundary[cell]
                if parent >= 0:
                    boundary[parent] ^= cut
                count = cut.bit_count()
                if count == 2:
                    two.add(tuple(values(cut)))
                    if len(two) >= limit:
                        check()
                        return tuple(sorted(two))
                elif count == 3 and len(three) < limit:
                    three.add(tuple(values(cut)))
            check()
    # Three-edge boundaries must not fill the budget before removing a seam
    # exposes the stronger two-edge bottlenecks in later spanning forests.
    return tuple(sorted(two)) + tuple(sorted(three))[:max(0, limit-len(two))]


def _small_matching_supports(rows, check=lambda: None):
    """Exact injective tile supports for at most six conditional endpoint cells.

    Hall's condition is cheap here: at most 63 nonempty cell subsets. A
    deficient subset forbids every assignment. A tight subset reserves all its
    tiles, so cells outside it cannot use them. These removals are exact: forcing
    any unsupported cell/tile edge leaves a deficient subset of the other cells,
    which was a tight subset containing that tile before it was forced.
    """
    rows = tuple(rows)
    n = len(rows)
    if n > 6:
        raise ValueError('Conditional cut matching supports at most six cells.')
    unions = [0]*(1 << n)
    supported = list(rows)
    for subset in range(1, 1 << n):
        check()
        bit = subset & -subset
        union = unions[subset ^ bit] | rows[bit.bit_length()-1]
        unions[subset] = union
        size = subset.bit_count()
        count = union.bit_count()
        if count < size:
            return None
        if count == size:
            outside = ((1 << n)-1) ^ subset
            for cell in values(outside):
                supported[cell] &= ~union
    return tuple(supported)


def _complete_matching(rows, check=lambda: None, hint=None):
    """Find an injective physical-tile assignment, or prove that none exists.

    Every missing matching edge is repaired by a complete alternating-path
    search. A previous complete matching is only a checked starting hint; it
    can never certify a narrowed, widened or sibling graph by itself. Each
    examined row/edge and path update calls check(), bounding cancellation and
    allowing the caller to count the additional reasoning work separately.
    """
    rows = tuple(rows); n = len(rows)
    matched_cell = [-1]*n; matched_tile = {}
    if hint is not None:
        for cell, tile in enumerate(hint[:n]):
            check()
            if tile >= 0 and rows[cell] & (1 << tile) and tile not in matched_tile:
                matched_cell[cell] = tile; matched_tile[tile] = cell
    for start in sorted(range(n), key=lambda cell: (rows[cell].bit_count(), cell)):
        check()
        if matched_cell[start] >= 0:continue
        pending = deque([start]); seen = {start}; parents = {}; free = -1
        while pending and free < 0:
            cell = pending.popleft()
            for tile in values(rows[cell]):
                check()
                if tile in parents:continue
                parents[tile] = cell
                owner = matched_tile.get(tile)
                if owner is None:
                    free = tile;break
                if owner not in seen:
                    seen.add(owner);pending.append(owner)
        if free < 0:return None
        while free >= 0:
            check()
            cell = parents[free];previous = matched_cell[cell]
            matched_cell[cell] = free;matched_tile[free] = cell
            free = previous
    check()
    return tuple(matched_cell)


def _complete_matching_supports(rows, check=lambda: None, hint=None):
    """All injective assignment supports, from one matching and alternating paths.

    Matched tile vertices contract into their owning cell. Every non-matching
    choice points to that owner; unmatched tiles point to a spare vertex which
    points to every cell. Alternating cycles (including exchanges ending at a
    spare tile) are exactly the strongly connected components of this graph.
    Thus an edge is supported iff its endpoints share a component. Cancellation
    during matching or SCC construction returns no partially filtered rows.
    """
    rows = tuple(rows)
    matching = _complete_matching(rows, check, hint)
    if matching is None:return None, None
    return _matching_supports(rows, matching, check), matching


def _matching_supports(rows, matching, check=lambda: None):
    """Complete alternating SCC support filtering after a verified matching."""
    n = len(rows);owners = {tile:cell for cell,tile in enumerate(matching)}
    graph = [0]*(n+1);reverse = [0]*(n+1)
    for cell,row in enumerate(rows):
        for tile in values(row):
            check();other = owners.get(tile,n)
            graph[cell] |= 1 << other;reverse[other] |= 1 << cell
    # A spare tile takes a cell's former matched tile when an exchange ends.
    # Without spare tiles this vertex has no incoming arcs and changes nothing.
    graph[n] = (1 << n)-1
    for cell in range(n):reverse[cell] |= 1 << n
    order=[];seen=0
    for root in range(n+1):
        check()
        if seen & (1 << root):continue
        seen |= 1 << root;stack=[root]
        while stack:
            check();cell=stack[-1];unseen=graph[cell]&~seen
            if not unseen:order.append(stack.pop())
            else:
                bit=unseen&-unseen;seen |= bit;stack.append(bit.bit_length()-1)
    component=[-1]*(n+1);assigned=0
    for root in reversed(order):
        check()
        if assigned & (1 << root):continue
        pending=1 << root;assigned |= pending
        while pending:
            check();bit=pending&-pending;pending ^= bit;cell=bit.bit_length()-1
            component[cell]=root;unseen=reverse[cell]&~assigned
            assigned |= unseen;pending |= unseen
    supported=[]
    for cell,row in enumerate(rows):
        keep=0
        for tile in values(row):
            check()
            if component[cell]==component[owners.get(tile,n)]:keep |= 1 << tile
        supported.append(keep)
    check()
    return tuple(supported)


CUT_CONTINUATION_STATS=('cut_continuation_hits','cut_continuation_states_reused','cut_continuation_states_saved','cut_continuation_queries_completed')
STAT_KEYS += CUT_CONTINUATION_STATS


class AdvancedPruner:
    """Integer-bitset propagator; return changed, with domain zero as failure.

    Default bounds: cuts have at most three live seams and 256 joint mask
    vectors; regions contain two to four cells, at most 4096 search nodes and
    four uncached regions per call. Incomplete searches do not narrow domains.
    check() may raise at any time; prior individual deletions remain sound.
    """
    def __init__(self, problem, target='gold', options=None):
        if target not in ('gold', 'edges'):
            raise ValueError('Target must be gold or edges.')
        self.problem, self.target = problem, target
        self.options = normalize_options(options)
        self.native_graph,self.native_graph_reason=select_graph_helpers(self.options['native'])
        self._cyclic_edge_blocks=(self._packed_edge_blocks if self.native_graph is not None else cyclic_edge_blocks)
        self.stats = dict.fromkeys(STAT_KEYS, 0)
        self.n = problem.n
        self.masks = tuple(tuple(map(int, row)) for row in problem.masks)
        self.segments = tuple(tuple(((a[0]+r)%3*11+a[1]-1,
                                     (b[0]+r)%3*11+b[1]-1)
                                    for a, b in tile['segments'])
                              for tile in problem.tiles for r in range(3))
        self.heads = tuple(tuple(a for a, _ in pairs) for pairs in self.segments)
        self.port_codes = _port_supports(self.segments)
        self.nonempty = tuple(sum(1 << code for code, row in enumerate(self.masks)
                                  if row[side]) for side in range(3))
        self.adj = [[] for _ in range(self.n)]
        self.neighbor_cells = [set() for _ in range(self.n)]
        for eid, (a, sa, b, sb) in enumerate(problem.edges):
            self.adj[a].append((b, sa, sb, eid))
            self.adj[b].append((a, sb, sa, eid))
            self.neighbor_cells[a].add(b)
            self.neighbor_cells[b].add(a)
        regions = {tuple(sorted((a, b))) for a, _, b, _ in problem.edges
                   if self.n > 2}
        faces = {}
        for cell, info in enumerate(problem.board.cells):
            face = getattr(info, 'face', None)
            if face is not None:
                faces.setdefault(face, []).append(cell)
        regions.update(tuple(cells) for cells in faces.values() if 1 < len(cells) < self.n)
        self.cut_regions = tuple((region, tuple(eid for eid, (a, _, b, _) in enumerate(problem.edges)
                                                if (a in region) != (b in region)))
                                 for region in sorted(regions))
        self._local_ports, self._strand_cache = {}, {}
        self._possible_gold=PossibleGoldGraph(problem) if self.options.get('reversible_gold_graph',False) else None
        self._native_forests = None
        self._native_strands = None
        self._native_biconnected = None
        self._buckets, self._cut_cache, self._regional_cache = {}, {}, {}
        self._dynamic_cut_cache = {}
        self._cut_pending = {}
        self._cut_inventory_cache = {}
        self._cut_inventory_matching = None
        self._hall_certificates = HallCertificateCache()
        self.cut_matching_work_limit = 200000
        self.cut_support_work_limit = 200000
        self.cut_vector_limit, self.region_node_limit = 256, 4096
        self.region_size_limit, self.region_misses_per_call = 4, 4
        self.last_region = None
        self._biconnected_cache, self._local_edges, self._local_certificates = {}, {}, {}
        self._segment_supports = None
        self.separator = None
        self.hierarchical = None
        self.on_narrow = None
        self.before_hierarchical = None
        self.multi_port = None
        self.endpoint_supply = None
        self.regional_cache_dir = None

    def reasoning_snapshot(self):
        """Trusted local continuation data; never an externally supplied proof."""
        records=[]
        for (edges,involved,inventory),pending in self._cut_pending.items():
            records.append({'edges':list(edges),'involved':[[c,hex(m)] for c,m in involved],
                'inventory':[hex(x) for x in inventory],
                'support':[hex(pending['support'][c]) for c in range(self.n)],
                'completed':[[i,[[c,hex(m)] for c,m in current],crossings]
                             for i,current,crossings in sorted(pending['completed'])]})
        return {'version':1,'fingerprint':self.problem.fingerprint,'target':self.target,'cuts':records}

    def restore_reasoning(self,payload):
        if (not isinstance(payload,dict) or payload.get('version')!=1
                or payload.get('fingerprint')!=self.problem.fingerprint or payload.get('target')!=self.target
                or not isinstance(payload.get('cuts'),list) or len(payload['cuts'])>4):
            raise ValueError('Reasoning continuation scope or bounds differ.')
        restored={};full=(1<<(3*self.n))-1
        for record in payload['cuts']:
            edges=tuple(record['edges'])
            if not edges or len(edges)>6 or len(set(edges))!=len(edges) or any(type(x)is not int or not 0<=x<len(self.problem.edges) for x in edges):
                raise ValueError('Malformed cut continuation boundary.')
            involved=tuple((c,int(m,16)) for c,m in record['involved'])
            expected=tuple(sorted({c for e in edges for c in (self.problem.edges[e][0],self.problem.edges[e][2])}))
            if tuple(c for c,m in involved)!=expected or any(m<=0 or m&~full for c,m in involved):
                raise ValueError('Malformed cut continuation domains.')
            inventory=tuple(int(x,16) for x in record['inventory'])
            support=tuple(int(x,16) for x in record['support'])
            if len(inventory)!=self.n or len(support)!=self.n or any(x<=0 or x>>self.n for x in inventory) or any(x<0 or x&~full for x in support):
                raise ValueError('Malformed cut continuation inventory/supports.')
            if any(_tile_choices(m)!=inventory[c] for c,m in involved):raise ValueError('Cut domains and inventory differ.')
            limits=dict(involved);completed=set()
            if len(record['completed'])>8192:raise ValueError('Cut continuation state cap exceeded.')
            for index,current,crossings in record['completed']:
                state=tuple((c,int(m,16)) for c,m in current)
                if (type(index)is not int or not 0<=index<=len(edges) or type(crossings)is not int
                        or not 0<=crossings<=11*len(edges) or tuple(c for c,m in state)!=expected
                        or any(m<=0 or m&~limits[c] for c,m in state)):
                    raise ValueError('Malformed completed cut state.')
                completed.add((index,state,crossings))
            restored[(edges,involved,inventory)]={'support':dict(enumerate(support)),'completed':completed}
        self._cut_pending=restored

    def _packed_edge_blocks(self, vertex_count, edges, check):
        # These endpoints originate only from validated tile ports and board
        # cells. array conversion is range-checked, never a truncating ctypes
        # cast; chain flattens exact pairs without per-edge Python generators.
        check()
        packed=array('i',chain.from_iterable(edges))
        if packed.itemsize!=4:
            return self.native_graph.cyclic_edge_blocks(vertex_count,edges,check)
        return self.native_graph.cyclic_edge_blocks_packed(vertex_count,packed.tobytes(),check)

    def _apply(self, domains, allowed, kind, check):
        changed = False
        for cell, support in allowed:
            check()
            reduced = domains[cell] & support
            if reduced != domains[cell]:
                self.stats[kind+'_prunes'] += domains[cell].bit_count()-reduced.bit_count()
                domains[cell] = reduced
                changed = True
                if not reduced:
                    self.stats[kind+'_failures'] += 1
                    break
        return changed

    def _possible_allowed(self,domains,kind,check):
        try:
            state=self._possible_gold.update(domains,check)
            if kind=='strand':return self._possible_gold.strands(state,check)
            return self._possible_gold.blocks(state,self._cyclic_edge_blocks,check)
        finally:self.stats.update(self._possible_gold.stats)

    def _local_forest(self, domain, check):
        cached = self._local_ports.get(domain)
        if cached is not None:
            return cached
        if self.native_graph is not None:
            if self._native_forests is None:
                self._native_forests = self.native_graph.prepare_local_forests(self.segments, check)
            result = self._native_forests.calculate(domain, check)
            # Publish only a fully completed calculation; interrupted native
            # calls leave the domain cache untouched.
            _put(self._local_ports, domain, result, 1024)
            return result
        dsu = _DSU(33)
        used = sum(1 << port for port, codes in enumerate(self.port_codes) if domain & codes)
        components = used.bit_count()
        for index, code in enumerate(values(domain)):
            if index % 32 == 0:
                check()
            for a, b in self.segments[code]:
                components -= dsu.join(a, b)
            # All possible ports are now connected. Additional candidate edges
            # cannot change this local union graph, so dense domains are cheap.
            if components <= 1:
                break
        forest = tuple((port, dsu.find(port)) for port in values(used)
                       if port != dsu.find(port))
        result = forest, used
        _put(self._local_ports, domain, result, 1024)
        return result

    def prune_strands(self, domains, check=lambda: None):
        """Retain codes whose every strand can belong to one common union component.

        Optional disconnected ports are not mandatory vertices. We intersect
        each cell's eligible component choices, rather than demanding that the
        entire optimistic union graph be connected.
        """
        if self.target != 'gold' or not all(domains):
            return False
        self.stats['strand_checks'] += 1
        key = tuple(domains)
        allowed = self._strand_cache.get(key)
        if allowed is None and self._possible_gold is not None:
            allowed=self._possible_allowed(domains,'strand',check);_put(self._strand_cache,key,allowed,4)
        # Full native setup amortizes on Diamond boards (at least ten cells);
        # tiny standalone fixtures retain the cheaper existing local path.
        if allowed is None and self.n>=10 and getattr(self.native_graph,'supports_strands',False) is True:
            if self._native_strands is None:
                self._native_strands=self.native_graph.prepare_strands(self.n,self.segments,self.problem.edges,check)
            allowed=self._native_strands.allowed(domains,check)
            _put(self._strand_cache,key,allowed,4)
        if allowed is None:
            dsu = _DSU(33*self.n) if self.native_graph is None else None
            graph, used = [], []
            append = graph.append
            for cell, domain in enumerate(domains):
                check()
                forest, ports = self._local_forest(domain, check)
                used.append(ports)
                if dsu is None:
                    base = 33*cell
                    for a, b in forest:
                        append(base+a); append(base+b)
                else:
                    for a, b in forest:
                        dsu.join(33*cell+a, 33*cell+b)
            for a, sa, b, sb in self.problem.edges:
                check()
                for p in range(11):
                    self.stats['gold_graph_seam_tests']+=1
                    u, v = 11*sa+p, 11*sb+10-p
                    if used[a] >> u & 1 and used[b] >> v & 1:
                        if dsu is None:
                            append(33*a+u); append(33*b+v)
                        else:
                            dsu.join(33*a+u, 33*b+v)
            if dsu is None:
                check()
                packed = array('i', graph)
                if packed.itemsize == 4:
                    component = self.native_graph.union_roots_packed(33*self.n, packed.tobytes(), check)
                    find = component.__getitem__
                else:
                    dsu = _DSU(33*self.n)
                    for index in range(0, len(graph), 2):
                        if index % 512 == 0:
                            check()
                        dsu.join(graph[index], graph[index+1])
                    find = dsu.find
            else:
                find = dsu.find
            choices, common = [], None
            for cell, domain in enumerate(domains):
                check()
                roots = {p: find(33*cell+p) for p in values(used[cell])}
                distinct = set(roots.values())
                if len(distinct) == 1:
                    by_component = {next(iter(distinct)): domain}
                else:
                    by_component = {}
                    for index, code in enumerate(values(domain)):
                        if index % 32 == 0:
                            check()
                        hosting = {roots[p] for p in self.heads[code]}
                        if len(hosting) == 1:
                            root = next(iter(hosting))
                            by_component[root] = by_component.get(root, 0) | (1 << code)
                choices.append(by_component)
                common = set(by_component) if common is None else common.intersection(by_component)
                if not common:
                    allowed = ((cell, 0),)
                    break
            else:
                allowed = tuple((cell, sum(mapping[root] for root in common))
                                for cell, mapping in enumerate(choices))
            _put(self._strand_cache, key, allowed, 4)
        return self._apply(domains, allowed, 'strand', check)

    def _local_block_certificate(self, domain, check):
        """Sparse graph with the same connectivity after deleting any one port.

        Each local cyclic block may be replaced by a cycle through its vertices;
        keep each bridge. Both the old and replacement block stay connected
        after any one vertex deletion. Therefore adjoining external seam edges
        preserves global vertex-biconnected membership exactly. This reduction
        does not approximate optional-port or articulation behavior.
        """
        cached=self._local_certificates.get(domain)
        if cached is not None:
            return cached
        if self.native_graph is not None:
            if self._native_forests is None:
                self._native_forests = self.native_graph.prepare_local_forests(self.segments, check)
            result = self._native_forests.certificate(domain, check)
            _put(self._local_certificates, domain, result, 1024)
            return result
        pairs=self._local_edges.get(domain)
        if pairs is None:
            pairs=tuple(pair for pair,support in self._segment_supports if domain&support)
            _put(self._local_edges,domain,pairs,1024)
        ports=sum(1<<p for p,codes in enumerate(self.port_codes) if domain&codes)
        used=tuple(values(ports));prefix=pairs[:2*len(used)]
        blocks=self._cyclic_edge_blocks(33,prefix,check)
        # If this subset already proves all possible ports biconnected, adding
        # remaining local edges cannot change the local block structure.
        if blocks and len(set(blocks))==1 and blocks[0]>=0 and {v for pair in prefix for v in pair}==set(used):
            certificate=tuple(zip(used,used[1:]+used[:1]))
        else:
            if len(prefix)!=len(pairs):
                blocks=self._cyclic_edge_blocks(33,pairs,check)
            groups={};certificate=[]
            for pair,block in zip(pairs,blocks):
                if block<0:
                    certificate.append(pair)
                else:
                    groups.setdefault(block,set()).update(pair)
            for group in groups.values():
                vertices=tuple(sorted(group))
                certificate.extend(zip(vertices,vertices[1:]+vertices[:1]))
            certificate=tuple(certificate)
        result=certificate,ports
        _put(self._local_certificates,domain,result,1024)
        return result

    def prune_biconnected(self, domains, check=lambda: None):
        """Require every mandatory segment AND seam edge to share one cyclic block.

        The union graph includes all possible local connections and seam ports.
        A Gold arrangement is one simple cycle in this abstract endpoint graph
        (drawing crossings are not junctions). Thus its edges lie in one cyclic
        biconnected block. Optional blocks need not be used; each cell instead
        supplies candidate block choices, which must have a common host.
        """
        if self.target != 'gold' or not all(domains):
            return False
        self.stats['biconnected_checks'] += 1
        key=tuple(domains);allowed=self._biconnected_cache.get(key)
        if allowed is None and self._possible_gold is not None:
            allowed=self._possible_allowed(domains,'biconnected',check);_put(self._biconnected_cache,key,allowed,4)
        if allowed is None and self.n>=10 and getattr(self.native_graph,'supports_biconnected',False) is True:
            if self._native_biconnected is None:
                self._native_biconnected=self.native_graph.prepare_biconnected(self.n,self.segments,self.problem.edges,check)
            allowed=self._native_biconnected.allowed(domains,check)
            _put(self._biconnected_cache,key,allowed,4)
        if allowed is None:
            if self._segment_supports is None and self.native_graph is None:
                supports={}
                for code,pairs in enumerate(self.segments):
                    for pair in pairs:
                        pair=tuple(sorted(pair));supports[pair]=supports.get(pair,0)|(1<<code)
                self._segment_supports=tuple(supports.items())
            graph=[];used=[];append=graph.append
            native=self.native_graph is not None
            for cell,domain in enumerate(domains):
                check()
                pairs,ports=self._local_block_certificate(domain,check)
                base=33*cell
                if native:
                    for a,b in pairs:
                        append(base+a);append(base+b)
                else:
                    graph.extend((base+a,base+b) for a,b in pairs)
                used.append(ports)
            # Dense cell/port addressing preserves edge zero and distinguishes
            # absent seams with -1. Repeated references retain the last edge.
            seam_ids=[-1]*(33*self.n)
            for a,sa,b,sb in self.problem.edges:
                check()
                for p in range(11):
                    self.stats['gold_graph_seam_tests']+=1
                    u,v=11*sa+p,11*sb+10-p
                    if used[a]>>u&1 and used[b]>>v&1:
                        if native:
                            eid=len(graph)//2;append(33*a+u);append(33*b+v)
                        else:
                            eid=len(graph);append((33*a+u,33*b+v))
                        seam_ids[33*a+u]=eid;seam_ids[33*b+v]=eid
            if native:
                # Keep endpoint order and immutable, range-checked native input.
                check()
                packed=array('i',graph)
                if packed.itemsize==4:
                    blocks=self.native_graph.cyclic_edge_blocks_packed(33*self.n,packed.tobytes(),check)
                else:
                    blocks=cyclic_edge_blocks(33*self.n,list(zip(graph[::2],graph[1::2])),check)
            else:
                blocks=self._cyclic_edge_blocks(33*self.n,graph,check)
            choices=[];common=None
            for cell,domain in enumerate(domains):
                check()
                # Every chosen port uses its seam edge. When all those edges
                # share one block, each actual segment joins two vertices in
                # that block too, even if the certificate replaced that edge.
                offset=33*cell
                port_blocks=[blocks[e] if e>=0 else -1 for e in seam_ids[offset:offset+33]]
                all_blocks={port_blocks[p] for p in values(used[cell])}
                if len(all_blocks)==1 and -1 not in all_blocks:
                    mapping={next(iter(all_blocks)):domain}
                else:
                    mapping={}
                    for index,code in enumerate(values(domain)):
                        if index%32==0:
                            check()
                        hosting=set()
                        for a,b in self.segments[code]:
                            for port in (a,b):
                                hosting.add(port_blocks[port])
                        if len(hosting)==1 and -1 not in hosting:
                            block=next(iter(hosting));mapping[block]=mapping.get(block,0)|(1<<code)
                choices.append(mapping)
                common=set(mapping) if common is None else common.intersection(mapping)
                if not common:
                    allowed=((cell,0),);break
            else:
                allowed=tuple((cell,sum(mapping[block] for block in common)) for cell,mapping in enumerate(choices))
            _put(self._biconnected_cache,key,allowed,4)
        return self._apply(domains,allowed,'biconnected',check)

    def _mask_buckets(self, domain, side, reversed_mask=False):
        key = domain, side, reversed_mask
        cached = self._buckets.get(key)
        if cached is None:
            cached = {}
            for code in values(domain):
                mask = self.masks[code][side]
                if reversed_mask:
                    mask = reverse_mask(mask)
                cached[mask] = cached.get(mask, 0) | (1 << code)
            _put(self._buckets, key, cached, 2048)
        return cached

    def _cut_support(self, domains, edge_ids, check):
        involved = tuple(sorted({cell for eid in edge_ids
                                 for cell in (self.problem.edges[eid][0], self.problem.edges[eid][2])}))
        inventory = tuple(_tile_choices(domain) for domain in domains)
        # A conditional endpoint choice can consume pieces needed anywhere on
        # the board. Outside orientations matter here only through inventory.
        key = edge_ids, tuple((cell, domains[cell]) for cell in involved), inventory
        cached = self._cut_cache.get(key)
        if cached is not None:
            return cached
        tables, product = [], 1
        for eid in edge_ids:
            check()
            a, sa, b, sb = self.problem.edges[eid]
            left = self._mask_buckets(domains[a], sa)
            right = self._mask_buckets(domains[b], sb, True)
            masks = sorted(left.keys() & right.keys())
            product *= len(masks)
            if product > (max(4096,self.cut_vector_limit) if self.options['wide_cuts'] else self.cut_vector_limit):
                return None
            tables.append((a, b, tuple((m, left[m], right[m]) for m in masks)))
        resumable=self.options.get('resumable_reasoning',False)
        pending=self._cut_pending.get(key) if resumable else None
        if pending is None:
            pending={'support':dict.fromkeys(range(self.n),0),'completed':set()}
            if resumable:
                _put(self._cut_pending,key,pending,4)
        else:self.stats['cut_continuation_hits']+=1
        support=pending['support']
        work = 0; support_work = 0; support_capped = False
        def support_step():
            nonlocal support_work
            if support_work >= self.cut_support_work_limit:raise _Budget
            support_work += 1;self.stats['cut_global_support_steps'] += 1
            check()
        def matching_step():
            nonlocal work
            if work >= self.cut_matching_work_limit:raise _Budget
            work += 1;self.stats['cut_global_matching_steps'] += 1
            check()
        def global_supports(rows):
            nonlocal support_capped
            check();self.stats['cut_global_matching_checks'] += 1
            if rows in self._cut_inventory_cache:
                self.stats['cut_global_matching_cache_hits'] += 1
                allowed = self._cut_inventory_cache[rows]
            else:
                matching = _complete_matching(rows, matching_step, self._cut_inventory_matching)
                allowed = None
                if matching is None and self.options['conditional_hall']:
                    witness=self._hall_certificates.remember(rows,matching_step)
                    self.stats['conditional_hall_certificates']+=int(witness is not None)
                if matching is not None:
                    self._cut_inventory_matching = matching
                    try:allowed = _matching_supports(rows, matching, support_step)
                    except _Budget:
                        # Completed feasibility still proves the R21 vector.
                        # Use its local supports and unrestricted outside tiles;
                        # no incomplete SCC deduction or cache entry escapes.
                        self.stats['cut_global_support_budget_skips'] += 1
                        support_capped = True
                        return rows
                check()
                _put(self._cut_inventory_cache, rows, allowed, 128)
            if allowed is None:self.stats['cut_global_matching_failures'] += 1
            return allowed
        completed_states=pending['completed'] if resumable else set()
        def visit(index, current, crossings):
            check()
            state=(index,tuple(sorted(current.items())),crossings)
            if (resumable or self.options['wide_cuts']) and state in completed_states:
                self.stats['wide_cut_reuses']+=1
                self.stats['cut_continuation_states_reused']+=int(resumable)
                return
            def complete():
                if (resumable or self.options['wide_cuts']) and len(completed_states)<8192:
                    completed_states.add(state)
                    self.stats['cut_continuation_states_saved']+=int(resumable)
            if self.options['conditional_hall']:
                rows=list(inventory)
                for c,d in current.items():rows[c]=_tile_choices(d)
                if self._hall_certificates.impossible(rows,matching_step):
                    self.stats['conditional_hall_hits']+=1
                    complete();return
            if index == len(tables):
                if crossings and crossings % 2 == 0:
                    # Seam masks condition the domains jointly. Hall filtering
                    # on broader parent domains can miss piece collisions here.
                    self.stats['cut_matching_checks'] += 1
                    allowed = _small_matching_supports(
                        (_tile_choices(current[cell]) for cell in involved), check)
                    if allowed is None:
                        self.stats['cut_matching_failures'] += 1
                        complete();return
                    conditional = list(inventory)
                    for cell, tiles in zip(involved, allowed):conditional[cell] = tiles
                    globally_allowed = global_supports(tuple(conditional))
                    if globally_allowed is None:complete();return
                    for cell, tiles in enumerate(globally_allowed):
                        domain = current.get(cell, domains[cell])
                        keep = domain & _orientation_choices(tiles)
                        self.stats['cut_global_matching_support_prunes'] += (
                            domain & _orientation_choices(conditional[cell]) & ~keep).bit_count()
                        if cell in current:
                            local_keep = domain & _orientation_choices(conditional[cell])
                            self.stats['cut_matching_support_prunes'] += (domain & ~local_keep).bit_count()
                            support[cell] |= keep
                        else:
                            # The key binds outside TILE rows, not rotations.
                            # Cache all orientations of supported outside tiles;
                            # _apply intersects the caller's current domains.
                            support[cell] |= _orientation_choices(tiles)
                complete();return
            a, b, masks = tables[index]
            for mask, left, right in masks:
                da, db = current[a] & left, current[b] & right
                if da and db:
                    narrowed = current.copy()
                    narrowed[a], narrowed[b] = da, db
                    visit(index+1, narrowed, crossings+mask.bit_count())
            complete()
            if self.options['wide_cuts']:self.stats['wide_cut_states']+=1
        try:visit(0, {cell: domains[cell] for cell in involved}, 0)
        except _Budget:
            self.stats['cut_global_matching_budget_skips'] += 1
            return None
        result = tuple(support.items())
        if resumable:
            self._cut_pending.pop(key,None)
            self.stats['cut_continuation_queries_completed']+=1
        if not support_capped:_put(self._cut_cache, key, result, 32)
        return result

    def prune_cuts(self, domains, check=lambda: None):
        """Correlate up to three live exits, preserving one orientation per cell.

        A Gold cycle visiting every tile crosses every proper tile region a
        positive even number of times. The table intersects ALL cut-side mask
        requirements on each endpoint cell; independent seam/bridge tests lose
        this correlation. Parallel geometric seams remain separate variables.
        """
        if self.target != 'gold' or not all(domains):
            return False
        possible = tuple(bool(domains[a] & self.nonempty[sa] and
                              domains[b] & self.nonempty[sb])
                         for a, sa, b, sb in self.problem.edges)
        seen, changed = set(), False
        # Keep the existing static-region allowance. Newly discovered regions
        # receive a separate bounded allowance, so they cannot crowd it out.
        for dynamic in (False, True):
            if dynamic:
                boundaries = self._dynamic_cut_cache.get(possible)
                if boundaries is None:
                    boundaries = small_cut_boundaries(self.n, self.problem.edges, possible, check)
                    # Publish only after discovery returns; cancelled traversal
                    # leaves no certificate. Exact topology keys handle siblings.
                    _put(self._dynamic_cut_cache, possible, boundaries, 8)
                    self.stats['dynamic_cut_topologies'] += 1
                else:
                    self.stats['dynamic_cut_cache_hits'] += 1
            else:
                boundaries = (boundary for _, boundary in self.cut_regions)
            checks = 0
            for boundary in boundaries:
                check()
                live = tuple(eid for eid in boundary if possible[eid])
                if not 2 <= len(live) <= (6 if self.options['wide_cuts'] else 3) or live in seen:
                    continue
                seen.add(live)
                support = self._cut_support(domains, live, check)
                if support is None:
                    continue
                self.stats['cut_checks'] += 1
                before = self.stats['cut_prunes'], self.stats['cut_failures']
                changed |= self._apply(domains, support, 'cut', check)
                if dynamic:
                    self.stats['dynamic_cut_checks'] += 1
                    self.stats['dynamic_cut_prunes'] += self.stats['cut_prunes']-before[0]
                    self.stats['dynamic_cut_failures'] += self.stats['cut_failures']-before[1]
                checks += 1
                if not all(domains):
                    return changed
                if checks >= 8:
                    break
        return changed

    def _region_key(self, domains, region):
        inventory = 0
        for cell in region:
            inventory |= _tile_choices(domains[cell])
        outside = tuple(sorted(set().union(*(self.neighbor_cells[c] for c in region))-set(region)))
        return (region, tuple(domains[c] for c in region), inventory,
                tuple((c, domains[c]) for c in outside))

    def _solve_region(self, domains, key, check):
        region, _, inventory, boundary = key
        excluded = _orientation_choices(inventory)
        outside = {cell: domain & ~excluded for cell, domain in boundary}
        supports = dict.fromkeys(region, 0)
        assigned, nodes, completions = {}, 0, 0
        def visit(unfilled, used_tiles, remaining):
            nonlocal nodes, completions
            check()
            nodes += 1
            if nodes > self.region_node_limit:
                raise _Budget
            if not unfilled:
                completions += 1
                for cell, code in assigned.items():
                    supports[cell] |= 1 << code
                return
            cell = min(unfilled, key=lambda c: (domains[c].bit_count(), c))
            for code in values(domains[cell]):
                tile = 1 << (code//3)
                if used_tiles & tile:
                    continue
                updated = remaining.copy()
                valid = True
                # Include open sides: helpers remain sound on generic fixtures.
                for side in range(3):
                    if self.problem.neighbors[cell, side] < 0 and self.masks[code][side]:
                        valid = False
                        break
                if not valid:
                    continue
                for other, side, other_side, _ in self.adj[cell]:
                    expected = reverse_mask(self.masks[code][side])
                    if other in assigned:
                        if self.masks[assigned[other]][other_side] != expected:
                            valid = False
                            break
                    elif other in updated:
                        updated[other] &= self._mask_buckets(updated[other], other_side).get(expected, 0)
                        if not updated[other]:
                            valid = False
                            break
                if valid:
                    assigned[cell] = code
                    visit(unfilled-{cell}, used_tiles | tile, updated)
                    del assigned[cell]
        if all(outside.values()):
            visit(set(region), 0, outside)
        return tuple(supports.items()), completions, nodes

    def prune_region(self, domains, cells, check=lambda: None):
        """Exhaustively solve one small, fixed-inventory region, or do nothing.

        Each adjacent outside cell must jointly support all incident region
        seams, but outside cells need not agree with one another. This ignores
        outside strand pairings and therefore cannot incorrectly exclude a
        Gold completion. The full neighboring domains are nevertheless keyed.
        """
        region = tuple(sorted(set(cells)))
        if not 2 <= len(region) <= self.region_size_limit or not all(domains[c] for c in region):
            return False
        key = self._region_key(domains, region)
        if key[2].bit_count() != len(region):
            return False
        self.stats['regional_checks'] += 1
        result = self._regional_cache.get(key)
        if result is None:
            try:
                result = self._solve_region(domains, key, check)
            except _Budget:
                self.stats['regional_budget_skips'] += 1
                return False
            _put(self._regional_cache, key, result, 128)
            self.stats['regional_cache_entries'] = len(self._regional_cache)
            self.stats['regional_assignments'] += result[1]
        else:
            self.stats['regional_cache_hits'] += 1
        self.last_region = {'cells': region, 'complete': True, 'assignments': result[1],
                            'nodes': result[2], 'outside_pairing_assumptions': False}
        return self._apply(domains, result[0], 'regional', check)

    def _regions(self, domains, check):
        eligible = {c for c, d in enumerate(domains) if 0 < _tile_choices(d).bit_count() <= 4}
        seen, candidates = set(), []
        # Regions consisting solely of singleton domains cannot yield useful
        # candidate support reductions; start only at an undecided cell. Every
        # connected eligible region with a non-singleton still has such a seed.
        for first in sorted(c for c in eligible if domains[c].bit_count() > 1):
            work = [(first,)]
            while work:
                check()
                region = work.pop()
                if region in seen:
                    continue
                seen.add(region)
                inventory = 0
                for c in region:
                    inventory |= _tile_choices(domains[c])
                if inventory.bit_count() > self.region_size_limit:
                    continue
                if len(region) >= 2 and inventory.bit_count() == len(region) and any(domains[c].bit_count() > 1 for c in region):
                    candidates.append(region)
                if len(region) < self.region_size_limit:
                    neighbors = set().union(*(self.neighbor_cells[c] for c in region)) & eligible
                    for other in sorted(neighbors-set(region), reverse=True):
                        work.append(tuple(sorted((*region, other))))
        return sorted(candidates, key=lambda r: (sum(domains[c].bit_count() for c in r), len(r), r))

    def prune(self, domains, check=lambda: None):
        if len(domains) != self.n:
            raise ValueError('Expected one integer domain per cell.')
        check()
        if not all(domains):
            return False
        changed = False
        if self.options['strand_connectivity']:
            changed |= self.prune_strands(domains, check)
        if all(domains) and self.options['biconnected']:
            changed |= self.prune_biconnected(domains, check)
        if all(domains) and self.options['multi_seam_cuts']:
            changed |= self.prune_cuts(domains, check)
        if all(domains) and self.options['regional_cache']:
            misses, count = 0, 0
            for region in self._regions(domains, check):
                check()
                key = self._region_key(domains, region)
                miss = key not in self._regional_cache
                if miss and misses >= self.region_misses_per_call:
                    continue
                changed |= self.prune_region(domains, region, check)
                misses += miss
                count += 1
                if not all(domains) or count >= 16:
                    break
        if all(domains) and self.options['separator_regions']:
            if self.separator is None:
                self.separator=SeparatorPruner(self.problem,self.target)
            try:
                changed |= self.separator.prune(domains,check)
            finally:
                self.stats.update(self.separator.stats)
        if all(domains) and self.options.get('extended_regions', False) and self.target=='gold':
            if self.before_hierarchical is not None:self.before_hierarchical(domains)
            if self.multi_port is None:
                self.multi_port=MultiPortCutPruner(self.problem)
                self.multi_port.max_cuts=24
                self.multi_port.max_candidates=8
                self.endpoint_supply=EndpointSupplyPruner(self.problem)
                self.endpoint_supply.max_candidates=12
            dependencies=tuple(range(self.n))
            def publish_trial(trial):
                modified=False
                for cell,mask in enumerate(trial):
                    check()
                    if mask!=domains[cell]:
                        domains[cell]&=mask;modified=True
                        if self.on_narrow is not None:self.on_narrow(cell,domains[cell],dependencies)
                        if not domains[cell]:break
                return modified
            try:
                trial=domains.copy();self.multi_port.prune(trial,check)
                changed|=publish_trial(trial)
                if all(domains):
                    for region,_ in self.cut_regions[:4]:
                        trial=domains.copy();self.endpoint_supply.prune_region(trial,region,check)
                        changed|=publish_trial(trial)
                        if not all(domains):break
            finally:
                for key,value in self.multi_port.stats.items():
                    if key.startswith('multi_port_'):self.stats[key]=value
                for key,value in self.endpoint_supply.stats.items():
                    if key.startswith('endpoint_supply_'):self.stats[key]=value
        if all(domains) and any(self.options.get(k,False) for k in ('hierarchical_regions','extended_regions','symbolic_regions','symbolic_inventory')):

            if self.hierarchical is None:
                self.hierarchical=HierarchicalRegionPruner(self.problem,self.target)
            if any(self.options.get(k,False) for k in ('extended_regions','symbolic_regions','symbolic_inventory')) and not self.hierarchical.extended_regions:
                self.hierarchical.enable_advanced(self.regional_cache_dir)
            if self.before_hierarchical is not None:self.before_hierarchical(domains)
            if self.hierarchical._network is not None:
                self.hierarchical._network.direct_diagrams=self.options.get('symbolic_regions',False)
                self.hierarchical._network.symbolic_inventory=self.options.get('symbolic_inventory',False)
            self.hierarchical.on_narrow=self.on_narrow
            try:
                changed |= self.hierarchical.prune(domains,check)
            finally:
                self.stats.update(self.hierarchical.stats)
                self.stats.update(self.hierarchical.extended_stats)
                if self.hierarchical._disk_store is not None:
                    store=self.hierarchical._disk_store
                    self.stats.update(regional_disk_hits=store.hits,regional_disk_writes=store.writes,regional_disk_rejections=store.rejections)
                self.hierarchical.on_narrow=None
        return changed
