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

from constraint_pruning import values, _tile_choices, _orientation_choices
from search_options import normalize_options
from validator import reverse_mask
from constraint_native import select_graph_helpers
from constraint_separator import SeparatorPruner, STAT_KEYS as SEPARATOR_STAT_KEYS
from constraint_regions import HierarchicalRegionPruner, STAT_KEYS as HIERARCHICAL_STAT_KEYS

STAT_KEYS = ('strand_checks', 'strand_prunes', 'strand_failures',
             'cut_checks', 'cut_prunes', 'cut_failures',
             'regional_checks', 'regional_prunes', 'regional_failures',
             'regional_cache_hits', 'regional_cache_entries',
             'regional_assignments', 'regional_budget_skips',
             'biconnected_checks', 'biconnected_prunes', 'biconnected_failures') + SEPARATOR_STAT_KEYS + HIERARCHICAL_STAT_KEYS


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
        self._buckets, self._cut_cache, self._regional_cache = {}, {}, {}
        self.cut_vector_limit, self.region_node_limit = 256, 4096
        self.region_size_limit, self.region_misses_per_call = 4, 4
        self.last_region = None
        self._biconnected_cache, self._local_edges, self._local_certificates = {}, {}, {}
        self._segment_supports = None
        self.separator = None
        self.hierarchical = None
        self.on_narrow = None
        self.before_hierarchical = None

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

    def _local_forest(self, domain, check):
        cached = self._local_ports.get(domain)
        if cached is not None:
            return cached
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
        if allowed is None:
            dsu, used = _DSU(33*self.n), []
            for cell, domain in enumerate(domains):
                check()
                forest, ports = self._local_forest(domain, check)
                used.append(ports)
                for a, b in forest:
                    dsu.join(33*cell+a, 33*cell+b)
            for a, sa, b, sb in self.problem.edges:
                check()
                for p in range(11):
                    u, v = 11*sa+p, 11*sb+10-p
                    if used[a] >> u & 1 and used[b] >> v & 1:
                        dsu.join(33*a+u, 33*b+v)
            choices, common = [], None
            for cell, domain in enumerate(domains):
                check()
                roots = {p: dsu.find(33*cell+p) for p in values(used[cell])}
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
        if allowed is None:
            if self._segment_supports is None:
                supports={}
                for code,pairs in enumerate(self.segments):
                    for pair in pairs:
                        pair=tuple(sorted(pair));supports[pair]=supports.get(pair,0)|(1<<code)
                self._segment_supports=tuple(supports.items())
            graph=[];used=[]
            for cell,domain in enumerate(domains):
                check()
                pairs,ports=self._local_block_certificate(domain,check)
                graph.extend((33*cell+a,33*cell+b) for a,b in pairs)
                used.append(ports)
            seam_ids={}
            for a,sa,b,sb in self.problem.edges:
                check()
                for p in range(11):
                    u,v=11*sa+p,11*sb+10-p
                    if used[a]>>u&1 and used[b]>>v&1:
                        eid=len(graph);graph.append((33*a+u,33*b+v))
                        seam_ids[a,u]=eid;seam_ids[b,v]=eid
            blocks=self._cyclic_edge_blocks(33*self.n,graph,check)
            choices=[];common=None
            for cell,domain in enumerate(domains):
                check()
                # Every chosen port uses its seam edge. When all those edges
                # share one block, each actual segment joins two vertices in
                # that block too, even if the certificate replaced that edge.
                all_blocks={blocks[seam_ids[cell,p]] if (cell,p) in seam_ids else -1 for p in values(used[cell])}
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
                                edge=seam_ids.get((cell,port))
                                hosting.add(-1 if edge is None else blocks[edge])
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
        key = edge_ids, tuple((cell, domains[cell]) for cell in involved)
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
            if product > self.cut_vector_limit:
                return None
            tables.append((a, b, tuple((m, left[m], right[m]) for m in masks)))
        support = dict.fromkeys(involved, 0)
        def visit(index, current, crossings):
            check()
            if index == len(tables):
                if crossings and crossings % 2 == 0:
                    for cell, domain in current.items():
                        support[cell] |= domain
                return
            a, b, masks = tables[index]
            for mask, left, right in masks:
                da, db = current[a] & left, current[b] & right
                if da and db:
                    narrowed = current.copy()
                    narrowed[a], narrowed[b] = da, db
                    visit(index+1, narrowed, crossings+mask.bit_count())
        visit(0, {cell: domains[cell] for cell in involved}, 0)
        result = tuple(support.items())
        _put(self._cut_cache, key, result, 32)
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
        seen, changed, checks = set(), False, 0
        for _, boundary in self.cut_regions:
            check()
            live = tuple(eid for eid in boundary if possible[eid])
            if len(live) not in (2, 3) or live in seen:
                continue
            seen.add(live)
            support = self._cut_support(domains, live, check)
            if support is None:
                continue
            self.stats['cut_checks'] += 1
            changed |= self._apply(domains, support, 'cut', check)
            checks += 1
            if not all(domains) or checks >= 8:
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
        if all(domains) and self.options.get('hierarchical_regions', False):
            if self.hierarchical is None:
                self.hierarchical=HierarchicalRegionPruner(self.problem,self.target)
            if self.before_hierarchical is not None:self.before_hierarchical(domains)
            self.hierarchical.on_narrow=self.on_narrow
            try:
                changed |= self.hierarchical.prune(domains,check)
            finally:
                self.stats.update(self.hierarchical.stats)
                self.hierarchical.on_narrow=None
        return changed
