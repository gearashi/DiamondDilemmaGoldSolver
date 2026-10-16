"""Sound, reusable edge-puzzle propagators; no puzzle search or GPU work.

Domains are integer bitsets of orientation codes 3*tile+rotation. prune() returns
whether domains changed; a zero domain denotes contradiction. Every deletion is
independently justified, so a cooperative check() exception may safely retain
partially filtered domains. These rules apply to both the edge and Gold targets.

The AllDifferent filter finds a perfect cell/tile matching, then retains exactly
its matched edges and edges in alternating cycles (SCCs of the residual graph).
See also the primary implementation reference:
https://github.com/google/or-tools/blob/stable/ortools/sat/all_different.cc
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from functools import lru_cache

from validator import reverse_mask
from constraint_native import select_hall
from constraint_explanations import removal_witness, failure_witness
from constraint_parity import ParityPruner, STAT_KEYS as PARITY_STAT_KEYS

STAT_KEYS = ('pruner_table_builds', 'pruner_table_reuses', 'hall_checks', 'hall_prunes', 'hall_failures', 'hall_explanation_checks', 'hall_explanation_witnesses', 'hall_explanation_fallbacks',
             'region_checks', 'region_prunes', 'region_failures',
             'cluster_checks', 'cluster_prunes', 'cluster_failures',
             'cluster_embeddings', 'forced_bonds', 'forced_clusters',
             'neighborhood_checks', 'neighborhood_prunes', 'neighborhood_failures') + PARITY_STAT_KEYS


def values(bits):
    while bits:
        low = bits & -bits
        yield low.bit_length()-1
        bits ^= low


@lru_cache(maxsize=2048)
def _reverse(mask):
    return reverse_mask(mask)


def boundary_minimum(counts):
    """Minimum unpaired physical edge incidences, including the empty mask.

    Internal joins remove one m and one reverse(m), or two identical masks when
    m is self-reverse. Pairing ignores geometry and tile ownership and therefore
    gives only a lower bound on a patch's necessary boundary size.
    """
    seen, result = set(), 0
    for mask, count in counts.items():
        if mask in seen:
            continue
        other = _reverse(mask)
        seen.update((mask, other))
        result += count % 2 if other == mask else abs(count-counts.get(other, 0))
    return result


# Twelve orientation bits encode four physical tiles. A small lookup gathers
# each nonempty triplet into one bit, avoiding a Python iteration per rotation.
_TILE_GATHER = tuple(sum(bool(chunk & (7 << (3*t))) << t for t in range(4))
                     for chunk in range(4096))


@lru_cache(maxsize=8192)
def _tile_choices(domain):
    result = 0
    # Sparse late-search domains should not scan dozens of empty chunks.
    if domain.bit_count() <= domain.bit_length()//12:
        for code in values(domain):
            result |= 1 << (code//3)
        return result
    shift = 0
    while domain:
        result |= _TILE_GATHER[domain & 4095] << shift
        domain >>= 12
        shift += 4
    return result


_ORIENTATION_SCATTER = tuple(sum(7 << (3*t) for t in range(4) if chunk & (1 << t))
                             for chunk in range(16))


@lru_cache(maxsize=8192)
def _orientation_choices(tiles):
    """Expand physical-tile bits to their three orientation bits."""
    result, shift = 0, 0
    while tiles:
        result |= _ORIENTATION_SCATTER[tiles & 15] << shift
        tiles >>= 4
        shift += 12
    return result


def _remember(cache, key, value, limit):
    """Bound a pure exact-key cache; eviction only causes recomputation."""
    if key not in cache and len(cache) >= limit:
        cache.pop(next(iter(cache)))
    cache[key] = value


def _distinct_supports(choices):
    """Hall's condition for at most three distinct neighboring cells."""
    if any(not row for row in choices):
        return False
    if len(choices) < 2:
        return True
    a, b = choices[:2]
    if (a | b).bit_count() < 2:
        return False
    if len(choices) == 2:
        return True
    c = choices[2]
    return ((a | c).bit_count() >= 2 and (b | c).bit_count() >= 2
            and (a | b | c).bit_count() >= 3)


@dataclass(frozen=True)
class PrunerTables:
    """Completed puzzle-only deductions; mutable search state stays private.

    Every field contains only immutable values. Binding includes every input
    used to derive clusters and regional boundaries; parity has its own exact
    binding. Original cell domains are deliberately absent because these tables
    hold no deductions under narrowed search or job assumptions.
    """
    binding: tuple
    regions: tuple
    bonds: tuple
    clusters: tuple
    cluster_masks: tuple
    parity: object
    stars: tuple | None = None
    cluster_supports: tuple | None = None


class Pruner:
    """Immutable puzzle deductions plus per-call Hall/cluster/region filtering.

    Default regions are individual macrofaces (where board.cells expose face)
    and the entire board. Explicit regions may also be supplied as cell sets.
    Forced bonds are conservatively disabled on boards with any open side.
    *_prunes counts removed orientation values, not branches or full boards.
    Neighborhood Hall checks default to center domains of at most 24 values;
    neighborhood_limit=None enables the more expensive unrestricted rule.
    """
    def __init__(self, problem, check=lambda: None, *, regions=None, neighborhood=True,
                 neighborhood_limit=24, parity=True, native='python', tables=None):
        self.problem, self.n, self.check = problem, problem.n, check
        if neighborhood_limit is not None and (type(neighborhood_limit) is not int or neighborhood_limit < 1):
            raise ValueError('neighborhood_limit must be positive or None for unrestricted checks.')
        self.neighborhood = bool(neighborhood)
        self.neighborhood_limit = neighborhood_limit
        self._stars = None
        self._matching = None
        self.on_narrow = None
        self._hall_cache = {}
        self._neighborhood_cache = {}
        self.native_hall,self.native_hall_reason=select_hall(native)
        self._inventory_cache = {}
        self.stats = dict.fromkeys(STAT_KEYS, 0)
        def rows(source):
            result = []
            for row in source:
                self.check()
                result.append(tuple(map(int, row)))
            return tuple(result)
        self.masks, self.neighbors, self.sides = map(rows, (problem.masks, problem.neighbors, problem.sides))
        self.tile_bits = tuple(7 << (3*tile) for tile in range(self.n))
        self.base_masks = tuple(self.masks[3*tile] for tile in range(self.n))
        self.check()
        if regions is None:
            faces = defaultdict(list)
            for cell, record in enumerate(problem.board.cells):
                if hasattr(record, 'face'):
                    faces[record.face].append(cell)
            regions = list(faces.values()) + [range(self.n)]
        region_cells = []
        seen_regions = set()
        for region in regions:
            self.check()
            cells = tuple(sorted(set(region)))
            if not cells or any(type(cell) is not int or not 0 <= cell < self.n for cell in cells):
                raise ValueError('Regions must be nonempty sets of valid cell indices.')
            if cells in seen_regions:
                continue
            seen_regions.add(cells)
            region_cells.append(cells)
        binding = (self.n, self.masks, self.neighbors, self.sides, tuple(region_cells), bool(parity))
        if tables is not None:
            if not isinstance(tables, PrunerTables) or tables.binding != binding:
                raise ValueError('Pruner tables differ from current puzzle masks, topology or regions.')
            self.parity = ParityPruner(problem, tables=tables.parity, check=check) if parity else None
            self.regions = tuple((cells, dict(boundary), size) for cells, boundary, size in tables.regions)
            self.bonds, self.clusters, self._cluster_masks = tables.bonds, tables.clusters, tables.cluster_masks
            self._cluster_supports = tables.cluster_supports
            if self._cluster_supports is None:
                # Complete older in-memory tables before publishing the new
                # immutable index; cancelled construction publishes nothing.
                self._cluster_supports = self._build_cluster_supports()
                tables = replace(tables, cluster_supports=self._cluster_supports)
            self._cluster_cache = [{} for _ in self.clusters]
            self.stats.update(pruner_table_reuses=1, forced_bonds=len(self.bonds), forced_clusters=len(self.clusters))
            self.tables = tables
            self._stars = tables.stars
            return
        self.parity = ParityPruner(problem, check=check) if parity else None
        self.regions = []
        for cells in region_cells:
            self.check()
            members = set(cells)
            boundary = {cell: tuple(side for side, nb in enumerate(self.neighbors[cell]) if nb not in members)
                        for cell in cells}
            boundary = {cell: sides for cell, sides in boundary.items() if sides}
            self.regions.append((cells, boundary, sum(map(len, boundary.values()))))
        self.regions = tuple(self.regions)
        self.bonds, self.clusters = self._build_clusters()
        self.stats['forced_bonds'] = len(self.bonds)
        self.stats['forced_clusters'] = len(self.clusters)
        self._cluster_cache = [{} for _ in self.clusters]
        self._cluster_masks = []
        for members, bits, embeddings in self.clusters:
            self.check()
            self._cluster_masks.append(tuple(tuple((cell, 1 << code) for cell, code in embedding)
                                             for embedding in embeddings))
        self._cluster_masks = tuple(self._cluster_masks)
        self._cluster_supports = self._build_cluster_supports()
        self.tables = PrunerTables(binding,
            tuple((cells, tuple(boundary.items()), size) for cells, boundary, size in self.regions),
            self.bonds, self.clusters, self._cluster_masks,
            self.parity.tables if self.parity is not None else None,
            cluster_supports=self._cluster_supports)
        self.stats['pruner_table_builds'] = 1

    def _build_cluster_supports(self):
        """Transpose finite embeddings into immutable per-cell support bitsets.

        Each bit identifies one complete embedding. The absent mask permits a
        cell to hold an unrelated tile only when a viable embedding omits it.
        Tables depend on puzzle geometry, never on mutable search domains.
        """
        result = []
        for embeddings in self._cluster_masks:
            all_embeddings = (1 << len(embeddings))-1
            rows, present = [{} for _ in range(self.n)], [0]*self.n
            for index, embedding in enumerate(embeddings):
                self.check()
                bit = 1 << index
                for cell, orientation in embedding:
                    present[cell] |= bit
                    rows[cell][orientation] = rows[cell].get(orientation, 0) | bit
            immutable_rows = []
            for used, row in zip(present, rows):
                self.check()
                immutable_rows.append((all_embeddings ^ used, tuple(row.items()), sum(row)))
            result.append((all_embeddings, tuple(immutable_rows)))
        return tuple(result)

    def _build_clusters(self):
        if any(nb < 0 for row in self.neighbors for nb in row):
            return (), ()
        occurrences = defaultdict(list)
        for tile, masks in enumerate(self.base_masks):
            self.check()
            for side, mask in enumerate(masks):
                occurrences[mask].append((tile, side))
        bonds = []
        for mask in sorted(occurrences):
            self.check()
            reverse = _reverse(mask)
            if mask < reverse and len(occurrences[mask]) == len(occurrences.get(reverse, ())) == 1:
                bonds.append((*occurrences[mask][0], *occurrences[reverse][0]))
            elif mask == reverse and len(occurrences[mask]) == 2:
                bonds.append((*occurrences[mask][0], *occurrences[mask][1]))
        adjacency = defaultdict(list)
        for a, sa, b, sb in bonds:
            adjacency[a].append((sa, b, sb))
            adjacency[b].append((sb, a, sa))
        clusters, seen = [], set()
        for root in sorted(adjacency):
            if root in seen:
                continue
            members, pending = set(), [root]
            while pending:
                self.check()
                tile = pending.pop()
                if tile in members:
                    continue
                members.add(tile)
                pending.extend(other for _, other, _ in adjacency[tile] if other not in members)
            seen.update(members)
            embeddings = []
            for anchor in range(self.n):
                for rotation in range(3):
                    self.check()
                    placements = {root: (anchor, rotation)}
                    owners = {anchor: root}
                    pending = [root]
                    valid = True
                    while pending and valid:
                        self.check()
                        tile = pending.pop()
                        cell, rot = placements[tile]
                        for side, other, other_side in adjacency[tile]:
                            mapped_side = (side+rot) % 3
                            next_cell = self.neighbors[cell][mapped_side]
                            next_rot = (self.sides[cell][mapped_side]-other_side) % 3
                            proposed = (next_cell, next_rot)
                            if other in placements:
                                if placements[other] != proposed:
                                    valid = False
                                    break
                            elif next_cell in owners:
                                valid = False
                                break
                            else:
                                placements[other] = proposed
                                owners[next_cell] = other
                                pending.append(other)
                    if valid:
                        embeddings.append(tuple(sorted((cell, 3*tile+rot) for tile, (cell, rot) in placements.items())))
                        self.stats['cluster_embeddings'] += 1
            # Different anchors describe different placements of the root tile.
            bits = sum(self.tile_bits[tile] for tile in members)
            clusters.append((tuple(sorted(members)), bits, tuple(embeddings)))
        return tuple(bonds), tuple(clusters)

    def _narrow(self, domains, cell, allowed, kind, dependencies=None):
        old = domains[cell]
        new = old & allowed
        if new == old:
            return False
        domains[cell] = new
        if dependencies is not None and self.on_narrow is not None:self.on_narrow(cell,new,dependencies)
        self.stats[kind+'_prunes'] += (old ^ new).bit_count()
        return True

    def _fail(self, domains, cell, kind, dependencies=None):
        self.stats[kind+'_failures'] += 1
        return self._narrow(domains, cell, 0, kind,dependencies)

    def prune(self, domains):
        self.check()
        if (len(domains) != self.n or any(type(domain) is not int or domain < 0
                                        or domain >> (3*self.n) for domain in domains)):
            raise ValueError('Domains must contain one valid orientation bitset per cell.')
        if any(domain == 0 for domain in domains):
            return False
        changed = self.prune_hall(domains)
        if 0 in domains:
            return changed
        changed = self.prune_clusters(domains) or changed
        if 0 in domains:
            return changed
        changed = self.prune_regions(domains) or changed
        if 0 in domains:
            return changed
        if self.neighborhood:
            changed = self.prune_neighborhood(domains, max_domain_values=self.neighborhood_limit) or changed
        if 0 in domains or self.parity is None:
            return changed
        try:
            return self.parity.prune(domains, self.check) or changed
        finally:
            # Keep counters accurate even when a cooperative stop interrupts a
            # pass after sound reductions have already changed some domains.
            self.stats.update(self.parity.stats)

    def prune_hall(self, domains):
        """Exact AllDifferent filtering with checked matching and graph reuse.

        The cache key is the entire cell-to-physical-tile graph, not orientation
        domains or an assumption that search only narrows. A previous matching
        supplies only still-allowed edges; every missing edge is repaired before
        SCC deductions. All cached conclusions are complete immutable values.
        """
        self.stats['hall_checks'] += 1
        rows = []
        for domain in domains:
            self.check()
            rows.append(_tile_choices(domain))
        signature = tuple(rows)
        allowed = self._hall_cache.get(signature)
        if allowed is None:
            if self.native_hall is not None:
                filtered,matching,failed=self.native_hall.filter(rows,self._matching,self.check)
                if filtered is None:
                    witness=failure_witness(rows,self.check) if self.on_narrow is not None else None
                    return self._fail(domains,failed,'hall',witness)
                self._matching=matching
                allowed=tuple(_orientation_choices(row) for row in filtered)
            else:
                expanded = {}
                choices = []
                for row in rows:
                    self.check()
                    if row not in expanded:
                        expanded[row] = tuple(values(row))
                    choices.append(expanded[row])
                matched_cell, matched_tile = [-1]*self.n, [-1]*self.n
                if self._matching is not None:
                    for cell, tile in enumerate(self._matching):
                        self.check()
                        if rows[cell] & (1 << tile):
                            matched_cell[cell], matched_tile[tile] = tile, cell
                for start in sorted(range(self.n), key=lambda cell: (len(choices[cell]), cell)):
                    self.check()
                    if matched_cell[start] >= 0:
                        continue
                    pending = deque([start])
                    parents = [-1]*self.n
                    seen_cells = {start}
                    free = -1
                    while pending and free < 0:
                        self.check()
                        cell = pending.popleft()
                        for tile in choices[cell]:
                            if parents[tile] >= 0:
                                continue
                            parents[tile] = cell
                            owner = matched_tile[tile]
                            if owner < 0:
                                free = tile
                                break
                            if owner not in seen_cells:
                                seen_cells.add(owner)
                                pending.append(owner)
                    if free < 0:
                        return self._fail(domains, start, 'hall',tuple(sorted(seen_cells)) if self.on_narrow is not None else None)
                    while free >= 0:
                        cell = parents[free]
                        previous = matched_cell[cell]
                        matched_cell[cell], matched_tile[free] = free, cell
                        free = previous
                self._matching = tuple(matched_cell)
                # Contract matched tile->cell arcs. Group identical source rows so
                # dense repeated domains do not construct the same adjacency again.
                grouped = {}
                for cell, row in enumerate(rows):
                    grouped[row] = grouped.get(row, 0) | (1 << cell)
                mapped, reverse = {}, [0]*self.n
                for row, sources in grouped.items():
                    self.check()
                    targets = 0
                    for tile in expanded[row]:
                        owner = matched_tile[tile]
                        targets |= 1 << owner
                        reverse[owner] |= sources
                    mapped[row] = targets
                graph = [mapped[row] for row in rows]
                # Bitset DFS skips already visited outgoing edges in one operation.
                # Finishing order plus reverse reachability is the same Kosaraju
                # decomposition as the scalar implementation.
                order, seen = [], 0
                for root in range(self.n):
                    if seen & (1 << root):
                        continue
                    seen |= 1 << root
                    stack = [root]
                    while stack:
                        self.check()
                        cell = stack[-1]
                        unseen = graph[cell] & ~seen
                        if not unseen:
                            order.append(stack.pop())
                        else:
                            bit = unseen & -unseen
                            seen |= bit
                            stack.append(bit.bit_length()-1)
                component, assigned = [-1]*self.n, 0
                for root in reversed(order):
                    if assigned & (1 << root):
                        continue
                    pending = 1 << root
                    assigned |= pending
                    while pending:
                        self.check()
                        bit = pending & -pending
                        pending ^= bit
                        cell = bit.bit_length()-1
                        component[cell] = root
                        unseen = reverse[cell] & ~assigned
                        assigned |= unseen
                        pending |= unseen
                component_tiles = {}
                for cell, label in enumerate(component):
                    component_tiles[label] = component_tiles.get(label, 0) | (1 << matched_cell[cell])
                filtered = tuple(row & component_tiles[component[cell]] for cell, row in enumerate(rows))
                allowed = tuple(_orientation_choices(row) for row in filtered)
            # The GAC output has the same perfect matchings and is a fixed point
            # of this filter, so it may be cached under its own graph as well.
            for key in (signature, filtered):
                _remember(self._hall_cache, key, allowed, 16)
        changed = False;reason_budget=8
        for cell, keep in enumerate(allowed):
            self.check();dependencies=None
            removed=_tile_choices(domains[cell]&~keep) if self.on_narrow is not None else 0
            if self.on_narrow is not None and removed:
                witnesses=[]
                for tile in values(removed):
                    if reason_budget<=0:break
                    reason_budget-=1;self.stats['hall_explanation_checks']+=1
                    witness=removal_witness(rows,self._matching,cell,tile,self.check)
                    if witness is None:break
                    witnesses.append(witness)
                if len(witnesses)==removed.bit_count():
                    dependencies=tuple(sorted({c for witness in witnesses for c in witness}))
                    self.stats['hall_explanation_witnesses']+=len(witnesses)
                else:self.stats['hall_explanation_fallbacks']+=1;dependencies=tuple(range(self.n))
            changed = self._narrow(domains, cell, keep, 'hall',dependencies) or changed
        return changed

    def prune_clusters(self, domains):
        changed = False
        for index, (members, tile_bits, embeddings) in enumerate(self.clusters):
            self.check()
            self.stats['cluster_checks'] += 1
            # Other tiles do not affect which embeddings of this cluster are
            # possible. Their values are retained unless every embedding uses
            # the cell. Project ALL cells, including absent cluster placements.
            signature = tuple(domain & tile_bits for domain in domains)
            cache = self._cluster_cache[index]
            allowed = cache.get(signature)
            if allowed is None:
                viable, rows = self._cluster_supports[index]
                if not viable:
                    return self._fail(domains, 0, 'cluster') or changed
                for domain, (absent, supports, full) in zip(signature, rows):
                    self.check()
                    if domain & full == full:
                        continue
                    possible = absent
                    for bit, embeddings_with_value in supports:
                        if domain & bit:
                            possible |= embeddings_with_value
                    viable &= possible
                    if not viable:
                        return self._fail(domains, 0, 'cluster') or changed
                allowed = []
                for absent, supports, full in rows:
                    self.check()
                    keep = ~tile_bits if absent & viable else 0
                    for bit, embeddings_with_value in supports:
                        if embeddings_with_value & viable:
                            keep |= bit
                    allowed.append(keep)
                allowed = tuple(allowed)
                # Removing unsupported placements does not remove any viable
                # embedding. The projected output is therefore also a fixed
                # point of this one cluster filter.
                filtered = tuple(row & keep for row, keep in zip(signature, allowed))
                for key in (signature, filtered):
                    _remember(cache, key, allowed, 8)
            for cell, keep in enumerate(allowed):
                self.check()
                changed = self._narrow(domains, cell, keep, 'cluster') or changed
                if not domains[cell]:
                    self.stats['cluster_failures'] += 1
                    return changed
        return changed

    def prune_regions(self, domains):
        changed = False
        choices = []
        for domain in domains:
            self.check()
            choices.append(_tile_choices(domain))
        for cells, boundary, boundary_size in self.regions:
            self.check()
            possible = 0
            for cell in cells:
                possible |= choices[cell]
            # Exactly |region| tile IDs means AllDifferent fixes the inventory,
            # even though their positions and rotations remain undecided.
            if possible.bit_count() != len(cells):
                continue
            self.stats['region_checks'] += 1
            cached = self._inventory_cache.get(possible)
            if cached is None:
                inventory = Counter(mask for tile in values(possible) for mask in self.base_masks[tile])
                cached = (inventory, boundary_minimum(inventory))
                _remember(self._inventory_cache, possible, cached, 64)
            inventory, minimum = cached
            if minimum > boundary_size:
                return self._fail(domains, cells[0], 'region') or changed
            forced, forced_by_cell = Counter(), {}
            for cell, sides in boundary.items():
                if domains[cell].bit_count() == 1:
                    code = domains[cell].bit_length()-1
                    items = Counter(self.masks[code][side] for side in sides)
                    forced.update(items)
                    forced_by_cell[cell] = items
            for cell, sides in boundary.items():
                self.check()
                already = forced_by_cell.get(cell, {})
                keep = 0
                # Account for this proposed cell plus other singleton boundary
                # cells. Every unpaired remaining incidence must fit into the
                # as-yet-unselected part of the patch boundary.
                for index, code in enumerate(values(domains[cell])):
                    if index % 32 == 0:
                        self.check()
                    chosen = forced.copy()
                    chosen.subtract(already)
                    chosen.update(self.masks[code][side] for side in sides)
                    remaining = inventory.copy()
                    remaining.subtract(chosen)
                    free_boundary = boundary_size-sum(chosen.values())
                    if (free_boundary >= 0 and all(count >= 0 for count in remaining.values())
                            and boundary_minimum(remaining) <= free_boundary):
                        keep |= 1 << code
                changed = self._narrow(domains, cell, keep, 'region') or changed
                if not domains[cell]:
                    self.stats['region_failures'] += 1
                    return changed
        return changed

    def _build_stars(self):
        by_side = [defaultdict(int) for _ in range(3)]
        for code, row in enumerate(self.masks):
            self.check()
            for side, mask in enumerate(row):
                by_side[side][mask] |= 1 << code
        profiles, stars = {}, []
        for cell in range(self.n):
            self.check()
            grouped = defaultdict(list)
            for side, neighbor in enumerate(self.neighbors[cell]):
                if neighbor >= 0:
                    grouped[neighbor].append((side, self.sides[cell][side]))
            star = []
            for neighbor, seams in grouped.items():
                relation = tuple(seams)
                if relation not in profiles:
                    support = []
                    for code, row in enumerate(self.masks):
                        if code % 32 == 0:
                            self.check()
                        allowed = (1 << (3*self.n))-1
                        for side, other_side in relation:
                            allowed &= by_side[other_side].get(_reverse(row[side]), 0)
                        support.append(allowed)
                    profiles[relation] = tuple(support)
                star.append((neighbor, profiles[relation]))
            stars.append(tuple(star))
        return tuple(stars)

    def prune_neighborhood(self, domains, *, max_domain_values=None):
        """Keep a candidate only if its neighbors can use distinct other tiles.

        Parallel seams are intersected before projecting orientations to tile
        IDs, so one orientation must satisfy every seam to the same neighbor.
        The test is exact for this relaxed star, not for the entire puzzle.
        A per-pass snapshot makes cached supports safe after domain reductions.
        On larger boards, completed exact star results survive calls in a private
        512-entry cache. The key includes every neighboring domain and immutable
        support profile; widening or changing a parallel seam cannot hit it.
        Tiny boards retain the cheaper per-pass cache to avoid eviction overhead.
        The regular propagation pass limits this optional strengthening to small
        center domains; skipping wider stars weakens filtering, never coverage.
        Direct calls default to checking every center domain.
        """
        eligible = [cell for cell, domain in enumerate(domains)
                    if max_domain_values is None or domain.bit_count() <= max_domain_values]
        if not eligible:
            return False
        if self._stars is None:
            # Publish only a fully constructed immutable tuple. An exception
            # inside construction leaves both the old record and _stars intact.
            self._stars = self._build_stars()
            self.tables = replace(self.tables, stars=self._stars)
        persistent_stars = self.n >= 32
        star_cache = self._neighborhood_cache if persistent_stars else {}
        snapshot, cache, changed = tuple(domains), {}, False
        for cell in eligible:
            self.check()
            star = self._stars[cell]
            if not star:
                continue
            signature = (snapshot[cell], tuple((snapshot[neighbor], id(profile)) for neighbor, profile in star))
            if signature in star_cache:
                changed = self._narrow(domains, cell, star_cache[signature], 'neighborhood') or changed
                if not domains[cell]:
                    self.stats['neighborhood_failures'] += 1
                    return changed
                continue
            keep = 0
            for index, code in enumerate(values(snapshot[cell])):
                if index % 32 == 0:
                    self.check()
                self.stats['neighborhood_checks'] += 1
                choices = []
                forbidden = 1 << (code//3)
                for neighbor, profile in star:
                    key = (snapshot[neighbor], profile[code])
                    if key not in cache:
                        cache[key] = _tile_choices(snapshot[neighbor] & profile[code])
                    choices.append(cache[key] & ~forbidden)
                if _distinct_supports(choices):
                    keep |= 1 << code
            # The whole star was checked. No interrupted candidate scan can
            # publish an incomplete result, even when earlier stars completed.
            if persistent_stars:
                _remember(star_cache, signature, keep, 512)
            else:
                star_cache[signature] = keep
            changed = self._narrow(domains, cell, keep, 'neighborhood') or changed
            if not domains[cell]:
                self.stats['neighborhood_failures'] += 1
                return changed
        return changed
