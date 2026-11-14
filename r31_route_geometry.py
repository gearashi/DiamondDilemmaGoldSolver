"""Exact support geometry and connected cycle augmentation for four-swap exchanges."""
from collections import Counter
from bisect import bisect_right
from functools import lru_cache
from itertools import combinations, permutations

import numpy as np

from r30_cycles import _contact_rules, _group_rules, canonical_cycles


def proved_connected(bound, target, family):
    family = tuple(sorted(family))
    if family not in ((2, 2, 3), (2, 2, 2, 2)) or not _contact_rules(bound, target):
        return False
    three, four = _group_rules(bound)
    return three if len(family) == 3 else four


@lru_cache(maxsize=4)
def cut_index(family):
    """For each possible contact edge, precompute the partitions it connects on each cut."""
    from r30_multicenter import orders
    all_orders = orders(family)
    size = sum(family)
    pairs = tuple(combinations(range(size), 2))
    partitions = {}
    for index, order in enumerate(all_orders):
        groups = tuple(sorted(tuple(sorted(g)) for g in canonical_cycles(range(size), order)))
        partitions[groups] = partitions.get(groups, 0) | (1 << index)
    width = len(family)
    cuts = []
    # Each proper cut has a unique side excluding the first cycle.
    for mask in range(1, 1 << (width - 1)):
        bits = [0] * len(pairs)
        for groups, represented in partitions.items():
            side = {c for j, group in enumerate(groups[1:]) if mask & (1 << j) for c in group}
            for e, (i, j) in enumerate(pairs):
                if (i in side) != (j in side):
                    bits[e] |= represented
        cuts.append(tuple(bits))
    return all_orders, pairs, tuple(cuts)


@lru_cache(maxsize=8192)
def connected_mask(family, edges):
    all_orders, _, cuts = cut_index(family)
    live = (1 << len(all_orders)) - 1
    for cut in cuts:
        bridged = 0
        for e in edges:
            bridged |= cut[e]
        live &= bridged
        if not live:
            break
    return live


def eligible_orders(close, cells, family):
    all_orders, pairs, _ = cut_index(family)
    edges = tuple(e for e, (a, b) in enumerate(pairs) if close[cells[a], cells[b]])
    mask = connected_mask(family, edges)
    while mask:
        bit = mask & -mask
        yield all_orders[bit.bit_length() - 1]
        mask ^= bit


def connected(group, close):
    if not group:
        return True
    reached = {0}
    pending = [0]
    while pending:
        i = pending.pop()
        for j, cycle in enumerate(group):
            if j not in reached and any(close[a, b] for a in group[i] for b in cycle):
                reached.add(j)
                pending.append(j)
    return len(reached) == len(group)


def parent(group, close):
    """Delete the lexicographically last non-articulation cycle; every leaf has one parent."""
    for i in reversed(range(len(group))):
        smaller = group[:i] + group[i+1:]
        if connected(smaller, close):
            return smaller
    raise ValueError("Disconnected input to canonical parent.")


def cycle_options(available, length, boundary):
    """Generate exactly once by the least cell in the nonempty attachment intersection."""
    for anchor in sorted(available & boundary):
        later = sorted(x for x in available if x != anchor and (x not in boundary or x > anchor))
        for tail in combinations(later, length-1):
            first, *rest = sorted((anchor, *tail))
            for direction in permutations(rest):
                yield (first, *direction)


def cycle_mapping(group):
    return {a: cycle[(i+1) % len(cycle)] for cycle in group for i, a in enumerate(cycle)}


class ConnectedJoin:
    """Stream a typed connected family on a cell pool without materializing half catalogues.

    Each child extends its unique connected parent. The last layer prices the
    physical seam fit with a free choice of rotations per edge. This upper
    bound is exact for that relaxation and includes all inter-cycle edges.
    """
    def __init__(self, clusters, cells, family, target, *, actor_bounds=True):
        self.stars = clusters
        self.bound = clusters.bound
        self.family = tuple(sorted(family))
        source = tuple(cells)
        if len(set(source))!=len(source) or any(type(x) is not int and not isinstance(x,np.integer) for x in source):
            raise ValueError("Expected distinct integer cell IDs.")
        self.pool = frozenset(map(int,source))
        if (self.family not in ((2, 2, 3), (2, 2, 2, 2)) or len(self.pool) < sum(self.family)
                or not self.pool.issubset(self.bound.model.free)):
            raise ValueError("Invalid connected exchange family or pool.")
        self.target = target
        self.close = clusters.close
        self.near = {c: frozenset(i for i in self.pool if self.close[c,i]) for c in self.pool}
        self.edges = [(a,b) for a, _, b, _ in self.bound.model.problem.edges]
        self.incident = {c: tuple(i for i,e in enumerate(self.edges) if c in e) for c in self.pool}
        masks = self.bound.masks[self.bound.tiles,0]
        self.fit = (masks[:,None,:,None] == self.bound.model.reverse[masks[None,:,None,:]]).any(axis=(2,3))
        self.base = sum(int(self.fit[a,b]) for a,b in self.edges)
        self.work = Counter()
        self.buckets = lru_cache(maxsize=48)(self._bucket)
        from r31_route_price import DefectPrices
        self.actors = DefectPrices(self.bound) if actor_bounds else None

    def covered_swap_prefix(self, group):
        if not hasattr(self,'_checked_prefixes'):
            from r32_route_pool import load_prefix_certificates
            self._checked_prefixes = load_prefix_certificates(self.bound,self.target,self.family)
            self._verified_prefixes = {}
        group = tuple(group)
        if group not in self._checked_prefixes:
            return False
        if group not in self._verified_prefixes:
            from r32_route_pool import defect_cover
            try:
                result = defect_cover(self.bound,group,self.bound.model.free,3,208)
            except ImportError:
                return False
            self._verified_prefixes[group] = result['closed']
        if self._verified_prefixes[group]:
            self.work['certified_defect_prefixes'] += 1
            return True
        return False

    def _attachments(self, group, occupied, remaining):
        available = self.pool - occupied
        boundary = set().union(*(self.near[c] for c in occupied)) if occupied else available
        for length in sorted(set(remaining)):
            for cycle in cycle_options(available, length, boundary):
                child = tuple(sorted((*group, cycle)))
                if parent(child, self.close) == group:
                    yield cycle, child

    def catalogue_cells(self, length, anchor):
        # Only cycles containing anchor are needed. Two directed triples over
        # the same support may have very different standalone prices.
        remaining = sorted(self.pool-{anchor})
        if length==2:
            rows = np.column_stack((np.full(len(remaining),anchor),remaining))
            rows.sort(axis=1)
        else:
            pairs = np.asarray(list(combinations(remaining,2)),dtype=int)
            rows = np.column_stack((np.full(len(pairs),anchor),pairs))
            rows.sort(axis=1)
            rows = np.concatenate((rows,rows[:,(0,2,1)]))
        return rows

    def catalogue_prices(self, rows):
        length = rows.shape[1]
        # Directed cycle source vector. First count each moved cell's
        # individual substitutions, then correct every shared physical edge.
        gain = np.zeros(len(rows),dtype=np.int16)
        n = self.bound.n
        neighbors = self.bound.neighbors[:n].copy()
        fit = np.pad(self.fit,((0,1),(0,1))).astype(np.int16)
        for j in range(length):
            c = rows[:,j]
            d = rows[:,(j+1)%length]
            for side in range(3):
                other = neighbors[c,side]
                gain += fit[d,other]-fit[c,other]
            for i in range(j):
                a = rows[:,i]
                b = rows[:,(i+1)%length]
                shared = (neighbors[a] == c[:,None]).sum(axis=1)
                gain += (fit[b,d]-fit[b,c]-fit[a,d]+fit[a,c])*shared
        indices = np.lexsort((*rows.T[::-1],-gain))
        self.work['catalogue_rows'] += len(rows)
        return -gain[indices], indices

    def _bucket(self, length, anchor):
        rows = self.catalogue_cells(length,anchor)
        prices,indices = self.catalogue_prices(rows)
        return prices.tolist(),rows[indices]

    def attachment_bonuses(self,available,sources):
        donors = np.array(sorted(available),dtype=int)
        best = {}
        for c in available:
            values = np.zeros(len(donors),dtype=int)
            for e in self.incident[c]:
                a,b = self.edges[e]
                p = b if c==a else a
                if p in sources:
                    before, after = p,sources[p]
                    values += self.fit[donors,after].astype(int)-self.fit[donors,before]
                    values -= int(self.fit[c,after])-int(self.fit[c,before])
            best[c] = int(values.max())
        return best

    def _last(self,group,sources,score,length):
        available = self.pool - sources.keys()
        boundary = set().union(*(self.near[c] for c in sources)) & available
        # The change in a last cycle's standalone price comes only from
        # physical edges crossing to the chosen parent. For each free cell
        # maximize that correction over its possible incoming donors.
        best = self.attachment_bonuses(available,sources)
        for anchor in sorted(boundary):
            can_use = {x for x in available if x==anchor or x not in boundary or x>anchor}
            if len(can_use)<length:
                continue
            bonus = best[anchor]+sum(sorted((best[x] for x in can_use if x!=anchor),reverse=True)[:length-1])
            prices,rows = self.buckets(length,anchor)
            end = bisect_right(prices,score+bonus-self.target)
            self.work['sorted_rows_skipped'] += len(rows)-end
            self.work['empty_buckets'] += end==0
            for row in rows[:end]:
                cycle = tuple(map(int,row))
                if not can_use.issuperset(cycle):
                    continue
                child = tuple(sorted((*group,cycle)))
                if parent(child,self.close)==group:
                    yield cycle,child

    def _visit(self, group, sources, score, remaining, price):
        if self.actors and self.family==(2,2,3) and tuple(map(len,group))==(2,2):
            if self.actors.actor_upper(group,sum(remaining),pool=self.pool)<self.target:
                self.work['forced_swap_pairs_closed'] += 1
                return
        if tuple(map(len,group))==(2,2) and self.family==(2,2,3) and self.covered_swap_prefix(group):
            return
        children = self._attachments(group, set(sources), remaining)
        final = len(remaining) == 1
        if final and price:
            children = self._last(group,sources,score,remaining[0])
        possible = False
        for cycle, child in children:
            moves = cycle_mapping((cycle,))
            change = sources | moves
            new_score = score
            for e in {i for c in cycle for i in self.incident[c]}:
                a,b = self.edges[e]
                new_score += int(self.fit[change.get(a,a), change.get(b,b)]) - int(self.fit[sources.get(a,a), sources.get(b,b)])
            self.work['attachments_priced'] += 1
            if final:
                if price and new_score < self.target:
                    self.work['last_pruned'] += 1
                    continue
                possible = True
                cells = tuple(sorted(change))
                index = {c:i for i,c in enumerate(cells)}
                order = tuple(index[change[c]] for c in cells)
                self.work['exact_star_queries'] += 1
                if self.stars.price(cells,order) >= 2*self.target:
                    yield cells, order
            else:
                pending = list(remaining)
                pending.remove(len(cycle))
                yield from self._visit(child, change, new_score, pending, price)
        if final and not possible:
            self.work['parents_without_candidate'] += 1

    def root(self,cycle,*,price=True):
        """Search one canonical singleton-root subtree, including star-shaped completions."""
        cycle = tuple(cycle)
        if (len(set(cycle))!=len(cycle) or not self.pool.issuperset(cycle)
                or len(cycle) not in self.family or cycle[0]!=min(cycle)):
            raise ValueError("Invalid connected-cycle root.")
        moves = cycle_mapping((cycle,))
        gain = sum(int(self.fit[moves.get(a,a),moves.get(b,b)])-int(self.fit[a,b])
                   for a,b in self.edges if a in moves or b in moves)
        pending = list(self.family)
        pending.remove(len(cycle))
        yield from self._visit((cycle,),moves,self.base+gain,pending,price)

    def survivors(self, *, price=True):
        """Results cover connected endpoints; full-family use needs proved_connected."""
        yield from self._visit((), {}, self.base, self.family, price)
