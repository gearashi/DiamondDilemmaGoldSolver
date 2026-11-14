"""Connected cycle generation shared across physical centers, with exact two-swap seam pricing."""
from collections import Counter, defaultdict
from functools import lru_cache
from itertools import combinations
import numpy as np

from r31_route_geometry import connected, cycle_mapping, cycle_options, parent


def adjacent_mask(group, cycle, near):
    cells = set().union(*(near[c] for c in cycle))
    return sum(1 << i for i, g in enumerate(group) if any(x in cells for x in g))


def graph_masks(group, near):
    return tuple(adjacent_mask(group, cycle, near) & ~(1 << i) for i, cycle in enumerate(group))


@lru_cache(maxsize=1024)
def owns_attachment(graph, attached, insertion):
    """After inserting the new cycle, no lexicographically later cycle may be removable."""
    n = len(graph)
    if not attached:
        return False
    rows = [row | ((1 << n) if attached & (1 << i) else 0) for i,row in enumerate(graph)]
    rows.append(attached)
    for remove in range(insertion, n):
        available = ((1 << (n+1))-1) ^ (1 << remove)
        seen, frontier = 0, 1 << n
        while frontier:
            bit = frontier & -frontier
            frontier ^= bit
            seen |= bit
            frontier |= rows[bit.bit_length()-1] & available & ~seen
        if seen == available:
            return False
    return True


class Catalog:
    def __init__(self, join, family):
        self.pool = join.pool
        self.near = join.near
        self.close = join.close
        self.family = family
        self.work = Counter()

    def _children(self, group, length):
        """Index attachments by their exact contacts, order and canonical parent."""
        used = {c for g in group for c in g}
        graph = graph_masks(group, self.near)
        # Unique minimum identifies insertion order as cycles have disjoint cells.
        cells = sorted(self.pool - used)
        by_first = defaultdict(list)
        for c in cells:
            signature = tuple(i for i,g in enumerate(group) if any(c in self.near[t] for t in g))
            by_first[sum(1<<i for i in signature)].append(c)
        from itertools import combinations_with_replacement, product, permutations
        from bisect import bisect_left
        for sigs in combinations_with_replacement(by_first, length):
            attached = 0
            for s in sigs:
                attached |= s
            if not attached:
                continue
            low = min(by_first[s][0] for s in sigs)
            high = min(by_first[s][-1] for s in sigs)
            mins = [g[0] for g in group]
            possible = {i for i in range(bisect_left(mins,low),bisect_left(mins,high)+1)
                        if owns_attachment(graph,attached,i)}
            if not possible:
                self.work['wrong_owner_buckets'] += 1
                continue
            groups = [combinations(by_first[s],sigs.count(s)) for s in dict.fromkeys(sigs)]
            for chunks in product(*groups):
                chosen = tuple(sorted(x for row in chunks for x in row))
                if bisect_left(mins,chosen[0]) in possible:
                    for tail in permutations(chosen[1:]):
                        yield (chosen[0], *tail)

    def children(self, group, length):
        return self._children(group,length)


class PhysicalPrices:
    def __init__(self, join):
        self.join = join

    def delta(self, baseline, cycle):
        moves = cycle_mapping((cycle,))
        fit = self.join.fit
        def identity(c):
            return baseline.get(c,c)
        delta = 0
        for e in {e for c in cycle for e in self.join.incident[c]}:
            a,b = self.join.edges[e]
            aa,bb = identity(a),identity(b)
            delta += int(fit[moves.get(a,aa),moves.get(b,bb)]) - int(fit[aa,bb])
        return delta

    def one_score(self, group):
        moves = cycle_mapping(group)
        return self.join.base + sum(int(self.join.fit[moves.get(a,a),moves.get(b,b)])-int(self.join.fit[a,b])
                                    for e in {e for c in moves for e in self.join.incident[c]}
                                    for a,b in (self.join.edges[e],))

    def pair_catalog(self, group, pairs):
        """Compute standalone swap deltas relative to a common prefix only once."""
        fixed = cycle_mapping(group)
        score = self.one_score(group)
        join = self.join
        rows = np.array(pairs,dtype=int)
        a,b = rows.T
        fit = np.pad(join.fit.astype(np.int16),((0,1),(0,1)))
        changed = np.arange(join.bound.n+1)
        for c,d in fixed.items():
            changed[c] = d
        nb = join.bound.neighbors
        gain = np.zeros(len(rows),dtype=np.int16)
        for src,dst in ((a,b),(b,a)):
            for side in range(3):
                old_nb = changed[nb[src,side]]
                gain += fit[dst,old_nb] - fit[src,old_nb]
        # The double update on an internal seam of a swap must be corrected once.
        # Swapping the two endpoints preserves the unordered compatibility on that seam.
        internal = (nb[a]==b[:,None]).sum(axis=1)
        gain += (2*fit[a,b]-fit[a,a]-fit[b,b])*internal
        order = np.argsort(-gain,kind='stable')
        values = dict(zip(pairs,map(int,gain)))
        return dict(score=score,rows=rows[order],gains=gain[order],values=values,
                    available=tuple(sorted(set(rows.ravel().tolist()))),fit=fit)

    def partners(self, catalog, first, work):
        """Stop the sorted final catalogue as soon as even its best cross-edge correction fails."""
        fit = catalog['fit']
        join = self.join
        a,b = first
        incident = defaultdict(list)
        allowed = set(catalog['available'])-set(first)
        for c,d in ((a,b),(b,a)):
            for n in join.bound.neighbors[c]:
                if n in allowed:
                    incident[int(n)].append((c,d))
        values = []
        for n,changed in incident.items():
            correction = sum(fit[d,catalog['available']] - fit[c,catalog['available']] - fit[d,n]+fit[c,n]
                             for c,d in changed)
            values.append(max(0,int(correction.max())))
        cap = sum(sorted(values,reverse=True)[:2])
        extra = catalog['score'] + catalog['values'][first]
        from bisect import bisect_left
        end = len(catalog['gains'])-bisect_left(catalog['gains'][::-1],join.target-extra-cap)
        work['sorted_partner_rows_skipped'] += len(catalog['gains'])-end
        for row,gain in zip(catalog['rows'][:end],catalog['gains'][:end]):
            c,d = map(int,row)
            if c in first or d in first or first >= (c,d):
                continue
            correction=0
            for new,donor in ((c,d),(d,c)):
                for old,moved in incident.get(new,()):
                    correction+=int(fit[moved,donor]-fit[old,donor]-fit[moved,new]+fit[old,new])
            work['joint_physical_prices']+=1
            if extra+int(gain)+correction>=join.target:
                yield c,d

    def pair_rows(self, group, remaining, work):
        """Yield only threshold-eligible disjoint cycle pairs. Corrections use shared physical seams."""
        join = self.join
        baseline = cycle_mapping(group)
        cells = sorted(join.pool - baseline.keys())
        catalogs = {}
        for width in set(remaining):
            options = list(cycle_options(set(cells),width,set(cells)))
            catalogs[width] = sorted(((self.delta(baseline,cycle),cycle) for cycle in options),
                                      key=lambda x:(-x[0],x[1]))
            work['individual_cycle_prices'] += len(options)
        score = self.one_score(group)
        fit = join.fit
        for gain, first in catalogs[remaining[0]]:
            moved = cycle_mapping((first,))
            neighbors = [(c,n) for c in first for n in join.bound.neighbors[c]
                         if n in join.pool and n not in moved and n not in baseline]
            # A cross seam's mixed difference is bounded by its maximum over the incoming
            # physical donor, relative to the common parent. Each can be counted once.
            incident = defaultdict(list)
            for c,n in neighbors:
                incident[int(n)].append(c)
            if incident:
                optimistic = []
                for n,touch in incident.items():
                    terms = sum(fit[moved[c],cells].astype(int) - fit[c,cells].astype(int) -
                                int(fit[moved[c],n]) + int(fit[c,n]) for c in touch)
                    optimistic.append(max(0,int(terms.max())))
                bonus = sum(sorted(optimistic,reverse=True)[:remaining[1]])
            else:
                bonus = 0
            for at,(other_gain,second) in enumerate(catalogs[remaining[1]]):
                if score+gain+other_gain+bonus < join.target:
                    work['suffix_rows_skipped'] += len(catalogs[remaining[1]])-at
                    break
                if remaining[0] == remaining[1] and second <= first:
                    continue
                if set(first).intersection(second):
                    continue
                new = cycle_mapping((second,))
                correction = 0
                for n in set(second).intersection(incident):
                    for c in incident[n]:
                        correction += int(fit[moved[c],new[n]]) - int(fit[c,new[n]]) - int(fit[moved[c],n]) + int(fit[c,n])
                work['joint_physical_prices'] += 1
                if score+gain+other_gain+correction >= join.target:
                    yield tuple(sorted((*group,first,second)))


def donor_bridge_upper(stars, fixed, possible):
    """Bound all single-cycle extensions over a pool through their actual cross-star interfaces.

    A bridge is required only to select this diagnostic. For every possible attached
    cycle the component star sums are reusable and the exact mixed correction is local.
    """
    fixed = tuple(sorted(fixed))
    def subset_score(group):
        m = cycle_mapping(group)
        c = tuple(sorted(m))
        return stars.price(c,tuple(c.index(m[t]) for t in c))
    base = subset_score(fixed)
    used = set(cycle_mapping(fixed))
    b = stars.bound
    front = set().union(*(set(stars.affected[c]) for c in used))
    samples = Counter()
    cache = {}
    upper = -1
    def local(m, cells):
        sig = tuple(sorted(m.items())),tuple(cells)
        if sig not in cache:
            pieces = b.pieces.copy()
            for c,d in m.items():
                pieces[c] = b.tiles[d]
            a=np.asarray(cells,dtype=int)
            cache[sig]=int(b.stars(pieces[a],pieces[b.neighbors[a]]).sum())
        return cache[sig]
    left=cycle_mapping(fixed)
    for cyc in possible:
        if used.intersection(cyc):
            continue
        group=tuple(sorted((*fixed,cyc)))
        if not connected(group,stars.close):
            continue
        right=cycle_mapping((cyc,))
        shared=sorted(front & set().union(*(set(stars.affected[c]) for c in cyc)))
        if not shared:
            continue
        # The two components interact only in these shared star factors.
        mix=local(left|right,shared)-local(left,shared)-local(right,shared)+int(b.base[shared].sum())
        upper=max(upper,base+subset_score((cyc,))-b.total+mix)
        samples['partners']+=1
    return upper,dict(samples)


def bridge_upper(stars, group):
    """Split a contact-graph bridge and recompute only the actual shared star factors."""
    close = stars.close
    nodes = tuple(range(len(group)))
    edges = [(i,j) for i,j in combinations(nodes,2) if any(close[a,b] for a in group[i] for b in group[j])]
    def scored(g):
        m = cycle_mapping(g)
        cells = tuple(sorted(m))
        return stars.price(cells, tuple(cells.index(m[c]) for c in cells))
    for removed in edges:
        side = {removed[0]}
        for _ in nodes:
            for i,j in edges:
                if (i,j)!=removed and (i in side or j in side):
                    side.update((i,j))
        if len(side)==len(nodes):
            continue
        left,right = [tuple(g for i,g in enumerate(group) if (i in side)==pick) for pick in (True,False)]
        # All star factors outside this intersection depend on at most one half.
        ml,mr = cycle_mapping(left),cycle_mapping(right)
        affected_l = set().union(*(set(stars.affected[c]) for c in ml))
        affected_r = set().union(*(set(stars.affected[c]) for c in mr))
        both = np.array(sorted(affected_l & affected_r),dtype=int)
        b = stars.bound
        def contribution(mapping):
            pieces = b.pieces.copy()
            for c,d in mapping.items():
                pieces[c] = b.tiles[d]
            return int(b.stars(pieces[both],pieces[b.neighbors[both]]).sum())
        mixed = contribution(ml|mr)-contribution(ml)-contribution(mr)+int(b.base[both].sum())
        return scored(left) + scored(right) - b.total + mixed
    return None


def traverse(joins, *, pair=True, buckets=True, bridge=False, together=True, cover=False):
    """Existence/optimum within a finite pool. Centers keep separate exact endpoint sets."""
    first = joins[0]
    if any(j.pool!=first.pool or j.family!=first.family or not np.array_equal(j.close,first.close) for j in joins):
        raise ValueError("Connected traversal needs identical geometry, family and pool.")
    catalog = Catalog(first,first.family)
    work = Counter()
    pricing = [PhysicalPrices(j) for j in joins]
    def visit(group, active):
        work['geometry_prefixes'] += 1
        pending = list(first.family)
        used = set(cycle_mapping(group))
        for g in group:
            pending.remove(len(g))
        if not pending:
            return
        if cover and len(pending)==2:
            from r32_route_pool import defect_cover
            viable = []
            for i in active:
                test = defect_cover(joins[i].bound,group,first.pool,sum(pending),joins[i].target)
                work['overlap_calls'] += 1
                if test['closed']:
                    work['overlap_closed'] += 1
                else:
                    viable.append(i)
            active = viable
            if not active:
                return
        if pair and len(pending)==2:
            candidates = defaultdict(list)
            for ix in active:
                for child in pricing[ix].pair_rows(group,pending,work):
                    if connected(child,first.close) and parent(parent(child,first.close),first.close)==group:
                        candidates[child].append(ix)
            if not candidates:
                work['joint_parents_closed'] += 1
        else:
            candidates = {}
            for width in set(pending):
                if buckets:
                    choices = catalog.children(group,width)
                else:
                    boundary = set().union(*(first.near[c] for c in used))
                    choices = (g for g in cycle_options(first.pool-used,width,boundary)
                               if parent(tuple(sorted((*group,g))),first.close)==group)
                for cycle in choices:
                    work['owned_attachments'] += 1
                    child = tuple(sorted((*group,cycle)))
                    candidates[child] = active
        for child, owners in candidates.items():
            if len(child)!=len(first.family):
                yield from visit(child,owners)
                continue
            mapping = cycle_mapping(child)
            cells = tuple(sorted(mapping))
            order = tuple(cells.index(mapping[c]) for c in cells)
            for ix in owners:
                j = joins[ix]
                upper = bridge_upper(j.stars,child) if bridge else None
                if upper is None:
                    upper = j.stars.price(cells,order)
                    work['full_stars'] += 1
                else:
                    work['bridge_bounds'] += 1
                if upper >= 2*j.target:
                    yield ix,cells,order
    def rows():
        scopes = [tuple(range(len(joins)))] if together else [(i,) for i in range(len(joins))]
        for active in scopes:
            for width in sorted(set(first.family)):
                for g in cycle_options(first.pool,width,first.pool):
                    yield from visit((g,), active)
        work.update(catalog.work)
    return rows(),work
