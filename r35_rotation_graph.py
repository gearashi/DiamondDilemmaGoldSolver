"""Exact threshold propagation and structural elimination for three rotations."""
from collections import deque
import heapq
import math


def support_rows(table, cache):
    # Only 512 binary 3-by-3 tables exist, including transposes.
    key = table.astype(bool, copy=False).tobytes()
    if key not in cache:
        rows = [sum(1 << t for t in range(3) if key[3*r+t]) for r in range(3)]
        cache[key] = tuple(sum(1 << r for r,row in enumerate(rows) if row & mask) for mask in range(8))
    return cache[key]


class RotationThreshold:
    def __init__(self, domains, edges, cache, work):
        self.work = work
        self.tables = [[] for _ in domains]
        for a, b, raw in edges:
            self.tables[a].append((b, support_rows(raw, cache)))
            self.tables[b].append((a, support_rows(raw.T, cache)))
        self.base = tuple(sum(1 << r for r in row) for row in domains)
        self.near = [tuple(sorted({c, *(v for v, _ in row)})) for c, row in enumerate(self.tables)]
        self.affected = [tuple(sorted({v for u in row for v in self.near[u]})) for row in self.near]
        self.base_keys = [tuple(self.base[i] for i in row) for row in self.affected]
        self.base_losses = [None]*len(domains)
        self.recent_losses = [None]*len(domains)
        self.incoming = [{} for _ in domains]
        self.prices = []
        for cell,edges in enumerate(self.tables):
            row = [0,0,0]
            for other,support in edges:
                self.incoming[other].setdefault(cell,[]).append(support)
                allowed = support[self.base[other]]
                for r in range(3):
                    row[r] += (allowed >> r) & 1
            self.work['threshold_supports'] += len(edges)
            self.prices.append(tuple(row))
        maxima = [self.maxima(row,mask) for row,mask in zip(self.prices,self.base)]
        self.stars = [score for score,_ in maxima]
        self.best = [rotations for _,rotations in maxima]
        self.total = sum(self.stars)

    def maxima(self, row, mask):
        self.work['threshold_stars'] += 1
        best = max(row[r] for r in range(3) if mask & (1 << r))
        return best, sum(1 << r for r in range(3) if mask & (1 << r) and row[r] == best)

    def losses(self, c, masks, prices, stars, best):
        key = tuple(masks[i] for i in self.affected[c])
        if key == self.base_keys[c]:
            prior = self.base_losses[c]
        else:
            last = self.recent_losses[c]
            prior = last[1] if last is not None and last[0] == key else None
        if prior is not None:
            self.work['threshold_condition_hits'] += 1
            return prior
        mask = masks[c]
        rotations = [r for r in range(3) if mask & (1 << r)]
        losses = [0,0,0]
        for r in rotations:
            losses[r] = stars[c] - prices[c][r]
        for other,tables in self.incoming[c].items():
            previous = [support[mask] for support in tables]
            self.work['threshold_supports'] += len(tables)*(1+len(rotations))
            if len(tables) == 1 and other != c:
                # A single seam decreases its neighbor's maximum by exactly
                # one iff every formerly maximizing rotation loses support.
                self.work['threshold_argmax_checks'] += len(rotations)
                for r in rotations:
                    lost = previous[0] & ~tables[0][1 << r]
                    losses[r] += int(best[other] & ~lost == 0)
            else:
                for r in rotations:
                    lost = [old & ~support[1 << r] for old,support in zip(previous,tables)]
                    if other == c:
                        losses[r] += sum((bits >> r) & 1 for bits in lost)
                    else:
                        row = [prices[other][t]-sum((bits >> t) & 1 for bits in lost) for t in range(3)]
                        alternative,_ = self.maxima(row,masks[other])
                        losses[r] += stars[other] - alternative
        if key == self.base_keys[c]:
            self.base_losses[c] = losses
        else:
            self.recent_losses[c] = key, losses
        return losses

    def reduce(self, floor):
        # Every threshold starts at the original domains, including a lower
        # threshold following a failed stricter one.
        if self.total < 2*floor:
            return None
        max_degree = max(map(len,self.tables),default=0)
        if floor <= 0 or self.total-2*floor >= 2*max_degree:
            return [tuple(r for r in range(3) if mask & (1 << r)) for mask in self.base]
        # Each doubled bound is integral even when a caller supplies a
        # fractional target.
        slack = self.total - math.ceil(2*floor)
        masks = list(self.base)
        stars = self.stars.copy()
        best = self.best.copy()
        prices = self.prices.copy()
        queue = deque(i for i, m in enumerate(masks) if m.bit_count() > 1)
        queued = set(queue)
        # With unchanged two-hop neighborhoods each conditioned loss is fixed.
        # A drop in global slack can still invalidate a distant rotation.
        watchers = [set() for _ in range(2*max_degree+1)]
        watched = [-1]*len(masks)
        def schedule(c):
            if watched[c] >= 0:
                watchers[watched[c]].discard(c)
                watched[c] = -1
            if c not in queued and masks[c].bit_count() > 1:
                queued.add(c)
                queue.append(c)
        while queue:
            c = queue.popleft()
            queued.remove(c)
            mask = masks[c]
            near = self.near[c]
            # For each incident seam at most one point can disappear at its
            # center and one at its neighbor. This also covers parallel seams.
            largest = 2*len(self.tables[c])
            keep = mask
            if largest > slack:
                losses = self.losses(c,masks,prices,stars,best)
                largest = max(losses)
                for r,loss in enumerate(losses):
                    if loss > slack:
                        keep &= ~(1 << r)
                masks[c] = keep
            if not keep:
                return None
            if keep == mask:
                watched[c] = largest
                watchers[largest].add(c)
                continue
            self.work['threshold_removed'] += mask.bit_count() - keep.bit_count()
            previous = slack
            before = sum(stars[i] for i in near)
            for other,tables in self.incoming[c].items():
                row = list(prices[other])
                for support in tables:
                    lost = support[mask] & ~support[keep]
                    for t in range(3):
                        row[t] -= (lost >> t) & 1
                self.work['threshold_supports'] += 2*len(tables)
                prices[other] = tuple(row)
            for i in near:
                stars[i],best[i] = self.maxima(prices[i],masks[i])
            slack += sum(stars[i] for i in near) - before
            if slack < 0:
                return None
            for i in self.affected[c]:
                schedule(i)
            for level in range(slack+1, min(previous, len(watchers)-1)+1):
                for i in sorted(watchers[level]):
                    self.work['threshold_remote_events'] += 1
                    schedule(i)
        return [tuple(r for r in range(3) if mask & (1 << r)) for mask in masks]


def elimination_plan(widths, pairs, work, cap, prefix=(), residual=None, *, retain=True):
    # Eliminating a variable joins its neighbors. Only their products change;
    # ties always use the lower physical cell. A capped plan can retain the
    # remaining neighborhoods and unchanged prices for a later higher budget.
    prices = [None]*len(widths)
    if residual is None:
        neighbors = [{c} for c in range(len(widths))]
        for a,b in pairs:
            neighbors[a].add(b)
            neighbors[b].add(a)
    else:
        neighbors = [set() for _ in widths]
        for c,row,price in residual:
            neighbors[c].update(row)
            prices[c] = price
    def cost(c):
        work['plan_neighbor_products'] += 1
        return math.prod(widths[i] for i in neighbors[c])
    queue = []
    for c,row in enumerate(neighbors):
        if row:
            if prices[c] is None:
                prices[c] = cost(c)
            queue.append((prices[c],c))
    heapq.heapify(queue)
    steps = list(prefix)
    while queue:
        product,cell = heapq.heappop(queue)
        if product != prices[cell] or not neighbors[cell]:
            continue
        scope = tuple(sorted(neighbors[cell]))
        steps.append((cell,scope,product))
        remaining = neighbors[cell] - {cell}
        neighbors[cell].clear()
        for other in remaining:
            neighbors[other].discard(cell)
            neighbors[other].update(remaining)
            if product <= cap:
                prices[other] = cost(other)
                heapq.heappush(queue,(prices[other],other))
            else:
                prices[other] = None
        if product > cap:
            saved = (tuple((c,tuple(sorted(row)),prices[c]) for c,row in enumerate(neighbors) if row)
                     if retain and len(steps) < len(widths) else None)
            return tuple(steps),saved
    return tuple(steps),None
