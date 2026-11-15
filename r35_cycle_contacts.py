"""Score connected two- and three-cycle moves with exact compressed rotation interfaces."""
from collections import Counter, defaultdict
from itertools import combinations
from math import prod

import numpy as np

from r34_swap_packing import pack_swaps


def contact_graph(rows):
    """An edge between disjoint moves lists their actual common board seams."""
    by_edge = defaultdict(list)
    masks = [sum(1 << int(c) for c in row[0]) for row in rows]
    neighbors = [dict() for _ in rows]
    for i, row in enumerate(rows):
        common = defaultdict(list)
        for edge in row[4]:
            for j in by_edge[edge]:
                if not masks[i] & masks[j]:
                    common[j].append(edge)
            by_edge[edge].append(i)
        for j, edges in common.items():
            neighbors[i][j] = neighbors[j][i] = tuple(sorted(edges))
    return neighbors


class Prices:
    def __init__(self, rows, before, neighbors, work):
        self.rows = rows
        self.before = before
        self.neighbors = neighbors
        self.work = work
        self.interfaces = {}

    def interface(self, i, edges):
        key = i, edges
        if key in self.interfaces:
            self.work['interface_hits'] += 1
            return self.interfaces[key]
        _, _, _, gain, boundary, _ = self.rows[i]
        # First tie at the best delta reproduces full Cartesian argmax in row order.
        representatives = {}
        for r in range(len(gain)):
            signature = tuple((int(boundary[e][0][r]), int(boundary[e][1][r])) for e in edges)
            old = representatives.get(signature)
            if old is None or gain[r] > gain[old]:
                representatives[signature] = r
        indices = np.array(sorted(representatives.values()), dtype=int)
        self.interfaces[key] = indices
        self.work['interface_builds'] += 1
        return indices

    def price(self, indices, *, compress=True):
        n = len(indices)
        cross = [(a, b, self.neighbors[i][j]) for a, i in enumerate(indices)
                 for b, j in enumerate(indices[:a]) if j in self.neighbors[i]]
        interfaces = [set() for _ in indices]
        for a, b, edges in cross:
            interfaces[a].update(edges)
            interfaces[b].update(edges)
        keep = [self.interface(i, tuple(sorted(interfaces[a]))) if compress else np.arange(len(self.rows[i][3]))
                for a, i in enumerate(indices)]
        widths = tuple(map(len, keep))
        def correction(a, b, edges):
            terms = np.zeros((widths[a], widths[b]), dtype=np.int16)
            for edge in edges:
                ma, olda = self.rows[indices[a]][4][edge]
                mb, oldb = self.rows[indices[b]][4][edge]
                left = ma[keep[a]][:, None]
                right = mb[keep[b]][None, :]
                terms += ((left == right).astype(np.int16) + self.before[edge]
                          - olda[keep[a]][:, None] - oldb[keep[b]][None, :])
            return terms

        unary = [self.rows[i][3][keep[a]] for a,i in enumerate(indices)]
        self.work[f'raw_{n}_rotation_products'] += prod(len(self.rows[i][3]) for i in indices)
        self.work[f'compressed_{n}_rotation_products'] += prod(widths)
        if n == 3 and len(cross) == 2 and compress:
            # The leaf rotations are independent at a fixed rotation of the
            # middle cycle. Choose the same earliest maximizing Cartesian tuple.
            center, = {cross[0][0], cross[0][1]} & {cross[1][0], cross[1][1]}
            score = unary[center].copy()
            fitted = {center: np.arange(widths[center])}
            for a,b,edges in cross:
                values = correction(a,b,edges)
                leaf = a if b == center else b
                if leaf == a:
                    values = values.T
                values += unary[leaf]
                fitted[leaf] = values.argmax(axis=1)
                score += values[np.arange(widths[center]),fitted[leaf]]
                self.work['conditional_leaf_entries'] += values.size
            value = int(score.max())
            best = min(tuple(int(fitted[a][r]) for a in range(n)) for r in np.flatnonzero(score == value))
            self.work['conditional_middle_rows'] += len(score)
        else:
            grid = np.zeros(widths, dtype=np.int16)
            for a,values in enumerate(unary):
                shape = [1]*n
                shape[a] = widths[a]
                grid += values.reshape(shape)
            for a,b,edges in cross:
                shape = [1]*n
                shape[a],shape[b] = widths[a],widths[b]
                values = correction(a,b,edges)
                # The lower-dimensional index b occurs before a in the grid.
                grid += values.T.reshape(shape)
            self.work[f'priced_{n}_rotation_products'] += grid.size
            best = np.unravel_index(grid.argmax(),widths)
            value = int(grid[best])
        codes = {}
        for a, i in enumerate(indices):
            _, cells, orientations, _, _, _ = self.rows[i]
            codes.update(zip(cells, map(int, orientations[keep[a][best[a]]])))
        return value, codes


def connected_triples(rows, graph):
    for middle, nbr in enumerate(graph):
        for a, b in combinations(nbr, 2):
            if rows[a][0] & rows[b][0]:
                continue
            if a in graph[b] and middle != min(a, b, middle):
                continue
            # A path has one center, a triangle is emitted by its least vertex.
            yield (middle, a, b)


def choose(model, options):
    items = sorted(options.values(), key=lambda item: -item[0])
    selected = pack_swaps(model, [tuple(codes) for _, codes in items], [gain for gain, _ in items])
    return sum(items[i][0] for i in selected), [items[i][1] for i in selected]


def corrected_moves(model, rows, before, singles, max_group, *, compress=True):
    work = Counter()
    graph = contact_graph(rows)
    work['all_row_pairs'] = len(rows)*(len(rows)-1)//2
    edges = [(i,j,neighbors[j]) for i,neighbors in enumerate(graph) for j in sorted(neighbors) if i>j]
    work['disjoint_touching_pairs'] = len(edges)
    prices = Prices(rows, before, graph, work)
    # Use the shipped candidates throughout. Adding moves can change a greedy
    # large-component decision; compare packed witnesses at each completed stage.
    answer = choose(model, singles)
    best_pairs = {}
    options = singles.copy()

    def save(gain, codes):
        key = frozenset(codes)
        if gain > options.get(key, (0, None))[0]:
            options[key] = gain, codes

    for positive in (False, True):
        if positive and max_group == 1:
            break
        replaced = False
        for i, j, common in edges:
            gi, gj = rows[i][5], rows[j][5]
            if (min(gi, gj) > 0) != positive:
                continue
            cutoff = max(0, gi, gj)
            if gi+gj+sum(1+int(before[e]) for e in common) <= cutoff:
                continue
            value, codes = prices.price((i, j), compress=compress)
            work['positive_pairs_priced' if positive else 'mixed_pairs_priced'] += 1
            if value > cutoff:
                work['positive_pair_improvements' if positive else 'mixed_pair_improvements'] += 1
                best_pairs[i,j] = value
                size = frozenset(codes)
                if value > options.get(size, (0,None))[0]:
                    save(value, codes)
                    replaced = True
        if replaced:
            packed = choose(model, options)
            if packed[0] > answer[0] or (not positive and packed[0] == answer[0]):
                answer = packed
        work['gain_pairs' if positive else 'gain_released'] = answer[0]
    if max_group == 3:
        any_added = False
        for indices in connected_triples(rows, graph):
            gains = [rows[i][5] for i in indices]
            threshold = max(0, *gains)
            upper = sum(gains)
            for i,j in combinations(indices, 2):
                common = graph[i].get(j, ())
                upper += sum(1+int(before[e]) for e in common)
                if common:
                    threshold = max(threshold, best_pairs.get((max(i,j),min(i,j)),0))
                else:
                    threshold = max(threshold, rows[i][5] + rows[j][5])
            footprint = frozenset().union(*(rows[i][0] for i in indices))
            threshold = max(threshold, options.get(footprint,(0,None))[0])
            work['connected_triples'] += 1
            if upper <= threshold:
                continue
            value, codes = prices.price(indices, compress=compress)
            work['joint_triples_priced'] += 1
            if value > threshold:
                save(value,codes)
                work['triple_improvements'] += 1
                any_added = True
        if any_added:
            packed = choose(model, options)
            if packed[0] > answer[0]:
                answer = packed
        work['gain_triples'] = answer[0]
    return answer[0], answer[1], dict(work)
