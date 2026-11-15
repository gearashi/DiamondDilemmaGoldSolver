"""Retain rank-stable profitable swaps across successive legal boards and active sets."""
from collections import Counter, OrderedDict

import numpy as np


class SwapCatalog:
    def __init__(self, model, rebuild_fraction=1.0):
        self.model = model
        self.rebuild_fraction = rebuild_fraction
        self.board = None
        self.allowed = np.zeros(model.n, dtype=bool)
        self.buckets = [np.empty(0, dtype=int) for _ in range(7)]
        self.rotations = np.empty((model.n*model.n, 2), dtype=np.int8)
        self.work = Counter()
        self.ranked = None
        self.packs = OrderedDict()
        self.pack_key_bytes = 0

    def noncontacting(self, ranked, separator_depth, stats, solver):
        """An ordered endpoint/weight list and the geometry determine this pack.

        The current codes of the replacements can differ. They are applied by
        the caller on the current board, followed by fresh signed seam prices.
        """
        left,right,_,_,weights = ranked
        if not len(weights):
            return ()
        if len(weights) == 1:
            self.work['singleton_packs'] += 1
            return (0,)
        # Store the actual ordered integer columns for exact membership.
        packed = np.asarray((left,right,weights),dtype='<i4').tobytes()
        key = (solver,separator_depth,packed)
        answer = self.packs.get(key)
        if answer is not None:
            self.packs.move_to_end(key)
            self.work['nonempty_packs_reused'] += 1
            chosen,metrics = answer
        else:
            pairs = list(zip(left.tolist(),right.tolist()))
            metrics = Counter()
            chosen = tuple(solver(self.model,pairs,weights.tolist(),separator_depth=separator_depth,stats=metrics))
            self.work['nonempty_packs_solved'] += 1
            if len(packed) <= 262144:
                self.packs[key] = (chosen,metrics)
                self.pack_key_bytes += len(packed)
                while len(self.packs)>32 or self.pack_key_bytes>262144:
                    oldest,_ = self.packs.popitem(last=False)
                    self.pack_key_bytes -= len(oldest[2])
                self.work['peak_cached_graph_key_bytes'] = max(
                    self.work['peak_cached_graph_key_bytes'],self.pack_key_bytes)
        # Public work counts describe logical search decisions, so cold and
        # resumed trajectories agree. Catalog counters count physical calls.
        if stats is not None:
            for key,value in metrics.items():
                stats[key] = stats.get(key,0)+value
        return chosen

    def _affected(self, board, active):
        if self.board is None:
            return active.copy()
        changed = self.board != board
        dirty = changed | (active & ~self.allowed)
        # A neighbor cares only about the mask of the facing side. A donor's
        # physical identity and its current orientation matter at its own cell.
        for side in range(3):
            neighbors = self.model.problem.neighbors[:, side]
            changed_mask = (self.model.masks[self.board, side] != self.model.masks[board, side])
            dirty[neighbors[changed_mask & (neighbors >= 0)]] = True
        return dirty & active

    def rank(self, prices, board, cells):
        """Return position pairs, physical replacement codes and gains in released tie order.

        Call prices.update(board) first. Its state can also have been used by a
        different repair since the preceding call to this catalog.
        """
        model = self.model
        cells = np.sort(np.asarray(cells, dtype=int))
        active = np.zeros(model.n, dtype=bool)
        active[cells] = True
        dirty = self._affected(board, active)
        affected = np.flatnonzero(dirty)
        size = len(cells)
        possible = size*(size-1)//2
        clean = size-len(affected)
        required = possible-clean*(clean-1)//2
        rebuild = self.board is None or (possible > 0 and required >= self.rebuild_fraction*possible)
        self.work['calls'] += 1
        self.work['available_pairs'] += possible
        self.work['affected_unordered_pairs'] += required
        if self.ranked is not None and not len(affected) and np.array_equal(active,self.allowed):
            self.work['unchanged_rankings'] += 1
            self.board = board.copy()
            return self.ranked
        if rebuild:
            self.work['full_rebuilds'] += 1
            i,j = np.triu_indices(size, 1)
            left,right = cells[i],cells[j]
        else:
            self.work['incremental_updates'] += 1
            clean_cells = cells[~dirty[cells]]
            within_a,within_b = np.triu_indices(len(affected),1)
            cross_a = np.repeat(affected,len(clean_cells))
            cross_b = np.tile(clean_cells,len(affected))
            # Dirty/dirty combinations and dirty/clean products are disjoint.
            left = np.concatenate((affected[within_a],np.minimum(cross_a,cross_b)))
            right = np.concatenate((affected[within_b],np.maximum(cross_a,cross_b)))
        self.work['pairs_reconsidered'] += len(left)
        nonadjacent = ~np.any(model.problem.neighbors[left] == right[:,None], axis=1)
        left,right = left[nonadjacent],right[nonadjacent]
        self.work['nonadjacent_gains_evaluated'] += len(left)
        at_left = prices.prices[left,board[right]//3]
        at_right = prices.prices[right,board[left]//3]
        max_left = at_left.max(axis=1)
        max_right = at_right.max(axis=1)
        tile,rotation = board//3,board%3
        old = prices.prices[np.arange(model.n),tile,rotation]
        gain = max_left+max_right-old[left]-old[right]
        positive = (max_left>=0) & (max_right>=0) & (gain>0)
        new_keys = left[positive]*model.n+right[positive]
        gains = gain[positive]
        self.rotations[new_keys,0] = at_left[positive].argmax(axis=1)
        self.rotations[new_keys,1] = at_right[positive].argmax(axis=1)
        invalid = dirty | ~active
        for weight in range(1,7):
            changed_keys = new_keys[gains==weight]
            if rebuild:
                self.buckets[weight] = np.sort(changed_keys)
            else:
                old_keys = self.buckets[weight]
                keep = ~(invalid[old_keys//model.n] | invalid[old_keys%model.n])
                retained = old_keys[keep]
                self.work['positive_rows_retained'] += len(retained)
                if not len(changed_keys):
                    self.buckets[weight] = retained
                    continue
                added = np.sort(changed_keys)
                slots = np.searchsorted(retained,added)+np.arange(len(added))
                combined = np.empty(len(retained)+len(added),dtype=int)
                combined[slots] = added
                original = np.ones(len(combined),dtype=bool)
                original[slots] = False
                combined[original] = retained
                self.buckets[weight] = combined
        self.board = board.copy()
        self.allowed = active
        ordered = self.buckets[:0:-1]
        keys = np.concatenate(ordered)
        weights = np.repeat(np.arange(6,0,-1), [len(bucket) for bucket in ordered])
        a,b = keys//model.n, keys%model.n
        self.ranked = (a,b,3*tile[b]+self.rotations[keys,0],3*tile[a]+self.rotations[keys,1],weights)
        for array in self.ranked:
            array.flags.writeable = False
        return self.ranked
