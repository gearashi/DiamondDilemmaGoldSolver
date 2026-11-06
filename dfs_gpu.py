"""Constructive CUDA depth-first search with persistent stacks and exact seam pruning.

Each active replica owns a disjoint fixed prefix and visits candidate lists in
random cyclic order without repeating leaves. Surplus lanes remain idle. GPU launches are bounded by a per-replica work budget.
Pending complete boards freeze until their CPU validation is acknowledged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from gpu_engine import reverse11, score_boards_cpu
ROOT = Path(__file__).resolve().parent


def fill_order(neighbors, start=0):
    """Favor cells with the most already assigned neighbors, then closing frontier gaps."""
    neighbors = np.asarray(neighbors)
    chosen, remaining = [int(start)], set(range(160)) - {int(start)}
    occupied = {int(start)}
    while remaining:
        assigned_counts = [sum(int(nb) in occupied for nb in neighbors[c]) for c in range(160)]
        def priority(cell):
            future_pressure = sum(assigned_counts[int(nb)] for nb in neighbors[cell] if int(nb) not in occupied)
            return assigned_counts[cell], future_pressure, -cell
        nxt = max(remaining, key=priority)
        chosen.append(nxt)
        occupied.add(nxt)
        remaining.remove(nxt)
    return np.array(chosen, np.int16)


def normalize_prefixes(prefixes):
    """Return (jobs,160) int16 rows in fill-order sequence, padded with trailing -1."""
    if isinstance(prefixes, np.ndarray) and prefixes.ndim == 2:
        raw = prefixes
        if raw.shape[1] > 160 or raw.dtype.kind not in 'iu':
            raise ValueError('Prefixes must be integer rows of at most160 codes.')
        if np.any(raw < -1) or np.any(raw > 479):
            raise ValueError('Prefix codes must be 0..479, with trailing -1 padding only.')
        lengths = np.sum(raw >= 0, axis=1).astype(np.int16)
        if np.any((raw >= 0) != (np.arange(raw.shape[1])[None, :] < lengths[:, None])):
            raise ValueError('Prefix padding must be trailing -1 values.')
        rows = np.full((len(raw), 160), -1, np.int16)
        rows[:, :raw.shape[1]] = raw
        return rows, lengths
    rows_list = list(prefixes)
    rows = np.full((len(rows_list), 160), -1, np.int16)
    lengths = np.zeros(len(rows_list), np.int16)
    for i, row in enumerate(rows_list):
        values = np.asarray(row)
        if values.ndim != 1 or len(values) > 160 or (values.size and values.dtype.kind not in 'iu'):
            raise ValueError('Each prefix must be an integer code sequence of length0..160.')
        if np.any(values < -1) or np.any(values > 479):
            raise ValueError('Prefix codes must be 0..479, with trailing -1 padding only.')
        length = int(np.sum(values >= 0))
        if np.any((values >= 0) != (np.arange(len(values)) < length)):
            raise ValueError('Prefix padding must be trailing -1 values.')
        rows[i, :length] = values[:length]
        lengths[i] = length
    return rows, lengths


def ensure_disjoint_prefixes(prefixes):
    """Reject duplicate prefixes and ancestor/descendant overlaps in lexicographic order."""
    ordered = sorted(prefixes)
    for left, right in zip(ordered, ordered[1:]):
        if len(left) <= len(right) and right[:len(left)] == left:
            raise ValueError('DFS jobs overlap: duplicate or ancestor/descendant prefixes.')


class DFS:
    """Disjoint fixed-prefix DFS jobs: states0active,1pending,2exhausted,3idle.

    The caller's SSD ledger owns job IDs and historical completed jobs. This class
    prevents overlaps among currently active/pending jobs and new assignments.
    Loading an exhausted lane is allowed; the ledger must never reassign past work.
    """
    CHECKPOINT_VERSION = 3
    DEVICE_FIELDS = ('boards', 'used', 'depths', 'maxdepths', 'states', 'floors',
                     'firsts', 'lengths', 'cursors', 'shifts', 'rng', 'counters')

    def __init__(self, masks, neighbors, neighbor_sides, n=3840, seed=20261007,
                 root_codes=None, order=None, prefixes=None, gold_data=None, gpu_features=None):
        initial_rows = self._prepare_host(masks, neighbors, neighbor_sides, n, root_codes, order, prefixes)
        self._prepare_gold(gold_data)
        from gpu_reasoning import prepare_policy
        prepare_policy(self, gpu_features, gold_data)
        os.environ.setdefault('CUPY_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'DiamondDilemmaGoldSolver-cupy-cache'))
        import cupy as cp
        self.cp = cp
        name = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name']
        self.device = name.decode() if isinstance(name, bytes) else str(name)
        host_rng = np.random.default_rng(seed)
        self._prepare_tables()
        self.tables = [cp.asarray(v) for v in (self.host_faces, self.host_reverse, self.neighbors, self.neighbor_sides)]
        self.pool = cp.asarray(self.host_pool)
        self.single_offsets = cp.asarray(self.host_single_offsets)
        self.pair_offsets = cp.asarray(self.host_pair_offsets)
        self.device_order = cp.asarray(self.order)
        self.gold_partners = cp.asarray(self.host_gold_partners)
        self.gold_ranks = cp.asarray(self.order_rank.astype(np.int16))
        self.gold_checked = cp.zeros(n, np.uint8)
        self.device_policy = cp.asarray(self.host_policy)
        self.module = cp.RawModule(code=(ROOT / 'dfs_kernels.cu').read_text(encoding='utf-8-sig'),
                                  options=('--std=c++17', '-DREASONING_FLAGS='+str(self.feature_bits)), name_expressions=('dfs_search',))
        self.kernel = self.module.get_function('dfs_search')
        self.boards = cp.full((160, n), -1, np.int16)
        self.used = cp.zeros((5, n), np.uint32)
        self.depths = cp.zeros(n, np.int16)
        self.maxdepths = cp.zeros(n, np.int16)
        self.states = cp.full(n, 3, np.uint8)
        self.floors = cp.zeros(n, np.int16)
        self.firsts = cp.zeros((160, n), np.uint16)
        self.lengths = cp.zeros((160, n), np.uint16)
        self.cursors = cp.full((160, n), 65535, np.uint16)
        self.shifts = cp.zeros((160, n), np.uint16)
        self.rng = cp.asarray(host_rng.integers(1, 2**32, n, dtype=np.uint32))
        self.counters = cp.zeros((3, n), np.uint64)
        self.prefix_codes = np.full((n, 160), -1, np.int16)
        self.prefixes = [None] * n
        self.roots = np.full(n, -1, np.int16)
        if len(initial_rows):
            self.load_prefixes(initial_rows, lanes=np.arange(len(initial_rows), dtype=np.int32))

    def _prepare_host(self, masks, neighbors, neighbor_sides, n, root_codes, order, prefixes):
        """Shared integer input validation; no device or search work occurs here."""
        if type(n) is not int or not 1 <= n <= 131072:
            raise ValueError('Replica count must be in1..131072.')
        self.n = n
        raw_masks, raw_neighbors, raw_sides = map(np.asarray, (masks, neighbors, neighbor_sides))
        if raw_masks.shape != (480, 3) or raw_masks.dtype.kind not in 'iu' or np.any(raw_masks < 0) or np.any(raw_masks > 2047):
            raise ValueError('Expected480x3 integer masks of11bits.')
        if raw_neighbors.shape != (160, 3) or raw_sides.shape != (160, 3):
            raise ValueError('Expected160x3 neighbor arrays.')
        if raw_neighbors.dtype.kind not in 'iu' or raw_sides.dtype.kind not in 'iu':
            raise ValueError('Neighbor arrays must contain integers.')
        if np.any(raw_neighbors < 0) or np.any(raw_neighbors >= 160) or np.any(raw_sides < 0) or np.any(raw_sides >= 3):
            raise ValueError('Invalid neighbor index.')
        self.masks = np.ascontiguousarray(raw_masks, dtype=np.uint16)
        self.neighbors = np.ascontiguousarray(raw_neighbors, dtype=np.int16)
        self.neighbor_sides = np.ascontiguousarray(raw_sides, dtype=np.uint8)
        for a in range(160):
            if len(set(map(int, self.neighbors[a]))) != 3:
                raise ValueError('Board must have three distinct neighbors per cell.')
            for side in range(3):
                b, t = int(self.neighbors[a, side]), int(self.neighbor_sides[a, side])
                if a == b or self.neighbors[b, t] != a or self.neighbor_sides[b, t] != side:
                    raise ValueError('Neighbor map must be reciprocal.')
        raw_order = fill_order(self.neighbors) if order is None else np.asarray(order)
        if raw_order.shape != (160,) or raw_order.dtype.kind not in 'iu' or not np.array_equal(np.sort(raw_order), np.arange(160)):
            raise ValueError('Fill order must contain all160 cells exactly once.')
        self.order = np.asarray(raw_order, np.int16)
        self.order_rank = np.argsort(self.order)
        if root_codes is not None and prefixes is not None:
            raise ValueError('Supply either root_codes or prefixes, not both.')
        if root_codes is not None:
            roots = np.asarray(root_codes)
            if roots.shape != (n,) or roots.dtype.kind not in 'iu' or np.any(roots < 0) or np.any(roots > 479):
                raise ValueError('Root orientation codes must have shape(n,) and lie in0..479.')
            initial_prefixes = roots[:, None]
        elif prefixes is None:
            # At most480 unique root jobs; surplus lanes are idle, never repeated roots.
            initial_prefixes = np.arange(min(n, 480), dtype=np.int16)[:, None]
        else:
            initial_prefixes = prefixes
        initial_rows, initial_lengths = normalize_prefixes(initial_prefixes)
        if len(initial_rows) > n:
            raise ValueError('More prefix jobs than GPU lanes.')
        ensure_disjoint_prefixes([tuple(map(int, row[:length])) for row, length in zip(initial_rows, initial_lengths)])
        self._verify_prefix_rows(initial_rows, initial_lengths)
        self.fingerprint = hashlib.sha256(self.masks.tobytes() + self.neighbors.tobytes()
                                         + self.neighbor_sides.tobytes() + self.order.tobytes()).hexdigest()
        return initial_rows

    def _prepare_gold(self, data):
        """Explicit segment pairs distinguish Gold from edge-only DFS/checkpoints."""
        self.host_gold_partners = np.full((480, 33), 255, np.uint8)
        self.gold_segments = 0
        self.gold_fingerprint = ''
        if data is None:
            return
        from validator import normalize_tiles, orientation_masks
        tiles = normalize_tiles(data)
        if len(tiles) != 160 or not np.array_equal(np.asarray(orientation_masks(tiles), np.uint16), self.masks):
            raise ValueError('Gold segments must describe exactly the supplied 160 tile masks.')
        for tile, item in enumerate(tiles):
            self.gold_segments += len(item['segments'])
            for rotation in range(3):
                for left, right in item['segments']:
                    a = ((left[0] + rotation) % 3) * 11 + left[1] - 1
                    b = ((right[0] + rotation) % 3) * 11 + right[1] - 1
                    self.host_gold_partners[3 * tile + rotation, a] = b
                    self.host_gold_partners[3 * tile + rotation, b] = a
        self.gold_fingerprint = hashlib.sha256(self.host_gold_partners.tobytes()).hexdigest()

    def _prepare_tables(self):
        """Shared exact candidate buckets, independent of the device API."""
        palette = np.unique(np.concatenate((self.masks.ravel(), reverse11(self.masks).ravel())))
        self.host_faces = np.searchsorted(palette, self.masks).astype(np.uint16)
        self.host_reverse = np.searchsorted(palette, reverse11(self.masks)).astype(np.uint16)
        self.colors = len(palette)
        code_arrays = [np.arange(480, dtype=np.int16)]
        single_keys = np.concatenate([side * self.colors + self.host_faces[:, side].astype(np.int32) for side in range(3)])
        single_order = np.argsort(single_keys, kind='stable')
        codes = np.tile(np.arange(480, dtype=np.int16), 3)
        code_arrays.append(codes[single_order])
        self.host_single_offsets = np.zeros(3 * self.colors + 1, np.int32)
        self.host_single_offsets[1:] = np.cumsum(np.bincount(single_keys, minlength=len(self.host_single_offsets) - 1), dtype=np.int32)
        self.host_single_offsets += 480
        pair_keys = np.concatenate([(pair * self.colors + self.host_faces[:, a].astype(np.int32)) * self.colors + self.host_faces[:, b]
            for pair, (a, b) in enumerate(((0, 1), (0, 2), (1, 2)))])
        pair_order = np.argsort(pair_keys, kind='stable')
        code_arrays.append(codes[pair_order])
        self.host_pair_offsets = np.zeros(3 * self.colors * self.colors + 1, np.int32)
        self.host_pair_offsets[1:] = np.cumsum(np.bincount(pair_keys, minlength=len(self.host_pair_offsets) - 1), dtype=np.int32)
        self.host_pair_offsets += 1920
        self.host_pool = np.concatenate(code_arrays)

    def _verify_prefix_rows(self, rows, lengths):
        if not len(rows):
            return
        reverse = reverse11(self.masks)
        for start in range(0, len(rows), 2048):
            batch = rows[start:start + 2048]
            count = lengths[start:start + 2048]
            assigned = np.arange(160)[None, :] < count[:, None]
            if np.any((batch >= 0) != assigned) or np.any(batch < -1) or np.any(batch > 479):
                raise ValueError('Prefix codes and lengths disagree.')
            pieces = np.sort(np.where(assigned, batch // 3, 160), axis=1)
            if np.any((pieces[:, :-1] == pieces[:, 1:]) & (pieces[:, :-1] < 160)):
                raise ValueError('A prefix repeats a tile.')
            board = batch[:, self.order_rank]
            for a in range(160):
                for sa in range(3):
                    b, sb = int(self.neighbors[a, sa]), int(self.neighbor_sides[a, sa])
                    if a >= b:
                        continue
                    valid = (board[:, a] >= 0) & (board[:, b] >= 0)
                    if np.any(self.masks[board[valid, a], sa] != reverse[board[valid, b], sb]):
                        raise ValueError('A fixed prefix contains a mismatching seam.')

    def available_lanes(self):
        flags = self.states.get()
        return np.flatnonzero((flags == 2) | (flags == 3))

    def load_prefixes(self, prefixes, lanes=None):
        """Assign disjoint jobs to idle/exhausted lanes; cumulative counters and RNG persist."""
        rows, lengths = normalize_prefixes(prefixes)
        flags = self.states.get()
        if lanes is None:
            ids = np.flatnonzero((flags == 2) | (flags == 3))[:len(rows)]
        else:
            raw = np.asarray(lanes)
            if raw.ndim != 1 or (raw.size and raw.dtype.kind not in 'iu'):
                raise ValueError('Lane IDs must be a one-dimensional integer sequence.')
            ids = raw.astype(np.int32)
        if len(ids) != len(rows) or np.any(ids < 0) or np.any(ids >= self.n) or len(set(ids.tolist())) != len(ids):
            raise ValueError('Insufficient available lanes or invalid lane IDs.')
        if np.any((flags[ids] != 2) & (flags[ids] != 3)):
            raise ValueError('Only idle or exhausted lanes can receive new jobs.')
        if not len(ids):
            return ids
        incoming = [tuple(map(int, row[:length])) for row, length in zip(rows, lengths)]
        # Completed-job history is owned by the SSD ledger, not this bounded lane pool.
        active = [self.prefixes[int(i)] for i in np.flatnonzero((flags == 0) | (flags == 1))]
        ensure_disjoint_prefixes(active + incoming)
        self._verify_prefix_rows(rows, lengths)
        boards = rows[:, self.order_rank]
        if getattr(self, 'gpu_features', {}).get('shared_gold'):
            for board in boards[:64]:
                literals = [(int(cell), int(code)) for cell, code in enumerate(board) if code >= 0]
                if literals: self.add_gold_exclusion(literals)
        used = np.zeros((len(ids), 5), np.uint32)
        valid = rows >= 0
        pieces = np.where(valid, rows // 3, 0)
        bits = np.left_shift(np.uint32(1), (pieces % 32).astype(np.uint32))
        for word in range(5):
            used[:, word] = np.bitwise_or.reduce(np.where(valid & (pieces // 32 == word), bits, np.uint32(0)), axis=1)
        cp = self.cp
        self.gold_checked[ids] = 0
        self.boards[:, ids] = cp.asarray(boards.T.copy())
        self.used[:, ids] = cp.asarray(used.T.copy())
        self.depths[ids] = cp.asarray(lengths)
        self.floors[ids] = cp.asarray(lengths)
        self.maxdepths[ids] = cp.maximum(self.maxdepths[ids], cp.asarray(lengths))
        self.states[ids] = cp.asarray((lengths == 160).astype(np.uint8))
        self.firsts[:, ids] = 0
        self.lengths[:, ids] = 0
        self.cursors[:, ids] = 65535
        self.shifts[:, ids] = 0
        self.prefix_codes[ids] = rows
        self.roots[ids] = rows[:, 0]
        for lane, prefix in zip(ids, incoming):
            self.prefixes[int(lane)] = prefix
        return ids

    def step(self, nodes=128):
        if type(nodes) is not int or not 1 <= nodes <= 512:
            raise ValueError('Use a bounded work budget of1..512 per launch.')
        cp = self.cp
        held = self._pause_cpu_owned()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        try:
            self.kernel(((self.n + 127) // 128,), (128,), (self.boards, self.used, self.depths,
                self.maxdepths, self.floors, self.states, self.firsts, self.lengths, self.cursors, self.shifts,
                self.rng, self.counters, self.device_order, *self.tables, self.single_offsets,
                self.pair_offsets, self.pool, self.gold_partners, self.gold_ranks, self.gold_checked, self.device_policy, np.int32(self.feature_bits),
                np.int32(self.gold_segments), np.int32(self.colors), np.int32(self.n), np.int32(nodes)))
            end.record()
            end.synchronize()
            return float(cp.cuda.get_elapsed_time(start, end))
        finally:
            self._restore_cpu_owned(held)

    def pending(self):
        return np.flatnonzero(self.states.get() == 1)

    def has_pending(self):
        return bool(self.cp.any(self.states == 1).get())

    def acknowledge(self, indices):
        """Continue strictly inside a validated candidate's assigned prefix subtree."""
        raw = np.asarray(indices)
        if raw.ndim != 1 or (raw.size and raw.dtype.kind not in 'iu'):
            raise ValueError('Acknowledgement indices must be integers.')
        ids = raw.astype(np.int32)
        if np.any(ids < 0) or np.any(ids >= self.n) or len(set(ids.tolist())) != len(ids):
            raise ValueError('Invalid acknowledgement indices.')
        if not len(ids):
            return 0
        if np.any(self.states[ids].get() != 1) or np.any(self.depths[ids].get() != 160):
            raise ValueError('Only pending complete boards can be acknowledged.')
        floors = self.floors[ids].get()
        terminal = ids[floors == 160]
        if len(terminal):
            self.boards[:, terminal] = -1
            self.used[:, terminal] = 0
            self.depths[terminal] = 0
            self.states[terminal] = 2
        resume = ids[floors < 160]
        if len(resume):
            if getattr(self, 'dynamic_order', False):
                from gpu_reasoning import frame_cells
                leaf = frame_cells(self.firsts[-1, resume].get(), self.shifts[-1, resume].get()).astype(int)
            else:
                leaf = int(self.order[-1])
            if getattr(self, 'dynamic_order', False):
                code = np.array([int(self.boards[int(cell), int(lane)].get()) for cell, lane in zip(leaf, resume)], np.int16)
            else:
                code = self.boards[leaf, resume].get()
            used = self.used[:, resume].get()
            tiles = code // 3
            for word in range(5):
                selected = tiles // 32 == word
                used[word, selected] &= np.bitwise_not(np.left_shift(np.uint32(1), (tiles[selected] % 32).astype(np.uint32)))
            self.used[:, resume] = self.cp.asarray(used)
            if getattr(self, 'dynamic_order', False):
                for cell, lane in zip(leaf, resume): self.boards[int(cell), int(lane)] = -1
            else:
                self.boards[leaf, resume] = -1
            self.depths[resume] = 159
            self.states[resume] = 0
        return len(ids)

    def _pause_cpu_owned(self):
        service = getattr(self, 'cpu_assist', None)
        if service is None:return []
        held = service.before_step(self)
        previous = [(int(lane), int(self.states[int(lane)].get())) for lane in held]
        for lane, _ in previous:self.states[lane] = 3
        return previous

    def _restore_cpu_owned(self, held):
        for lane, state in held:self.states[lane] = state

    def drain_cpu_candidates(self):
        service = getattr(self, 'cpu_assist', None)
        if service is None:return []
        candidates = service.candidates[:];service.candidates.clear()
        return candidates

    def cpu_assist_stats(self):
        service = getattr(self, 'cpu_assist', None)
        return {} if service is None else dict(service.stats)

    def reject_current_subtree(self, lane, assignment):
        """Backtrack only a scope that a completed CPU proof rejected."""
        if int(self.states[lane].get()) != 0 or list(map(int,self.boards[:,lane].get())) != list(assignment):
            raise ValueError('CPU rejection no longer names the active GPU subtree.')
        depth, floor = int(self.depths[lane].get()), int(self.floors[lane].get())
        if depth == floor:
            self.boards[:,lane] = -1; self.used[:,lane] = 0
            self.depths[lane] = 0; self.states[lane] = 2
        else:
            if getattr(self, 'dynamic_order', False):
                from gpu_reasoning import frame_cells
                cell=int(frame_cells(self.firsts[depth-1,lane].get(), self.shifts[depth-1,lane].get()))
            else:cell=int(self.order[depth-1])
            tile=int(self.boards[cell,lane].get())//3
            self.boards[cell,lane] = -1
            self.used[tile//32,lane] = np.uint32(int(self.used[tile//32,lane].get()) & ~(1 << (tile%32)))
            self.depths[lane] = depth-1
        self.gold_checked[lane] = 0

    def install_cpu_candidate(self, lane, assignment, codes):
        """Publish a checked CPU endgame through the ordinary pending-board path."""
        service=getattr(self,'cpu_assist',None)
        if service is None or not service.problem.validate(codes)['valid']:
            raise ValueError('CPU endgame is not a Gold certificate.')
        if int(self.states[lane].get())!=0 or list(map(int,self.boards[:,lane].get()))!=list(assignment):
            raise ValueError('CPU solution does not own this GPU subtree.')
        if any(code>=0 and codes[cell]!=code for cell,code in enumerate(assignment)):
            raise ValueError('CPU solution changes its owned GPU prefix.')
        depth=int(self.depths[lane].get())
        if int(self.cursors[depth,lane].get())!=65535:
            raise ValueError('CPU endgame must own an untouched current GPU frame.')
        from gpu_reasoning import pack_frame
        partial=list(assignment)
        arrays={name:getattr(self,name)[:,lane].get() for name in ('firsts','lengths','cursors','shifts')}
        remaining=[int(cell) for cell in self.order if partial[int(cell)]<0]
        for frame,cell in enumerate(remaining,depth):
            known=[side for side in range(3) if partial[int(self.neighbors[cell,side])]>=0]
            needs=[int(self.host_reverse[partial[int(self.neighbors[cell,side])],int(self.neighbor_sides[cell,side])]) for side in known[:2]]
            if not known:first,length=0,480
            elif len(known)==1:
                key=known[0]*self.colors+needs[0];first=int(self.host_single_offsets[key]);length=int(self.host_single_offsets[key+1])-first
            else:
                pair={(0,1):0,(0,2):1,(1,2):2}[tuple(known[:2])]
                key=(pair*self.colors+needs[0])*self.colors+needs[1]
                first=int(self.host_pair_offsets[key]);length=int(self.host_pair_offsets[key+1])-first
            selected=np.flatnonzero(self.host_pool[first:first+length]==codes[cell])
            if len(selected)!=1:raise ValueError('CPU candidate is absent from its exact GPU bucket.')
            shift=int(selected[0]);packed_first,packed_shift=(pack_frame(first,shift,cell) if self.dynamic_order else (first,shift))
            arrays['firsts'][frame]=packed_first;arrays['shifts'][frame]=packed_shift
            arrays['lengths'][frame]=length;arrays['cursors'][frame]=1
            partial[cell]=codes[cell]
        for name,values in arrays.items():getattr(self,name)[:,lane]=self.cp.asarray(values)
        self.boards[:,lane]=self.cp.asarray(np.asarray(codes,np.int16))
        self.used[:,lane]=self.cp.asarray(np.full(5,0xffffffff,np.uint32))
        self.depths[lane]=160;self.maxdepths[lane]=160;self.states[lane]=1;self.gold_checked[lane]=255

    def _install_cpu_restrictions(self, proof):
        service = getattr(self, 'cpu_assist', None)
        if service is None or proof.get('fingerprint') != self.fingerprint or proof.get('status') != 'propagated':
            raise ValueError('CPU restriction is not bound to this GPU engine.')
        # Only a completed result produced by this private service is accepted.
        key=tuple(proof['assignment'])
        if service.completed_probes.get(key) != proof:
            raise ValueError('CPU restriction was not independently established by this engine.')
        from gpu_reasoning import append_verified_cut
        fixed=tuple((cell,code) for cell,code in enumerate(key) if code>=0)
        used={code//3 for _,code in fixed}
        added=0
        for cell, mask in enumerate(proof['domains']):
            if key[cell]>=0:continue
            for code in range(480):
                if code//3 in used or mask & (1<<code):continue
                # Skip elementary fixed-neighbor failures: the shader already checks them.
                if any(key[int(nb)]>=0 and self.host_faces[code,side] != self.host_reverse[key[int(nb)],int(self.neighbor_sides[cell,side])]
                       for side,nb in enumerate(self.neighbors[cell])):continue
                if append_verified_cut(self, tuple(sorted((*fixed,(cell,code))))):added+=1
                if added>=16:return added
        return added

    def _upload_policy(self):
        self.device_policy.set(self.host_policy)

    def add_gold_exclusion(self, literals):
        from gpu_reasoning import install_gold_cut
        return install_gold_cut(self, literals)

    def selected_stack(self, arrays):
        """Decode assigned cells, then complete each partial order canonically."""
        count = len(arrays['depths'])
        chosen = np.tile(self.order, (count, 1))
        if not getattr(self, 'dynamic_order', False): return chosen
        from gpu_reasoning import frame_cells
        decoded = frame_cells(arrays['firsts'], arrays['shifts']).T
        for lane in range(count):
            depth, floor = int(arrays['depths'][lane]), int(arrays['floors'][lane])
            if arrays['states'][lane] not in (0, 1): continue
            end = depth + int(depth < 160 and arrays['cursors'][depth, lane] != 65535)
            chosen[lane, floor:end] = decoded[lane, floor:end]
            head = list(map(int, chosen[lane, :end]))
            if len(set(head)) != end or any(cell >= 160 for cell in head):
                raise ValueError('DFS dynamic stack repeats a selected cell.')
            remaining = [int(cell) for cell in self.order if cell not in set(head)]
            chosen[lane, end:] = remaining
        return chosen

    def _validate_host(self, arrays, prefix_rows, check_stack=True):
        n = arrays['boards'].shape[1]
        depths, floors, states = arrays['depths'], arrays['floors'], arrays['states']
        maxima = arrays['maxdepths']
        if np.any(depths < 0) or np.any(depths > 160) or np.any(floors < 0) or np.any(floors > 160) or np.any(states > 3):
            raise ValueError('Malformed DFS state, depth or prefix floor.')
        active, pending = states == 0, states == 1
        done, idle = states == 2, states == 3
        living = active | pending
        if (np.any(active & ((depths < floors) | (depths >= 160))) or np.any(pending & (depths != 160))
                or np.any((done | idle) & (depths != 0)) or np.any(idle & (floors != 0))):
            raise ValueError('DFS state/depth/floor inconsistency.')
        if np.any(maxima < depths) or np.any(maxima < floors) or np.any(maxima > 160):
            raise ValueError('Invalid DFS maximum depths.')
        self._verify_prefix_rows(prefix_rows, floors)
        if np.any(idle & np.any(prefix_rows != -1, axis=1)):
            raise ValueError('Idle lane contains an assigned prefix.')
        tuples = [tuple(map(int, row[:floor])) for row, floor in zip(prefix_rows[living], floors[living])]
        ensure_disjoint_prefixes(tuples)
        reverse = reverse11(self.masks)
        selected_order = self.selected_stack(arrays)
        for start in range(0, n, 2048):
            stop = min(n, start + 2048)
            board = arrays['boards'][:, start:stop].T
            ordered = np.take_along_axis(board, selected_order[start:stop], axis=1)
            depth = depths[start:stop]
            assigned = np.arange(160)[None, :] < depth[:, None]
            if np.any((ordered >= 0) != assigned) or np.any(ordered < -1) or np.any(ordered > 479):
                raise ValueError('DFS board does not match its assigned stack prefix.')
            pieces = np.sort(np.where(assigned, ordered // 3, 160), axis=1)
            if np.any((pieces[:, :-1] == pieces[:, 1:]) & (pieces[:, :-1] < 160)):
                raise ValueError('DFS board reuses a tile.')
            fixed = (np.arange(160)[None, :] < floors[start:stop, None]) & living[start:stop, None]
            if np.any(ordered[fixed] != prefix_rows[start:stop][fixed]):
                raise ValueError('DFS changed a fixed prefix.')
            actual_piece = np.where(assigned, ordered // 3, 0)
            bits = np.left_shift(np.uint32(1), (actual_piece % 32).astype(np.uint32))
            for word in range(5):
                expected = np.bitwise_or.reduce(np.where(assigned & (actual_piece // 32 == word), bits, np.uint32(0)), axis=1)
                if not np.array_equal(expected, arrays['used'][word, start:stop]):
                    raise ValueError('DFS used-tile bitset differs from its board.')
            for a in range(160):
                for sa in range(3):
                    b, sb = int(self.neighbors[a, sa]), int(self.neighbor_sides[a, sa])
                    if a >= b:
                        continue
                    valid = (board[:, a] >= 0) & (board[:, b] >= 0)
                    if np.any(self.masks[board[valid, a], sa] != reverse[board[valid, b], sb]):
                        raise ValueError('DFS assigned a mismatching seam.')
        if np.any(arrays['rng'] == 0):
            raise ValueError('DFS RNG state may not be zero.')
        if not check_stack:
            return n
        if getattr(self, 'dynamic_order', False):
            self._validate_dynamic_frames(arrays, selected_order, living, active)
            return n
        # Check all entered stack frames against the exact candidate bucket selected
        # by their already assigned neighbors; this catches skipped/repeated cursors.
        for depth, cell in enumerate(self.order):
            ancestor = living & (floors <= depth) & (depth < depths)
            current = active & (depths == depth) & (arrays['cursors'][depth] != 65535)
            lanes = np.flatnonzero(ancestor | current)
            if not len(lanes):
                continue
            known = [s for s in range(3) if self.order_rank[self.neighbors[cell, s]] < depth]
            if not known:
                first = np.zeros(len(lanes), np.int32)
                length = np.full(len(lanes), 480, np.int32)
            else:
                needs = []
                for side in known[:2]:
                    nb, ns = self.neighbors[cell, side], self.neighbor_sides[cell, side]
                    needs.append(self.host_reverse[arrays['boards'][nb, lanes], ns])
                if len(known) == 1:
                    key = known[0] * self.colors + needs[0].astype(np.int32)
                    first = self.host_single_offsets[key]
                    length = self.host_single_offsets[key + 1] - first
                else:
                    pair = {(0, 1): 0, (0, 2): 1, (1, 2): 2}[tuple(known[:2])]
                    key = (pair * self.colors + needs[0].astype(np.int32)) * self.colors + needs[1]
                    first = self.host_pair_offsets[key]
                    length = self.host_pair_offsets[key + 1] - first
            cursor = arrays['cursors'][depth, lanes].astype(np.int32)
            shift = arrays['shifts'][depth, lanes].astype(np.int32)
            if (not np.array_equal(arrays['firsts'][depth, lanes], first)
                    or not np.array_equal(arrays['lengths'][depth, lanes], length)
                    or np.any(cursor > length) or np.any((length == 0) & (shift != 0))
                    or np.any((length > 0) & (shift >= length))):
                raise ValueError('DFS checkpoint contains an inconsistent candidate-list cursor.')
            selected = ancestor[lanes]
            if np.any(cursor[selected] == 0) or np.any(length[selected] == 0):
                raise ValueError('Assigned DFS frame has no consumed candidate.')
            chosen = self.host_pool[first[selected] + (cursor[selected] - 1 + shift[selected]) % length[selected]]
            if not np.array_equal(chosen, arrays['boards'][cell, lanes[selected]]):
                raise ValueError('DFS stack cursor does not identify its assigned child.')
        return n

    def _validate_dynamic_frames(self, arrays, chosen, living, active):
        for lane in np.flatnonzero(living):
            depth, floor = int(arrays['depths'][lane]), int(arrays['floors'][lane])
            rank = np.empty(160, int); rank[chosen[lane]] = np.arange(160)
            end = depth + int(active[lane] and arrays['cursors'][depth, lane] != 65535)
            for frame in range(floor, end):
                cell = int(chosen[lane, frame])
                known = [side for side in range(3) if rank[int(self.neighbors[cell, side])] < frame]
                needs = [int(self.host_reverse[arrays['boards'][int(self.neighbors[cell, side]), lane],
                                               int(self.neighbor_sides[cell, side])]) for side in known[:2]]
                if not known:
                    first, length = 0, 480
                elif len(known) == 1:
                    key = known[0]*self.colors + needs[0]
                    first = int(self.host_single_offsets[key]); length = int(self.host_single_offsets[key+1])-first
                else:
                    pair = {(0,1):0,(0,2):1,(1,2):2}[tuple(known[:2])]
                    key = (pair*self.colors+needs[0])*self.colors+needs[1]
                    first = int(self.host_pair_offsets[key]); length = int(self.host_pair_offsets[key+1])-first
                stored_first = int(arrays['firsts'][frame,lane]) & 4095
                shift = int(arrays['shifts'][frame,lane]) & 4095
                cursor = int(arrays['cursors'][frame,lane])
                if stored_first != first or (int(arrays['lengths'][frame,lane]) & 32767) != length or cursor > length or (not length and shift) or (length and shift >= length):
                    raise ValueError('DFS dynamic checkpoint has an inconsistent candidate-list cursor.')
                if int(arrays['lengths'][frame,lane]) & 32768:
                    if not getattr(self, 'gpu_features', {}).get('forced') or cursor > 1:
                        raise ValueError('DFS forced frame consumed more than its unique candidate.')
                    used = {int(arrays['boards'][int(other),lane])//3 for other in chosen[lane,:frame]}
                    supported = []
                    for code in map(int, self.host_pool[first:first+length]):
                        if code//3 in used: continue
                        if getattr(self, 'gpu_features', {}).get('symmetry') and code//3 == 0 and not self.host_policy[cell]: continue
                        if all(self.host_faces[code,side] == self.host_reverse[
                                arrays['boards'][int(self.neighbors[cell,side]),lane],int(self.neighbor_sides[cell,side])]
                               for side in known): supported.append(code)
                    if len(supported) != 1 or int(self.host_pool[first+shift]) != supported[0]:
                        raise ValueError('DFS forced checkpoint frame is not a proved singleton at its saved shift.')
                if frame < depth:
                    if not cursor or not length or int(self.host_pool[first+(cursor-1+shift)%length]) != int(arrays['boards'][cell,lane]):
                        raise ValueError('DFS dynamic cursor does not identify its assigned child.')

    def verify(self, indices=None):
        ids = np.arange(self.n) if indices is None else np.asarray(indices)
        if ids.ndim != 1 or ids.dtype.kind not in 'iu' or not len(ids) or np.any(ids < 0) or np.any(ids >= self.n):
            raise ValueError('Verification indices must name existing lanes.')
        ids = ids.astype(np.int32)
        arrays = {key: (getattr(self, key)[:, ids] if getattr(self, key).ndim == 2 else getattr(self, key)[ids]).get()
                  for key in self.DEVICE_FIELDS}
        return self._validate_host(arrays, self.prefix_codes[ids])

    def snapshot_checkpoint(self):
        """Capture independent CPU arrays; the caller may add SSD-ledger metadata fields."""
        self.cp.cuda.get_current_stream().synchronize()
        payload = {key: getattr(self, key).get() for key in self.DEVICE_FIELDS}
        payload.update(dfs_version=np.int32(self.CHECKPOINT_VERSION), gold_fingerprint=np.array(self.gold_fingerprint), fingerprint=np.array(self.fingerprint),
                       order=self.order.copy(), roots=self.roots.copy(), prefix_codes=self.prefix_codes.copy())
        if getattr(self, 'cpu_assist', None) is not None:
            payload['cpu_assist_state'] = np.array(json.dumps(self.cpu_assist.snapshot(), separators=(',', ':')))
        for value in payload.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return payload

    @staticmethod
    def write_checkpoint(path, payload, compressed=False):
        from gpu_engine import GPU
        return GPU.write_checkpoint(path, payload, compressed=compressed)

    def checkpoint(self, path, compressed=False):
        started = time.perf_counter()
        payload = self.snapshot_checkpoint()
        capture = time.perf_counter() - started
        result = self.write_checkpoint(path, payload, compressed=compressed)
        result.update(capture_seconds=capture, total_seconds=time.perf_counter() - started)
        return result

    def read_checkpoint(self, archive):
        """Validate every saved lane on the CPU, independently of allocated GPU count."""
        if str(archive['fingerprint']) != self.fingerprint or not np.array_equal(archive['order'], self.order):
            raise ValueError('DFS checkpoint puzzle, geometry or fill order differs.')
        saved_gold = str(archive['gold_fingerprint']) if 'gold_fingerprint' in archive else ''
        if saved_gold and saved_gold != self.gold_fingerprint:
            raise ValueError('Gold checkpoint target or segment connections differ; edge-only search cannot reuse Gold-pruned coverage.')
        version = int(archive['dfs_version']) if 'dfs_version' in archive else 1
        if version not in (1, 2, self.CHECKPOINT_VERSION):
            raise ValueError('Unsupported DFS checkpoint version.')
        if version == 3 and 'gold_fingerprint' not in archive:
            raise ValueError('DFS v3 checkpoint is missing its Gold target binding.')
        states = archive['states']
        if states.ndim != 1 or not 0 <= len(states) <= 131072:
            raise ValueError('Malformed DFS checkpoint lane count.')
        count = len(states)
        arrays = {}
        for field in self.DEVICE_FIELDS:
            if field == 'floors' and version == 1:
                arrays[field] = np.ones(count, np.int16)
                continue
            value = archive[field]
            reference = getattr(self, field)
            shape = (reference.shape[0], count) if reference.ndim == 2 else (count,)
            if value.shape != shape or value.dtype != reference.dtype:
                raise ValueError('Malformed DFS checkpoint field: ' + field)
            arrays[field] = value
        if version == 1:
            roots = archive['roots']
            if roots.shape != (count,) or roots.dtype.kind not in 'iu' or np.any(roots < 0) or np.any(roots > 479):
                raise ValueError('Malformed legacy root jobs.')
            prefixes = np.full((count, 160), -1, np.int16)
            prefixes[:, 0] = roots
            ensure_disjoint_prefixes([tuple([int(code)]) for code in roots])
        else:
            prefixes = archive['prefix_codes']
            if prefixes.shape != (count, 160) or prefixes.dtype != np.int16:
                raise ValueError('Malformed fixed-prefix checkpoint metadata.')
            if 'roots' in archive and not np.array_equal(archive['roots'], prefixes[:, 0]):
                raise ValueError('Checkpoint roots differ from fixed prefixes.')
        self._validate_host(arrays, prefixes)
        if getattr(self, 'cpu_assist', None) is not None and 'cpu_assist_state' not in archive:
            raise ValueError('CPU-assisted checkpoint is missing its owned endgame state.')
        if 'cpu_assist_state' in archive:
            if getattr(self, 'cpu_assist', None) is None:raise ValueError('CPU-owned GPU work requires CPU assistance to resume.')
            arrays['cpu_assist_state'] = archive['cpu_assist_state'].copy()
        arrays.update(prefix_codes=prefixes, roots=prefixes[:, 0].copy(),
                      dfs_version=np.int32(self.CHECKPOINT_VERSION),
                      fingerprint=np.array(self.fingerprint), gold_fingerprint=np.array(self.gold_fingerprint), order=self.order.copy())
        return arrays

    def _install_lanes(self, payload, ids):
        cp = self.cp
        self.gold_checked[ids] = 0
        for field in self.DEVICE_FIELDS:
            target = getattr(self, field)
            if target.ndim == 2:
                target[:, ids] = cp.asarray(payload[field])
            else:
                target[ids] = cp.asarray(payload[field])
        self.prefix_codes[ids] = payload['prefix_codes']
        self.roots[ids] = payload['roots']
        for lane, state, row, floor in zip(ids, payload['states'], payload['prefix_codes'], payload['floors']):
            self.prefixes[int(lane)] = None if state == 3 else tuple(map(int, row[:floor]))

    def restore_checkpoint(self, payload, *, _cpu_owner_payloads=None):
        """Install a validated exact-size checkpoint; no random restart is performed."""
        checked = self.read_checkpoint(payload)
        if len(checked['states']) != self.n:
            raise ValueError('Checkpoint lane count differs; systematic resizing requires JobLedger.resume.')
        if 'cpu_assist_state' in checked:
            from gpu_cpu_assist import CPUAssist
            decoded=json.loads(str(checked['cpu_assist_state']))
            CPUAssist.validate_ownership(decoded, [checked] if _cpu_owner_payloads is None else _cpu_owner_payloads)
            if decoded.get('fingerprint')!=self.fingerprint:raise ValueError('CPU assistance checkpoint binding differs.')
        self._install_lanes(checked, np.arange(self.n, dtype=np.int32))
        if 'cpu_assist_state' in checked:
            self.cpu_assist.restore(decoded)
        return True

    def restore_lanes(self, payload, lanes):
        """Restore paused exact stacks into idle/exhausted lanes without resetting cursors."""
        raw = np.asarray(lanes)
        if raw.ndim != 1 or (raw.size and raw.dtype.kind not in 'iu'):
            raise ValueError('Lane IDs must be a one-dimensional integer sequence.')
        ids = raw.astype(np.int32)
        if np.any(ids < 0) or np.any(ids >= self.n) or len(np.unique(ids)) != len(ids):
            raise ValueError('Invalid restore lane IDs.')
        checked = self.read_checkpoint(payload)
        if len(checked['states']) != len(ids) or np.any(~np.isin(checked['states'], [0, 1])):
            raise ValueError('Paused states must be active/pending and match destination lane count.')
        flags = self.states.get()
        if np.any(~np.isin(flags[ids], [2, 3])):
            raise ValueError('Only idle/exhausted lanes may receive paused work.')
        incoming = [tuple(map(int, row[:floor])) for row, floor in zip(checked['prefix_codes'], checked['floors'])]
        active = [self.prefixes[int(i)] for i in np.flatnonzero(np.isin(flags, [0, 1]))]
        ensure_disjoint_prefixes(active + incoming)
        self._install_lanes(checked, ids)
        return ids

    def resume(self, path):
        with np.load(path, allow_pickle=False) as archive:
            if any(key.startswith('exact_paused_') or key in ('exact_bank_version', 'exact_retired_counters') for key in archive):
                raise ValueError('This checkpoint has systematic lane-bank state; use JobLedger.resume to preserve it.')
            return self.restore_checkpoint(archive)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--replicas', type=int, default=3840)
    parser.add_argument('--nodes', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--output', type=Path, default=ROOT / 'dfs-runtime')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    from geometry import build_board
    from validator import orientation_masks, validate_arrangement
    from render_board import render
    data = json.loads((ROOT / 'data' / 'tiles.json').read_text(encoding='utf-8-sig'))
    geometry = build_board()
    dfs = DFS(orientation_masks(data), geometry.neighbor_cells, geometry.neighbor_sides, n=args.replicas, seed=args.seed, gold_data=data)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / 'checkpoint.npz'
    if args.resume and checkpoint.exists():
        dfs.resume(checkpoint)
    started, last_check, last_save = time.monotonic(), -10.0, 0.0
    checked, solved = 0, False
    status = {}
    try:
        while time.monotonic() - started < args.seconds and not solved:
            dfs.step(args.nodes)
            elapsed = time.monotonic() - started
            if elapsed - last_check > 2:
                flags = dfs.states.get()
                ids = np.flatnonzero(flags == 1)
                for idx in ids:
                    board = dfs.boards[:, int(idx)].get().astype(int).tolist()
                    report = validate_arrangement(data, board)
                    if report['matched_edges'] != 240:
                        raise RuntimeError('Complete DFS candidate failed independent seam validation.')
                    checked += 1
                    record = {'codes': board, 'validation': report, 'elapsed_seconds': elapsed,
                              'seed': args.seed, 'method': 'CUDA constructive DFS', 'replica': int(idx)}
                    (output / f'candidate-{checked:06d}.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
                    if report['valid']:
                        (output / 'solution.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
                        render(output / 'solution.html', data['tiles'], board, report)
                        solved = True
                        break
                if not solved and len(ids):
                    dfs.acknowledge(ids)
                dfs.verify(np.arange(min(16, dfs.n)))
                status = {'state': 'solved' if solved else 'running', 'elapsed_seconds': elapsed,
                          'gpu': dfs.device, 'replicas': dfs.n, 'max_depth': int(dfs.maxdepths.max().get()),
                          'depth_median': float(np.median(dfs.depths.get())), 'pending': int(np.sum(flags == 1)),
                          'exhausted_root_jobs': int(np.sum(flags == 2)), 'idle_lanes': int(np.sum(flags == 3)), 'candidates_checked': checked,
                          'counters': dfs.counters.get().sum(axis=1).astype(int).tolist()}
                (output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
                print(json.dumps(status), flush=True)
                last_check = elapsed
                if np.all((flags == 2) | (flags == 3)):
                    break
            if elapsed - last_save >= 30:
                dfs.checkpoint(checkpoint)
                last_save = elapsed
            if (output / 'stop.request').exists():
                break
    except KeyboardInterrupt:
        pass
    finally:
        dfs.verify(np.arange(min(64, dfs.n)))
        dfs.checkpoint(checkpoint)
        status['state'] = 'solved' if solved else 'stopped'
        status['elapsed_seconds'] = time.monotonic() - started
        (output / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
