"""Standalone CUDA replica search for Diamond Dilemma Gold.

CUDA performs every search move. NumPy only initializes, validates, and saves states.
A score of 240 establishes seam compatibility, not the required single closed loop;
validator.validate_arrangement is the authoritative complete-puzzle check.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
CELLS = 160
VERSION = 3


def reverse11(values):
    """Reverse the 11 boundary-position bits; bit zero denotes position one."""
    x = np.asarray(values, dtype=np.uint16)
    result = np.zeros_like(x)
    for bit in range(11):
        result |= ((x >> bit) & 1) << (10 - bit)
    return result


def legal_boards(boards):
    """Validate integer orientation codes and exact use of all 160 tiles."""
    b = np.asarray(boards)
    if b.ndim == 1:
        b = b[None, :]
    if b.ndim != 2 or b.shape[1] != CELLS or b.dtype.kind not in 'iu':
        raise ValueError('Boards must be an integer array with shape (k,160).')
    if len(b) < 1 or np.any(b < 0) or np.any(b >= CELLS * 3):
        raise ValueError('Orientation codes must be between 0 and 479.')
    if not np.all(np.sort(b // 3, axis=1) == np.arange(CELLS)):
        raise ValueError('Every board must use each tile exactly once.')
    return np.ascontiguousarray(b, dtype=np.int16)


def score_boards_cpu(masks, neighbors, neighbor_sides, boards):
    """Independent full rescoring using raw masks, without CUDA palette/index tables."""
    b = legal_boards(boards)
    masks = np.asarray(masks, dtype=np.uint16)
    # Reverse the tiny raw-mask table once, not every replica vector at every seam.
    reversed_masks = reverse11(masks)
    neighbors = np.asarray(neighbors)
    neighbor_sides = np.asarray(neighbor_sides)
    scores = np.zeros(len(b), dtype=np.int32)
    for cell in range(CELLS):
        for side in range(3):
            other = int(neighbors[cell, side])
            if cell < other:
                target = reversed_masks[b[:, other], int(neighbor_sides[cell, side])]
                scores += masks[b[:, cell], side] == target
    return scores


class GPU:
    def __init__(self, masks, neighbors, neighbor_sides, n=4096, seed=20261007, cache_slots=64):
        if isinstance(n, bool) or int(n) != n or not 1 <= int(n) <= 131072:
            raise ValueError('Replica count must be an integer from 1 to 131072.')
        self.n = int(n)
        if isinstance(cache_slots, bool) or int(cache_slots) != cache_slots or not 0 <= int(cache_slots) <= 256:
            raise ValueError('cache_slots must be an integer in 0..256; zero disables duplicate rejection.')
        self.cache_slots = int(cache_slots)
        if self.cache_slots * CELLS * self.n > 2147483647:
            raise ValueError('Requested cache shape exceeds the CUDA indexing limit; reduce cache_slots or replicas.')
        raw = np.asarray(masks)
        if raw.shape != (480, 3) or raw.dtype.kind not in 'iu' or np.any(raw < 0) or np.any(raw > 2047):
            raise ValueError('masks must contain 480x3 integer masks of 11 bits.')
        self.masks = np.ascontiguousarray(raw, dtype=np.uint16)
        ng = np.asarray(neighbors)
        ns = np.asarray(neighbor_sides)
        if ng.shape != (160, 3) or ng.dtype.kind not in 'iu' or np.any(ng < 0) or np.any(ng >= 160):
            raise ValueError('neighbors must be a 160x3 integer array of cell indices.')
        if ns.shape != (160, 3) or ns.dtype.kind not in 'iu' or np.any(ns < 0) or np.any(ns > 2):
            raise ValueError('neighbor_sides must be a 160x3 array with values 0,1,2.')
        self.neighbors = np.ascontiguousarray(ng, dtype=np.int16)
        self.neighbor_sides = np.ascontiguousarray(ns, dtype=np.uint8)
        for a in range(160):
            if len(set(map(int, ng[a]))) != 3 or a in ng[a]:
                raise ValueError('Board must be a simple cubic graph without self edges.')
            for s in range(3):
                b, t = int(ng[a, s]), int(ns[a, s])
                if ng[b, t] != a or ns[b, t] != s:
                    raise ValueError('Board neighbor relationships must be reciprocal.')
        self.fingerprint = hashlib.sha256(self.masks.tobytes() + self.neighbors.tobytes() + self.neighbor_sides.tobytes()).hexdigest()
        self.host_rng = np.random.default_rng(seed)
        os.environ.setdefault('CUPY_CACHE_DIR', str(Path(tempfile.gettempdir()) / 'DiamondDilemmaGoldSolver-cupy-cache'))
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError('No CUDA device was detected.')
            self.cp = cp
            props = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
            name = props['name']
            self.device = name.decode() if isinstance(name, bytes) else str(name)
        except (ImportError, RuntimeError) as exc:
            raise RuntimeError('The GPU search requires CuPy and a working NVIDIA CUDA driver. No CPU search fallback is used.') from exc
        palette = np.unique(np.concatenate((self.masks.ravel(), reverse11(self.masks).ravel())))
        faces = np.searchsorted(palette, self.masks).astype(np.uint16)
        reversed_faces = np.searchsorted(palette, reverse11(self.masks)).astype(np.uint16)
        self.colors = len(palette)
        table_bytes = (faces.nbytes + reversed_faces.nbytes + self.neighbors.nbytes
                       + self.neighbor_sides.nbytes + (3 * self.colors * self.colors + 1) * 4
                       + 1440 * 2 + CELLS * 480 * 8)
        self.estimated_device_bytes = self.n * (1049 + 330 * self.cache_slots) + table_bytes
        free_bytes, _ = cp.cuda.runtime.memGetInfo()
        reserve_bytes = 512 * 1024 * 1024
        if self.estimated_device_bytes + reserve_bytes > free_bytes:
            raise RuntimeError(f'Requested GPU state needs about {self.estimated_device_bytes / 2**30:.2f} GiB '
                               f'plus a 512 MiB reserve, but only {free_bytes / 2**30:.2f} GiB is free. '
                               'Reduce replicas or cache_slots.')
        self.tables = [cp.asarray(v) for v in (faces, reversed_faces, self.neighbors, self.neighbor_sides)]
        key_parts = []
        code_parts = []
        for pair, (s, t) in enumerate(((0, 1), (0, 2), (1, 2))):
            key_parts.append((pair * self.colors + faces[:, s].astype(np.int32)) * self.colors + faces[:, t])
            code_parts.append(np.arange(480, dtype=np.int16))
        keys = np.concatenate(key_parts)
        codes = np.concatenate(code_parts)
        order = np.argsort(keys, kind='stable')
        offsets = np.zeros(3 * self.colors * self.colors + 1, dtype=np.int32)
        offsets[1:] = np.cumsum(np.bincount(keys, minlength=len(offsets) - 1), dtype=np.int32)
        self.pair_offsets = cp.asarray(offsets)
        self.pair_codes = cp.asarray(codes[order])
        self.module = cp.RawModule(code=(ROOT / 'kernels.cu').read_text(encoding='utf-8-sig'), options=('--std=c++17',), name_expressions=('search_moves', 'test_delta', 'score_boards', 'test_cache_probe'))
        self.search_kernel = self.module.get_function('search_moves')
        self.delta_kernel = self.module.get_function('test_delta')
        self.score_kernel = self.module.get_function('score_boards')
        self.cache_probe_kernel = self.module.get_function('test_cache_probe')
        self.rng = cp.asarray(self.host_rng.integers(1, 2**32, self.n, dtype=np.uint32))
        self.counters = cp.zeros((3, self.n), dtype=np.uint64)
        self.pending = cp.zeros(self.n, dtype=np.uint8)
        self.temps = cp.ones(self.n, dtype=np.float32)
        self.arms = np.arange(self.n) % 8
        self.guidance = cp.asarray(np.array([0, .25, .5, .75, .9, 1, .9, .6], np.float32)[self.arms])
        self.guide_prob = self.guidance
        self.initialized = False
        # Fixed independent Zobrist table keeps the search RNG sequence unchanged by cache allocation.
        self._zobrist_host = np.random.default_rng(0xD1A601D).integers(0, 2**64, (CELLS, 480), dtype=np.uint64)
        self.zobrist = cp.asarray(self._zobrist_host)
        self.current_hashes = cp.zeros(self.n, np.uint64)
        self.cache_boards = cp.full((self.cache_slots, CELLS, self.n), -1, np.int16)
        self.cache_hashes = cp.zeros((self.cache_slots, self.n), np.uint64)
        self.cache_scores = cp.full((self.cache_slots, self.n), -1, np.int16)
        self.cache_counts = cp.zeros(self.n, np.uint16)
        self.cache_cursors = cp.zeros(self.n, np.uint16)
        self.cache_counters = cp.zeros((4, self.n), np.uint64)
        self.cache_resume_note = 'new_cache'

    def _cache_cpu_hash(self, boards, zobrist=None):
        table = self._zobrist_host if zobrist is None else zobrist
        b = np.asarray(boards)
        return np.bitwise_xor.reduce(table[np.arange(CELLS), b], axis=1)

    def _seed_cache(self, boards, scores, indices=None, clear=False):
        """A reseed starts a fresh bounded history for just those replicas."""
        ids = np.arange(self.n) if indices is None else np.asarray(indices, np.int32)
        if clear:
            self.cache_boards.fill(-1)
            self.cache_hashes.fill(0)
            self.cache_scores.fill(-1)
            self.cache_counts.fill(0)
            self.cache_cursors.fill(0)
            self.current_hashes.fill(0)
            self.cache_counters.fill(0)
        if not self.cache_slots:
            return
        hashes = self._cache_cpu_hash(boards)
        self.current_hashes[ids] = self.cp.asarray(hashes)
        self.cache_boards[0, :, ids] = self.cp.asarray(np.asarray(boards, np.int16))
        self.cache_hashes[0, ids] = self.cp.asarray(hashes)
        self.cache_scores[0, ids] = self.cp.asarray(np.asarray(scores, np.int16))
        self.cache_counts[ids] = 1
        self.cache_cursors[ids] = 1 % self.cache_slots
        self.cache_counters[3, ids] += 1

    def cache_stats(self):
        """Cumulative event counters; entries counts only currently retained states."""
        counters = self.cache_counters.get().sum(axis=1, dtype=np.uint64)
        memory = sum(getattr(self, field).nbytes for field in ('zobrist', 'current_hashes',
            'cache_boards', 'cache_hashes', 'cache_scores', 'cache_counts', 'cache_cursors', 'cache_counters'))
        return {'enabled': bool(self.cache_slots), 'slots_per_replica': self.cache_slots,
                'capacity': self.n * self.cache_slots, 'entries': int(self.cache_counts.sum().get()),
                'checks': int(counters[0]), 'hits': int(counters[1]), 'evictions': int(counters[2]),
                'inserts': int(counters[3]), 'memory_bytes': int(memory)}

    def _validate_cache_host(self, payload, boards, scores, slots, zobrist):
        counts, cursors = payload['cache_counts'], payload['cache_cursors']
        if np.any(counts > slots) or (slots and np.any(counts < 1)):
            raise ValueError('Invalid retained cache entry count.')
        if not slots:
            if np.any(counts) or np.any(cursors) or np.any(payload['current_hashes']):
                raise ValueError('Disabled cache contains active entries or fingerprints.')
            return 0
        if np.any(cursors >= slots) or np.any((counts < slots) & (cursors != counts)):
            raise ValueError('Invalid cache ring cursor.')
        if not np.array_equal(payload['current_hashes'], self._cache_cpu_hash(boards.T, zobrist)):
            raise ValueError('Current-board cache fingerprints failed verification.')
        total = 0
        # Limit temporary host allocations for full-population checkpoint verification.
        for first in range(0, len(counts), 128):
            last = min(len(counts), first + 128)
            block = payload['cache_boards'][:, :, first:last].transpose(0, 2, 1)
            retained = np.arange(slots)[:, None] < counts[None, first:last]
            b = block[retained]
            expected_scores = payload['cache_scores'][:, first:last][retained]
            expected_hashes = payload['cache_hashes'][:, first:last][retained]
            if not np.array_equal(self.score_cpu(b), expected_scores):
                raise ValueError('Retained full-board cache scores failed verification.')
            if not np.array_equal(self._cache_cpu_hash(b, zobrist), expected_hashes):
                raise ValueError('Retained full-board cache fingerprints failed verification.')
            total += len(b)
        ids = np.arange(len(counts))
        latest = (cursors.astype(np.int32) - 1) % slots
        if not np.array_equal(payload['cache_boards'][latest, :, ids], boards.T):
            raise ValueError('Latest retained cache entry differs from the current board.')
        if not np.array_equal(payload['cache_scores'][latest, ids], scores):
            raise ValueError('Latest retained cache result differs from the current score.')
        return total

    def verify_cache(self, indices=None):
        ids = np.arange(self.n) if indices is None else np.asarray(indices, np.int32)
        if ids.ndim != 1 or not len(ids) or np.any(ids < 0) or np.any(ids >= self.n):
            raise ValueError('Cache verification indices must name existing replicas.')
        payload = {'cache_counts': self.cache_counts[ids].get(),
            'cache_cursors': self.cache_cursors[ids].get(), 'current_hashes': self.current_hashes[ids].get(),
            'cache_boards': self.cache_boards[:, :, ids].get(),
            'cache_scores': self.cache_scores[:, ids].get(), 'cache_hashes': self.cache_hashes[:, ids].get()}
        return self._validate_cache_host(payload, self.boards[:, ids].get(), self.scores[ids].get(),
                                         self.cache_slots, self._zobrist_host)

    def score_cpu(self, boards):
        return score_boards_cpu(self.masks, self.neighbors, self.neighbor_sides, boards)

    def perturb(self, boards, strengths):
        b = legal_boards(boards).copy()
        strengths = np.broadcast_to(strengths, (len(b),))
        if np.any(strengths < 0) or np.any(strengths != np.floor(strengths)):
            raise ValueError('Perturbation strengths must be nonnegative integers.')
        for row, strength in enumerate(strengths):
            for _ in range(int(strength)):
                a, z = self.host_rng.choice(CELLS, 2, replace=False)
                ta, tz = int(b[row, a]) // 3, int(b[row, z]) // 3
                b[row, a] = tz * 3 + self.host_rng.integers(3)
                b[row, z] = ta * 3 + self.host_rng.integers(3)
        return b

    def initialize(self, seeds=None, perturbations=None):
        cp = self.cp
        if seeds is None:
            b = np.empty((self.n, CELLS), np.int16)
            for i in range(self.n):
                b[i] = self.host_rng.permutation(CELLS) * 3 + self.host_rng.integers(3, size=CELLS)
            self.seeds = b[:min(64, self.n)].copy()
            best = b.copy()
        else:
            self.seeds = legal_boards(seeds).copy()
            best = self.seeds[np.arange(self.n) % len(self.seeds)].copy()
            if perturbations is None:
                perturbations = np.array([0, 2, 4, 8, 12, 24, 40, 80])[np.arange(self.n) % 8]
            b = self.perturb(best, perturbations)
        scores = self.score_cpu(b)
        bestscores = self.score_cpu(best)
        improved = scores > bestscores
        best[improved] = b[improved]
        bestscores[improved] = scores[improved]
        perfect = bestscores == 240
        b[perfect] = best[perfect]
        scores[perfect] = 240
        self.pending = cp.asarray(perfect.astype(np.uint8))
        self.boards = cp.asarray(b.T.copy())
        self.bestboards = cp.asarray(best.T.copy())
        self.scores = cp.asarray(scores)
        self.bestscores = cp.asarray(bestscores)
        self.positions = cp.asarray(np.argsort(b // 3, axis=1).astype(np.int16).T.copy())
        self.initialized = True
        self._seed_cache(b, scores, clear=True)

    def step(self, steps=32):
        if not self.initialized:
            raise RuntimeError('Call initialize() or resume() before searching.')
        if isinstance(steps, bool) or int(steps) != steps or not 1 <= int(steps) <= 128:
            raise ValueError('Use 1..128 moves per bounded kernel launch (32 recommended).')
        cp = self.cp
        start, end = cp.cuda.Event(), cp.cuda.Event()
        start.record()
        self.search_kernel(((self.n + 127) // 128,), (128,), (
            self.boards, self.bestboards, self.positions, *self.tables,
            self.pair_offsets, self.pair_codes, np.int32(self.colors), self.guidance,
            self.temps, self.scores, self.bestscores, self.rng, self.counters,
            self.pending, self.zobrist, self.current_hashes, self.cache_boards, self.cache_hashes,
            self.cache_scores, self.cache_counts, self.cache_cursors, self.cache_counters,
            np.int32(self.cache_slots), np.int32(self.n), np.int32(steps)))
        end.record()
        end.synchronize()
        return float(cp.cuda.get_elapsed_time(start, end))

    def has_pending(self):
        """Synchronously detect a frozen 240-edge candidate after a bounded launch."""
        return bool(self.cp.any(self.pending).get())

    def pending_indices(self):
        """Return replicas whose immutable perfect candidate awaits loop validation."""
        if not self.initialized:
            raise RuntimeError('Call initialize() or resume() first.')
        return np.flatnonzero(self.pending.get())

    def acknowledge(self, indices):
        """Release candidates after independent CPU validation (including duplicates)."""
        if not self.initialized:
            raise RuntimeError('Call initialize() or resume() first.')
        raw = np.asarray(indices)
        if raw.ndim != 1 or (raw.size and raw.dtype.kind not in 'iu'):
            raise ValueError('Acknowledgement indices must be a one-dimensional integer sequence.')
        if np.any(raw < 0) or np.any(raw >= self.n):
            raise ValueError('Acknowledgement index is outside the replica range.')
        if not raw.size:
            return 0
        idx = np.unique(raw.astype(np.int32))
        count = int(self.cp.count_nonzero(self.pending[idx]).get())
        self.pending[idx] = 0
        return count

    def cool(self, elapsed):
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Elapsed seconds must be finite and nonnegative.')
        phase = (elapsed / 45.0 + np.arange(self.n) / self.n) % 1.0
        peaks = np.array([.35, .5, .7, .9, 1.2, 1.6, 2., 2.5])[self.arms]
        self.temps.set((.06 + peaks * (1.0 - phase)**2).astype(np.float32))

    def best_board(self):
        idx = int(self.cp.argmax(self.bestscores).get())
        return self.bestboards[:, idx].get(), int(self.bestscores[idx].get()), idx

    def reseed(self, seeds=None, fraction=.125, strengths=(2, 4, 8, 16, 32)):
        if not self.initialized:
            raise RuntimeError('Call initialize() first.')
        if not 0 < fraction <= 1:
            raise ValueError('Reseed fraction must be in (0,1].')
        if seeds is not None:
            self.seeds = legal_boards(seeds).copy()
        cp = self.cp
        available = np.flatnonzero(self.pending.get() == 0)
        count = min(len(available), max(1, int(self.n * fraction)))
        if not count:
            return np.empty(0, dtype=np.int32)
        ids = self.host_rng.choice(available, count, replace=False)
        base = self.seeds[self.host_rng.integers(0, len(self.seeds), size=count)].copy()
        b = self.perturb(base, self.host_rng.choice(strengths, count))
        scores = self.score_cpu(b)
        bestscores = self.score_cpu(base)
        improved = scores > bestscores
        base[improved] = b[improved]
        bestscores[improved] = scores[improved]
        perfect = bestscores == 240
        b[perfect] = base[perfect]
        scores[perfect] = 240
        self.pending[ids] = cp.asarray(perfect.astype(np.uint8))
        self.boards[:, ids] = cp.asarray(b.T.copy())
        self.scores[ids] = cp.asarray(scores)
        self.positions[:, ids] = cp.asarray(np.argsort(b // 3, axis=1).astype(np.int16).T.copy())
        # Historical bests survive a restart; callers can still inspect previous perfect candidates.
        old_best = self.bestscores[ids].get()
        replace = (bestscores > old_best) | perfect
        chosen = ids[replace]
        if len(chosen):
            self.bestboards[:, chosen] = cp.asarray(base[replace].T.copy())
            self.bestscores[chosen] = cp.asarray(bestscores[replace])
        self._seed_cache(b, scores, indices=ids)
        return ids

    def verify_device_scores(self, indices=None, include_best=True):
        idx = np.arange(self.n) if indices is None else np.asarray(indices, dtype=np.int32)
        if idx.ndim != 1 or np.any(idx < 0) or np.any(idx >= self.n) or not len(idx):
            raise ValueError('Verification indices must name existing replicas.')
        fields = [('boards', 'scores')]
        if include_best:
            fields.append(('bestboards', 'bestscores'))
        for boardfield, scorefield in fields:
            b = getattr(self, boardfield)[:, idx].get().T
            observed = getattr(self, scorefield)[idx].get()
            actual = self.score_cpu(b)
            if not np.array_equal(observed, actual):
                raise RuntimeError('GPU state failed independent CPU scoring: ' + boardfield)
        b = self.boards[:, idx].get().T
        positions = self.positions[:, idx].get().T
        if not np.array_equal(positions, np.argsort(b // 3, axis=1)):
            raise RuntimeError('GPU tile-position inverse failed validation.')
        if np.any(self.bestscores[idx].get() < self.scores[idx].get()):
            raise RuntimeError('Best scores fell below current scores.')
        pending = self.pending[idx].get()
        if np.any(pending > 1):
            raise RuntimeError('Invalid pending candidate flag.')
        flagged = pending != 0
        if np.any(flagged):
            if np.any(self.scores[idx].get()[flagged] != 240):
                raise RuntimeError('Pending candidate is not edge-perfect.')
            if not np.array_equal(self.boards[:, idx].get()[:, flagged],
                                  self.bestboards[:, idx].get()[:, flagged]):
                raise RuntimeError('Pending candidate differs from its frozen search state.')
        self.verify_cache(idx)
        return len(idx)

    def snapshot_checkpoint(self):
        """Capture a coherent independent CPU snapshot between bounded GPU launches.

        Call this on the thread that owns the search, before scheduling more moves.
        The returned arrays never alias live GPU or host state; write_checkpoint()
        can therefore serialize them on a background thread while searching resumes.
        """
        if not self.initialized:
            raise RuntimeError('No search state to save.')
        self.cp.cuda.get_current_stream().synchronize()
        payload = {
            'version': np.int32(VERSION), 'fingerprint': np.array(self.fingerprint),
            'boards': self.boards.get(), 'bestboards': self.bestboards.get(),
            'scores': self.scores.get(), 'bestscores': self.bestscores.get(),
            'rng': self.rng.get(), 'counters': self.counters.get(), 'pending': self.pending.get(),
            'temperatures': self.temps.get(), 'guidance': self.guidance.get(),
            'seed_pool': self.seeds.copy(),
            'host_rng': np.array(json.dumps(self.host_rng.bit_generator.state)),
            'cache_slots': np.int32(self.cache_slots), 'cache_zobrist': self._zobrist_host.copy(),
            'current_hashes': self.current_hashes.get(), 'cache_boards': self.cache_boards.get(),
            'cache_hashes': self.cache_hashes.get(), 'cache_scores': self.cache_scores.get(),
            'cache_counts': self.cache_counts.get(), 'cache_cursors': self.cache_cursors.get(),
            'cache_counters': self.cache_counters.get()}
        for value in payload.values():
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        return payload

    @staticmethod
    def write_checkpoint(path, payload, compressed=False):
        """Atomically write a CPU snapshot; no CUDA context or search lock is needed.

        Uncompressed NPZ avoids prolonged CPU compression of the complete board
        cache. Both formats preserve every V3 field and load with the same resume().
        Callers must serialize writes to the same destination to preserve age order.
        """
        from io_utils import replace_with_retry
        if type(compressed) is not bool:
            raise ValueError('compressed must be a boolean.')
        started = time.perf_counter()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='wb', prefix=path.name + '.', suffix='.tmp',
                                             dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                save = np.savez_compressed if compressed else np.savez
                save(stream, **payload)
                stream.flush()
                os.fsync(stream.fileno())
            file_bytes = temporary.stat().st_size
            replace_with_retry(temporary, path)
            temporary = None
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
        return {'compressed': compressed,
                'snapshot_bytes': sum(int(getattr(value, 'nbytes', 0)) for value in payload.values()),
                'file_bytes': file_bytes, 'write_seconds': time.perf_counter() - started}

    def checkpoint(self, path, compressed=False):
        """Synchronous convenience wrapper; use snapshot/write separately for background I/O."""
        started = time.perf_counter()
        payload = self.snapshot_checkpoint()
        capture_seconds = time.perf_counter() - started
        result = self.write_checkpoint(path, payload, compressed=compressed)
        result['capture_seconds'] = capture_seconds
        result['total_seconds'] = time.perf_counter() - started
        return result

    def resume(self, path):
        cp = self.cp
        with np.load(path, allow_pickle=False) as archive:
            version = int(archive['version'])
            if version not in (1, 2, VERSION) or str(archive['fingerprint']) != self.fingerprint:
                raise ValueError('Checkpoint uses a different puzzle, geometry, or format.')
            if archive['boards'].shape != (CELLS, self.n):
                return False
            expected = {
                'boards': ((CELLS, self.n), np.int16), 'bestboards': ((CELLS, self.n), np.int16),
                'scores': ((self.n,), np.int32), 'bestscores': ((self.n,), np.int32),
                'rng': ((self.n,), np.uint32), 'counters': ((3, self.n), np.uint64),
                'temperatures': ((self.n,), np.float32), 'guidance': ((self.n,), np.float32)}
            arrays = {}
            for field, (shape, dtype) in expected.items():
                value = archive[field]
                if value.shape != shape or value.dtype != dtype:
                    raise ValueError('Malformed checkpoint field: ' + field)
                # NPZ fields are owned eager arrays, valid after the archive closes.
                arrays[field] = value
            for boardfield, scorefield in [('boards', 'scores'), ('bestboards', 'bestscores')]:
                if not np.array_equal(self.score_cpu(arrays[boardfield].T), arrays[scorefield]):
                    raise ValueError('Checkpoint score verification failed: ' + boardfield)
            if np.any(arrays['bestscores'] < arrays['scores']):
                raise ValueError('Checkpoint has inconsistent best scores.')
            if version == 1:
                # Legacy checkpoints had no acknowledgement state. Present each
                # stored perfect candidate once before allowing it to change.
                arrays['pending'] = (arrays['bestscores'] == 240).astype(np.uint8)
                flagged = arrays['pending'] != 0
                arrays['boards'][:, flagged] = arrays['bestboards'][:, flagged]
                arrays['scores'][flagged] = 240
            else:
                if 'pending' not in archive:
                    raise ValueError('Checkpoint is missing pending candidate flags.')
                pending = archive['pending']
                if pending.shape != (self.n,) or pending.dtype != np.uint8 or np.any(pending > 1):
                    raise ValueError('Malformed checkpoint field: pending')
                arrays['pending'] = pending.copy()
                flagged = pending != 0
                if (np.any(arrays['scores'][flagged] != 240)
                        or not np.array_equal(arrays['boards'][:, flagged], arrays['bestboards'][:, flagged])):
                    raise ValueError('Checkpoint contains an inconsistent pending candidate.')
            if np.any(arrays['rng'] == 0):
                raise ValueError('Checkpoint RNG contains a zero state.')
            for field in ('temperatures', 'guidance'):
                if not np.all(np.isfinite(arrays[field])) or np.any(arrays[field] < 0):
                    raise ValueError('Invalid checkpoint ' + field)
            if np.any(arrays['guidance'] > 1):
                raise ValueError('Checkpoint guidance probabilities exceed one.')
            seeds = legal_boards(archive['seed_pool']).copy()
            rng_state = json.loads(str(archive['host_rng']))
            host_rng = np.random.default_rng()
            host_rng.bit_generator.state = rng_state
            cache_payload = None
            if version == 3:
                stored_slots = int(archive['cache_slots'])
                if not 0 <= stored_slots <= 256:
                    raise ValueError('Invalid checkpoint cache capacity.')
                cache_expected = {
                    'cache_zobrist': ((CELLS, 480), np.uint64),
                    'current_hashes': ((self.n,), np.uint64),
                    'cache_boards': ((stored_slots, CELLS, self.n), np.int16),
                    'cache_hashes': ((stored_slots, self.n), np.uint64),
                    'cache_scores': ((stored_slots, self.n), np.int16),
                    'cache_counts': ((self.n,), np.uint16),
                    'cache_cursors': ((self.n,), np.uint16),
                    'cache_counters': ((4, self.n), np.uint64)}
                cache_payload = {}
                for field, (shape, dtype) in cache_expected.items():
                    value = archive[field]
                    if value.shape != shape or value.dtype != dtype:
                        raise ValueError('Malformed checkpoint field: ' + field)
                    # Avoid a second full cache allocation during large-population resume.
                    cache_payload[field] = value
                self._validate_cache_host(cache_payload, arrays['boards'], arrays['scores'],
                                          stored_slots, cache_payload['cache_zobrist'])
        for field in ('boards', 'bestboards', 'scores', 'bestscores', 'rng', 'counters', 'pending'):
            setattr(self, field, cp.asarray(arrays[field]))
        self.temps.set(arrays['temperatures'])
        self.guidance.set(arrays['guidance'])
        self.positions = cp.asarray(np.argsort(arrays['boards'].T // 3, axis=1).astype(np.int16).T.copy())
        self.seeds, self.host_rng = seeds, host_rng
        self.initialized = True
        if cache_payload is not None and stored_slots == self.cache_slots:
            self._zobrist_host = cache_payload.pop('cache_zobrist')
            self.zobrist.set(self._zobrist_host)
            for field, value in cache_payload.items():
                getattr(self, field).set(value)
            self.cache_resume_note = 'restored'
        else:
            self._seed_cache(arrays['boards'].T, arrays['scores'], clear=True)
            if cache_payload is not None:
                # Capacity changes discard retained entries, not historical accounting.
                self.cache_counters += cp.asarray(cache_payload['cache_counters'])
            self.cache_resume_note = 'initialized_from_legacy' if cache_payload is None else 'reset_after_capacity_change'
        return True
