"""Collision-safe recent-arrangement cache tests and optional actual-puzzle benchmark."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
from gpu_engine import GPU
from test_gpu import fixture


def probe(gpu, a, b, ca, cb):
    cp = gpu.cp
    output = cp.empty(gpu.n, np.uint8)
    hashes = cp.empty(gpu.n, np.uint64)
    args = [cp.asarray(np.asarray(v, np.int16)) for v in (a, b, ca, cb)]
    gpu.cache_probe_kernel(((gpu.n + 127) // 128,), (128,), (gpu.boards, gpu.cache_boards,
        gpu.cache_hashes, gpu.cache_counts, gpu.zobrist, gpu.current_hashes, *args,
        output, hashes, np.int32(gpu.n)))
    return output.get(), hashes.get()


def run_tests():
    masks, neighbors, sides, reference = fixture()
    gpu = GPU(masks, neighbors, sides, n=16, seed=117, cache_slots=8)
    gpu.initialize()
    assert gpu.cache_stats()['entries'] == gpu.n
    assert gpu.verify_cache() == gpu.n
    gpu.temps.fill(3.0)
    for _ in range(6):
        gpu.step(128)
    gpu.verify_device_scores()
    stats = gpu.cache_stats()
    assert stats['checks'] > 0 and stats['hits'] > 0 and stats['evictions'] > 0, stats
    assert stats['entries'] == stats['capacity'] == 128
    # No identical full arrangements may coexist within one replica's retained history.
    history = gpu.cache_boards.get()
    for replica in range(gpu.n):
        assert len(np.unique(history[:, :, replica], axis=0)) == gpu.cache_slots
    with tempfile.TemporaryDirectory(prefix='diamond-cache-tests-') as directory:
        path = Path(directory) / 'cache.npz'
        gpu.checkpoint(path)
        other = GPU(masks, neighbors, sides, n=gpu.n, seed=919, cache_slots=8)
        assert other.resume(path)
        assert other.cache_resume_note == 'restored'
        assert gpu.cache_stats() == other.cache_stats()
        gpu.step(32)
        other.step(32)
        fields = ('boards', 'bestboards', 'scores', 'bestscores', 'rng', 'counters', 'pending',
                  'cache_boards', 'cache_hashes', 'cache_scores', 'cache_counts', 'cache_cursors',
                  'cache_counters', 'current_hashes')
        for name in fields:
            assert np.array_equal(getattr(gpu, name).get(), getattr(other, name).get()), name
        other.verify_device_scores()
        with np.load(path, allow_pickle=False) as saved:
            payload = {name: saved[name].copy() for name in saved.files}
        for version in (1, 2):
            legacy = {name: value for name, value in payload.items() if not name.startswith('cache_') and name != 'current_hashes'}
            legacy['version'] = np.int32(version)
            if version == 1:
                legacy.pop('pending')
            legacy_path = Path(directory) / f'legacy-v{version}.npz'
            np.savez_compressed(legacy_path, **legacy)
            assert other.resume(legacy_path)
            assert other.cache_resume_note == 'initialized_from_legacy'
            assert other.cache_stats()['entries'] == other.n
            other.verify_device_scores()
        changed = GPU(masks, neighbors, sides, n=gpu.n, seed=113, cache_slots=2)
        assert changed.resume(path)
        assert changed.cache_resume_note == 'reset_after_capacity_change'
        assert changed.cache_stats()['entries'] == changed.n
        assert np.array_equal(changed.cache_counters.get()[:3], payload['cache_counters'][:3])
        assert np.array_equal(changed.cache_counters.get()[3], payload['cache_counters'][3] + 1)
        changed.verify_device_scores()
        corrupt = dict(payload)
        corrupt['cache_hashes'] = corrupt['cache_hashes'].copy()
        corrupt['cache_hashes'][0, 0] ^= np.uint64(1)
        bad = Path(directory) / 'bad.npz'
        np.savez_compressed(bad, **corrupt)
        try:
            other.resume(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Corrupt retained fingerprint was accepted.')
    # Force every fingerprint to collide. Distinct full boards must still remain admissible.
    collision = GPU(masks, neighbors, sides, n=3, seed=331, cache_slots=2)
    collision.initialize()
    collision._zobrist_host.fill(0)
    collision.zobrist.fill(0)
    boards = collision.boards.get().T
    collision._seed_cache(boards, collision.scores.get(), clear=True)
    previous = boards.copy()
    previous[:, 0] = previous[:, 0] // 3 * 3 + (previous[:, 0] % 3 + 1) % 3
    collision.cache_boards[1] = collision.cp.asarray(previous.T.copy())
    collision.cache_scores[1] = collision.cp.asarray(collision.score_cpu(previous).astype(np.int16))
    collision.cache_counts.fill(2)
    collision.cache_cursors.fill(1)  # Slot zero is current; slot one is older.
    a = np.array([1, 0, 5], np.int16)
    b = np.array([1, 0, 6], np.int16)
    ca = np.array([boards[0, 1], previous[1, 0], boards[2, 6]], np.int16)
    cb = np.array([boards[0, 1], previous[1, 0], boards[2, 5]], np.int16)
    repeated, hashes = probe(collision, a, b, ca, cb)
    assert np.array_equal(repeated, [1, 1, 0]), repeated
    assert np.all(hashes == 0)
    collision.verify_device_scores()
    collision.step(128)
    collision.verify_device_scores()
    # Exact no-op destinations are rejected after acknowledging a perfect board;
    # all other pending replicas remain immutable.
    perfect = GPU(masks, neighbors, sides, n=8, seed=814, cache_slots=4)
    perfect.initialize(reference, perturbations=0)
    frozen = perfect.boards.get()
    assert perfect.has_pending()
    perfect.acknowledge([0])
    perfect.temps.fill(0)
    perfect.step(128)
    assert perfect.cache_stats()['hits'] > 0
    assert np.array_equal(perfect.boards[:, 1:].get(), frozen[:, 1:])
    assert np.all(perfect.pending[1:].get() == 1)
    perfect.verify_device_scores()
    # Reseeding resets only eligible replicas' ring histories and stores their new current state.
    pending_cache = perfect.cache_boards[:, :, 1:].get()
    ids = perfect.reseed(reference, fraction=1.0, strengths=(2,))
    assert np.array_equal(ids, [0])
    assert perfect.cache_counts[0].get() == 1
    assert np.array_equal(perfect.cache_boards[:, :, 1:].get(), pending_cache)
    perfect.verify_device_scores()
    disabled = GPU(masks, neighbors, sides, n=8, seed=914, cache_slots=0)
    disabled.initialize()
    disabled.step(128)
    disabled.verify_device_scores()
    assert disabled.cache_stats()['checks'] == disabled.cache_stats()['entries'] == 0
    return {'status': 'passed', 'device': gpu.device,
        'checks': ['retained full-board uniqueness', 'search-level no-op rejection',
                   'forced hash collision with distinct full boards', 'cached scores and incremental fingerprints',
                   'pending freeze and acknowledgement', 'reseed cache consistency',
                   'checkpoint deterministic cache replay', 'legacy v1/v2 cache initialization',
                   'capacity-change reset', 'corrupt-cache rejection', 'disabled mode'],
        'test_cache_stats': stats}


def benchmark(seconds=3.0, replicas=4096, cache_sizes=(0, 16, 64)):
    from geometry import build_board
    from validator import orientation_masks
    root = Path(__file__).resolve().parent
    data = json.loads((root / 'data' / 'tiles.json').read_text(encoding='utf-8-sig'))
    saved = json.loads((root / 'runtime' / 'best.json').read_text(encoding='utf-8-sig'))
    geometry = build_board()
    masks = orientation_masks(data)
    results = []
    for slots in cache_sizes:
        gpu = GPU(masks, geometry.neighbor_cells, geometry.neighbor_sides,
                  n=replicas, seed=621, cache_slots=slots)
        gpu.initialize(saved['codes'], perturbations=0)
        gpu.temps.fill(.3)
        for _ in range(10):
            gpu.step(32)
        before = gpu.counters.get().sum(axis=1, dtype=np.uint64)
        cache_before = gpu.cache_stats()
        started = time.monotonic()
        gpu_ms = 0.0
        launches = 0
        while time.monotonic() - started < seconds:
            gpu_ms += gpu.step(32)
            launches += 1
        wall = time.monotonic() - started
        counters = gpu.counters.get().sum(axis=1, dtype=np.uint64) - before
        stats = gpu.cache_stats()
        gpu.verify_device_scores(np.arange(min(16, gpu.n)))
        results.append({'cache_slots': slots, 'replicas': replicas, 'seconds': wall,
            'proposals_per_second': int(counters[0] / wall), 'accepted_per_second': int(counters[1] / wall),
            'cache_hits_during_benchmark': stats['hits'] - cache_before['hits'],
            'cache_checks_during_benchmark': stats['checks'] - cache_before['checks'],
            'hit_fraction_of_proposals': (stats['hits'] - cache_before['hits']) / max(1, int(counters[0])),
            'hit_fraction_of_cache_checks': (stats['hits'] - cache_before['hits']) / max(1, stats['checks'] - cache_before['checks']),
            'gpu_ms_per_launch': gpu_ms / launches, 'launches': launches,
            'best_edge_score': int(gpu.bestscores.max().get()), 'cache_stats': stats})
        cp = gpu.cp
        del gpu
        cp.get_default_memory_pool().free_all_blocks()
    return {'description': 'Short actual-puzzle throughput comparison from the same saved seed; not a convergence study.',
            'temperature': .3, 'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--benchmark-seconds', type=float, default=3)
    parser.add_argument('--benchmark-replicas', type=int, default=4096)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_tests()
    if args.benchmark:
        if not 0 < args.benchmark_seconds <= 10:
            parser.error('Benchmark seconds must be in (0,10].')
        result['benchmark'] = benchmark(args.benchmark_seconds, args.benchmark_replicas)
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
