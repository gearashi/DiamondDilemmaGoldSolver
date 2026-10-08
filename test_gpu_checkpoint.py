"""Targeted fast-checkpoint correctness tests and opt-in resource measurements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import zipfile
import numpy as np
from gpu_engine import GPU
from test_gpu import fixture
ROOT = Path(__file__).resolve().parent


def run_tests():
    masks, neighbors, sides, _ = fixture()
    gpu = GPU(masks, neighbors, sides, n=16, seed=117, cache_slots=8)
    gpu.initialize()
    gpu.step(128)
    snapshot = gpu.snapshot_checkpoint()
    preserved = {key: value.copy() for key, value in snapshot.items()}
    assert all(not value.flags.writeable for value in snapshot.values() if isinstance(value, np.ndarray))
    old_seeds = gpu.seeds.copy()
    gpu.seeds[0, 0] = gpu.seeds[0, 0] // 3 * 3 + (gpu.seeds[0, 0] % 3 + 1) % 3
    gpu._zobrist_host[0, 0] ^= np.uint64(1)
    assert np.array_equal(snapshot['seed_pool'], old_seeds)
    assert snapshot['cache_zobrist'][0, 0] != gpu._zobrist_host[0, 0]
    gpu.seeds = old_seeds
    gpu._zobrist_host[0, 0] ^= np.uint64(1)
    with tempfile.TemporaryDirectory(prefix='diamond-fast-checkpoint-') as directory:
        directory = Path(directory).resolve()
        assert directory.parent == Path(tempfile.gettempdir()).resolve()
        uncompressed = directory / 'state.npz'
        with ThreadPoolExecutor(max_workers=1) as worker:
            writing = worker.submit(GPU.write_checkpoint, uncompressed, snapshot)
            gpu.step(64)
            plain_stats = writing.result()
        for key in snapshot:
            assert np.array_equal(snapshot[key], preserved[key]), key
        assert plain_stats['compressed'] is False
        with zipfile.ZipFile(uncompressed) as archive:
            assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        other = GPU(masks, neighbors, sides, n=16, seed=713, cache_slots=8)
        assert other.resume(uncompressed)
        other.step(64)
        for key in ('boards', 'bestboards', 'scores', 'bestscores', 'rng', 'counters', 'cache_boards', 'cache_counters'):
            assert np.array_equal(getattr(gpu, key).get(), getattr(other, key).get()), key
        other.verify_device_scores()
        compressed = directory / 'compressed.npz'
        compressed_stats = GPU.write_checkpoint(compressed, snapshot, compressed=True)
        assert compressed_stats['compressed']
        with zipfile.ZipFile(compressed) as archive:
            assert all(info.compress_type == zipfile.ZIP_DEFLATED for info in archive.infolist())
        assert other.resume(compressed)
        other.verify_device_scores()
        old_hash = hashlib.sha256(uncompressed.read_bytes()).hexdigest()
        with patch('numpy.savez', side_effect=OSError('intentional failed-write test')):
            try:
                GPU.write_checkpoint(uncompressed, snapshot)
            except OSError:
                pass
            else:
                raise AssertionError('An interrupted write unexpectedly succeeded.')
        assert hashlib.sha256(uncompressed.read_bytes()).hexdigest() == old_hash
        assert not list(directory.glob('*.tmp'))
        # One worker preserves snapshot age order when saving while the search continues.
        latest = gpu.snapshot_checkpoint()
        with ThreadPoolExecutor(max_workers=1) as worker:
            first = worker.submit(GPU.write_checkpoint, uncompressed, snapshot)
            last = worker.submit(GPU.write_checkpoint, uncompressed, latest)
            first.result()
            last.result()
        with np.load(uncompressed, allow_pickle=False) as archive:
            for key in latest:
                assert np.array_equal(archive[key], latest[key]), key
        convenience = gpu.checkpoint(directory / 'convenience.npz')
        assert 'capture_seconds' in convenience and not convenience['compressed']
    try:
        GPU(masks, neighbors, sides, n=131072, cache_slots=256)
    except ValueError as exc:
        assert 'indexing limit' in str(exc)
    else:
        raise AssertionError('Unsafe cache index footprint was accepted.')
    return {'status': 'passed', 'checks': ['independent readonly snapshot', 'GPU advances during background CPU writing',
        'uncompressed default', 'compressed backward compatibility', 'deterministic resume from captured snapshot',
        'atomic write failure preserves previous checkpoint', 'unique temporary files cleaned',
        'single-worker snapshot ordering', 'GPU cache index preflight'],
        'small_uncompressed': plain_stats, 'small_compressed': compressed_stats}


def measure_existing_checkpoint():
    """Read the stopped run's real cache once and time uncompressed writing on the same drive."""
    source = ROOT / 'runtime' / 'checkpoint.npz'
    started = time.perf_counter()
    with np.load(source, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    load_seconds = time.perf_counter() - started
    # One temporary file is removed explicitly; the active checkpoint is never changed.
    fd, temporary_name = tempfile.mkstemp(prefix='checkpoint-speed-', suffix='.npz', dir=source.parent)
    import os
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        stats = GPU.write_checkpoint(temporary, payload)
        stats.update(source_compressed_bytes=source.stat().st_size, source_load_seconds=load_seconds,
                     replicas=int(payload['boards'].shape[1]))
        with np.load(temporary, allow_pickle=False) as archive:
            assert np.array_equal(archive['boards'], payload['boards'])
            assert np.array_equal(archive['cache_counters'], payload['cache_counters'])
        return stats
    finally:
        temporary.unlink(missing_ok=True)


def measure_largest(replicas=131072, seconds=2.0, steps=8):
    import cupy as cp
    from geometry import build_board
    from validator import orientation_masks
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    data = json.loads((ROOT / 'data' / 'tiles.json').read_text(encoding='utf-8-sig'))
    seed = json.loads((ROOT / 'runtime' / 'best.json').read_text(encoding='utf-8-sig'))['codes']
    geometry = build_board()
    free_before, total_bytes = cp.cuda.runtime.memGetInfo()
    started = time.perf_counter()
    gpu = GPU(orientation_masks(data), geometry.neighbor_cells, geometry.neighbor_sides,
              n=replicas, seed=841, cache_slots=64)
    gpu.initialize(seed, perturbations=0)
    setup_seconds = time.perf_counter() - started
    gpu.temps.fill(.3)
    for _ in range(4):
        gpu.step(steps)
    before = gpu.counters.get().sum(axis=1, dtype=np.uint64)
    started = time.perf_counter()
    milliseconds = []
    while time.perf_counter() - started < seconds:
        milliseconds.append(gpu.step(steps))
    elapsed = time.perf_counter() - started
    counters = gpu.counters.get().sum(axis=1, dtype=np.uint64) - before
    gpu.verify_device_scores(np.arange(8))
    free_after, _ = cp.cuda.runtime.memGetInfo()
    result = {'replicas': replicas, 'cache_slots': 64, 'setup_seconds': setup_seconds,
        'estimated_device_bytes': gpu.estimated_device_bytes, 'free_vram_before': int(free_before),
        'free_vram_after': int(free_after), 'total_vram': int(total_bytes),
        'seconds': elapsed, 'steps_per_launch': steps, 'max_launch_ms': max(milliseconds),
        'mean_launch_ms': sum(milliseconds) / len(milliseconds),
        'proposals_per_second': int(counters[0] / elapsed), 'accepted_per_second': int(counters[1] / elapsed),
        'best_edge_score': int(gpu.bestscores.max().get()), 'cache_stats': gpu.cache_stats()}
    del gpu
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--measure-disk', action='store_true')
    parser.add_argument('--measure-largest', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_tests()
    if args.measure_disk:
        result['real_checkpoint_write'] = measure_existing_checkpoint()
    if args.measure_largest:
        result['largest_configuration'] = measure_largest()
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered, flush=True)


if __name__ == '__main__':
    main()
