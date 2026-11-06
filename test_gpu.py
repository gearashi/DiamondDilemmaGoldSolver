"""Independent CPU/GPU checks, including neighboring swaps and restart replay.

Run: venv/Scripts/python.exe test_gpu.py
All test boards are synthetic fixtures; passing does not solve the real puzzle.
"""
import json
from pathlib import Path
import tempfile
import numpy as np
from geometry import build_board
from gpu_engine import GPU, legal_boards, reverse11, score_boards_cpu


def fixture():
    geometry = build_board()
    neighbors = np.asarray(geometry.neighbor_cells, np.int16)
    sides = np.asarray(geometry.neighbor_sides, np.uint8)
    rng = np.random.default_rng(817)
    faces = np.zeros((160, 3), np.uint16)
    # Assign one random mask per edge and its reversal at the other endpoint.
    for a, sa, b, sb in geometry.edges:
        mask = rng.integers(0, 2048, dtype=np.uint16)
        faces[a, sa] = mask
        faces[b, sb] = reverse11(mask)
    masks = np.stack([np.roll(face, rotation) for face in faces for rotation in range(3)])
    reference = np.arange(160, dtype=np.int16) * 3
    return masks, neighbors, sides, reference


def expect_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError('Expected malformed data to be rejected.')


def main():
    masks, neighbors, sides, reference = fixture()
    assert int(reverse11(np.uint16(1))) == 1024
    assert int(reverse11(np.uint16(1024))) == 1
    assert np.array_equal(reverse11(reverse11(np.arange(2048, dtype=np.uint16))), np.arange(2048))
    assert int(score_boards_cpu(masks, neighbors, sides, reference)[0]) == 240
    invalid = reference.copy()
    invalid[0] = invalid[1]
    expect_value_error(lambda: legal_boards(invalid))
    gpu = GPU(masks, neighbors, sides, n=192, seed=919)
    gpu.initialize(reference, perturbations=8)
    cp = gpu.cp
    assert gpu.verify_device_scores() == gpu.n
    device_scores = cp.empty(gpu.n, np.int32)
    gpu.score_kernel(((gpu.n + 127) // 128,), (128,), (gpu.boards, *gpu.tables, device_scores, np.int32(gpu.n)))
    assert np.array_equal(device_scores.get(), gpu.score_cpu(gpu.boards.get().T))
    rng = np.random.default_rng(615)
    b = gpu.boards.get().T
    a = rng.integers(0, 160, gpu.n, dtype=np.int16)
    z = rng.integers(0, 160, gpu.n, dtype=np.int16)
    z[:64] = a[:64]  # rotation only
    z[64:128] = neighbors[a[64:128], np.arange(64) % 3]  # adjacent cells
    newa = (b[np.arange(gpu.n), z] // 3 * 3 + rng.integers(0, 3, gpu.n)).astype(np.int16)
    newz = (b[np.arange(gpu.n), a] // 3 * 3 + rng.integers(0, 3, gpu.n)).astype(np.int16)
    newz[z == a] = newa[z == a]
    proposed = b.copy()
    proposed[np.arange(gpu.n), a] = newa
    proposed[np.arange(gpu.n), z] = newz
    deltas = cp.empty(gpu.n, np.int32)
    gpu.delta_kernel(((gpu.n + 127) // 128,), (128,), (gpu.boards, *gpu.tables,
        cp.asarray(a), cp.asarray(z), cp.asarray(newa), cp.asarray(newz), deltas, np.int32(gpu.n)))
    assert np.array_equal(deltas.get(), gpu.score_cpu(proposed) - gpu.score_cpu(b))
    gpu.initialize()
    timings = []
    for temperature in (0.0, 0.2, 1.5):
        gpu.temps.fill(temperature)
        for _ in range(3):
            timings.append(gpu.step(32))
        gpu.verify_device_scores()
    gpu.cool(123.5)
    gpu.reseed(reference, fraction=.3)
    gpu.verify_device_scores()
    with tempfile.TemporaryDirectory(prefix='diamond-gpu-test-') as directory:
        path = Path(directory) / 'state.npz'
        gpu.checkpoint(path)
        other = GPU(masks, neighbors, sides, n=gpu.n, seed=42)
        assert other.resume(path)
        other.verify_device_scores()
        gpu.step(32)
        other.step(32)
        for field in ('boards', 'bestboards', 'scores', 'bestscores', 'rng', 'counters'):
            assert np.array_equal(getattr(gpu, field).get(), getattr(other, field).get()), field
        gpu.verify_device_scores()
        other.verify_device_scores()
        with np.load(path, allow_pickle=False) as archive:
            corrupt = {key: archive[key].copy() for key in archive.files}
        corrupt['scores'][0] += 1
        bad = Path(directory) / 'bad.npz'
        np.savez_compressed(bad, **corrupt)
        expect_value_error(lambda: other.resume(bad))
    expect_value_error(lambda: gpu.step(129))
    expect_value_error(lambda: gpu.cool(float('nan')))
    result = {'status': 'passed', 'device': gpu.device, 'replicas': gpu.n,
        'delta_cases': gpu.n, 'cuda_search_moves': gpu.n * 32 * 11,
        'max_test_launch_ms': round(max(timings), 3),
        'checks': ['11-bit reversal', 'planted 240-edge fixture', 'tile uniqueness',
            'CUDA full scorer', 'same-cell and adjacent move deltas', 'search state invariants',
            'inverse tile positions', 'checkpoint deterministic replay', 'corrupt checkpoint rejection']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
