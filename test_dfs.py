"""Small CUDA DFS tests: partial-state invariants, freeze/ack, and checkpoint replay."""
import json
from pathlib import Path
import tempfile
import numpy as np
from dfs_gpu import DFS
from gpu_engine import score_boards_cpu
from test_gpu import fixture


def main():
    masks, neighbors, sides, reference = fixture()
    dfs = DFS(masks, neighbors, sides, n=32, seed=313)
    dfs.verify()
    launches = []
    for _ in range(8):
        launches.append(dfs.step(32))
        dfs.verify()
    with tempfile.TemporaryDirectory(prefix='diamond-dfs-tests-') as directory:
        path = Path(directory) / 'dfs.npz'
        dfs.checkpoint(path)
        other = DFS(masks, neighbors, sides, n=32, seed=919)
        other.resume(path)
        dfs.step(64)
        other.step(64)
        dfs.verify()
        other.verify()
        for field in ('boards', 'used', 'depths', 'maxdepths', 'states', 'rng', 'counters', 'cursors'):
            assert np.array_equal(getattr(dfs, field).get(), getattr(other, field).get()), field
    # Every edge is compatible in this synthetic fixture; uniqueness remains mandatory.
    easy_masks = np.zeros((480, 3), np.uint16)
    easy = DFS(easy_masks, neighbors, sides, n=8, seed=412)
    for _ in range(100):
        launches.append(easy.step(128))
        easy.verify()
        if len(easy.pending()) == easy.n:
            break
    assert len(easy.pending()) == easy.n, easy.depths.get()
    assert np.all(score_boards_cpu(easy_masks, neighbors, sides, easy.boards.get().T) == 240)
    frozen = easy.boards.get()
    before = easy.counters.get()
    easy.step(128)
    assert np.array_equal(frozen, easy.boards.get())
    assert np.array_equal(before, easy.counters.get())
    easy.acknowledge([0, 3])
    easy.verify()
    assert np.array_equal(easy.depths[[0, 3]].get(), [159, 159])
    assert np.all(easy.states[[1, 2, 4, 5, 6, 7]].get() == 1)
    easy.step(128)
    easy.verify()
    assert np.array_equal(easy.boards[:, [1, 2, 4, 5, 6, 7]].get(), frozen[:, [1, 2, 4, 5, 6, 7]])
    # No tile has the reversed mask 1024, so every root branch must exhaust immediately.
    impossible = DFS(np.ones((480, 3), np.uint16), neighbors, sides, n=7, seed=913)
    impossible.step(32)
    impossible.verify()
    assert np.all(impossible.states.get() == 2)
    assert np.all(impossible.depths.get() == 0)
    print(json.dumps({'status': 'passed', 'device': dfs.device,
        'max_test_launch_ms': max(launches), 'pending_solutions_checked': easy.n,
        'checks': ['partial exact seams', 'tile bitset and permutation', 'deterministic checkpoint replay',
                   'found candidates freeze', 'explicit acknowledgement continues DFS',
                   'incompatible root branches exhaust']}, indent=2))


if __name__ == '__main__':
    main()
