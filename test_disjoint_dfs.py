"""Finite end-to-end checks for disjoint CUDA DFS jobs and exact resume/refill."""
from itertools import permutations, product
import json
from pathlib import Path
import tempfile
import numpy as np
from dfs_gpu import DFS, normalize_prefixes, ensure_disjoint_prefixes
from geometry import build_board


def rejects(call):
    try:
        call()
    except ValueError:
        return
    raise AssertionError('Expected an invalid or overlapping job to be rejected.')


def compare(left, right):
    for field in DFS.DEVICE_FIELDS:
        assert np.array_equal(getattr(left, field).get(), getattr(right, field).get()), field
    assert np.array_equal(left.prefix_codes, right.prefix_codes)


def collect(dfs, limit=1000, seen=None):
    seen = set() if seen is None else seen
    found = []
    for _ in range(limit):
        dfs.step(128)
        dfs.verify()
        pending = dfs.pending()
        if len(pending):
            frozen = dfs.boards[:, pending].get()
            dfs.step(32)
            assert np.array_equal(frozen, dfs.boards[:, pending].get())
            for lane in pending:
                row = tuple(map(int, dfs.boards[:, int(lane)].get()[dfs.order]))
                assert row not in seen, 'Repeated complete arrangement across jobs or acknowledgements.'
                seen.add(row)
                found.append(row)
            dfs.acknowledge(pending)
            dfs.verify()
        flags = dfs.states.get()
        if np.all((flags == 2) | (flags == 3)):
            return found, seen
    raise AssertionError('Small finite prefix forest did not exhaust within the test budget.')


def main():
    board = build_board()
    masks = np.zeros((480, 3), np.uint16)
    args = (masks, board.neighbor_cells, board.neighbor_sides)
    rows, lengths = normalize_prefixes([[], [0, 3, -1]])
    assert np.array_equal(lengths, [0, 2])
    rejects(lambda: normalize_prefixes([[0, -1, 3]]))
    rejects(lambda: ensure_disjoint_prefixes([(0,), (0, 3)]))
    rejects(lambda: ensure_disjoint_prefixes([(), (0,)]))
    unique_roots = DFS(*args, n=512, seed=18)
    assert np.array_equal(unique_roots.roots[:480], np.arange(480))
    assert np.all(unique_roots.states[480:].get() == 3)
    unique_roots.verify()
    rejects(lambda: DFS(*args, n=2, root_codes=[0, 0]))
    idle = DFS(*args, n=4, prefixes=[], seed=91)
    assert np.all(idle.states.get() == 3)
    idle.verify()
    idle.step(128)
    assert not np.any(idle.counters.get())
    idle.load_prefixes([[0]], lanes=[1])
    rejects(lambda: idle.load_prefixes([[0]], lanes=[2]))
    rejects(lambda: idle.load_prefixes([[0, 3]], lanes=[2]))
    rejects(lambda: idle.load_prefixes([[]], lanes=[2]))
    rejects(lambda: idle.load_prefixes([[3]], lanes=[1]))
    rejects(lambda: idle.load_prefixes([[0, 1]], lanes=[2]))  # Same tile in two cells.
    idle.verify()
    full = list(map(int, np.arange(160) * 3))
    terminal = DFS(*args, n=2, prefixes=[full], seed=11)
    assert np.array_equal(terminal.states.get(), [1, 3])
    assert terminal.floors[0].get() == 160
    terminal.verify()
    terminal.acknowledge([0])
    terminal.step(128)
    assert terminal.states[0].get() == 2 and terminal.depths[0].get() == 0
    assert np.all(terminal.boards[:, 0].get() == -1)
    terminal.verify()
    empty = DFS(*args, n=2, prefixes=[[]], seed=401)
    assert empty.floors[0].get() == 0
    first = None
    for _ in range(100):
        empty.step(128)
        empty.verify()
        if len(empty.pending()):
            current = tuple(empty.boards[:, 0].get())
            if first is not None:
                assert current != first
                break
            first = current
            empty.acknowledge([0])
    else:
        raise AssertionError('Empty-root DFS failed to produce two distinct leaves.')
    # Two disjoint prefixes fix158tiles, leaving exactly2!*3^2=18 completions each.
    prefix0 = full[:158]
    prefix1 = prefix0.copy()
    prefix1[0] = 1
    dfs = DFS(*args, n=3, prefixes=[prefix0, prefix1], seed=431)
    expected = set()
    for prefix in (prefix0, prefix1):
        for tail in permutations((158, 159)):
            for rotations in product(range(3), repeat=2):
                expected.add(tuple(prefix + [3 * tile + rotation for tile, rotation in zip(tail, rotations)]))
    found, seen = collect(dfs)
    assert len(found) == 36 and set(found) == expected
    assert np.array_equal(dfs.states.get(), [2, 2, 3])
    assert np.all(dfs.boards.get() == -1) and np.all(dfs.used.get() == 0)
    assert np.array_equal(dfs.floors.get(), [158, 158, 0])
    # Refill only an exhausted lane with a new disjoint prefix; keep cumulative work counters.
    previous_counters = dfs.counters.get()
    prefix2 = prefix0.copy()
    prefix2[0] = 2
    dfs.load_prefixes([prefix2], lanes=[0])
    assert np.array_equal(dfs.counters.get(), previous_counters)
    dfs.step(7)
    with tempfile.TemporaryDirectory(prefix='disjoint-dfs-test-') as directory:
        directory = Path(directory).resolve()
        assert directory.parent == Path(tempfile.gettempdir()).resolve()
        path = directory / 'refill.npz'
        payload = dfs.snapshot_checkpoint()
        payload.update(exact_job_ids=np.array([201, 102, -1], np.int64), exact_next_job=np.int64(202),
                       exact_frontier_sha=np.array('test-frontier'))
        DFS.write_checkpoint(path, payload)
        resumed = DFS(*args, n=3, prefixes=[], seed=919)
        resumed.resume(path)
        compare(dfs, resumed)
        dfs.step(32)
        resumed.step(32)
        compare(dfs, resumed)
        resumed.verify()
        more, seen = collect(resumed, seen=seen)
        assert len(more) == 18 and len(seen) == 54
        resumed.checkpoint(directory / 'exhausted.npz')
        restored = DFS(*args, n=3, prefixes=[])
        restored.resume(directory / 'exhausted.npz')
        restored.verify()
        assert np.array_equal(restored.states.get(), [2, 2, 3])
        # Legacy unique-root states retain their one-code prefix; duplicates are rejected.
        legacy_gpu = DFS(*args, n=3, seed=231)
        legacy_gpu.step(4)
        legacy = legacy_gpu.snapshot_checkpoint()
        for key in ('dfs_version', 'prefix_codes', 'floors'):
            legacy.pop(key)
        DFS.write_checkpoint(directory / 'legacy.npz', legacy)
        restored.resume(directory / 'legacy.npz')
        compare(legacy_gpu, restored)
        with np.load(path, allow_pickle=False) as archive:
            corrupt = {key: archive[key] for key in archive.files}
        corrupt['floors'] = corrupt['floors'].copy()
        corrupt['floors'][0] = 157
        DFS.write_checkpoint(directory / 'bad.npz', corrupt)
        rejects(lambda: restored.resume(directory / 'bad.npz'))
    result = {'status': 'passed', 'device': dfs.device, 'finite_expected_leaves': 54,
              'finite_unique_leaves_found': len(seen), 'checks': ['unique default roots and idle surplus lanes',
              'overlap and tile-reuse rejection', 'prefix floor0 search', 'prefix floor160 terminal acknowledgement',
              'exact finite disjoint subtree exhaustion', 'pending candidate freeze', 'no repeated full boards',
              'exhaustion never enters sibling jobs', 'refill preserves cumulative counters',
              'checkpoint with extra SSD-ledger metadata', 'deterministic replay after refill',
              'legacy unique-root checkpoint', 'corrupt fixed-prefix rejection']}
    output = Path(__file__).resolve().parent / 'runtime' / 'disjoint-dfs-tests.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
