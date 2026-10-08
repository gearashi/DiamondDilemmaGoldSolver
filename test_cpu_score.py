"""CPU-only equivalence check for checkpoint scoring optimization; never imports CUDA."""
import json
from pathlib import Path
import time
import numpy as np
from geometry import build_board
from gpu_engine import legal_boards, reverse11, score_boards_cpu


def legacy_score(masks, neighbors, sides, boards):
    b = legal_boards(boards)
    scores = np.zeros(len(b), np.int32)
    for cell in range(160):
        for side in range(3):
            other = int(neighbors[cell, side])
            if cell < other:
                target = masks[b[:, other], int(sides[cell, side])]
                scores += masks[b[:, cell], side] == reverse11(target)
    return scores


def main():
    rng = np.random.default_rng(9031)
    geometry = build_board()
    neighbors = np.asarray(geometry.neighbor_cells)
    sides = np.asarray(geometry.neighbor_sides)
    count = 512
    boards = np.array([rng.permutation(160) * 3 + rng.integers(0, 3, 160) for _ in range(count)], np.int16)
    fixtures = [rng.integers(0, 2048, (480, 3), dtype=np.uint16),
                rng.choice(np.array([0, 1, 1024, 17, 1088, 2047], np.uint16), (480, 3)),
                np.zeros((480, 3), np.uint16)]
    old_seconds = new_seconds = 0.0
    for masks in fixtures:
        started = time.perf_counter()
        expected = legacy_score(masks, neighbors, sides, boards)
        old_seconds += time.perf_counter() - started
        started = time.perf_counter()
        actual = score_boards_cpu(masks, neighbors, sides, boards)
        new_seconds += time.perf_counter() - started
        assert np.array_equal(actual, expected)
    assert np.all(actual == 240)
    result = {'status': 'passed', 'cpu_only': True, 'boards_compared': count * len(fixtures),
              'checks': ['random full11bit masks', 'sparse asymmetric masks', 'all matching masks'],
              'legacy_seconds': old_seconds, 'optimized_seconds': new_seconds,
              'small_test_speed_ratio': old_seconds / new_seconds}
    output = Path(__file__).resolve().parent / 'runtime' / 'cpu-score-equivalence.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
