"""Incumbent initialization for fresh native sampling lanes (never pruning)."""
import numpy as np


def variants(problem, codes, seed, count):
    from constraint_lns import score
    original = list(map(int, codes))
    metrics = score(problem, original)
    if metrics['domain_violations']:
        raise ValueError('Incumbent violates the current domains.')
    rng = np.random.default_rng(seed)
    free = [cell for cell in range(problem.n) if cell not in problem.fixed]
    defects = sorted({cell for a, _, b, _ in metrics['unmatched_edges'] for cell in (a, b)} & set(free))
    pool = [original]
    seen = {tuple(original)}
    for attempt in range(count * 4):
        if len(pool) >= count or not free:
            break
        candidate = original.copy()
        for _ in range((1, 2, 4, 8)[attempt % 4]):
            a = int(rng.choice(defects or free))
            b = int(rng.choice(defects if defects and rng.integers(4) else free))
            if rng.integers(3) == 0:
                b = a
            for cell, tile in ((a, candidate[b] // 3), (b, candidate[a] // 3)):
                choices = [3 * tile + r for r in range(3) if 3 * tile + r in problem.domains[cell]]
                if not choices:
                    break
                if cell == a:
                    left = int(rng.choice(choices))
                else:
                    right = int(rng.choice(choices))
            else:
                if a == b:
                    candidate[a] = left
                else:
                    candidate[a], candidate[b] = left, right
        key = tuple(candidate)
        if key not in seen:
            seen.add(key)
            pool.append(candidate)
    return pool


def initialize_lanes(problem, engine, codes, seed):
    """Overwrite current and lane-best together before the first move.

    Lane RNG and the last quarter of native random boards remain as initialized.
    Equivalent boards can occupy different lanes, each with its original RNG.
    """
    from gpu_sampling import gold_score, score_board
    from gpu_incremental_sampling_round26 import CACHE_BASE, canonical_labels
    count = max(1, 3 * engine.r // 4)
    pool = variants(problem, codes, seed, min(count, 128))
    score = gold_score if engine.gold_total else score_board
    key = 'gold_objective' if engine.gold_total else 'objective'
    objectives = []
    for values in pool:
        record = score(problem, values)
        if record['domain_violations']:
            raise ValueError('Illegal seeded lane.')
        objectives.append(record[key])
    cache = None
    if engine.flags & 64 and engine.gold_total:
        cache = np.asarray([canonical_labels(problem, values) for values in pool], dtype=np.uint32)
    boards = np.asarray(pool, dtype=np.uint32)
    objectives = np.asarray(objectives, dtype=np.uint32)
    for start in range(0, count, 1024):
        stop = min(count, start + 1024)
        ids = np.arange(start, stop) % len(pool)
        values = np.ascontiguousarray(boards[ids].T)
        scores = np.ascontiguousarray(objectives[ids])
        labels = np.ascontiguousarray(cache[ids].T) if cache is not None else None
        if hasattr(engine, 'cp'):
            dev = engine.cp.asarray(values)
            engine.boards[:, start:stop] = dev
            engine.best[:, start:stop] = dev
            engine.meta[1:3, start:stop] = engine.cp.asarray(scores)
            if labels is not None:
                engine.meta[CACHE_BASE:, start:stop] = engine.cp.asarray(labels)
        else:
            for cell, row in enumerate(values):
                offset = (cell * engine.r + start) * 4
                for buffer in engine.buffers[:2]:
                    engine.queue.write_buffer(buffer, offset, row)
            for row in (1, 2):
                engine.queue.write_buffer(engine.buffers[2], (row * engine.r + start) * 4, scores)
            if labels is not None:
                for row, packed in enumerate(labels, CACHE_BASE):
                    engine.queue.write_buffer(engine.buffers[2], (row * engine.r + start) * 4, packed)
    if hasattr(engine, 'cp'):
        engine.cp.cuda.get_current_stream().synchronize()
    else:
        engine.queue.read_buffer(engine.buffers[2], 0, 4)
    return {'seeded_lanes': count, 'distinct_seeds': len(pool), 'random_controls': engine.r-count}

