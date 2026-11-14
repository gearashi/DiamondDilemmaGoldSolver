"""Resume a physical exchange shell and certify matched-edge thresholds over every rotation."""
import argparse
import hashlib
from functools import lru_cache
import itertools
import json
import math
from pathlib import Path
import sqlite3
import time

import numpy as np

from r27_incumbent import BoardModel, rotation_alternatives
from search_problem import SearchProblem


def derangements(size):
    return [p for p in itertools.permutations(range(size))
            if all(i != j for i, j in enumerate(p))]


@lru_cache(maxsize=4096)
def _allowed_bijections(remaining, diagonal):
    """Count bijections that avoid all surviving forbidden diagonal pairs."""
    return sum((-1)**i * math.comb(diagonal,i)*math.factorial(remaining-i)
               for i in range(diagonal+1))


def derangement_unrank(size, rank):
    if not 0<=rank<_allowed_bijections(size,size):
        raise ValueError("Derangement rank out of range.")
    available = set(range(size))
    order = []
    for row in range(size):
        for column in sorted(available-{row}):
            rest = available-{column}
            diagonal = sum(c>row for c in rest)
            ways = _allowed_bijections(len(rest),diagonal)
            if rank<ways:
                order.append(column)
                available = rest
                break
            rank -= ways
    return tuple(order)


class ExchangeBound:
    """Each cell may pick a rotation and independently match each neighbor rotation.

    Summing all cell maxima counts every actual matching seam twice. A changed
    tile affects only its own star and the stars centered on its neighbors.
    """
    def __init__(self, model, board):
        self.model = model
        self.board = model.legal_board(board)
        self.tiles = self.board // 3
        self.n = model.n
        self.neighbors = np.where(model.problem.neighbors < 0, self.n, model.problem.neighbors)
        self.neighbors = np.vstack((self.neighbors, np.full(3, self.n)))
        self.pieces = np.append(self.tiles, self.n)
        masks = model.masks.reshape(self.n, 3, 3)
        # A neighbor can put any one of its three unrotated sides on this seam.
        # Relax all domain restrictions here. The exact rotation solver restores them.
        support = np.zeros((2048, self.n + 1), dtype=np.int8)
        for t, sides in enumerate(masks[:, 0, :]):
            support[model.reverse[sides], t] = 1
        self.support = support
        self.masks = np.concatenate((masks, np.zeros((1, 3, 3), dtype=masks.dtype)))
        self.base = self.stars(self.pieces, self.pieces[self.neighbors])
        self.base[-1] = 0
        self.total = int(self.base.sum())

    def stars(self, center, neighbors):
        values = sum(self.support[self.masks[center, :, side], neighbors[..., side, None]]
                     for side in range(3))
        return values.max(axis=-1)

    def bounds(self, cells, order):
        cells = np.asarray(cells, dtype=int)
        touched = np.concatenate((cells, self.neighbors[cells].reshape(len(cells), -1)), axis=1)
        touched.sort(axis=1)
        unique = np.ones(touched.shape, dtype=bool)
        unique[:, 1:] = touched[:, 1:] != touched[:, :-1]
        unique &= touched != self.n
        touched_neighbors = self.neighbors[touched]
        tc = self.pieces[touched].copy()
        tn = self.pieces[touched_neighbors].copy()
        sources = self.tiles[cells[:, order]]
        for j in range(cells.shape[1]):
            t = cells[:, j, None]
            tc = np.where(touched == t, sources[:, j, None], tc)
            tn = np.where(touched_neighbors == t[..., None], sources[:, j, None, None], tn)
        delta = self.stars(tc, tn) - self.base[touched]
        return (self.total + (delta * unique).sum(axis=1)) // 2

    def witness_seed(self, cells, order):
        candidate = self.board.copy()
        incoming = self.tiles[np.asarray(cells)[list(order)]]
        for cell, tile in zip(cells, incoming):
            rotations = np.flatnonzero(self.model.legal[cell, 3*tile:3*tile+3])
            if not len(rotations):
                return None
            candidate[cell] = 3*tile + rotations[0]
        return candidate


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def domain_binding(model):
    digest = hashlib.sha256(model.problem.fingerprint.encode())
    digest.update(np.packbits(model.legal,bitorder="little").tobytes())
    return digest.hexdigest()


def combination_batch(movable, size, offset, count):
    """Seek by combinadic rank, then increment in colex order without replay."""
    from r28_exchange_cuda import unrank
    n = len(movable)
    positions = unrank(offset,n,size)
    rows = []
    for _ in range(count):
        rows.append([movable[i] for i in positions])
        for j in range(size):
            if positions[j]+1 < (positions[j+1] if j+1<size else n):
                positions[j] += 1
                positions[:j] = range(j)
                break
    return np.asarray(rows,dtype=int)


def run_shell(problem, board, size, target, database, *, seconds=0, batch=None,
              stop_on_hit=False, backend="cpu"):
    """Commit complete combination batches. A crash can replay the unfinished batch."""
    if (not 2 <= size <= problem.n or not 0 <= target <= len(problem.edges) or seconds < 0
            or backend not in {"cpu", "cuda"} or (batch is not None and batch < 1)):
        raise ValueError("Invalid shell, threshold, backend, batch or time limit.")
    bound = ExchangeBound(BoardModel(problem), board)
    order_count = _allowed_bijections(size,size)
    permutations = derangements(size) if size<=6 else None
    movable = sorted(set(range(problem.n)) - set(problem.fixed))
    combinations = math.comb(len(movable), size)
    total = combinations*order_count
    cuda = None
    if backend == "cuda":
        from r28_exchange_cuda import CudaShell
        cuda = CudaShell(bound, size)
    batch = batch or (262144 if cuda else 512)
    binding = domain_binding(bound.model)
    fingerprint = hashlib.sha256(json.dumps([2, binding, bound.tiles.tolist(),
        size, target], separators=(",", ":")).encode()).hexdigest()
    connection = sqlite3.connect(database, timeout=30)
    connection.execute("CREATE TABLE IF NOT EXISTS scope (key TEXT PRIMARY KEY, state TEXT NOT NULL)")
    connection.execute("CREATE TABLE IF NOT EXISTS optima (key TEXT PRIMARY KEY, result TEXT NOT NULL)")
    state = dict(offset=0, order_offset=0, tested=0, bounded=0, exact=0, reused=0, illegal=0, hits=0, best=None)
    from r28_exchange_rotations import RotationOracle
    oracle = RotationOracle(bound.model)
    from r29_move_cores import load_cores, permitted_sets
    cores = load_cores(connection, bound.model, board, target) if not cuda else []
    start = time.monotonic()
    def finish():
        return dict(state, total=total, complete=state["tested"]==total, objective="matched_edges", target=target)
    try:
        while True:
            # A single transaction owns the next batch. A second writer reads the
            # committed cursor after acquiring the lock; a crash rolls back both
            # its unfinished progress and cache inserts.
            connection.execute("BEGIN IMMEDIATE")
            saved = connection.execute("SELECT state FROM scope WHERE key=?", (fingerprint,)).fetchone()
            if saved:
                state = json.loads(saved[0])
            if (state["offset"] >= combinations or (seconds and time.monotonic()-start>=seconds)
                    or (stop_on_hit and state["best"])):
                connection.commit()
                break
            if permutations is None:
                # Large distances have too many orders for one checkpoint.
                # Seek directly to the selected piece ordering inside this combination.
                count = 1
                first_order = state.get("order_offset",0)
                end_order = min(first_order+min(batch,128),order_count)
                orders = (derangement_unrank(size,j) for j in range(first_order,end_order))
            else:
                count = min(batch, combinations-state["offset"])
                orders = permutations
                end_order = order_count
            cells = None if cuda else combination_batch(movable,size,state["offset"],count)
            if cores:
                cells = cells[permitted_sets(cells, cores)]
                state["core_skips"] = state.get("core_skips",0) + (count-len(cells)) * (end_order - (first_order if permutations is None else 0))
            for order in orders:
                if cuda:
                    survivors = cuda.survivors(state["offset"], count, order, target)
                    if survivors and np.any(bound.bounds(survivors, order)<target):
                        raise RuntimeError("CUDA and CPU exchange bounds disagree.")
                else:
                    survivors = cells[bound.bounds(cells, order)>=target] if len(cells) else []
                state["tested"] += count
                state["bounded"] += count-len(survivors)
                for moved in survivors:
                    seed = bound.witness_seed(moved, order)
                    if seed is None:
                        state["illegal"] += 1
                        continue
                    perm_key = binding + ":" + ",".join(map(str,seed//3))
                    prior = connection.execute("SELECT result FROM optima WHERE key=?", (perm_key,)).fetchone()
                    if prior:
                        item = json.loads(prior[0])
                        state["reused"] += 1
                    else:
                        answer = oracle.optimum(seed)
                        if answer is None:
                            answers = rotation_alternatives(bound.model, seed, k=1, seconds=30)
                            if not answers or not answers[0][1]["exact"]:
                                raise RuntimeError("Rotation optimum unresolved; resume this batch.")
                            candidate, stats = answers[0]
                            optimum, codes = stats["matched_edges"], candidate.tolist()
                        else:
                            optimum, witnesses = answer
                            codes = next(witnesses)
                        bound.model.legal_board(codes)
                        report = problem.validate(codes)
                        if report["matched_edges"] != optimum:
                            raise RuntimeError("Independent rotation scoring failed.")
                        item = dict(score=report["matched_edges"], components=report["line_components"],
                                    gold=report["valid"], codes=codes)
                        connection.execute("INSERT OR IGNORE INTO optima VALUES (?,?)",
                                            (perm_key, json.dumps(item)))
                        state["exact"] += 1
                    if item["score"] >= target:
                        state["hits"] += 1
                        previous = state["best"]
                        if previous is None or (item["score"], -item["components"]) > (previous["score"], -previous["components"]):
                            state["best"] = dict(item, cells=list(map(int,moved)), order=order)
            if end_order==order_count:
                state["offset"] += count
                state["order_offset"] = 0
            else:
                state["order_offset"] = end_order
            connection.execute("INSERT OR REPLACE INTO scope VALUES (?,?)", (fingerprint, json.dumps(state)))
            connection.commit()
            if stop_on_hit and state["best"]:
                break
        if state["offset"] == combinations and state["tested"] != total:
            raise RuntimeError("Inconsistent exchange cursor.")
        return dict(finish(), rotation_work=oracle.work)
    finally:
        connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', default='data/tiles.json')
    parser.add_argument('--incumbent', required=True)
    parser.add_argument('--distance', type=int, default=3)
    parser.add_argument('--target', type=int, default=208, help='Matched-edge threshold. At 240 check the returned component count for Gold.')
    parser.add_argument('--database', required=True, help='SQLite checkpoint, reusable across centers.')
    parser.add_argument('--seconds', type=float, default=0, help='0 runs the whole shell.')
    parser.add_argument('--backend', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--stop-on-hit', action='store_true', help='Stop after a checkpoint contains a seam-threshold witness; multiple loops qualify for this command.')
    args = parser.parse_args()
    state = run_shell(SearchProblem(read_json(args.tiles)), read_json(args.incumbent)['codes'],
                      args.distance, args.target, args.database, seconds=args.seconds,
                      stop_on_hit=args.stop_on_hit, backend=args.backend)
    print(json.dumps(state))


if __name__ == '__main__':
    main()
