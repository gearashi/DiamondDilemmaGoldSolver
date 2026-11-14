"""Search a chain of one-swap neighborhoods for a higher seam score.

Each center expands every pair and its nine destination rotation pairs, keeping
other cells fixed. SQLite retains the exact oriented centers already expanded.
A finite frontier width and a score floor make this a heuristic plateau walk.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from search_problem import SearchProblem


class PairScores:
    def __init__(self, model):
        self.model = model
        self.left, self.right = np.triu_indices(model.n, 1)
        # The lookup also supports fixtures with multiple seams between two cells.
        self.seams = []
        for a, sa, b, sb in model.problem.edges:
            if a > b:
                a, sa, b, sb = b, sb, a, sa
            i = a * (2*model.n-a-1)//2 + b-a-1
            self.seams.append((i, a, sa, b, sb))
        self.seams = np.asarray(self.seams, dtype=int).reshape(-1, 5).T

    def all(self, values, indices=None):
        """Evaluate every pair or only the requested pair indices."""
        model = self.model
        board = model.legal_board(values)
        n = model.n
        match = model.matches(board)
        scores = np.zeros((n, 3*n), dtype=np.int16)
        for s in range(3):
            others = model.problem.neighbors[:, s]
            valid = np.flatnonzero(others >= 0)
            side = model.problem.sides[valid, s]
            target = model.reverse[model.masks[board[others[valid]], side]]
            scores[valid] += model.masks[None, :, s] == target[:, None]
        scores -= scores[np.arange(n), board, None]
        a, b = self.left, self.right
        i, u, su, v, sv = self.seams
        if indices is not None:
            a, b = a[indices], b[indices]
            local = np.full(len(self.left), -1, dtype=int)
            local[indices] = np.arange(len(indices))
            keep = local[i] >= 0
            i = local[i[keep]]
            u, su, v, sv = u[keep], su[keep], v[keep], sv[keep]
        x = board[b, None]//3*3 + np.arange(3)
        y = board[a, None]//3*3 + np.arange(3)
        result = int(match.sum()) + scores[a[:, None], x][:, :, None] + scores[b[:, None], y][:, None, :]
        old_u = model.masks[board[u], su]
        old_v = model.reverse[model.masks[board[v], sv]]
        new_u = model.masks[x[i], su[:, None]]
        new_v = model.reverse[model.masks[y[i], sv[:, None]]]
        correction = (new_u[:, :, None] == new_v[:, None, :]).astype(np.int16)
        correction += (old_u == old_v)[:, None, None]
        correction -= (new_u == old_v[:, None])[:, :, None]
        correction -= (new_v == old_u[:, None])[:, None, :]
        np.add.at(result, i, correction)
        result[~(model.legal[a[:, None], x][:, :, None] & model.legal[b[:, None], y][:, None, :])] = -1
        return result


def packed(board):
    return np.asarray(board, dtype='<u2').tobytes()


def walk(model, starts, database, output, *, goal, floor, width, limit, seconds=0):
    max_score = len(model.a)
    gold_goal = goal == max_score
    if width < 1 or limit < 1 or seconds < 0 or not 0 <= floor <= goal <= max_score:
        raise ValueError("Invalid walk limits or target")
    scope = json.dumps([1, model.problem.fingerprint, model.problem.domains, goal, floor, width], separators=(',', ':'))
    signature = hashlib.sha256(scope.encode()).digest()
    evaluator = PairScores(model)
    db = sqlite3.connect(database, timeout=30)
    began = time.monotonic()

    def verified(raw, recorded):
        codes = np.frombuffer(raw, dtype='<u2').astype(int).tolist()
        model.legal_board(codes)
        validation = model.problem.validate(codes)
        if not validation['unique_tiles'] or validation['matched_edges'] != recorded:
            raise ValueError('Stored walk witness failed independent validation')
        return codes, validation

    def perfect(raw):
        previous = db.execute('SELECT gold FROM perfect_checks WHERE board=?', (raw,)).fetchone()
        if previous is not None:
            return bool(previous[0])
        _, validation = verified(raw, max_score)
        gold = bool(validation['valid'])
        db.execute('INSERT INTO perfect_checks VALUES (?,?)', (raw, int(gold)))
        return gold

    def gold_winner():
        # Older walk databases may already contain a multi-loop 240 and another
        # equally scoring Gold board. Check only previously unclassified centers.
        saved = db.execute('SELECT p.board FROM perfect_checks p JOIN centers c ON p.board=c.board '
                           'WHERE p.gold=1 AND c.score=? LIMIT 1', (max_score,)).fetchone()
        if saved is not None:
            return saved
        unknown = db.execute('SELECT c.board FROM centers c WHERE c.score=? AND NOT EXISTS '
                             '(SELECT 1 FROM perfect_checks p WHERE p.board=c.board)', (max_score,))
        for (raw,) in unknown:
            if perfect(raw):
                return (raw,)
        return None

    def insert(raw, score):
        return db.execute('INSERT OR IGNORE INTO centers VALUES (?,?,0,?)',
                          (raw, score, hashlib.sha256(raw).digest())).rowcount

    try:
        with db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('CREATE TABLE IF NOT EXISTS scope (id INTEGER PRIMARY KEY CHECK(id=1), binding BLOB NOT NULL)')
            db.execute('INSERT OR IGNORE INTO scope VALUES (1,?)', (signature,))
            if db.execute('SELECT binding FROM scope').fetchone()[0] != signature:
                raise ValueError('Walk database belongs to another puzzle or policy')
            db.execute('CREATE TABLE IF NOT EXISTS centers (board BLOB PRIMARY KEY, score INTEGER NOT NULL, done INTEGER NOT NULL, priority BLOB NOT NULL) WITHOUT ROWID')
            db.execute('CREATE INDEX IF NOT EXISTS scores ON centers(score DESC)')
            db.execute('CREATE INDEX IF NOT EXISTS queue ON centers(done,score DESC,priority,board)')
            db.execute('CREATE TABLE IF NOT EXISTS perfect_checks (board BLOB PRIMARY KEY, gold INTEGER NOT NULL CHECK(gold IN (0,1))) WITHOUT ROWID')
            db.execute('CREATE INDEX IF NOT EXISTS solved_walk ON perfect_checks(gold) WHERE gold=1')
            for values in starts:
                board = model.legal_board(values)
                insert(packed(board),int(model.matches(board).sum()))
            if db.execute('SELECT 1 FROM centers LIMIT 1').fetchone() is None:
                raise ValueError('A new walk needs at least one starting board')
        expanded = reused = 0
        while expanded < limit and (not seconds or time.monotonic()-began < seconds):
            with db:
                db.execute('BEGIN IMMEDIATE')
                if gold_goal:
                    if gold_winner():
                        break
                elif db.execute('SELECT 1 FROM centers WHERE score>=? LIMIT 1',(goal,)).fetchone():
                    break
                row = db.execute('SELECT board,score FROM centers WHERE done=0 ORDER BY score DESC,priority,board LIMIT 1').fetchone()
                if row is None:
                    break
                center = np.frombuffer(row[0], dtype='<u2')
                grid = evaluator.all(center)
                eligible = np.flatnonzero(grid.ravel() >= floor)
                rng = np.random.default_rng(int.from_bytes(hashlib.sha256(row[0]).digest()[:8], 'little'))
                rng.shuffle(eligible)
                order = eligible[np.argsort(-grid.ravel()[eligible], kind='stable')]
                added = 0
                for flat in order:
                    p, r, s = np.unravel_index(flat, grid.shape)
                    score = int(grid[p, r, s])
                    # A tied Gold witness could lie beyond the ordinary frontier
                    # width. Test every perfect candidate from this expansion.
                    if added>=width and not (gold_goal and score==max_score):
                        break
                    a, b = evaluator.left[p], evaluator.right[p]
                    child = center.copy()
                    child[a], child[b] = center[b]//3*3+r, center[a]//3*3+s
                    raw = packed(child)
                    is_gold = gold_goal and score==max_score and perfect(raw)
                    if added<width or is_gold:
                        changed = insert(raw,score)
                        added += changed
                        reused += 1-changed
                    if is_gold:
                        break
                db.execute('UPDATE centers SET done=1 WHERE board=?', (row[0],))
                expanded += 1
        # Serialize selecting and atomically publishing the current best across
        # every writer sharing this database. A slower publisher cannot replace
        # a newer winning board with a lower score or an equally scoring non-Gold.
        with db:
            db.execute('BEGIN IMMEDIATE')
            winner = gold_winner()
            if winner:
                raw,best = winner[0],max_score
            else:
                raw,best = db.execute('SELECT board,score FROM centers ORDER BY score DESC LIMIT 1').fetchone()
            codes, validation = verified(raw,best)
            if winner and not validation['valid']:
                raise ValueError('Stored Gold witness failed independent validation')
            solved = bool(validation['valid'])
            atomic_json(output, {'codes':codes, 'score':best, 'solved':solved, 'validation':validation})
            counts = dict(db.execute('SELECT done,COUNT(*) FROM centers GROUP BY done'))
        return {'best':best,'solved':solved,'goal_reached':solved if gold_goal else best>=goal,
                'expanded_this_run':expanded,'expanded_total':counts.get(1,0),'frontier':counts.get(0,0),
                'duplicates_seen_before_admission_limit':reused,'scored_per_center':len(evaluator.left)*9,
                'seconds':round(time.monotonic()-began,3),'witness':str(output)}
    finally:
        db.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tiles', type=Path)
    p.add_argument('boards', type=Path, nargs='+')
    p.add_argument('--database', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--goal', type=int, default=208, help='Seam milestone. At 240, require one validated Gold loop.')
    p.add_argument('--floor', type=int, default=205)
    p.add_argument('--width', type=int, default=16)
    p.add_argument('--limit', type=int, default=10000)
    p.add_argument('--seconds', type=float, default=0)
    a = p.parse_args()
    model = BoardModel(SearchProblem(json.loads(a.tiles.read_text(encoding='utf-8-sig'))))
    starts = [json.loads(f.read_text(encoding='utf-8-sig'))['codes'] for f in a.boards]
    print(json.dumps(walk(model, starts, a.database, a.output, goal=a.goal, floor=a.floor, width=a.width, limit=a.limit, seconds=a.seconds)))


if __name__ == '__main__':
    main()
