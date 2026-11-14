"""SQLite-backed regional enumeration, boundary indexes and empty join suffixes."""
from collections import defaultdict
from contextlib import closing
import hashlib
import json
import sqlite3

from r27_structure_filters import bits
from r27_structure_region import _strand_key
from validator import reverse_mask


def packed(value):
    return json.dumps(value, separators=(',', ':'))


def signature(value):
    return hashlib.sha256(packed(value).encode()).hexdigest()


def _row(value):
    used, ports, pairs = json.loads(value)
    return used, tuple(ports), tuple(map(tuple, pairs))


class RegionalCache:
    """Immutable row bindings plus an atomic DFS cursor per puzzle/region/domain."""

    def __init__(self, path):
        self.db = sqlite3.connect(path, timeout=30)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS r29_region (
                id TEXT PRIMARY KEY, cursor TEXT, done INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS r29_row (
                region TEXT, pattern TEXT, n TEXT NOT NULL, example TEXT NOT NULL,
                PRIMARY KEY(region,pattern)
            );
            CREATE TABLE IF NOT EXISTS r29_index (
                region TEXT, edge INTEGER, labels TEXT NOT NULL,
                PRIMARY KEY(region,edge)
            );
            CREATE TABLE IF NOT EXISTS r29_empty (id TEXT PRIMARY KEY);
        """)

    def close(self):
        self.db.close()

    def binding(self, problem, part, ds):
        return signature(['regional-history-v1', problem.fingerprint, [[c, ds[c]] for c in part]])

    def table(self, problem, part, ds, rank, budget, work):
        binding = self.binding(problem, part, ds)
        entry = self.db.execute('SELECT cursor,done FROM r29_region WHERE id=?', (binding,)).fetchone()
        if not entry or not entry[1]:
            if not entry:
                work.counters['table_starts'] += 1
                work.counters['region_nodes'] += 1
            cursor = json.loads(entry[0]) if entry else [[], [0]]
            done, cursor = self._advance(problem, part, ds, rank, binding, cursor, budget, work)
            self.db.execute('INSERT OR REPLACE INTO r29_region VALUES (?,?,?)',
                            (binding, packed(cursor), int(done)))
            if not done:
                return binding, None
        else:
            work.counters['table_hits'] += 1
        query = 'SELECT pattern,n,example FROM r29_row WHERE region=?'
        return binding, {_row(key): (int(n), tuple(map(tuple, json.loads(ex))))
                         for key, n, ex in self.db.execute(query, (binding,))}

    def _advance(self, problem, part, ds, rank, binding, cursor, budget, work):
        order = sorted(part, key=lambda c: (ds[c].bit_count(), c))
        choices = [tuple(bits(ds[c])) for c in order]
        inside = set(part)
        near = defaultdict(list)
        for e, (a, sa, b, sb) in enumerate(problem.edges):
            if a in inside:
                near[a].append((sa, b, sb, e, False))
            if b in inside:
                near[b].append((sb, a, sa, e, True))
        path, next_index = cursor
        assignment = dict(zip(order, path))
        used = sum(1 << (v // 3) for v in path)
        increments = {}
        for _ in range(budget):
            if not next_index:
                break
            depth = len(path)
            if depth == len(order):
                key = _strand_key(problem, near, assignment, inside, rank, False, work, used)
                if key is not None:
                    if key not in increments:
                        increments[key] = [0, tuple(assignment.items())]
                    increments[key][0] += 1
            elif next_index[-1] < len(choices[depth]):
                value = choices[depth][next_index[-1]]
                next_index[-1] += 1
                c = order[depth]
                tile = 1 << (value // 3)
                work.counters['placement_choices'] += 1
                if used & tile or any(b in assignment and
                        reverse_mask(int(problem.masks[value, s])) != int(problem.masks[assignment[b], t])
                        for s, b, t, _, _ in near[c]):
                    continue
                assignment[c] = value
                path.append(value)
                next_index.append(0)
                used |= tile
                work.counters['region_nodes'] += 1
                continue
            next_index.pop()
            if path:
                del assignment[order[depth - 1]]
                used ^= 1 << (path.pop() // 3)
        for key, (count, example) in increments.items():
            params = binding, packed(key)
            previous = self.db.execute('SELECT n FROM r29_row WHERE region=? AND pattern=?', params).fetchone()
            if previous:
                self.db.execute('UPDATE r29_row SET n=? WHERE region=? AND pattern=?',
                                (str(int(previous[0]) + count), *params))
            else:
                self.db.execute('INSERT INTO r29_row VALUES (?,?,?,?)', (*params, str(count), packed(example)))
        return not next_index, [path, next_index]

    def labels(self, binding, edge, build, work):
        item = self.db.execute('SELECT labels FROM r29_index WHERE region=? AND edge=?', (binding, edge)).fetchone()
        if item:
            work.counters['index_hits'] += 1
            return tuple(json.loads(item[0]))
        labels = tuple(build())
        self.db.execute('INSERT INTO r29_index VALUES (?,?,?)', (binding, edge, packed(labels)))
        work.counters['index_builds'] += 1
        return labels

    def empty(self, key):
        return self.db.execute('SELECT 1 FROM r29_empty WHERE id=?', (key,)).fetchone() is not None

    def record_empty(self, key):
        self.db.execute('INSERT OR IGNORE INTO r29_empty VALUES (?)', (key,))


def cached_page(path, page, *args, **kwargs):
    with closing(RegionalCache(path)) as store, store.db:
        store.db.execute('BEGIN IMMEDIATE')
        return page(*args, store=store, **kwargs)
