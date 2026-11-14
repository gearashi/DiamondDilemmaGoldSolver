"""Threshold-independent physical-swap graph with resumable breadth-first routes."""
import hashlib
import itertools
import json
import sqlite3
import time
import zlib

import numpy as np

from r27_incumbent import BoardModel, rotation_alternatives
from r28_exchange import ExchangeBound, domain_binding
from r28_exchange_rotations import RotationOracle


def identity(board):
    return ",".join(str(int(code) // 3) for code in board)


class SwapGraph:
    def __init__(self, db, model):
        self.db = db
        self.model = model
        self.binding = domain_binding(model)
        self.oracle = RotationOracle(model)
        self.pairs = np.array(list(itertools.combinations(model.free, 2)), dtype=int).reshape(-1, 2)
        db.executescript("""
            CREATE TABLE IF NOT EXISTS optima(key TEXT PRIMARY KEY, result TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS walk_bounds(board TEXT PRIMARY KEY, lowest INTEGER, bounds BLOB);
            CREATE TABLE IF NOT EXISTS walk_edges(board TEXT, neighbor TEXT, PRIMARY KEY(board,neighbor));
            CREATE TABLE IF NOT EXISTS walk_imported(board TEXT,floor INTEGER,PRIMARY KEY(board,floor));
            CREATE TABLE IF NOT EXISTS walk_roots(scope TEXT PRIMARY KEY, best TEXT);
            CREATE TABLE IF NOT EXISTS walk_queue(scope TEXT, board TEXT, parent TEXT, depth INTEGER,
                closed INTEGER DEFAULT 0, PRIMARY KEY(scope,board));
            CREATE INDEX IF NOT EXISTS walk_fifo ON walk_queue(scope,closed,depth);
            CREATE TABLE IF NOT EXISTS walk_ties(board TEXT PRIMARY KEY, requested INTEGER);
        """)
        self.legacy_scopes = []
        if db.execute("SELECT 1 FROM sqlite_master WHERE name='centers'").fetchone():
            for scope, in db.execute("SELECT DISTINCT scope FROM centers WHERE scope LIKE ?", (self.binding+'/%',)):
                fields = scope.split('/')
                if len(fields)==4 and fields[1].isdigit():
                    self.legacy_scopes.append((int(fields[1]),scope))
        if self.legacy_scopes:
            db.execute("CREATE INDEX IF NOT EXISTS optimum_physical ON optimum(perm,scope)")
        self.work = dict(centers=0, star=0, solved=0, reused=0, exact_fallback=0,
                         legacy_centers=0, star_evaluations=0, cached_stars=0, reused_arcs=0, tie_solves=0)

    def key(self, board):
        return self.binding + ":" + identity(board)

    def optimum(self, key, seed, floor, tie_limit=1):
        db = self.db
        saved = db.execute("SELECT result FROM optima WHERE key=?", (key,)).fetchone()
        known = json.loads(saved[0]) if saved else None
        if known is None:
            # v1.9 kept null witnesses for some exact scores below its active floor.
            old = db.execute("SELECT 1 FROM sqlite_master WHERE name='optimum'").fetchone()
            if old:
                legacy = db.execute("SELECT score,comps,codes FROM optimum WHERE perm=? AND scope LIKE ? "
                                    "ORDER BY comps IS NULL,comps LIMIT 1",
                                    (key.split(':',1)[1], self.binding + '/%')).fetchone()
                if legacy:
                    score, components, encoded = legacy
                    codes = json.loads(encoded) if encoded else None
                    if codes is not None:
                        self.check(key, codes, score)
                    known = dict(score=score, components=components, codes=codes)
        if not saved and known and known.get('codes'):
            known['gold'] = self.model.problem.validate(known['codes'])['valid']
            db.execute("INSERT OR IGNORE INTO optima VALUES (?,?)", (key,json.dumps(known)))
        if known and known['score'] < floor:
            self.work['reused'] += 1
            return known
        sampled = db.execute("SELECT requested FROM walk_ties WHERE board=?", (key,)).fetchone()
        if known and known.get('codes') and tie_limit <= (sampled[0] if sampled else 1):
            self.work['reused'] += 1
            return known
        if known and known.get('codes'):
            self.work['tie_solves'] += 1
        result = self.oracle.optimum(seed, ties=tie_limit > 1, limit=tie_limit)
        tried = tie_limit
        if result is None:
            self.work['exact_fallback'] += 1
            choices = rotation_alternatives(self.model, seed, k=1, seconds=120)
            if not choices or not choices[0][1]['exact']:
                raise RuntimeError("Unresolved rotation optimum; resume this batch.")
            score = choices[0][1]['matched_edges']
            boards = [choices[0][0].tolist()]
            tried = 1
        else:
            score, boards = result
        if known and score != known['score']:
            raise RuntimeError("Cached and recomputed rotation optima disagree.")
        best = known if known and known.get('codes') else None
        for codes in boards:
            report = self.check(key, codes, score)
            if best is None or report['line_components'] < best['components']:
                best = dict(score=score, components=report['line_components'], codes=codes, gold=report['valid'])
        if best is None:
            raise RuntimeError("Exact rotation solver returned no witness.")
        if 'gold' not in best:
            best['gold'] = self.check(key, best['codes'], score)['valid']
        db.execute("INSERT OR REPLACE INTO optima VALUES (?,?)", (key, json.dumps(best)))
        db.execute("INSERT OR REPLACE INTO walk_ties VALUES (?,?)",
                   (key, max(tried, sampled[0] if sampled else 0)))
        self.work['solved'] += 1
        return best

    def check(self, key, codes, score):
        self.model.legal_board(codes)
        report = self.model.problem.validate(codes)
        if self.key(codes) != key or report['matched_edges'] != score:
            raise RuntimeError("Cached physical identity or independently scored witness disagrees.")
        return report

    def neighbors(self, key, board, floor):
        db = self.db
        cached = db.execute("SELECT lowest,bounds FROM walk_bounds WHERE board=?", (key,)).fetchone()
        old_floor = cached[0] if cached else len(self.model.problem.edges) + 1
        imported = db.execute("SELECT 1 FROM walk_imported WHERE board=? AND floor=?",(key,floor)).fetchone()
        for legacy_floor, scope in sorted(self.legacy_scopes):
            if legacy_floor != floor or old_floor <= floor:
                continue
            if imported:
                self.work['reused_arcs'] += 1
                break
            perm = key.split(':', 1)[1]
            if not db.execute("SELECT 1 FROM centers WHERE scope=? AND perm=? AND closed=1", (scope,perm)).fetchone():
                continue
            members = dict(db.execute("SELECT perm,codes FROM centers WHERE scope=?", (scope,)))
            tiles = perm.split(',')
            for a,b in self.pairs:
                tiles[a],tiles[b] = tiles[b],tiles[a]
                child_perm = ','.join(tiles)
                tiles[a],tiles[b] = tiles[b],tiles[a]
                if child_perm in members:
                    child = self.binding + ':' + child_perm
                    self.optimum(child,json.loads(members[child_perm]),floor)
                    db.executemany("INSERT OR IGNORE INTO walk_edges VALUES (?,?)",((key,child),(child,key)))
            db.execute("INSERT OR IGNORE INTO walk_imported VALUES (?,?)",(key,floor))
            self.work['legacy_centers'] += 1
            break
        else:
            legacy_floor = None
        if floor < old_floor and legacy_floor != floor:
            bound = ExchangeBound(self.model, board)
            if cached:
                upper = np.frombuffer(zlib.decompress(cached[1]), dtype='<i4')
                if len(upper) != len(self.pairs):
                    raise RuntimeError("Invalid cached swap-bound vector.")
                self.work['cached_stars'] += len(upper)
            else:
                chunks = [bound.bounds(self.pairs[i:i+512], (1,0)) for i in range(0,len(self.pairs),512)]
                upper = np.concatenate(chunks).astype('<i4') if chunks else np.array([], dtype='<i4')
                self.work['star_evaluations'] += len(upper)
            self.work['star'] += int((upper < floor).sum())
            for a,b in self.pairs[(upper >= floor) & (upper < old_floor)]:
                seed = bound.witness_seed((int(a),int(b)), (1,0))
                if seed is None:
                    continue
                child = self.key(seed)
                item = self.optimum(child, seed, floor)
                # Store the edge even when below this floor. On a lower-floor pass
                # its star was already considered and will lie outside the new band.
                if item.get('codes') is None:
                    # Archive the score-only legacy hit, filling its witness on demand.
                    db.execute("INSERT OR IGNORE INTO walk_legacy VALUES (?,?,?)",
                               (child, item['score'], json.dumps(seed.tolist())))
                db.executemany("INSERT OR IGNORE INTO walk_edges VALUES (?,?)", ((key,child),(child,key)))
            if cached:
                db.execute("UPDATE walk_bounds SET lowest=? WHERE board=?", (floor,key))
            else:
                db.execute("INSERT INTO walk_bounds VALUES (?,?,?)",(key,floor,zlib.compress(upper.tobytes())))
        else:
            self.work['reused_arcs'] += 1
        for child, raw, score, encoded_seed in db.execute(
                "SELECT e.neighbor,o.result,l.score,l.seed FROM walk_edges e "
                "LEFT JOIN optima o ON o.key=e.neighbor LEFT JOIN walk_legacy l ON l.board=e.neighbor "
                "WHERE e.board=? ORDER BY e.neighbor", (key,)):
            if raw:
                item = json.loads(raw)
            elif score is not None and score >= floor:
                item = self.optimum(child,json.loads(encoded_seed),floor)
            else:
                continue
            if item['score'] >= floor:
                yield child, item


def _route(db, scope, key):
    path = []
    while key:
        path.append(key)
        key, = db.execute("SELECT parent FROM walk_queue WHERE scope=? AND board=?", (scope,key)).fetchone()
    return list(reversed(path))


def walk(problem, codes, database, floor, goal, *, max_centers=20, seconds=0, tie_limit=8, widest=False,
         max_swaps=None, potentials=True):
    """A BFS at each floor gives a shortest path; descending closed floors certify the bottleneck."""
    if not 0 <= floor < goal <= len(problem.edges) or tie_limit < 1 or max_centers < 0 or seconds < 0:
        raise ValueError("Invalid plateau range or budget.")
    if max_swaps is not None and (type(max_swaps) is not int or max_swaps<0):
        raise ValueError("Nonnegative swap limit required.")
    started = time.monotonic()
    model = BoardModel(problem)
    start = model.legal_board(codes)
    db = sqlite3.connect(database, timeout=120)
    try:
        graph = SwapGraph(db,model)
        db.execute("CREATE TABLE IF NOT EXISTS walk_legacy(board TEXT PRIMARY KEY,score INTEGER,seed TEXT)")
        db.execute("BEGIN IMMEDIATE")
        root = graph.key(start)
        initial = graph.optimum(root,start,0,tie_limit)
        if initial['score'] < floor:
            raise ValueError("The best rotation of the starting board is below the floor.")
        db.commit()
        levels = range(min(initial['score'],goal-1),floor-1,-1) if widest else [floor]
        for level in levels:
            binding = f'{root}/{level}/{goal}'
            if max_swaps is not None:
                binding += f'/depth/{max_swaps}'
                graph.work.setdefault('potential_closed',0)
                graph.work.setdefault('depth_closed',0)
            scope = hashlib.sha256(binding.encode()).hexdigest()
            db.execute("BEGIN IMMEDIATE")
            db.execute("INSERT OR IGNORE INTO walk_queue(scope,board,parent,depth) VALUES (?,?,NULL,0)", (scope,root))
            db.execute("INSERT OR IGNORE INTO walk_roots VALUES (?,?)",(scope,root))
            db.commit()
            while True:
                db.execute("BEGIN IMMEDIATE")
                best_key, = db.execute("SELECT best FROM walk_roots WHERE scope=?", (scope,)).fetchone()
                def get(key):
                    return json.loads(db.execute("SELECT result FROM optima WHERE key=?",(key,)).fetchone()[0])
                best = get(best_key)
                row = db.execute("SELECT board,depth FROM walk_queue WHERE scope=? AND closed=0 "
                                 "ORDER BY depth,rowid LIMIT 1",(scope,)).fetchone()
                expired = seconds and time.monotonic()-started >= seconds
                if not row or best['score'] >= goal or graph.work['centers'] >= max_centers or expired:
                    db.commit()
                    break
                key,depth = row
                board = get(key)['codes']
                skip = max_swaps is not None and depth>=max_swaps
                if skip:
                    graph.work['depth_closed'] += 1
                elif max_swaps is not None and potentials:
                    from r30_bounds import budget_upper, defect_packings
                    bound = ExchangeBound(model,board)
                    skip = budget_upper(bound,defect_packings(bound),max_swaps-depth)<goal
                    graph.work['potential_closed'] += int(skip)
                for child,item in (() if skip else graph.neighbors(key,board,level)):
                    db.execute("INSERT OR IGNORE INTO walk_queue(scope,board,parent,depth) VALUES (?,?,?,?)",
                               (scope,child,key,depth+1))
                    if item['score'] >= best['score'] and tie_limit > 1:
                        item = graph.optimum(child,item['codes'],level,tie_limit)
                    if (item['score'], -item['components']) > (best['score'], -best['components']):
                        best_key,best = child,item
                db.execute("UPDATE walk_roots SET best=? WHERE scope=?", (best_key,scope))
                db.execute("UPDATE walk_queue SET closed=1 WHERE scope=? AND board=?", (scope,key))
                db.commit()
                graph.work['centers'] += 1
            if row or best['score'] >= goal:
                break
        db.execute("BEGIN IMMEDIATE")
        best = graph.optimum(best_key, best['codes'], 0, tie_limit)
        counts = dict(db.execute("SELECT closed,count(*) FROM walk_queue WHERE scope=? GROUP BY closed",(scope,)))
        path = _route(db,scope,best_key)
        swaps = []
        for before, after in zip(path,path[1:]):
            a,b = [key.split(':',1)[1].split(',') for key in (before,after)]
            swaps.append([i for i,(x,y) in enumerate(zip(a,b)) if x!=y])
        db.commit()
        reached = best['score'] >= goal
        return dict(graph.work, objective="matched_edges", target=goal, score=best['score'],
                    components=best['components'], codes=best['codes'], gold=problem.validate(best['codes'])['valid'],
                    floor=level, closed=counts.get(1,0), open=counts.get(0,0), swaps=swaps, reached=reached,
                    shortest_at_floor=len(swaps) if reached else None,
                    widest_floor=(initial['score'] if len(path)==1 else level) if widest and reached else None,
                    rotation_work=graph.oracle.work)
    finally:
        db.close()
