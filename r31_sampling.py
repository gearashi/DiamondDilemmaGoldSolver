"""Reusable native sampler prefixes and disjoint fresh-board jobs."""
import hashlib
import json
import math
import sqlite3
import threading
import time
import uuid
from contextlib import closing, contextmanager
from pathlib import Path

import numpy as np


def ranked_start(model, seed):
    """Unrank a scrambled index into unrestricted physical boards and rotations."""
    from r30_fresh import fresh_start
    if seed < 0:
        raise ValueError("Start index must be nonnegative")
    if not model.legal.all():
        return fresh_start(model, np.random.default_rng(seed))
    n = model.n
    rotations = 3 ** n
    size = math.factorial(n) * rotations
    if seed >= size:
        raise ValueError("Unrestricted initial board ranks exhausted")
    salt = model.problem.fingerprint.encode()
    width = (size.bit_length() + 7) // 8
    a, b = [int.from_bytes(hashlib.shake_256(salt + key).digest(width), "big") % size
            for key in (b"initial-stride-v1", b"initial-offset-v1")]
    while math.gcd(a, size) != 1:
        a += 1
    perm, orient = divmod((a * seed + b) % size, rotations)
    pool = list(range(n))
    codes = []
    while pool:
        perm, j = divmod(perm, len(pool))
        orient, r = divmod(orient, 3)
        codes.append(3 * pool.pop(j) + r)
    return np.array(codes, dtype=int)


def _open(database):
    return sqlite3.connect(database, timeout=60)


def _schema(conn):
    conn.execute("""CREATE TABLE IF NOT EXISTS r31_jobs (
        scope TEXT, seed INTEGER, owner TEXT, expires REAL, result TEXT, gold INTEGER,
        PRIMARY KEY(scope, seed))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS r31_best (
        scope TEXT PRIMARY KEY, score INTEGER, components INTEGER, gold INTEGER, seed INTEGER, result TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS r31_pending (
        scope TEXT, seed INTEGER, state TEXT, PRIMARY KEY(scope,seed))""")
    conn.execute("CREATE INDEX IF NOT EXISTS r31_job_gold ON r31_jobs(scope, gold)")
    conn.execute("""CREATE TABLE IF NOT EXISTS r31_sampler (
        scope TEXT, seed INTEGER, launches INTEGER, state TEXT,
        PRIMARY KEY(scope, seed, launches))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS r34_seed_pages (
        scope TEXT, first_seed INTEGER, best_seed INTEGER NOT NULL,
        PRIMARY KEY(scope, first_seed))""")
    conn.execute("""CREATE INDEX IF NOT EXISTS r34_unfinished_seeds ON r31_jobs(scope, expires, seed)
                    WHERE result IS NULL""")
    conn.execute("""CREATE TABLE IF NOT EXISTS r34_final_archives (
        request TEXT, dispatches INTEGER, result TEXT NOT NULL, checksum TEXT NOT NULL,
        PRIMARY KEY(request,dispatches))""")
    conn.commit()


def _claim(conn, scope, seed, ttl):
    owner = uuid.uuid4().hex
    now = time.time()
    with conn:
        changed = conn.execute("""INSERT INTO r31_jobs VALUES(?,?,?,?,NULL,0)
            ON CONFLICT(scope,seed) DO UPDATE SET owner=excluded.owner,expires=excluded.expires
            WHERE result IS NULL AND expires<=?""", (scope, seed, owner, now + ttl, now))
    return owner if changed.rowcount else None


@contextmanager
def _heartbeat(database, scope, seed, owner, ttl):
    stop = threading.Event()
    def beat():
        with closing(_open(database)) as conn:
            while not stop.wait(ttl / 3):
                with conn:
                    row = conn.execute("""UPDATE r31_jobs SET expires=?
                        WHERE scope=? AND seed=? AND owner=? AND result IS NULL""",
                        (time.time() + ttl, scope, seed, owner))
                if not row.rowcount:
                    return
    thread = threading.Thread(target=beat, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join()


class _ClaimClosed(InterruptedError):
    pass


SEEDS_PER_PAGE = 128


def _completed_seed_best(conn, scope, start, stop):
    """Return the winning seed from a wholly committed interval of at most one page."""
    first = start - start % SEEDS_PER_PAGE
    full = start == first and stop - start == SEEDS_PER_PAGE
    known = conn.execute("SELECT best_seed FROM r34_seed_pages WHERE scope=? AND first_seed=?",
                         (scope, first)).fetchone()
    if known and start <= known[0] < stop:
        return known[0]
    if not known:
        count = conn.execute("""SELECT count(*) FROM r31_jobs
            WHERE scope=? AND seed>=? AND seed<? AND result IS NOT NULL""", (scope, start, stop)).fetchone()[0]
        if count != stop - start:
            return None
    best = conn.execute("""SELECT seed FROM r31_jobs WHERE scope=? AND seed>=? AND seed<?
        AND result IS NOT NULL ORDER BY gold DESC, json_extract(result,'$.score') DESC,
        json_extract(result,'$.components'), seed LIMIT 1""", (scope, start, stop)).fetchone()[0]
    if full:
        with conn:
            conn.execute("INSERT OR IGNORE INTO r34_seed_pages VALUES(?,?,?)", (scope, first, best))
    return best


def run_seeds(database, scope, seed, starts, run_one, *, ttl=120, progress=None, checkpointed=False, output=None):
    """Claim whole disjoint seeds, publish fenced results, and stop dispatch on shared Gold."""
    if starts < 0 or seed < 0 or not math.isfinite(ttl) or ttl <= 0:
        raise ValueError("Invalid range or lease")
    database = Path(database)
    database.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    busy = False
    best = None
    def consider(row):
        nonlocal best
        def key(v):
            return bool(v['gold']), v['score'], -v['components'], -v['seed']
        if best is None or key(row) > key(best):
            best = row
    with closing(_open(database)) as conn:
        _schema(conn)
        def publish():
            if output:
                from io_utils import atomic_json
                top = conn.execute("SELECT result FROM r31_best WHERE scope=?", (scope,)).fetchone()
                if top:
                    item = json.loads(top[0])
                    atomic_json(output, dict(codes=item["codes"], matched_edges=item["score"],
                                line_components=item["components"], valid=bool(item["gold"]), seed=item["seed"]))
        index = seed
        checked_until = seed
        while True:
            gold = conn.execute("SELECT result FROM r31_jobs WHERE scope=? AND gold=1 LIMIT 1", (scope,)).fetchone()
            if gold:
                consider(json.loads(gold[0]))
                break
            if starts and index >= seed + starts:
                if not busy:
                    break
                time.sleep(.1)
                busy = False
                index = checked_until = seed
                continue
            if index >= checked_until:
                if not starts and index > seed:
                    expired = conn.execute("""SELECT min(seed) FROM r31_jobs WHERE scope=? AND expires<=?
                        AND seed>=? AND seed<? AND result IS NULL""", (scope, time.time(), seed, index)).fetchone()[0]
                    if expired is not None:
                        index = expired
                checked_until = (index // SEEDS_PER_PAGE + 1) * SEEDS_PER_PAGE
                if starts:
                    checked_until = min(checked_until, seed + starts)
                winner = _completed_seed_best(conn, scope, index, checked_until)
                if winner is not None:
                    raw = conn.execute("SELECT result FROM r31_jobs WHERE scope=? AND seed=?",
                                       (scope, winner)).fetchone()[0]
                    consider(json.loads(raw))
                    index = checked_until
                    continue
            job = index
            index += 1
            row = conn.execute("SELECT result FROM r31_jobs WHERE scope=? AND seed=?", (scope, job)).fetchone()
            if row and row[0]:
                consider(json.loads(row[0]))
                continue
            owner = _claim(conn, scope, job, ttl)
            if owner is None:
                busy = True
                continue
            try:
                with _heartbeat(database, scope, job, owner, ttl):
                    def save(state):
                        with conn:
                            conn.execute("BEGIN IMMEDIATE")
                            active = conn.execute("SELECT 1 FROM r31_jobs WHERE scope=? AND seed=? AND owner=?",
                                                  (scope, job, owner)).fetchone()
                            gold = conn.execute("SELECT 1 FROM r31_jobs WHERE scope=? AND gold=1", (scope,)).fetchone()
                            if not active or gold:
                                raise _ClaimClosed()
                            conn.execute("INSERT OR REPLACE INTO r31_pending VALUES(?,?,?)",
                                         (scope, job, json.dumps(state)))
                    if checkpointed:
                        pending = conn.execute("SELECT state FROM r31_pending WHERE scope=? AND seed=?",
                                               (scope, job)).fetchone()
                        item = run_one(job, json.loads(pending[0]) if pending else None, save)
                    else:
                        item = run_one(job)
                    item = dict(item, seed=job)
                    with conn:
                        changed = conn.execute("""UPDATE r31_jobs SET owner=NULL,expires=NULL,result=?,gold=?
                            WHERE scope=? AND seed=? AND owner=? AND result IS NULL""",
                            (json.dumps(item), int(item['gold']), scope, job, owner))
                        if changed.rowcount:
                            top = conn.execute("""INSERT INTO r31_best VALUES (?,?,?,?,?,?)
                                ON CONFLICT(scope) DO UPDATE SET score=excluded.score,components=excluded.components,
                                gold=excluded.gold,seed=excluded.seed,result=excluded.result WHERE
                                (excluded.gold,excluded.score,-excluded.components,-excluded.seed)>
                                (r31_best.gold,r31_best.score,-r31_best.components,-r31_best.seed)""",
                                (scope,item["score"],item["components"],int(item["gold"]),job,json.dumps(item)))
                            if top.rowcount:
                                publish()
                            conn.execute("DELETE FROM r31_pending WHERE scope=? AND seed=?", (scope, job))
                if changed.rowcount:
                    done += 1
                    consider(item)
                    if progress:
                        progress(item)
                else:
                    busy = True
            except _ClaimClosed:
                busy = True
            finally:
                with conn:
                    conn.execute("UPDATE r31_jobs SET expires=0 WHERE scope=? AND seed=? AND owner=? AND result IS NULL",
                                 (scope, job, owner))
        sql = "SELECT count(*) FROM r31_jobs WHERE scope=? AND seed>=? AND result IS NOT NULL"
        args = (scope, seed)
        if starts:
            sql += " AND seed<?"
            args += (seed + starts,)
        total = conn.execute(sql, args).fetchone()[0]
        if output:
            with conn:
                conn.execute("BEGIN IMMEDIATE")
                publish()
    return dict(best=best, completed=done, reused=total-done)


@contextmanager
def _prefix_claim(conn, path, scope, seed, dispatches, *, final_request=None):
    """Serialize extensions of one sampler stream while unrelated seeds continue."""
    lock_scope = 'sampling:' + scope
    owner = None
    while owner is None:
        if final_request is None:
            exact = conn.execute(
                'SELECT 1 FROM r31_sampler WHERE scope=? AND seed=? AND launches=?',
                (scope, seed, dispatches)).fetchone()
        else:
            exact = conn.execute('SELECT 1 FROM r34_final_archives WHERE request=? AND dispatches=?',
                                 (final_request, dispatches)).fetchone()
        if exact:
            yield
            return
        owner = _claim(conn, lock_scope, seed, 120)
        if owner is None:
            time.sleep(.1)
    try:
        with _heartbeat(path, lock_scope, seed, owner, 120):
            yield
    finally:
        with conn:
            conn.execute('DELETE FROM r31_jobs WHERE scope=? AND seed=? AND owner=?',
                         (lock_scope, seed, owner))



def _archive_request(problem, opts):
    from constraint_work import digest
    names = ('gpu_sampling_resume_round25.py', 'gpu_sampling_state_round25.py', 'gpu_sampling.py',
             'gpu_incremental_sampling_round26.py', 'constraint_lns.py', 'search_problem.py', 'validator.py',
             'sampling_round24.cu' if opts['backend'] == 'cuda' else 'sampling_round24.wgsl')
    sources = {name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in names}
    return digest([1, problem.fingerprint, [list(row) for row in problem.domains],
                   sorted(problem.fixed.items()), opts, sources])


def _read_final(conn, request, dispatches, replicas, problem):
    from constraint_work import digest
    from gpu_sampling import score_board
    row = conn.execute("SELECT result,checksum FROM r34_final_archives WHERE request=? AND dispatches=?",
                       (request, dispatches)).fetchone()
    if row is None:
        return None
    result = json.loads(row[0])
    if digest([request, dispatches, result]) != row[1]:
        raise ValueError("Corrupt finalized sampling archive.")
    launches = result['reused_dispatches']
    if (type(launches) is not int or not 0 <= launches <= dispatches or
            result['move_proposals'] != launches * replicas * 64 or result['new_proposals'] != 0):
        raise ValueError("Final archive dispatch counts disagree.")
    rows = result['rows']
    if len(rows) > 8 or len({tuple(item['codes']) for item in rows}) != len(rows):
        raise ValueError("Invalid finalized sampling archive.")
    for item in rows:
        if item != {'codes':item['codes'], **score_board(problem, item['codes'])}:
            raise ValueError("Finalized sampling hint failed independent rescoring.")
    solution = result['solution']
    if solution is not None and (score_board(problem, solution)['domain_violations'] or not problem.validate(solution)['valid']):
        raise ValueError("Finalized sampling solution failed Gold validation.")
    return result


def sample_archive(problem, database, *, seed, backend, replicas, dispatches):
    """Restore native prefixes for extensions and independently verified archives for exact replays."""
    from constraint_work import digest
    from gpu_sampling_resume_round25 import SamplerSession
    if type(dispatches) is not int or dispatches < 0 or backend not in ('cuda', 'webgpu'):
        raise ValueError("Choose a GPU backend and nonnegative dispatch count")
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = dict(backend=backend, replicas=replicas, seed=seed, target='gold',
                experiments={'replica_exchange': True, 'move_tabu': True})
    request = _archive_request(problem, opts)
    with closing(_open(path)) as conn:
        _schema(conn)
        cached = _read_final(conn, request, dispatches, replicas, problem)
        if cached is not None:
            return cached
        session = SamplerSession(problem, **opts)
        try:
            scope = hashlib.sha256(json.dumps([backend, session.config], sort_keys=True).encode()).hexdigest()
            with _prefix_claim(conn, path, scope, seed, dispatches, final_request=request):
                cached = _read_final(conn, request, dispatches, replicas, problem)
                if cached is not None:
                    return cached
                row = conn.execute("""SELECT state FROM r31_sampler WHERE scope=? AND seed=? AND launches<=?
                                    ORDER BY launches DESC LIMIT 1""", (scope, seed, dispatches)).fetchone()
                if row:
                    session.close()
                    session = SamplerSession(problem, **opts, resume=json.loads(row[0]))
                before = session.counts['move_proposals']
                reused = session.launches
                while session.launches < dispatches and not session.done:
                    limit = min(dispatches, 64 * (session.launches // 64 + 1))
                    while session.launches < limit and not session.done:
                        session.advance(60, max_dispatches=limit-session.launches)
                    state = session.snapshot()
                    with conn:
                        conn.execute("INSERT OR IGNORE INTO r31_sampler VALUES(?,?,?,?)",
                                     (scope, seed, session.launches, json.dumps(state)))
                session.finish_phase()
                report = session.report()
                final = dict(rows=report['top_samples'], solution=session.solution,
                             move_proposals=report['move_proposals'], new_proposals=0, reused_dispatches=session.launches)
                with conn:
                    conn.execute('INSERT OR IGNORE INTO r34_final_archives VALUES(?,?,?,?)',
                                 (request, dispatches, json.dumps(final), digest([request, dispatches, final])))
                return dict(final, new_proposals=report['move_proposals']-before, reused_dispatches=reused)
        finally:
            session.close()
