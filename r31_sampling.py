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
                index = seed
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
def _prefix_claim(conn, path, scope, seed, dispatches):
    """Serialize extensions of one sampler stream while unrelated seeds continue."""
    lock_scope = 'sampling:' + scope
    owner = None
    while owner is None:
        exact = conn.execute(
            'SELECT 1 FROM r31_sampler WHERE scope=? AND seed=? AND launches=?',
            (scope, seed, dispatches)).fetchone()
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



def sample_archive(problem, database, *, seed, backend, replicas, dispatches):
    """Restore an unfinalized prefix; collect the requested terminal archive separately."""
    from gpu_sampling_resume_round25 import SamplerSession
    if dispatches < 0 or backend not in ('cuda', 'webgpu'):
        raise ValueError("Choose a GPU backend and nonnegative dispatch count")
    path = Path(database)
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = dict(backend=backend, replicas=replicas, seed=seed, target='gold',
                experiments={'replica_exchange': True, 'move_tabu': True})
    session = SamplerSession(problem, **opts)
    try:
        scope = hashlib.sha256(json.dumps([backend, session.config], sort_keys=True).encode()).hexdigest()
        with closing(_open(path)) as conn:
            _schema(conn)
            with _prefix_claim(conn, path, scope, seed, dispatches):
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
        return dict(rows=report['top_samples'], solution=session.solution,
                    move_proposals=report['move_proposals'],
                    new_proposals=report['move_proposals']-before, reused_dispatches=reused)
    finally:
        session.close()
