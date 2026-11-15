"""Fenced, disjoint ranked-page ownership shared by the loss and exchange solvers."""
from contextlib import closing, contextmanager
from dataclasses import dataclass
import sqlite3
import threading
import time
from uuid import uuid4


class LostRankLease(RuntimeError):
    pass


@dataclass(frozen=True)
class RankClaim:
    lo: int
    hi: int
    token: str


def covered_end(db, table, scope, start, stop):
    """Return the first missing rank without acquiring a write lock."""
    if table not in ('r30_pages','r32_joint_pages'):
        raise ValueError("Unknown ranked-page table")
    row=db.execute(f"SELECT hi FROM {table} WHERE scope=? AND lo<=? ORDER BY lo DESC LIMIT 1",
                   (scope,start)).fetchone()
    cursor=max(start,row[0]) if row else start
    if cursor<stop:
        for lo,hi in db.execute(f"SELECT lo,hi FROM {table} WHERE scope=? AND lo>=? AND lo<? ORDER BY lo",
                                (scope,cursor,stop)):
            if lo>cursor:
                break
            cursor=max(cursor,hi)
    return min(stop,cursor)


class RankedLeases:
    def __init__(self, db, family, scope, ttl=90, private=False, lifetime=None):
        self.db = db
        self.family = family
        self.scope = scope
        self.ttl = ttl
        self.private = private
        self.persistent = lifetime is not None
        self._active = None
        self._thread = None
        self._stop = threading.Event()
        if lifetime is not None:
            lifetime.callback(self.close)
        self.filename = next(row[2] for row in db.execute("PRAGMA database_list") if row[1]=="main")
        db.execute("CREATE TABLE IF NOT EXISTS r34_rank_leases("
                   "family TEXT, scope TEXT, lo INTEGER, hi INTEGER, token TEXT, expires REAL, "
                   "PRIMARY KEY(family,scope,lo))")
        db.execute("CREATE INDEX IF NOT EXISTS r34_rank_tokens ON r34_rank_leases(token)")

    def take(self, lo, hi, *, token=None, batch_time=None):
        """Caller holds BEGIN IMMEDIATE and checked finished coverage. Return a claim or a busy end."""
        if not self.db.in_transaction or not lo<hi:
            raise ValueError("Rank claims need a nonempty interval and a write transaction")
        now=time.time() if batch_time is None else batch_time
        key=(self.family,self.scope)
        if batch_time is None:
            self.purge(now)
        prior=self.db.execute("SELECT hi FROM r34_rank_leases WHERE family=? AND scope=? AND lo<=? "
                              "ORDER BY lo DESC LIMIT 1",(*key,lo)).fetchone()
        if prior and prior[0]>lo:
            return None,min(hi,prior[0])
        following=self.db.execute("SELECT lo FROM r34_rank_leases WHERE family=? AND scope=? AND lo>? AND lo<? "
                                  "ORDER BY lo LIMIT 1",(*key,lo,hi)).fetchone()
        if following:
            hi=following[0]
        claim=RankClaim(lo,hi,token or uuid4().hex)
        self.db.execute("INSERT INTO r34_rank_leases VALUES(?,?,?,?,?,?)",(*key,lo,hi,claim.token,float("inf") if self.private else now+self.ttl))
        return claim,lo

    def purge(self, now):
        self.db.execute("DELETE FROM r34_rank_leases WHERE family=? AND scope=? AND expires<=?",
                        (self.family,self.scope,now))

    def fence(self, claim):
        if not self.db.in_transaction:
            raise ValueError("Fence verification and page publication need one transaction")
        row=self.db.execute("SELECT 1 FROM r34_rank_leases WHERE family=? AND scope=? AND lo=? "
                            "AND hi=? AND token=? AND expires>?",
                            (self.family,self.scope,claim.lo,claim.hi,claim.token,time.time())).fetchone()
        if not row:
            raise LostRankLease("This ranked page is no longer owned")

    def fence_uncovered(self, claim, table, lo, hi):
        if table not in ('r30_pages','r32_joint_pages') or not claim.lo<=lo<hi<=claim.hi:
            raise ValueError("Invalid ranked page publication")
        self.fence(claim)
        row=self.db.execute(f"SELECT hi FROM {table} WHERE scope=? AND lo<? ORDER BY lo DESC LIMIT 1",
                            (self.scope,hi)).fetchone()
        if row and row[0]>lo:
            raise LostRankLease("Another solver already published part of this interval")

    def wait(self, rank, deadline=float("inf"), ready=None):
        """Wait without a writer lock for a live owner, publication or expiry."""
        delay=0.025
        while time.monotonic()<deadline:
            if ready is not None and ready():
                return
            row=self.db.execute("SELECT hi,expires FROM r34_rank_leases WHERE family=? AND scope=? AND lo<=? "
                                "ORDER BY lo DESC LIMIT 1",(self.family,self.scope,rank)).fetchone()
            if not row or row[0]<=rank or row[1]<=time.time():
                return
            time.sleep(max(0,min(delay,deadline-time.monotonic(),row[1]-time.time())))
            delay=min(0.4,2*delay)

    def release(self, claim):
        self.db.execute("DELETE FROM r34_rank_leases WHERE family=? AND scope=? AND lo=? AND token=?",
                        (self.family,self.scope,claim.lo,claim.token))

    def _renew(self):
        while not self._stop.wait(self.ttl/3):
            token=self._active
            if token is None:
                continue
            with closing(sqlite3.connect(self.filename,timeout=max(1,min(5,self.ttl/3)),
                                         isolation_level=None)) as db:
                now=time.time()
                try:
                    db.execute("UPDATE r34_rank_leases SET expires=? WHERE token=? AND expires>?",
                               (now+self.ttl,token,now))
                except sqlite3.OperationalError:
                    continue

    def close(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread=None

    @contextmanager
    def computing(self, claim):
        """One renewable owner covers the current page or sparse finite pool."""
        self._active=claim.token
        if self.filename and self._thread is None:
            self._stop.clear()
            self._thread=threading.Thread(target=self._renew,daemon=True)
            self._thread.start()
        try:
            yield
        finally:
            self._active=None
            if not self.persistent:
                self.close()
            with self.db:
                self.db.execute("DELETE FROM r34_rank_leases WHERE token=?",(claim.token,))
