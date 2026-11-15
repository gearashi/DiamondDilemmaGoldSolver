"""Ownership for CPU residuals. Only the current lease may publish a native decision."""
from contextlib import contextmanager, nullcontext
import time

from r31_sampling import _claim, _heartbeat, _jobs_schema


class LostQueryLease(Exception):
    pass


@contextmanager
def query_lease(db, database, key, deadline=float('inf'), *, ttl=60):
    if str(database)==':memory:':
        ttl=float('inf')
    _jobs_schema(db)
    owner=None
    while time.monotonic()<deadline:
        owner=_claim(db,key,0,ttl)
        if owner is not None:
            break
        time.sleep(min(.05,max(0,deadline-time.monotonic())))
    if owner is None:
        yield None
        return
    def guard():
        # Call inside the same writer transaction as the progress or result.
        if not db.in_transaction:
            raise RuntimeError('Fencing a result requires a writer transaction.')
        valid=db.execute("""SELECT 1 FROM r31_jobs WHERE scope=? AND seed=0 AND owner=?
                            AND expires>?""",(key,owner,time.time())).fetchone()
        if valid is None:
            raise LostQueryLease()
    try:
        pulse=nullcontext() if str(database)==':memory:' else _heartbeat(database,key,0,owner,ttl)
        with pulse:
            yield guard
    finally:
        with db:
            db.execute("DELETE FROM r31_jobs WHERE scope=? AND seed=0 AND owner=?",(key,owner))
