"""Fenced, resumable pages for a single scored-neighbor expansion."""
import json
import time
import uuid
import zlib

import numpy as np

from r28_exchange import ExchangeBound
from r34_query_ownership import query_lease


class ExpansionStopped(Exception):
    pass


def schema(db):
    db.executescript("""
        CREATE TABLE IF NOT EXISTS walk_star_pages(
            board TEXT NOT NULL, first_pair INTEGER NOT NULL, base INTEGER NOT NULL, deltas BLOB NOT NULL,
            PRIMARY KEY(board,first_pair));
        CREATE TABLE IF NOT EXISTS walk_vectors(
            board TEXT PRIMARY KEY, lowest INTEGER NOT NULL, base INTEGER NOT NULL, deltas BLOB NOT NULL);
        CREATE TABLE IF NOT EXISTS walk_band_cursor(
            board TEXT PRIMARY KEY, generation TEXT NOT NULL, low INTEGER NOT NULL, high INTEGER NOT NULL,
            next_pair INTEGER NOT NULL, legacy_scope TEXT);
    """)


def begin(db,guard):
    if not db.in_transaction:
        db.execute("BEGIN IMMEDIATE")
    guard()


def find_legacy(graph,key,floor):
    physical=key.split(':',1)[1]
    for old,scope in graph.legacy_scopes:
        if old==floor and graph.db.execute(
                "SELECT 1 FROM centers WHERE scope=? AND perm=? AND closed=1",(scope,physical)).fetchone():
            return scope
    return None


def lowest(db,key):
    row=db.execute("""SELECT min(lowest) FROM (SELECT lowest FROM walk_bounds WHERE board=?
                      UNION ALL SELECT lowest FROM walk_vectors WHERE board=?)""",(key,key)).fetchone()
    return row[0]


def stored_vector(graph,key):
    n=len(graph.pairs)
    exact=graph.db.execute("SELECT base,deltas FROM walk_vectors WHERE board=?",(key,)).fetchone()
    if exact is not None:
        delta=np.frombuffer(zlib.decompress(exact[1]),dtype=np.int8)
        if len(delta)!=n:raise ValueError('Invalid saved plateau delta dimensions.')
        return exact[0],delta.astype(np.int32)+exact[0]
    return None


def exact_parent(graph,key,board,parent):
    if not parent or parent.split(':',1)[0]!=graph.binding:return None
    donor=np.array([int(v) for v in parent.split(':',1)[1].split(',')])
    changed=set(np.flatnonzero(donor!=np.asarray(board)//3).tolist())
    if len(changed)!=2:return None
    a,b=changed
    child=np.asarray(board)//3
    if donor[a]!=child[b] or donor[b]!=child[a]:return None
    saved=stored_vector(graph,parent)
    if saved is None:return None
    base,doubled=saved
    # A candidate swap changes the stars at both endpoints and their neighbors.
    # Overlapping either original changed piece's two-step neighborhood can
    # change this delta; outside that fringe the exact doubled delta is equal.
    for _ in range(2):
        changed.update(int(nb) for c in list(changed) for nb in graph.model.problem.neighbors[c] if nb>=0)
    return doubled-base,np.array(sorted(changed))


def encoded_deltas(values):
    if len(values) and (values.min() < -128 or values.max() >= 128):
        raise ArithmeticError('Local star delta exceeded its storage range.')
    return zlib.compress(values.astype(np.int8).tobytes())


def materialize(graph,key,board,guard,deadline,*,parent=None):
    db=graph.db
    exact=stored_vector(graph,key)
    if exact is not None:
        graph.work['cached_stars']+=len(graph.pairs)
        return lowest(db,key),exact[1]//2
    saved=db.execute("SELECT lowest,bounds FROM walk_bounds WHERE board=?",(key,)).fetchone()
    if saved:
        vector=np.frombuffer(zlib.decompress(saved[1]),dtype='<i4')
        if len(vector)!=len(graph.pairs):raise ValueError('Invalid saved plateau bound dimensions.')
        graph.work['cached_stars']+=len(vector)
        return saved[0],vector
    count=len(graph.pairs)
    bound=ExchangeBound(graph.model,board)
    prior=exact_parent(graph,key,board,parent)
    parts=[]
    for start in range(0,count,512):
        page=graph.pairs[start:start+512]
        raw=db.execute("SELECT base,deltas FROM walk_star_pages WHERE board=? AND first_pair=?",(key,start)).fetchone()
        if raw is not None:
            delta=np.frombuffer(zlib.decompress(raw[1]),dtype=np.int8)
            if len(delta)!=len(page) or raw[0]!=bound.total:
                raise ValueError('Invalid partial plateau star page.')
            graph.work['cached_stars']+=len(page)
        else:
            if time.monotonic()>=deadline:raise ExpansionStopped()
            if prior is None:
                doubled=bound.bounds(page,(1,0),doubled=True)
                priced=len(page)
            else:
                old,fringe=prior
                dirty=np.isin(page,fringe).any(axis=1)
                doubled=old[start:start+len(page)]+bound.total
                if dirty.any():doubled[dirty]=bound.bounds(page[dirty],(1,0),doubled=True)
                priced=int(dirty.sum())
                graph.work['transported_stars']=graph.work.get('transported_stars',0)+len(page)-priced
            delta=doubled-bound.total
            begin(db,guard)
            db.execute("INSERT INTO walk_star_pages VALUES(?,?,?,?)",(key,start,bound.total,encoded_deltas(delta)))
            db.commit()
            graph.work['star_evaluations']+=priced
        parts.append(delta)
    delta=np.concatenate(parts) if parts else np.empty(0,dtype=np.int8)
    vector=(delta.astype(np.int32)+bound.total)//2
    high=len(graph.model.problem.edges)+1
    begin(db,guard)
    db.execute("INSERT INTO walk_vectors VALUES(?,?,?,?)",(key,high,bound.total,encoded_deltas(delta)))
    db.execute("DELETE FROM walk_star_pages WHERE board=?",(key,))
    db.commit()
    return high,vector


def advance(graph,database,key,board,floor,*,deadline=float('inf'),candidate_limit=None,ttl=20,parent=None):
    """Return true only after all candidates capable of entering this band are resolved."""
    db=graph.db
    with query_lease(db,database,'r35-plateau:'+key,deadline,ttl=ttl) as guard:
        if guard is None:return False
        try:
            while True:
                existing=lowest(db,key)
                if (existing is not None and existing<=floor) or db.execute(
                        "SELECT 1 FROM walk_imported WHERE board=? AND floor=?",(key,floor)).fetchone():
                    graph.work['reused_arcs']+=1
                    return True
                job=db.execute("SELECT generation,low,high,next_pair,legacy_scope FROM walk_band_cursor WHERE board=?",(key,)).fetchone()
                low=job[1] if job else floor
                legacy=job[4] if job else find_legacy(graph,key,floor)
                if legacy:
                    high=job[2] if job else len(graph.model.problem.edges)+1
                    eligible=np.arange(len(graph.pairs))
                    members=dict(db.execute("SELECT perm,codes FROM centers WHERE scope=?",(legacy,)))
                    physical=key.split(':',1)[1].split(',')
                else:
                    high,upper=materialize(graph,key,board,guard,deadline,parent=parent)
                    if job and high!=job[2]:raise ValueError('Plateau band high-water mark changed unexpectedly.')
                    eligible=np.flatnonzero((upper>=low)&(upper<high))
                    if not job:graph.work['star']+=int((upper<low).sum())
                    bound=ExchangeBound(graph.model,board)
                if job:
                    generation,_,_,position,_=job
                else:
                    generation=uuid.uuid4().hex
                    position=0
                    begin(db,guard)
                    db.execute("INSERT INTO walk_band_cursor VALUES(?,?,?,?,?,?)",
                               (key,generation,low,high,0,legacy))
                    db.commit()
                # Each durable cursor is an offset in this exact qualifying band.
                for first in range(position,len(eligible)):
                    if time.monotonic()>=deadline or (candidate_limit is not None and graph.work['swap_candidates']>=candidate_limit):
                        raise ExpansionStopped()
                    a,b=map(int,graph.pairs[eligible[first]])
                    if legacy:
                        physical[a],physical[b]=physical[b],physical[a]
                        neighbor=','.join(physical)
                        physical[a],physical[b]=physical[b],physical[a]
                        encoded=members.get(neighbor)
                        seed=np.array(json.loads(encoded)) if encoded else None
                    else:
                        seed=bound.witness_seed((a,b),(1,0))
                    item=None
                    if seed is not None:
                        child=graph.key(seed)
                        item=graph.optimum(child,seed,low)
                    begin(db,guard)
                    if item is not None:
                        if item.get('codes') is None:
                            db.execute("INSERT OR IGNORE INTO walk_legacy VALUES(?,?,?)",(child,item['score'],json.dumps(seed.tolist())))
                        db.executemany("INSERT OR IGNORE INTO walk_edges VALUES(?,?)",((key,child),(child,key)))
                    updated=db.execute("UPDATE walk_band_cursor SET next_pair=? WHERE board=? AND generation=? AND next_pair=?",
                                       (first+1,key,generation,first))
                    if updated.rowcount!=1:raise RuntimeError('A stale plateau candidate cannot publish.')
                    db.commit()
                    graph.work['swap_candidates']+=1
                begin(db,guard)
                if legacy:
                    db.execute("INSERT OR IGNORE INTO walk_imported VALUES(?,?)",(key,low))
                    graph.work['legacy_centers']+=1
                else:
                    db.execute("UPDATE walk_bounds SET lowest=min(lowest,?) WHERE board=?",(low,key))
                    db.execute("UPDATE walk_vectors SET lowest=min(lowest,?) WHERE board=?",(low,key))
                db.execute("DELETE FROM walk_band_cursor WHERE board=? AND generation=?",(key,generation))
                db.commit()
                if low<=floor:return True
        except ExpansionStopped:
            db.rollback()
            return False
        except BaseException:
            db.rollback()
            raise
