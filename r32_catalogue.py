"""Persist reusable attachment rows and center-specific ordering for connected search."""
from functools import lru_cache
import hashlib
import json
import zlib

import numpy as np


def _store(array):
    return zlib.compress(np.ascontiguousarray(array,dtype='<i4').tobytes(),1)


def _restore(blob, shape):
    return np.frombuffer(zlib.decompress(blob),dtype='<i4').reshape(shape)


def install(join,db,binding,owner):
    pool = hashlib.sha256(json.dumps(sorted(join.pool)).encode()).hexdigest()
    db.execute("""CREATE TABLE IF NOT EXISTS r32_catalogue(
        version INTEGER,pool TEXT,length INTEGER,anchor INTEGER, prices_for TEXT,
        body BLOB, checksum BLOB, PRIMARY KEY(version,pool,length,anchor,prices_for))""")
    def read(length,anchor,tag,shape):
        row = db.execute('SELECT body,checksum FROM r32_catalogue WHERE version=1 AND pool=? AND length=? AND anchor=? AND prices_for=?',
                         (pool,length,anchor,tag)).fetchone()
        if row is None:
            return None
        if hashlib.sha256(row[0]).digest()!=row[1]:
            raise ValueError("Attachment catalogue checksum mismatch.")
        return _restore(row[0],shape)

    def save(length,anchor,tag,array):
        blob = _store(array)
        db.execute('INSERT OR IGNORE INTO r32_catalogue VALUES(1,?,?,?,?,?,?)',
                   (pool,length,anchor,tag,blob,hashlib.sha256(blob).digest()))

    def bucket(length,anchor):
        raw = read(length,anchor,'geometry',(-1,length))
        if raw is None:
            raw = join.catalogue_cells(length,anchor)
            save(length,anchor,'geometry',raw)
            join.work['catalogue_cells_built'] += len(raw)
        else:
            join.work['catalogue_cells_reused'] += len(raw)
        tag = binding+':'+owner
        ranked = read(length,anchor,tag,(-1,2))
        if ranked is None:
            prices,indices = join.catalogue_prices(raw)
            ranked = np.column_stack((prices,indices))
            save(length,anchor,tag,ranked)
        else:
            join.work['catalogue_prices_reused'] += len(ranked)
        return ranked[:,0].tolist(),raw[ranked[:,1]]
    join.buckets = lru_cache(maxsize=48)(bucket)
