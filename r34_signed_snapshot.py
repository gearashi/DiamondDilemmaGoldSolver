"""Compact phase-bound dart prices for resuming a phase-first repair trajectory."""
import base64
import hashlib
import json
import zlib

import numpy as np


def snapshot(signed):
    # A SignedRepair owns one fixed phase vector. Its private prices only grow.
    cached=getattr(signed,'_price_snapshot',None)
    if cached is not None and cached['count']==len(signed.prices):
        return cached.copy()
    darts=sorted(signed.prices)
    count=len(signed.dart_phases)
    if darts:
        packed=np.packbits(np.stack([signed.prices[d] for d in darts]),axis=1).tobytes()
    else:
        packed=b''
    body=np.asarray(darts,dtype='<u2').tobytes()+zlib.compress(packed,1)
    checksum=hashlib.sha256(signed.scope.encode()+body).hexdigest()
    saved=dict(count=len(darts),width=count,checksum=checksum,data=base64.b64encode(body).decode('ascii'))
    signed._price_snapshot=saved
    return saved.copy()


def restore(signed,state):
    count=state['count']
    width=len(signed.dart_phases)
    if type(count) is not int or not 0<=count<=width or state['width']!=width:
        raise ValueError('Invalid saved phase price dimensions.')
    body=base64.b64decode(state['data'],validate=True)
    if hashlib.sha256(signed.scope.encode()+body).hexdigest()!=state['checksum']:
        raise ValueError('Saved phase prices failed validation.')
    darts=np.frombuffer(body[:2*count],dtype='<u2')
    if len(darts)!=count or list(darts)!=sorted(set(darts)) or any(d>=width for d in darts):
        raise ValueError('Invalid saved phase price indices.')
    stride=(width+7)//8
    decoded=zlib.decompress(body[2*count:])
    if len(decoded)!=count*stride:
        raise ValueError('Invalid saved phase price matrix.')
    packed=np.frombuffer(decoded,np.uint8).reshape(count,stride)
    prices=np.unpackbits(packed,axis=1,count=width).view(np.bool_)
    signed.prices.update((int(d),v) for d,v in zip(darts,prices))
    signed._price_snapshot=state.copy()


def archive(db,state):
    """Store a shared immutable price image in the same transaction as its cursor."""
    if 'dart_prices' not in state:
        return state
    raw=json.dumps(state['dart_prices'],separators=(',',':'),sort_keys=True)
    key=hashlib.sha256(raw.encode()).hexdigest()
    db.execute('CREATE TABLE IF NOT EXISTS r34_price_blobs(sha TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    db.execute('INSERT OR IGNORE INTO r34_price_blobs VALUES(?,?)',(key,raw))
    result=state.copy()
    del result['dart_prices']
    result['dart_price_blob']=key
    return result


def expand(db,state):
    """Read earlier inline checkpoints and validate shared images on newer ones."""
    if 'dart_price_blob' not in state:
        return state
    key=state.pop('dart_price_blob')
    row=db.execute('SELECT payload FROM r34_price_blobs WHERE sha=?',(key,)).fetchone()
    if row is None or hashlib.sha256(row[0].encode()).hexdigest()!=key:
        raise ValueError('Missing or damaged trajectory price snapshot.')
    state['dart_prices']=json.loads(row[0])
    return state
