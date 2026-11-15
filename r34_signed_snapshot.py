"""Compact phase-bound dart prices for resuming a phase-first repair trajectory."""
import base64
import hashlib
import zlib

import numpy as np


def snapshot(signed):
    darts=sorted(signed.prices)
    count=len(signed.dart_phases)
    if darts:
        packed=np.packbits(np.stack([signed.prices[d] for d in darts]),axis=1).tobytes()
    else:
        packed=b''
    body=np.asarray(darts,dtype='<u2').tobytes()+zlib.compress(packed,1)
    checksum=hashlib.sha256(signed.scope.encode()+body).hexdigest()
    return dict(count=len(darts),width=count,checksum=checksum,data=base64.b64encode(body).decode('ascii'))


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
