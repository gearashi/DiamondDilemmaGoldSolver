"""Trusted local sampler snapshots with complete RNG/tabu/exchange metadata.

Hints are independently checked before use. These snapshots prove no exclusion
or search coverage, and exact trajectories require the same runtime binding.
"""
from __future__ import annotations
import base64,hashlib,json,zlib
from pathlib import Path
import numpy as np
from constraint_work import digest

MAX_BYTES=256*1024*1024

def binding(problem,table,engine):
    from gpu_sampling import CUDA_SOURCE
    root=Path(__file__).resolve().parent;backend=engine.runtime_identity()['backend']
    source=(root/('sampling_round24.cu' if backend=='cuda' else 'sampling_round24.wgsl')).read_bytes() if engine.flags else (CUDA_SOURCE.encode() if backend=='cuda' else (root/'sampling_kernels.wgsl').read_bytes())
    return {'version':1,'fingerprint':problem.fingerprint,'table_sha256':hashlib.sha256(table.tobytes()).hexdigest(),
        'replicas':engine.r,'cells':engine.n,'free':engine.f,'meta_rows':engine.meta_rows,'flags':engine.flags,'gold_total':engine.gold_total,
        'runtime':engine.runtime_identity(),'kernel_sha256':hashlib.sha256(source).hexdigest()}

def encode(problem,table,engine):
    arrays=engine.snapshot_arrays();raw=b''.join(np.asarray(a,dtype='<u4').tobytes() for a in arrays)
    if len(raw)>MAX_BYTES:raise ValueError('Sampler state exceeds snapshot bound.')
    value={'sampling_state_version':1,'binding':binding(problem,table,engine),'bytes':len(raw),
           'sha256':hashlib.sha256(raw).hexdigest(),'compressed':base64.b64encode(zlib.compress(raw,1)).decode('ascii')}
    value['digest']=digest(value);return value

def decode(problem,table,engine,value):
    if (not isinstance(value,dict) or set(value)!={'sampling_state_version','binding','bytes','sha256','compressed','digest'}
            or type(value['sampling_state_version'])is not int or value['sampling_state_version']!=1
            or value['digest']!=digest({k:v for k,v in value.items() if k!='digest'})):raise ValueError('Invalid sampler checkpoint integrity.')
    if value['binding']!=binding(problem,table,engine):raise ValueError('Sampler backend, device/runtime, policy, replica count or input changed; exact sampler resume is unavailable.')
    size=(2*engine.n+engine.meta_rows)*engine.r*4
    if type(value['bytes'])is not int or value['bytes']!=size or not 0<size<=MAX_BYTES:raise ValueError('Invalid sampler checkpoint dimensions.')
    if not isinstance(value['compressed'],str) or len(value['compressed'])>MAX_BYTES*2:raise ValueError('Oversized sampler checkpoint.')
    decoder=zlib.decompressobj()
    try:raw=decoder.decompress(base64.b64decode(value['compressed'],validate=True),size+1)
    except (ValueError,zlib.error) as exc:raise ValueError('Malformed sampler compressed bytes.') from exc
    if len(raw)!=size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail or hashlib.sha256(raw).hexdigest()!=value['sha256']:
        raise ValueError('Malformed sampler checkpoint bytes.')
    all_rows=np.frombuffer(raw,dtype='<u4').reshape(2*engine.n+engine.meta_rows,engine.r)
    arrays=[all_rows[:engine.n].copy(),all_rows[engine.n:2*engine.n].copy(),all_rows[2*engine.n:].copy()]
    for boards in arrays[:2]:
        if np.any(boards>=3*problem.n) or not np.array_equal(np.sort(boards//3,axis=0),np.broadcast_to(np.arange(problem.n,dtype=np.uint32)[:,None],boards.shape)):
            raise ValueError('Sampler checkpoint repeats/omits physical pieces.')
        if any(np.any(boards[cell]!=code) for cell,code in problem.fixed.items()):raise ValueError('Sampler checkpoint changed a fixed placement.')
    meta=arrays[2];maximum=(problem.n*(len(problem.edges)+1)+len(problem.edges))*(engine.gold_total+1)+engine.gold_total
    if np.any(meta[0]==0) or np.any(meta[1:3]>maximum):raise ValueError('Invalid sampler RNG or objective range.')
    if engine.flags&32:
        from gpu_sampling import _temperature_values
        values=_temperature_values(list(map(float,meta[28,:8].view(np.float32))))
        expected=np.resize(values,engine.r).view(np.uint32)
        if not np.array_equal(meta[28],expected):raise ValueError('Sampler checkpoint has inconsistent per-group temperature ladders.')
    return arrays

def restore(problem,table,engine,value):
    arrays=decode(problem,table,engine,value) # Every host validation precedes device writes.
    engine.restore_arrays(arrays)
