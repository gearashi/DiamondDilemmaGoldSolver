"""Bounded live-state CUDA checkpoint codec; unused tails normalize to zero.

Only future-unread padding is omitted. Domain ownership and the complete runner
state belong to the enclosing generation, not to individual shared chunks.
"""
from __future__ import annotations
import base64,hashlib,json,struct,zlib
import numpy as np

MAX_STATE_BYTES=512*1024*1024
MAX_COLUMN_BYTES=2*1024*1024
MAX_HEADER_BYTES=65536

def canonical_bytes(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')
def sha(data):return hashlib.sha256(data).hexdigest()

def _ranges(engine,data,lane):
    n,w,rw=engine.n,engine.words,engine.reason_words;depth,trail,assigned,unions=(int(data[i,lane]) for i in (1,2,9,10))
    if not (0<=depth<=n and 0<=trail<=n*3*n and 0<=assigned<=n and 0<=unions<=16*n):raise ValueError('Invalid live prefix bounds.')
    lengths={'trail':trail,'assign_log':assigned,'union_child':unions,'union_parent':unions,'union_size':unions,
             'frame_cell':depth,'frame_code':depth,'frame_trail':depth,'frame_assign':depth,'frame_union':depth,
             'frame_remaining':depth*w,'frame_reason':depth*n*rw,'frame_failure':depth*rw}
    if getattr(engine,'proof_transport',False) and (int(data[0,lane])==2 or int(data[6,lane])==3):
        proof_depth=int(data[engine.offsets['proof_meta'],lane])
        if not 0<=proof_depth<=n:raise ValueError('Invalid proof failure prefix.')
        lengths['frame_cell']=lengths['frame_code']=max(depth,proof_depth)
    fields=sorted(engine.offsets.items(),key=lambda item:item[1]);ranges=[]
    for pos,(name,start) in enumerate(fields):
        full=(fields[pos+1][1] if pos+1<len(fields) else engine.rows)-start
        if name in ('clause_cell','clause_code'):
            # Preserve every ring slot's declared contents, including an old
            # inactive slot. Never infer occupancy from a prefix count alone.
            for slot in range(32):
                count=int(data[engine.offsets['clause_length']+slot,lane])
                if not 0<=count<=n:raise ValueError('Invalid checkpoint clause length.')
                if count:ranges.append((start+slot*n,count))
        elif name=='precise_why':
            keys=data[engine.offsets['precise_keys']:engine.offsets['precise_keys']+engine.reason_slots,lane]
            ranges.extend((start+slot*rw,rw) for slot,key in enumerate(keys) if key not in (0,0xffffffff))
        else:
            length=lengths.get(name,full)
            if not 0<=length<=full:raise ValueError('Checkpoint live range exceeds allocation.')
            if length:ranges.append((start,length))
    merged=[]
    for start,length in ranges:
        if merged and merged[-1][0]+merged[-1][1]==start:merged[-1]=(merged[-1][0],merged[-1][1]+length)
        else:merged.append((start,length))
    return merged

def encode_native(engine,snapshot,cache=None):
    data=snapshot['state'];cache=cache if cache is not None else {};next_cache={};columns=[];blobs={}
    metrics={'packed_bytes':0,'compressed_bytes':0,'reused_chunks':0,'omitted_bytes':0}
    if data.dtype!=np.uint32 or data.shape!=(engine.rows,engine.lanes):raise ValueError('Unexpected frozen GPU snapshot.')
    for lane in range(engine.lanes):
        ranges=_ranges(engine,data,lane);header=canonical_bytes(ranges)
        if len(header)>MAX_HEADER_BYTES:raise ValueError('Too many live state ranges.')
        words=np.concatenate([data[start:start+length,lane] for start,length in ranges]).astype('<u4',copy=False)
        raw=struct.pack('<I',len(header))+header+words.tobytes();identity=sha(raw)
        if len(raw)>MAX_COLUMN_BYTES:raise ValueError('Packed GPU column exceeds bound.')
        compressed=cache.get(identity)
        if compressed is None:compressed=zlib.compress(raw,1)
        else:metrics['reused_chunks']+=1
        next_cache[identity]=compressed;key=sha(compressed);blobs[key]=compressed
        columns.append({'sha256':key,'compressed_bytes':len(compressed),'packed_bytes':len(raw)})
        metrics['packed_bytes']+=len(raw);metrics['compressed_bytes']+=len(compressed)
        metrics['omitted_bytes']+=engine.rows*4-len(words)*4
    cache.clear();cache.update(next_cache)
    spec={k:snapshot[k] for k in ('schema','binding','cubes','delivered','rows')}
    spec.update(factored_native_version=1,shape=[engine.rows,engine.lanes],columns=columns)
    return spec,blobs,metrics

def validate_spec(spec):
    expected={'schema','binding','cubes','delivered','rows','factored_native_version','shape','columns'}
    if not isinstance(spec,dict) or set(spec)!=expected or type(spec['factored_native_version'])is not int or spec['factored_native_version']!=1:
        raise ValueError('Invalid factored native checkpoint.')
    shape=spec['shape']
    if (not isinstance(shape,list) or len(shape)!=2 or any(type(v)is not int or v<1 for v in shape)
            or shape[1]>512 or shape[0]*shape[1]*4>MAX_STATE_BYTES or spec['rows']!=shape[0]
            or not isinstance(spec['columns'],list) or len(spec['columns'])!=shape[1]
            or not isinstance(spec['cubes'],list) or len(spec['cubes'])!=shape[1]):raise ValueError('Invalid factored checkpoint dimensions.')
    for column in spec['columns']:
        if not isinstance(column,dict) or set(column) not in ({'sha256','compressed_bytes','packed_bytes'},{'sha256','compressed_bytes','packed_bytes','data'}):raise ValueError('Invalid native chunk descriptor.')
        if (not isinstance(column['sha256'],str) or len(column['sha256'])!=64 or any(x not in '0123456789abcdef' for x in column['sha256'])
                or any(type(column[k])is not int or not 1<=column[k]<=MAX_COLUMN_BYTES for k in ('compressed_bytes','packed_bytes'))):raise ValueError('Native chunk exceeds limits.')
    if any(column['packed_bytes']>shape[0]*4+MAX_HEADER_BYTES+4 or column['compressed_bytes']>column['packed_bytes']+4096 for column in spec['columns']):raise ValueError('Chunk sizes exceed reconstructed state bound.')
    return shape

def checked_blob(column,blob):
    if len(blob)!=column['compressed_bytes'] or sha(blob)!=column['sha256']:raise ValueError('Native chunk integrity failure.')
    return blob

def decode_native(spec,engine):
    shape=validate_spec(spec)
    if shape!=[engine.rows,engine.lanes] or spec['binding']!=engine.binding or spec['schema']!=engine.schema:
        raise ValueError('Factored GPU checkpoint binding differs.')
    data=np.zeros(shape,np.uint32)
    for lane,column in enumerate(spec['columns']):
        encoded=column.get('data')
        if not isinstance(encoded,str) or len(encoded)>MAX_COLUMN_BYTES*2:raise ValueError('Missing or oversized native chunk data.')
        blob=checked_blob(column,base64.b64decode(encoded,validate=True));decoder=zlib.decompressobj()
        raw=decoder.decompress(blob,column['packed_bytes']+1)
        if len(raw)!=column['packed_bytes'] or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:raise ValueError('Malformed compressed live-state chunk.')
        if len(raw)<4:raise ValueError('Missing live range header.')
        size=struct.unpack('<I',raw[:4])[0]
        if size>MAX_HEADER_BYTES or 4+size>len(raw):raise ValueError('Invalid live range header bound.')
        ranges=json.loads(raw[4:4+size]);cursor=4+size;end=0
        if not isinstance(ranges,list) or len(ranges)>8192:raise ValueError('Invalid live range list.')
        for pair in ranges:
            if (not isinstance(pair,list) or len(pair)!=2 or any(type(v)is not int for v in pair)
                    or pair[0]<end or pair[1]<1 or pair[0]+pair[1]>engine.rows):raise ValueError('Overlapping or out-of-bounds live range.')
            start,length=pair;following=cursor+length*4
            if following>len(raw):raise ValueError('Truncated live-state range.')
            data[start:start+length,lane]=np.frombuffer(raw[cursor:following],dtype='<u4');cursor=following;end=start+length
        if cursor!=len(raw):raise ValueError('Trailing live-state bytes.')
    return {**{k:spec[k] for k in ('schema','binding','cubes','delivered','rows')},'state':data,'state_sha256':sha(memoryview(data).cast('B'))}

def inline_native(spec,blobs):
    import copy
    validate_spec(spec);out=copy.deepcopy(spec)
    for column in out['columns']:column['data']=base64.b64encode(checked_blob(column,blobs[column['sha256']])).decode('ascii')
    return out
