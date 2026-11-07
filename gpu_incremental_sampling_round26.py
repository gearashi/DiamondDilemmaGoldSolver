"""Exact physical-strand cache layout for both native sampler backends.

Cache words are current-board proper-loop labels, never pruning evidence.
Labels use immutable physical segment IDs, so accepted exchanges move them
with the current board. Cold validation is independent of the native walker.
"""
from __future__ import annotations
import numpy as np

FLAG=64
COUNTER_BASE=29
CACHE_BASE=45
COUNTERS=('incremental_proposals','incremental_domain_terms','incremental_seam_terms',
          'incremental_cache_reads','incremental_component_starts','incremental_strand_steps',
          'incremental_full_fallbacks','incremental_cache_rebuilds')
MAX_SNAPSHOT_BYTES=256*1024*1024


def metadata_rows(flags,total):
    if flags&192:return CACHE_BASE+((total+1)//2 if flags&FLAG else 0)
    return 29 if flags&32 else 28 if flags&16 else 22 if flags&8 else 18 if flags else 4


def state_size(n,replicas,total,flags):
    if type(n)is not int or not 1<=n<=160 or type(replicas)is not int or not 1<=replicas<=131072 or type(total)is not int or not 0<=total<=16*n or type(flags)is not int or flags<0 or flags>255:raise ValueError('Invalid sampler state dimensions/policy.')
    rows=metadata_rows(flags,total)
    return {'meta_rows':rows,'bytes_per_replica':4*(2*n+rows),
            'snapshot_bytes':4*(2*n+rows)*replicas,
            'snapshot_replica_cap':MAX_SNAPSHOT_BYTES//(4*(2*n+rows))}


def prepare_table(problem,table,free,flags):
    base=25*problem.n+free+3*problem.n*problem.n
    total=int(table[base]) if len(table)>base else 0
    if not flags&FLAG or not total:return table,total
    if total!=problem.total_segments or not 0<total<65536:raise ValueError('Invalid incremental strand inventory.')
    nodes=np.full((3*problem.n,33),0xffffffff,np.uint32);node=0
    for tile,item in enumerate(problem.tiles):
        for left,right in item['segments']:
            for rotation in range(3):
                for side,position in (left,right):nodes[3*tile+rotation,((side+rotation)%3)*11+position-1]=node
            node+=1
    if node!=total:raise ValueError('Incremental segment inventory changed.')
    return np.concatenate((table,nodes.ravel())),total


def canonical_labels(problem,codes):
    """Independent physical-segment graph components from raw tile endpoints."""
    count=problem.total_segments;labels=np.zeros(count,np.uint16)
    ports={};offset=0;physical=[]
    for item in problem.tiles:
        physical.append(offset);offset+=len(item['segments'])
    for cell,code in enumerate(codes):
        tile,rotation=divmod(int(code),3)
        for segment,endpoints in enumerate(problem.tiles[tile]['segments']):
            node=physical[tile]+segment
            for side,position in endpoints:ports[(cell,(side+rotation)%3,position)]=node
    adjacency=[[] for _ in range(count)];missing=[0]*count
    for (cell,side,position),node in ports.items():
        nb=int(problem.neighbors[cell,side]);opposite=int(problem.sides[cell,side])
        other=ports.get((nb,opposite,12-position)) if nb<problem.n else None
        if other is None:missing[node]+=1
        else:adjacency[node].append(other)
    seen=set()
    for start in range(count):
        if start in seen:continue
        stack=[start];component=[];seen.add(start)
        while stack:
            node=stack.pop();component.append(node)
            for other in adjacency[node]:
                if other not in seen:seen.add(other);stack.append(other)
        if len(component)<count and all(len(adjacency[node])==2 and not missing[node] for node in component):
            labels[component]=min(component)+1
    packed=np.zeros((count+1)//2,np.uint32)
    for node,label in enumerate(labels):packed[node//2]|=np.uint32(int(label)<<(16*(node%2)))
    return packed


def validate_cache(problem,arrays,total,flags):
    if not flags&FLAG or not total:return 0
    boards,_,meta=arrays
    if meta.shape[0]!=metadata_rows(flags,total):raise ValueError('Invalid incremental cache layout.')
    for lane in range(boards.shape[1]):
        if not np.array_equal(meta[CACHE_BASE:,lane],canonical_labels(problem,boards[:,lane])):
            raise ValueError('Sampler checkpoint has incorrect physical-strand cache labels.')
    return boards.shape[1]*total


def validate_arrays(problem,arrays,replicas,total,flags):
    expected=((problem.n,replicas),(problem.n,replicas),(metadata_rows(flags,total),replicas))
    if not isinstance(arrays,(list,tuple)) or len(arrays)!=3 or any(not isinstance(a,np.ndarray) or a.dtype!=np.uint32 or a.shape!=shape for a,shape in zip(arrays,expected)):
        raise ValueError('Invalid incremental native array shapes/types.')
    for boards in arrays[:2]:
        if np.any(boards>=3*problem.n) or not np.array_equal(np.sort(boards//3,axis=0),np.broadcast_to(np.arange(problem.n,dtype=np.uint32)[:,None],boards.shape)):
            raise ValueError('Incremental native array has invalid physical inventory.')
        if any(np.any(boards[cell]!=code) for cell,code in problem.fixed.items()):raise ValueError('Incremental native array changed a fixed piece.')


def checked_rebuild(expected,rebuilt,*,best=False):
    if not np.array_equal(expected[2 if best else 1],rebuilt[1]):raise ValueError('Sampler checkpoint has an incorrect current/best objective.')
    if not best and not np.array_equal(expected[CACHE_BASE:],rebuilt[CACHE_BASE:]):raise ValueError('Sampler checkpoint has incorrect physical-strand cache labels.')


def validate_capacity(n,replicas,total,flags):
    result=state_size(n,replicas,total,flags)
    if flags&FLAG and result['snapshot_bytes']>MAX_SNAPSHOT_BYTES:
        raise ValueError(f"Incremental sampler supports at most {result['snapshot_replica_cap']} replicas within the durable snapshot bound.")
    return result


def charge_restore(before,after,stats):
    """Count scratch validation work without changing native search counters."""
    stats['restore_board_rebuilds']=stats.get('restore_board_rebuilds',0)+before.shape[1]
    for index,key in enumerate(COUNTERS):
        delta=sum(((int(after[29+2*index,lane])+(int(after[30+2*index,lane])<<32))-
                   (int(before[29+2*index,lane])+(int(before[30+2*index,lane])<<32)))%(1<<64) for lane in range(before.shape[1]))
        stats['restore_'+key]=stats.get('restore_'+key,0)+delta
