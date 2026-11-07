"""Bounded CUDA domain search with durable native state and portable work cubes.

Native checkpoints and coverage are trusted local progress, not imported proofs.
The app stops at its first independently validated solution and never acknowledges
one as an infeasible branch. A portable frontier intentionally discards scratch.
"""
from __future__ import annotations
import base64
from copy import deepcopy
import math
import time
import zlib
import numpy as np
from constraint_work import WorkScope, frontier_snapshot, restore_frontier, digest, CoverageLedger

COUNTERS=('decisions','domain_removals','support_checks','backjumps','learned_clauses','strand_joins','clause_units','solutions')
MAX_BYTES=512*1024*1024

def split_cubes(cubes,count):
    """Disjoint partitions; inspect each created row once, not every live row."""
    import heapq
    cubes=list(cubes)
    if len(cubes)>=count:return cubes
    heap=[];ticket=0
    def push(row):
        nonlocal ticket
        width,axis=max((mask.bit_count(),j) for j,mask in enumerate(row))
        heapq.heappush(heap,(-width,-ticket,axis,row));ticket+=1
    for row in cubes:push(row)
    while len(heap)<count and heap and heap[0][0]<-1:
        width,_,axis,row=heapq.heappop(heap);mask=row[axis];part=0
        for _ in range((-width)//2):bit=mask&-mask;part|=bit;mask^=bit
        for value in (part,mask):copy=list(row);copy[axis]=value;push(tuple(copy))
    return [item[3] for item in heap]

def pack_native(engine):
    payload=engine.checkpoint();array=payload.pop('state')
    payload['shape']=list(array.shape)
    payload['compressed_state']=base64.b64encode(zlib.compress(array.tobytes(),1)).decode('ascii')
    return payload

def unpack_native(payload,engine):
    payload=dict(payload)
    shape=payload.pop('shape',None)
    if shape!=[engine.rows,engine.lanes]:raise ValueError('Native domain checkpoint dimensions differ.')
    encoded=payload.pop('compressed_state',None)
    if not isinstance(encoded,str) or len(encoded)>MAX_BYTES*2:raise ValueError('Native domain checkpoint exceeds encoded bound.')
    expected=engine.rows*engine.lanes*4
    decoder=zlib.decompressobj();raw=decoder.decompress(base64.b64decode(encoded,validate=True),expected+1)
    if len(raw)!=expected or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:raise ValueError('Malformed compressed domain state.')
    payload['state']=np.frombuffer(raw,dtype=np.uint32).reshape(shape).copy()
    return payload

def solve(problem,seconds=0,seed=0,target='gold',resume=None,stop=None,progress=None,
          replicas=128,backend='auto',learning_state=None,options=None,work_frontier=None):
    from gpu_domain_round24 import DomainCUDA
    if target not in ('gold','edges'):raise ValueError('Invalid objective.')
    if type(replicas)is not int or not 1<=replicas<=512:raise ValueError('Domain CUDA supports at most512 lanes within a512MiB state budget; use128 initially.')
    if backend not in ('auto','cuda'):raise ValueError('Domain-trail search currently requires NVIDIA CUDA.')
    if not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid time limit.')
    stop=stop or (lambda:False);progress=progress or (lambda event:None);options=options or {}
    if any(not row for row in problem.domains):return {'status':'infeasible','complete':True,'nodes':0,'stats':{key:0 for key in COUNTERS},'checkpoint':None}
    scope=WorkScope.for_problem(problem,target);started=time.monotonic();engine=None;ledger=None
    queued=[scope.root];totals={key:0 for key in COUNTERS};completed=0;last_checkpoint=last_progress=started
    try:
        if resume is not None and work_frontier is not None:raise ValueError('Choose native resume or a portable frontier, not both.')
        if resume is not None:
            if not isinstance(resume,dict) or resume.get('version')!=1 or resume.get('digest')!=digest({k:v for k,v in resume.items() if k!='digest'}):raise ValueError('Invalid domain runner checkpoint checksum.')
            saved_scope=WorkScope.restore(resume['scope'])
            if saved_scope!=scope:raise ValueError('Domain runner scope differs.')
            remaining=restore_frontier(resume['frontier'],scope)
            queued=[tuple(int(x,16) for x in row) for row in resume['queue']]
            frontier_snapshot(scope,queued)
            totals=dict(resume['totals']);completed=resume['completed_lanes']
            if set(totals)!=set(COUNTERS) or any(type(v)is not int or v<0 for v in totals.values()) or type(completed)is not int or completed<0:raise ValueError('Invalid domain progress counters.')
            native=resume.get('native')
            if not native and frontier_snapshot(scope,queued)!=resume['frontier']:raise ValueError('Queued and portable remaining work disagree.')
            if stop():
                current={key:totals[key]+resume.get('active_stats',{}).get(key,0) for key in COUNTERS}
                return {'status':'stopped','complete':False,'nodes':current['decisions'],'stats':current,'checkpoint':resume}
            if native and len(native['cubes'])<=replicas:
                cubes=[tuple(int(x,16) for x in row) for row in native['cubes']]
                engine=DomainCUDA(problem,cubes,target=target,max_memory_bytes=MAX_BYTES)
                engine.restore(unpack_native(native,engine))
                if engine.delivered:raise ValueError('Enumeration checkpoints cannot be used as infeasibility searches.')
                actual=frontier_snapshot(scope,(*queued,*engine.remaining_domain_cubes()))
                if actual!=resume['frontier']:raise ValueError('Native state and portable remaining work disagree.')
            else:
                queued=list(remaining)
                if native:
                    for key,value in resume['active_stats'].items():
                        if key in totals:totals[key]+=value
        elif work_frontier is not None:queued=list(restore_frontier(work_frontier,scope))
        if learning_state and options.get('shared_coverage') and learning_state.get('_coverage_path'):
            ledger=CoverageLedger(learning_state['_coverage_path'],scope)
            queued,_=ledger.remaining(queued)
        if engine is None:queued=split_cubes(queued,replicas)
    except BaseException:
        if engine is not None:engine.close()
        if ledger is not None:ledger.close()
        raise
    state_revision=0;cached_revision=-1;cached_payload=None
    def stats():
        live=engine.stats() if engine else {}
        return {**{key:totals[key]+live.get(key,0) for key in COUNTERS},
            'gpu_domain_lanes':live.get('lanes',0),'gpu_domain_active':live.get('active',0),
            'gpu_domain_queued':len(queued),'gpu_domain_completed':completed+live.get('exhausted',0),
            'gpu_domain_state_bytes':live.get('state_bytes',0)}
    def checkpoint():
        nonlocal cached_revision,cached_payload
        # GPU dispatch, batch ownership, or accumulated totals invalidate this
        # private snapshot. A deadline crossed while saving can reuse it exactly.
        if cached_revision==state_revision:return cached_payload
        native=pack_native(engine) if engine else None
        payload={'version':1,'scope':scope.payload(),'queue':[[hex(x) for x in row] for row in queued],
            'native':native,'totals':dict(totals),'completed_lanes':completed,
            'active_stats':engine.stats() if engine else {},
            'frontier':frontier_snapshot(scope,(*queued,*(engine.remaining_domain_cubes() if engine else ())))}
        payload['digest']=digest(payload)
        cached_revision=state_revision;cached_payload=payload
        return payload
    try:
        work=256
        while True:
            if stop() or seconds and time.monotonic()-started>=seconds:
                current=stats();return {'status':'stopped' if stop() else 'timeout','complete':False,'nodes':current['decisions'],'stats':current,'checkpoint':checkpoint()}
            if engine is None and queued:
                batch,queued=queued[:replicas],queued[replicas:]
                state_revision+=1
                engine=DomainCUDA(problem,batch,target=target,max_memory_bytes=MAX_BYTES)
            if engine is None:
                return {'status':'infeasible','complete':True,'nodes':totals['decisions'],'stats':stats(),'checkpoint':checkpoint()}
            if stop() or seconds and time.monotonic()-started>=seconds:
                current=stats();return {'status':'stopped' if stop() else 'timeout','complete':False,'nodes':current['decisions'],'stats':current,'checkpoint':checkpoint()}
            state_revision+=1
            before=time.monotonic();states=engine.step(work);elapsed=time.monotonic()-before
            if elapsed>.2:work=max(16,work//2)
            elif elapsed<.03:work=min(4096,work*2)
            candidates=engine.candidates() if np.any(states==1) else []
            if candidates:
                item=candidates[0];current=stats()
                return {'status':'solved' if item['validation']['valid'] else 'edge_perfect','complete':True,'codes':item['codes'],'validation':item['validation'],
                        'nodes':current['decisions'],'stats':current,'checkpoint':checkpoint()}
            if np.all(states==2):
                if engine.delivered:raise RuntimeError('An enumerated lane cannot establish infeasibility.')
                if ledger:
                    for row in engine.cubes:ledger.record_completed(row,status='infeasible',complete=True,evidence={'kind':'cuda_domain_exhaustion'})
                current=engine.stats()
                for key in COUNTERS:totals[key]+=current[key]
                completed+=engine.lanes;engine.close();engine=None;state_revision+=1
                continue
            now=time.monotonic()
            if now-last_progress>=1:
                current=stats();event={'event':'progress','nodes':current['decisions'],'stats':current}
                boards=engine.state[engine.offsets['board']:engine.offsets['board']+problem.n].get()
                lane=int(np.argmax(np.sum(boards!=0xffffffff,axis=0)))
                event['codes']=[-1 if int(x)==0xffffffff else int(x) for x in boards[:,lane]]
                progress(event);last_progress=now
            if now-last_checkpoint>=30:
                current=stats()
                # Consumers own their event payload and may modify it. Keep the
                # reusable private snapshot isolated from callback mutations.
                progress({'event':'checkpoint','payload':deepcopy(checkpoint()),'nodes':current['decisions'],'stats':current})
                last_checkpoint=time.monotonic()
    finally:
        if engine is not None:engine.close()
        if ledger is not None:ledger.close()
