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

BASE_COUNTERS=('decisions','domain_removals','support_checks','backjumps','learned_clauses','strand_joins','clause_units','solutions')
FORK_COUNTERS=('fork_parents','fork_children','fork_root_removals','fork_root_checks','fork_reused_assignments','fork_reused_unions')
COUNTERS=BASE_COUNTERS+('joint_support_checks','joint_subset_checks','joint_exclusions','joint_word_scans','joint_tile_visits')+FORK_COUNTERS+('event_enqueues','event_cell_scans','event_inventory_checks','event_queue_word_scans','event_inventory_words')+('precise_probes','precise_hits','precise_fallbacks','precise_stores','precise_invalidations','precise_support_words','precise_support_values','precise_dropped')
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

def pack_native(engine,snapshot=None):
    payload=dict(snapshot) if snapshot is not None else engine.checkpoint();array=payload.pop('state')
    payload['shape']=list(array.shape)
    payload['compressed_state']=base64.b64encode(zlib.compress(array.tobytes(),1)).decode('ascii')
    return payload

def unpack_native(payload,engine):
    if isinstance(payload,dict) and 'factored_native_version' in payload:
        from gpu_checkpoint_codec_round25 import decode_native
        return decode_native(payload,engine)
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
    scope=WorkScope.for_problem(problem,target);started=time.monotonic();engine=None;ledger=None;exchange=None
    joint_support=options.get('gpu_joint_support',False)
    if type(joint_support)is not bool:raise ValueError('gpu_joint_support must be boolean.')
    event_propagation=options.get('gpu_event_propagation',False)
    if type(event_propagation)is not bool:raise ValueError('gpu_event_propagation must be boolean.')
    engine_options={'joint_support':True} if joint_support else {}
    if event_propagation:engine_options['event_propagation']=True
    precise_reasons=options.get('gpu_precise_reasons',False)
    if type(precise_reasons)is not bool:raise ValueError('gpu_precise_reasons must be boolean.')
    if precise_reasons:engine_options['precise_reasons']=True
    propagated_forks=options.get('gpu_propagated_forks',False)
    if type(propagated_forks)is not bool:raise ValueError('gpu_propagated_forks must be boolean.')
    incremental_checkpoint=options.get('gpu_incremental_checkpoint',False)
    if type(incremental_checkpoint)is not bool:raise ValueError('gpu_incremental_checkpoint must be boolean.')
    from gpu_checkpoint_round25 import is_reference,resolve_checkpoint,CheckpointStore,StoreCapacity,StoreFailure
    checkpoint_directory=(learning_state or {}).get('_gpu_checkpoint_dir')
    if (incremental_checkpoint or is_reference(resume)) and not checkpoint_directory:raise ValueError('Incremental GPU checkpoints require a run-local checkpoint directory.')
    if is_reference(resume):resume=resolve_checkpoint(resume,checkpoint_directory)
    storage_keys=('gpu_checkpoint_host_copies','gpu_checkpoint_reused_chunks','gpu_checkpoint_fallbacks','gpu_checkpoint_rollbacks',
                  'gpu_checkpoint_packed_bytes','gpu_checkpoint_compressed_bytes','gpu_checkpoint_omitted_bytes','gpu_checkpoint_store_bytes')
    storage=dict.fromkeys(storage_keys,0);saved_storage=resume.get('checkpoint_storage',{}) if isinstance(resume,dict) else {}
    if (not isinstance(saved_storage,dict) or not set(saved_storage)<=set(storage_keys) or any(type(v)is not int or v<0 for v in saved_storage.values())):raise ValueError('Invalid GPU checkpoint telemetry.')
    storage.update(saved_storage);recovery=resume.get('checkpoint_recovery') if isinstance(resume,dict) else None
    if recovery is not None:
        if not isinstance(recovery,dict) or type(recovery.get('generations_rolled_back'))is not int or recovery['generations_rolled_back']<1:raise ValueError('Invalid checkpoint recovery metadata.')
        storage['gpu_checkpoint_rollbacks']+=recovery['generations_rolled_back']
    checkpoint_store=None;storage_status='off';storage_reason=''
    lane_balance=options.get('gpu_lane_balance',False)
    if type(lane_balance)is not bool:raise ValueError('gpu_lane_balance must be boolean.')
    from gpu_lane_pool_round25 import LanePool,COUNTERS as LANE_COUNTERS,GAUGES as LANE_GAUGES
    saved_balance_stats=deepcopy(resume.get('balance_stats',{})) if isinstance(resume,dict) else {}
    if (not isinstance(saved_balance_stats,dict) or not set(saved_balance_stats)<=set(LANE_COUNTERS+LANE_GAUGES)
            or any(type(v)is not int or v<0 for v in saved_balance_stats.values())):raise ValueError('Invalid GPU lane balance statistics.')
    pool=None
    sharing=options.get('gpu_shared_clauses',False)
    if type(sharing)is not bool:raise ValueError('gpu_shared_clauses must be boolean.')
    dormant_sharing=deepcopy(resume.get('sharing')) if isinstance(resume,dict) else None
    if dormant_sharing is not None and not isinstance(dormant_sharing,dict):raise ValueError('Malformed GPU sharing checkpoint.')
    saved_sharing_stats=deepcopy(resume.get('sharing_stats',{})) if isinstance(resume,dict) else {}
    if not isinstance(saved_sharing_stats,dict):raise ValueError('Invalid saved GPU sharing telemetry.')
    if saved_sharing_stats:
        from gpu_shared_round25 import COUNTERS as SHARED_COUNTERS,GAUGES as SHARED_GAUGES
        if (not isinstance(saved_sharing_stats,dict) or not set(saved_sharing_stats)<=set(SHARED_COUNTERS+SHARED_GAUGES)
                or any(type(value)is not int or value<0 for value in saved_sharing_stats.values())):raise ValueError('Invalid saved GPU sharing telemetry.')
    portable_restart=False;preparing=False;prepared_owner=None;global_unsat=False
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
            saved_totals=resume['totals'];completed=resume['completed_lanes']
            if (not isinstance(saved_totals,dict) or not set(BASE_COUNTERS)<=set(saved_totals)<=set(COUNTERS)
                    or any(type(v)is not int or v<0 for v in saved_totals.values()) or type(completed)is not int or completed<0):
                raise ValueError('Invalid domain progress counters.')
            totals={**dict.fromkeys(COUNTERS,0),**saved_totals}
            native=resume.get('native')
            saved_balance=resume.get('lane_balance',False)
            if type(saved_balance)is not bool or saved_balance != ('lane_pool' in resume):raise ValueError('Invalid saved lane balance policy.')
            saved_forks=resume.get('propagated_forks',False);saved_preparing=resume.get('preparing',False)
            if type(saved_forks)is not bool or type(saved_preparing)is not bool or saved_preparing and (not saved_forks or not native):
                raise ValueError('Invalid saved root preparation policy.')
            saved_owner=resume.get('prepared_owner')
            if saved_owner is not None:
                if not saved_forks or not native:raise ValueError('Invalid prepared batch ownership.')
                saved_owner=tuple(int(x,16) for x in saved_owner);frontier_snapshot(scope,[saved_owner])
            if not native and frontier_snapshot(scope,queued)!=resume['frontier']:raise ValueError('Queued and portable remaining work disagree.')
            if stop():
                current={key:totals[key]+resume.get('active_stats',{}).get(key,0) for key in COUNTERS}
                return {'status':'stopped','complete':False,'nodes':current['decisions'],'stats':{**current,**saved_sharing_stats,**saved_balance_stats,**storage},'checkpoint':resume,'shared_conflicts':(dormant_sharing or {}).get('verified')}
            native_joint=native.get('binding',{}).get('joint_support',False) if native else False
            native_event=native.get('binding',{}).get('event_propagation',False) if native else False
            native_precise=native.get('binding',{}).get('precise_reasons',False) if native else False
            if type(native_joint)is not bool or type(native_event)is not bool or type(native_precise)is not bool:raise ValueError('Invalid saved domain feature policy.')
            if native and len(native['cubes'])<=replicas and native_joint==joint_support and saved_forks==propagated_forks and native_event==event_propagation and native_precise==precise_reasons and saved_balance==lane_balance:
                cubes=[tuple(int(x,16) for x in row) for row in native['cubes']]
                engine=DomainCUDA(problem,cubes,target=target,max_memory_bytes=MAX_BYTES,**engine_options)
                engine.restore(unpack_native(native,engine))
                preparing=saved_preparing;prepared_owner=saved_owner
                if preparing and (engine.lanes!=1 or bool(engine.cp.any(engine.state[1]!=0))):raise ValueError('Preparation checkpoint contains decisions or multiple lanes.')
                if not lane_balance and prepared_owner is not None and any(any(value&~allowed for value,allowed in zip(row,prepared_owner)) for row in engine.cubes):
                    raise ValueError('Prepared batch escapes its owner.')
                if engine.delivered:raise ValueError('Enumeration checkpoints cannot be used as infeasibility searches.')
                actual=frontier_snapshot(scope,(*queued,*engine.remaining_domain_cubes()))
                if actual!=resume['frontier']:raise ValueError('Native state and portable remaining work disagree.')
                if lane_balance:pool=LanePool.restore(scope,queued,engine,resume['lane_pool']);queued=pool.queue
            else:
                queued=list(remaining)
                if native:
                    # Changing propagation policy or lane count retains the exact
                    # portable remainder but intentionally reconstructs scratch.
                    portable_restart=True
                    for key,value in resume['active_stats'].items():
                        if key in totals:totals[key]+=value
        elif work_frontier is not None:queued=list(restore_frontier(work_frontier,scope))
        if learning_state and options.get('shared_coverage') and learning_state.get('_coverage_path'):
            ledger=CoverageLedger(learning_state['_coverage_path'],scope)
            if pool is None and not (resume and resume.get('lane_balance') and lane_balance):queued,_=ledger.remaining(queued) # Live owner groups retire only through their complete descendant set.
        if engine is None and not propagated_forks and not (resume and resume.get('lane_balance') and lane_balance and not portable_restart):queued=split_cubes(queued,replicas)
        if lane_balance and pool is None:
            if resume and resume.get('lane_balance') and not resume.get('native') and not portable_restart:
                pool=LanePool.restore(scope,queued,None,resume['lane_pool'])
            else:pool=LanePool(scope,queued,counters=saved_balance_stats)
            queued=pool.queue
        if incremental_checkpoint:
            try:
                checkpoint_store=CheckpointStore(checkpoint_directory,scope.payload(),previous=resume.get('checkpoint_origin') if isinstance(resume,dict) else None)
                storage['gpu_checkpoint_store_bytes']=checkpoint_store._usage()[0];storage_status='ready'
            except (StoreCapacity,StoreFailure,OSError) as exc:
                checkpoint_store=None;storage_status='legacy_fallback';storage_reason=str(exc);storage['gpu_checkpoint_fallbacks']+=1
        if sharing:
            from gpu_shared_round25 import GPUClauseExchange,SharingStopped
            exchange=GPUClauseExchange(problem,target,store=(learning_state or {}).get('_shared_conflicts'),resume=dormant_sharing)
    except BaseException:
        if engine is not None:engine.close()
        if ledger is not None:ledger.close()
        if exchange is not None:exchange.close()
        raise
    state_revision=0;cached_revision=-1;cached_payload=None;last_sharing=0.0
    def sharing_result():
        return {'shared_conflicts':exchange.verified_snapshot() if exchange else (dormant_sharing or {}).get('verified'),
                'gpu_sharing_status':exchange.store.status if exchange else 'off',
                'gpu_checkpoint_status':storage_status,'gpu_checkpoint_note':storage_reason,'checkpoint_recovery':recovery}
    def check_sharing():
        if stop() or seconds and time.monotonic()-started>=seconds:raise SharingStopped()
    def stats():
        live=engine.stats() if engine else {}
        return {**{key:totals[key]+live.get(key,0) for key in COUNTERS},**(exchange.public_stats() if exchange else saved_sharing_stats),**(pool.public_stats() if pool else saved_balance_stats),**storage,
            'gpu_domain_lanes':live.get('lanes',0),'gpu_domain_active':live.get('active',0),
            'gpu_domain_queued':len(queued),'gpu_domain_completed':completed+live.get('exhausted',0)-(pool.lanes.count(None) if pool else 0),
            'gpu_domain_state_bytes':live.get('state_bytes',0),'gpu_domain_portable_restart':int(portable_restart),
            'gpu_domain_preparing':int(preparing),'gpu_verified_root_unsat':int(global_unsat)}
    def checkpoint():
        nonlocal cached_revision,cached_payload,checkpoint_store,storage_status,storage_reason
        # GPU dispatch, batch ownership, or accumulated totals invalidate this
        # private snapshot. A deadline crossed while saving can reuse it exactly.
        if cached_revision==state_revision:return cached_payload
        frozen=engine.checkpoint() if incremental_checkpoint and engine else None
        if frozen is not None:storage['gpu_checkpoint_host_copies']+=1
        native=pack_native(engine) if engine and not incremental_checkpoint else None
        blobs={}
        if frozen is not None and checkpoint_store is not None:
            from gpu_checkpoint_codec_round25 import encode_native
            native,blobs,metrics=encode_native(engine,frozen,checkpoint_store.cache)
            storage['gpu_checkpoint_reused_chunks']+=metrics['reused_chunks']
            for source,key in [('packed_bytes','gpu_checkpoint_packed_bytes'),('compressed_bytes','gpu_checkpoint_compressed_bytes'),('omitted_bytes','gpu_checkpoint_omitted_bytes')]:storage[key]=metrics[source]
        elif frozen is not None:native=pack_native(engine,frozen)
        payload={'version':1,'scope':scope.payload(),'queue':[[hex(x) for x in row] for row in queued],
            'native':native,'totals':dict(totals),'completed_lanes':completed,
            'active_stats':(engine.stats(snapshot=frozen['state']) if frozen is not None else engine.stats()) if engine else {},
            'frontier':frontier_snapshot(scope,(*queued,*((engine.remaining_domain_cubes(snapshot=frozen['state']) if frozen is not None else engine.remaining_domain_cubes()) if engine else ())))}
        if exchange is not None:payload.update(sharing=exchange.snapshot(),sharing_stats=exchange.public_stats())
        elif dormant_sharing is not None:payload.update(sharing=deepcopy(dormant_sharing),sharing_stats=deepcopy(saved_sharing_stats))
        if pool is not None:payload.update(lane_balance=True,lane_pool=pool.snapshot(),balance_stats=pool.public_stats())
        elif saved_balance_stats:payload['balance_stats']=deepcopy(saved_balance_stats)
        if propagated_forks:payload.update(propagated_forks=True,preparing=preparing,prepared_owner=[hex(x) for x in prepared_owner] if prepared_owner is not None else None)
        if isinstance(resume,dict) and resume.get('checkpoint_origin') is not None:payload['checkpoint_origin']=deepcopy(resume['checkpoint_origin'])
        if incremental_checkpoint or saved_storage:payload['checkpoint_storage']=dict(storage)
        payload['digest']=digest(payload)
        if checkpoint_store is not None:
            try:
                reference=checkpoint_store.write(payload,blobs);storage_status='incremental'
                storage['gpu_checkpoint_store_bytes']=checkpoint_store._usage()[0];payload=reference
            except (StoreCapacity,StoreFailure,OSError) as exc:
                storage_status='legacy_fallback';storage_reason=str(exc);storage['gpu_checkpoint_fallbacks']+=1
                checkpoint_store=None
                if frozen is not None:payload['native']=pack_native(engine,frozen)
                payload['checkpoint_storage']=dict(storage);payload['digest']=digest({k:v for k,v in payload.items() if k!='digest'})
        cached_revision=state_revision;cached_payload=payload
        return payload
    def finish_verified_root():
        nonlocal engine,queued,preparing,prepared_owner,state_revision,global_unsat,pool
        if engine is not None:
            current=engine.stats()
            for key in COUNTERS:totals[key]+=current.get(key,0)
            engine.close();engine=None
        queued=[];preparing=False;prepared_owner=None;global_unsat=True;state_revision+=1
        if pool is not None:pool=LanePool(scope,[],counters=pool.public_stats());queued=pool.queue
        if ledger:ledger.record_completed(scope.root,status='infeasible',complete=True,evidence={'kind':'independently_verified_original_root_conflict'})
        saved=checkpoint();current=stats()
        return {'status':'infeasible','complete':True,'nodes':current['decisions'],'stats':current,'checkpoint':saved,
                'termination_evidence':'independently_verified_original_root_conflict',**sharing_result()}
    try:
        work=256
        while True:
            if stop() or seconds and time.monotonic()-started>=seconds:
                saved=checkpoint();current=stats();return {'status':'stopped' if stop() else 'timeout','complete':False,'nodes':current['decisions'],'stats':current,'checkpoint':saved,**sharing_result()}
            if exchange is not None and (engine is None or time.monotonic()-last_sharing>=.25):
                state_revision+=1
                try:contradiction=exchange.sync(engine,check_sharing)
                except SharingStopped:continue
                last_sharing=time.monotonic()
                if contradiction:return finish_verified_root()
            if engine is None and queued:
                count=1 if propagated_forks else replicas
                if pool is not None:batch=pool.take_batch(count);queued=pool.queue
                else:batch,queued=queued[:count],queued[count:]
                preparing=propagated_forks;prepared_owner=batch[0] if propagated_forks else None
                state_revision+=1
                engine=DomainCUDA(problem,batch,target=target,max_memory_bytes=MAX_BYTES,**engine_options)
            if engine is None:
                saved=checkpoint();return {'status':'infeasible','complete':True,'nodes':totals['decisions'],'stats':stats(),'checkpoint':saved,**sharing_result()}
            if stop() or seconds and time.monotonic()-started>=seconds:
                saved=checkpoint();current=stats();return {'status':'stopped' if stop() else 'timeout','complete':False,'nodes':current['decisions'],'stats':current,'checkpoint':saved,**sharing_result()}
            if exchange is not None and exchange.needs_initial_refresh(engine):
                state_revision+=1;exchange.refresh(engine)
            state_revision+=1
            before=time.monotonic()
            if preparing:
                states,ready=engine.propagate_root(work)
                if ready and not np.all(states==2):
                    parent_stats=engine.stats();remaining=engine.remaining_domain_cubes()
                    totals['fork_parents']+=1;totals['fork_root_removals']+=parent_stats['domain_removals']
                    totals['fork_root_checks']+=parent_stats['support_checks']
                    parts=split_cubes(remaining,replicas)
                    if len(parts)>1:
                        root_meta=engine.state[:11].get()
                        extras=queued[:max(0,replicas-len(parts))] if pool is not None else []
                        child=engine.fork_root(parts,max_memory_bytes=MAX_BYTES,extra_cubes=extras)
                        if pool is not None:pool.fork(len(parts),len(extras));queued=pool.queue;prepared_owner=None
                        totals['fork_children']+=len(parts)
                        totals['fork_reused_assignments']+=int(root_meta[9,0])*len(parts)
                        totals['fork_reused_unions']+=int(root_meta[10,0])*len(parts)
                        for key in COUNTERS:totals[key]+=parent_stats.get(key,0)
                        engine.close();engine=child;states=np.zeros(engine.lanes,dtype=int)
                    preparing=False
                elif np.all(states==2):preparing=False
            else:states=engine.step(work)
            elapsed=time.monotonic()-before
            if elapsed>.2:work=max(16,work//2)
            elif elapsed<.03:work=min(4096,work*2)
            candidates=engine.candidates() if np.any(states==1) else []
            if candidates:
                item=candidates[0];saved=checkpoint();current=stats()
                return {'status':'solved' if item['validation']['valid'] else 'edge_perfect','complete':True,'codes':item['codes'],'validation':item['validation'],
                        'nodes':current['decisions'],'stats':current,'checkpoint':saved,**sharing_result()}
            if pool is not None and not preparing:
                old,retired,finished=pool.retire(engine,states)
                for key in COUNTERS:totals[key]+=old.get(key,0)
                completed+=retired
                if ledger:
                    for row in finished:ledger.record_completed(row,status='infeasible',complete=True,evidence={'kind':'cuda_domain_owned_descendants_exhausted'})
                if propagated_forks and engine.lanes==1 and not any(owner is not None for owner in pool.lanes) and queued:
                    engine.close();engine=None;pool.lanes=[];preparing=False;prepared_owner=None;state_revision+=1
                    continue
                changed=pool.refill(engine)
                old=pool.donate(engine)
                if old is not None:
                    for key in COUNTERS:totals[key]+=old.get(key,0)
                    changed=True
                if changed and exchange is not None:exchange.invalidate()
                queued=pool.queue;state_revision+=1
                if not any(owner is not None for owner in pool.lanes):
                    engine.close();engine=None;pool.lanes=[];preparing=False;prepared_owner=None
                    continue
                states=engine.state[0].get()
            if np.all(states==2):
                if exchange is not None and queued:
                    state_revision+=1
                    try:contradiction=exchange.sync(engine,check_sharing)
                    except SharingStopped:continue
                    last_sharing=time.monotonic()
                    if contradiction:return finish_verified_root()
                if engine.delivered:raise RuntimeError('An enumerated lane cannot establish infeasibility.')
                if ledger:
                    for row in ([prepared_owner] if prepared_owner is not None else engine.cubes):ledger.record_completed(row,status='infeasible',complete=True,evidence={'kind':'cuda_domain_exhaustion'})
                current=engine.stats()
                for key in COUNTERS:totals[key]+=current.get(key,0)
                completed+=engine.lanes;engine.close();engine=None;preparing=False;prepared_owner=None;state_revision+=1
                continue
            now=time.monotonic()
            if now-last_progress>=1:
                current=stats();event={'event':'progress','nodes':current['decisions'],'stats':current,**sharing_result()}
                boards=engine.state[engine.offsets['board']:engine.offsets['board']+problem.n].get()
                lane=int(np.argmax(np.sum(boards!=0xffffffff,axis=0)))
                event['codes']=[-1 if int(x)==0xffffffff else int(x) for x in boards[:,lane]]
                progress(event);last_progress=now
            if now-last_checkpoint>=30:
                saved=checkpoint();current=stats()
                # Consumers own their event payload and may modify it. Keep the
                # reusable private snapshot isolated from callback mutations.
                progress({'event':'checkpoint','payload':deepcopy(saved),'nodes':current['decisions'],'stats':current,**sharing_result()})
                last_checkpoint=time.monotonic()
    finally:
        if engine is not None:engine.close()
        if ledger is not None:ledger.close()
        if exchange is not None:exchange.close()
