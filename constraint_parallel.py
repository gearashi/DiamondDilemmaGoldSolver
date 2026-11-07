"""Spawn-based exact DFS over a durable queue of disjoint remaining regions.

Every committed job checkpoint owns its unfinished search tree. Splitting turns
pending sibling choices into disjoint domain boxes, without changing the puzzle
fingerprint. Worker caches are private to a job: a proof made under a restricted
job root must never escape into another job. Shared SAT conflicts use a separate
original-domain verifier in every process; serialized claims are re-proved before
application or rebroadcast. A process failure retains its last
committed checkpoint, so uncommitted work may replay but coverage is not lost.
Workers use bounded adaptive leases; an idle peer can request a cooperative
checkpoint before the coordinator partitions its exact unfinished frontier.
The caller atomically persists the returned/progress checkpoint payloads.
"""
from __future__ import annotations
import copy
from collections import OrderedDict
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
import traceback
from constraint_dfs import solve as dfs_solve, _checkpoint_digest, _NogoodStore
from search_options import normalize_options, validate_workers
from constraint_round24 import MAX_GAUGE_KEYS as RESEARCH_MAX_GAUGES
from constraint_knowledge import KnowledgeStore
from constraint_shared_conflicts import VerifiedConflictStore, GAUGE_KEYS as SHARED_GAUGES
from constraint_mask_conflicts import VerifiedMaskConflictStore, GAUGE_KEYS as MASK_GAUGES, _digest as _mask_digest, MAX_RECORDS as MASK_MAX_RECORDS, MAX_LITERALS as MASK_MAX_LITERALS
from constraint_deep import CertificateStore
from constraint_policies import reorder_frontier,worker_options
from constraint_work import WorkScope,CoverageLedger,checkpoint_boxes,exclusion_box,subtract_many

VERSION=1
SLICE_SECONDS=.75
MAX_SLICE_SECONDS=3.0
MAX_PREPARATION_SECONDS=60.0
MAX_PREPARATION_STALLS=7
MIN_YIELD_SECONDS=.05
STOP_GRACE_SECONDS=5.0
JOB_CACHE_LIMIT=2
AFFINITY_SLICES=4


def _digest(payload):
    return hashlib.sha256(json.dumps({k:v for k,v in payload.items() if k!='digest'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _domain_hash(problem):
    return hashlib.sha256(json.dumps([sorted(set(row)) for row in problem.domains],separators=(',',':')).encode()).hexdigest()


def _stopped(stop):
    return bool(stop() if callable(stop) else stop.is_set() if stop is not None else False)


def _progress_fields(event):
    # A worker checkpoint owns only one job. Only coordinator snapshots may
    # appear under the runner's top-level durable checkpoint field.
    excluded={'event','nodes','stats','elapsed_seconds','search_options','root_symmetry','checkpoint','knowledge','shared_conflicts','mask_conflicts','certificates'}
    return {key:value for key,value in event.items() if key not in excluded}


def _validate_dfs(problem,checkpoint,target,options):
    # Zero budget validates structure/digest without consuming any branch.
    checked=dfs_solve(problem,seconds=0,target=target,resume=checkpoint,options=options)
    return checked['checkpoint']


def _restricted_mask_claims(payload,domains):
    """Restrict optional parent claims, never transport proof authority.

    A parent-root negative conjunction remains a useful *candidate* under a
    descendant root. Intersect its literals with the child, drop conjunctions
    already impossible there and reverify every retained claim in the child.
    Root-bound certificates cannot be relabelled and are deliberately omitted.
    """
    if not isinstance(payload,dict) or payload.get('digest')!=_mask_digest(payload):
        raise ValueError('Malformed parent mask-claim payload.')
    version=payload.get('mask_conflicts_version')
    keys={'mask_conflicts_version','fingerprint','target','original','initial','records','digest'}
    if version==2:keys.add('certificates')
    if type(version)is not int or version not in (1,2) or set(payload)!=keys:
        raise ValueError('Malformed parent mask-claim payload.')
    original=payload['original'];parent=payload['initial'];records=payload['records']
    if (not isinstance(original,list) or not isinstance(parent,list) or len(parent)!=len(domains)
            or len(original)!=len(domains) or any(not isinstance(v,str) for v in original+parent)):
        raise ValueError('Malformed parent mask-claim scope.')
    try:parent=[int(v,16) for v in parent];original=[int(v,16) for v in original]
    except ValueError:raise ValueError('Malformed parent mask-claim scope.') from None
    if any(type(child)is not int or child<=0 or child&~old or old<=0 or old&~source
           for child,old,source in zip(domains,parent,original)):
        raise ValueError('Child mask-claim scope must narrow its parent.')
    if not isinstance(records,list) or len(records)>MASK_MAX_RECORDS:
        raise ValueError('Malformed parent mask-claim records.')
    narrowed=[]
    for record in records:
        if not isinstance(record,list) or len(record)>MASK_MAX_LITERALS:
            raise ValueError('Malformed parent mask-claim record.')
        cells=set();result=[];impossible=False
        for item in record:
            if (not isinstance(item,list) or len(item)!=2 or type(item[0])is not int
                    or not 0<=item[0]<len(domains) or item[0] in cells or not isinstance(item[1],str)):
                raise ValueError('Malformed parent mask-claim literal.')
            cell,encoded=item;cells.add(cell)
            try:mask=int(encoded,16)
            except ValueError:raise ValueError('Malformed parent mask-claim literal.') from None
            if mask<=0 or mask&~parent[cell]:raise ValueError('Parent mask claim exceeds its scope.')
            mask&=domains[cell]
            if not mask:impossible=True
            elif mask!=domains[cell]:result.append([cell,hex(mask)])
        if not impossible:narrowed.append(result)
    result=copy.deepcopy(payload);result['initial']=[hex(v) for v in domains];result['records']=narrowed
    if version==2:result['certificates']=[]
    result['digest']=_mask_digest(result)
    return result


def _root_copy(checkpoint,domains):
    result=copy.deepcopy(checkpoint)
    claims=result.pop('local_mask_conflicts',None)
    if claims is not None and all(domains):
        if not isinstance(claims,dict) or claims.get('fingerprint')!=checkpoint.get('fingerprint') or claims.get('target')!=checkpoint.get('target'):
            raise ValueError('Parent mask-claim puzzle or objective differs.')
        result['local_mask_conflicts']=_restricted_mask_claims(claims,domains)
    result.update(owned_root_domains=[hex(value) for value in domains],nodes=0,stats={key:0 for key in result['stats']},elapsed_seconds=0.0,
                  stack=[{'domains':[hex(value) for value in domains],'cell':-1,'choices':[],'next':0}])
    result['digest']=_checkpoint_digest(result)
    return result


def partition_checkpoint(problem,checkpoint,*,parts=2,options=None):
    """Return disjoint DFS roots whose union is exactly the unfinished tree.

    An expanded frame contributes only choices[next:], because its selected
    child is represented by the next frame. The unexpanded top contributes its
    current domains. Already explored siblings are never reintroduced. This
    function resets child counters; its caller must retire parent counters once.
    Pending symmetry roots cannot be split before their preparation completes.
    """
    if type(parts) is not int or parts<1: raise ValueError('parts must be positive.')
    target=checkpoint.get('target') if isinstance(checkpoint,dict) else None
    checkpoint=_validate_dfs(problem,checkpoint,target,options)
    if checkpoint.get('root_symmetry',{}).get('state')=='pending':
        return [copy.deepcopy(checkpoint)]
    roots=[]
    for frame in checkpoint['stack']:
        domains=[int(value,16) for value in frame['domains']]
        if frame['cell']>=0:
            masks=([int(value,16) for value in frame['choice_masks']] if 'choice_masks' in frame else [1<<code for code in frame['choices']])
            pending=masks[frame['next']:]
            if not pending: continue
            domains[frame['cell']]=sum(pending)
        if any(value==0 for value in domains): continue
        roots.append(domains)
    while len(roots)<parts:
        choices=[(min(value.bit_count() for value in domains if value.bit_count()>1),index)
                 for index,domains in enumerate(roots) if any(value.bit_count()>1 for value in domains)]
        if not choices: break
        _,index=min(choices);domains=roots.pop(index)
        cell=min((i for i,value in enumerate(domains) if value.bit_count()>1),key=lambda i:domains[i].bit_count())
        values=[code for code in range(3*problem.n) if domains[cell]&(1<<code)]
        middle=len(values)//2
        for subset in (values[:middle],values[middle:]):
            child=domains.copy();child[cell]=sum(1<<code for code in subset);roots.append(child)
    return [_root_copy(checkpoint,domains) for domains in roots]


class _WorkerCaches:
    """Bounded job-local mutable state plus separately scoped immutable tables."""
    def __init__(self,problem,limit=JOB_CACHE_LIMIT):
        self.problem,self.limit=problem,limit;self.jobs=OrderedDict();self.support=OrderedDict();self.sat_assistants={}
        self.binding=KnowledgeStore(problem,target='edges')
        self.stats={'job_cache_hits':0,'job_cache_misses':0,'job_cache_evictions':0,'support_reuses':0,'shared_sat_models':0,'shared_sat_job_views':0,'shared_sat_clauses':0}
    def close(self):
        for helper in self.sat_assistants.values():helper.close()
        self.sat_assistants.clear()
    def attach_shared_sat(self,state,request):
        if request['options'].get('shared_sat_assistant'):
            from constraint_job_sat import SharedSatAssistant
            target=request['target']
            if target not in self.sat_assistants:
                self.sat_assistants[target]=SharedSatAssistant(self.problem,target)
                self.stats['shared_sat_models']+=1
            state['_shared_sat']=self.sat_assistants[target]
        return state
    def key(self,request):
        return (request['id'],request.get('region_key'),request['target'],
                json.dumps(request['options'],sort_keys=True,separators=(',',':')))
    def support_key(self,request):
        if not self.binding.matches(self.problem,'edges'):raise ValueError('Worker source domains changed.')
        return (self.problem.fingerprint,self.binding.domains_sha256,request['target'],
                json.dumps(request['checkpoint'].get('root_symmetry'),sort_keys=True,separators=(',',':')),
                tuple(request['checkpoint'].get('owned_root_domains',request['checkpoint'].get('stack',[{}])[0].get('domains',()))))
    def get(self,request):
        key=self.key(request)
        if not self.binding.matches(self.problem,'edges'):raise ValueError('Worker source domains changed.')
        if key in self.jobs:
            self.stats['job_cache_hits']+=1;state=self.jobs.pop(key);self.jobs[key]=state;return self.attach_shared_sat(state,request)
        self.stats['job_cache_misses']+=1;state={};support_key=self.support_key(request)
        if support_key in self.support:
            table=self.support.pop(support_key);self.support[support_key]=table
            state={'_nogoods':_NogoodStore(self.problem.fingerprint,request['target'],initial_domains=table.initial_domains),
                   '_support':table}
            self.stats['support_reuses']+=1
        self.jobs[key]=state
        while len(self.jobs)>self.limit:self.jobs.popitem(last=False);self.stats['job_cache_evictions']+=1
        return self.attach_shared_sat(state,request)
    def remember(self,request,state):
        self.stats['shared_sat_job_views']=sum(owner.views for owner in self.sat_assistants.values())
        self.stats['shared_sat_clauses']=sum(owner.helper.stats['sat_assist_clauses'] for owner in self.sat_assistants.values())
        table=state.get('_support')
        if table is not None:
            key=self.support_key(request);self.support[key]=table;self.support.move_to_end(key)
            while len(self.support)>self.limit:self.support.popitem(last=False)


def _choose_job(queue,slot,owners,last_jobs,*,priority=False):
    """Prefer a warm job, but rotate after bounded consecutive slices."""
    previous,streak=last_jobs.get(slot,(None,0))
    if priority:
        chosen=queue[0]
    elif previous in queue and streak<AFFINITY_SLICES:
        chosen=previous
    else:
        warm=[job for job in queue if job!=previous and owners.get(job)==slot]
        alternative=[job for job in queue if job!=previous]
        # After the affinity allowance, oldest waiting work wins even when
        # another cached job is warm; otherwise three jobs could starve one.
        chosen=((alternative or queue) if streak>=AFFINITY_SLICES else (warm or alternative or queue))[0]
    hit=owners.get(chosen)==slot
    queue.remove(chosen);owners[chosen]=slot
    last_jobs[slot]=(chosen,streak+1 if chosen==previous else 1)
    return chosen,hit


def _pending_work_hint(checkpoint):
    """A scheduling estimate only; exact partitioning remains authoritative."""
    width=0
    for frame in checkpoint.get('stack',()):
        if frame.get('cell',-1)>=0:
            choices=frame.get('choice_masks',frame.get('choices',()))
            width+=max(0,len(choices)-frame.get('next',0))
        else:
            width+=max((int(value,16).bit_count() for value in frame['domains']),default=0)
    return width


def _slice_budget(streak,*,slots,active,queued,remaining,stalled=0):
    # Longer leases retain warm engine state and amortize rebuilding/checkpoint
    # serialization. A separate cooperative signal still interrupts them when
    # another worker needs work. Keep the crash-replay window bounded.
    factor=2**min(2,max(0,streak)) if active+queued>=slots else 1
    ordinary=min(MAX_SLICE_SECONDS,SLICE_SECONDS*factor)
    preparation=(min(MAX_PREPARATION_SECONDS,SLICE_SECONDS*2**min(MAX_PREPARATION_STALLS,stalled)) if stalled>0 else 0.)
    return min(max(ordinary,preparation),max(0.,remaining))


def _idle_yield_delay(stalled,budget):
    ordinary=max(MIN_YIELD_SECONDS,SLICE_SECONDS*.5)
    # An idle peer must not repeatedly cancel the same preparatory proof before
    # its assigned finite allowance ends. Global Stop/deadline never uses this.
    return max(ordinary,min(MAX_PREPARATION_SECONDS,budget)) if stalled else ordinary

def _record_frontier_progress(stalls,job_id,before,after):
    """Track committed search progress, ignoring attempt counters and clocks.

    An interrupted propagator may increment nodes without committing a single
    branch. Only the authoritative stack (domains/choices/cursors) or verified
    root preparation state resets this job's longer-preparation allowance.
    """
    moved=(before.get('stack')!=after.get('stack')
           or before.get('root_symmetry')!=after.get('root_symmetry'))
    stalls[job_id]=0 if moved else min(MAX_PREPARATION_STALLS,stalls.get(job_id,0)+1)
    return moved


def _restore_preparation_stalls(payload,jobs):
    if not isinstance(payload,list) or len(payload)>len(jobs):
        raise ValueError('Malformed preparation scheduling hints.')
    restored={}
    for item in payload:
        if (not isinstance(item,list) or len(item)!=2 or type(item[0])is not int or item[0] not in jobs
                or item[0] in restored or type(item[1])is not int or not 0<=item[1]<=MAX_PREPARATION_STALLS):
            raise ValueError('Malformed preparation scheduling hints.')
        restored[item[0]]=item[1]
    return restored

class _WorkerSignal:
    """Picklable per-lease yield plus global cancellation; Event read/wait API."""
    def __init__(self,cancel,yield_request):
        self.cancel,self.yield_request=cancel,yield_request
    def is_set(self):
        return self.cancel.is_set() or self.yield_request.is_set()
    def wait(self,timeout=None):
        deadline=None if timeout is None else time.monotonic()+timeout
        while not self.is_set():
            remaining=None if deadline is None else deadline-time.monotonic()
            if remaining is not None and remaining<=0:return False
            self.cancel.wait(.01 if remaining is None else min(.01,remaining))
        return True


def _worker(problem,connection,cancel):
    caches=_WorkerCaches(problem);knowledge=None;shared_conflicts=None;mask_conflicts=None;certificates=None
    try:
        while True:
            request=connection.recv()
            if request is None: return
            job_id=request['id']
            def progress(event):
                connection.send({'kind':'progress','id':job_id,'pid':os.getpid(),'event':event})
            try:
                learning=caches.get(request)
                if request['options'].get('persistent_learning',True) and request['target']=='gold':
                    if knowledge is None:knowledge=KnowledgeStore(problem,target=request['target'])
                    payload=request.get('knowledge')
                    if payload is not None:
                        knowledge.merge_snapshot(payload)
                    learning['_knowledge']=knowledge
                if request['options'].get('shared_conflicts',False):
                    if shared_conflicts is None:
                        shared_conflicts=VerifiedConflictStore(problem,target=request['target'])
                    if not shared_conflicts.matches(problem,request['target']):
                        raise ValueError('Worker shared conflicts differ from original domains or target.')
                    payload=request.get('shared_conflicts')
                    if payload is not None:shared_conflicts.import_snapshot(payload)
                    learning['_shared_conflicts']=shared_conflicts
                if request['options'].get('smart_learning'):
                    if mask_conflicts is None:mask_conflicts=VerifiedMaskConflictStore(problem,request['target'],proof_transport=request['options'].get('proof_transport',False),resumable_proof_checking=request['options'].get('resumable_proof_checking',False),mixed_proof_dag=request['options'].get('mixed_proof_dag',False),proof_indexing=request['options'].get('proof_indexing',False))
                    mask_conflicts.configure_policy(proof_transport=request['options'].get('proof_transport',False),
                        resumable_proof_checking=request['options'].get('resumable_proof_checking',False),mixed_proof_dag=request['options'].get('mixed_proof_dag',False),proof_indexing=request['options'].get('proof_indexing',False))
                    mask_conflicts.configure_isolated_producer(request['options'].get('isolated_proof_producer',False))
                    payload=request.get('mask_conflicts')
                    if payload is not None:mask_conflicts.import_snapshot(payload)
                    learning['_mask_conflicts']=mask_conflicts
                if request['options'].get('proof_certificates'):
                    if certificates is None:certificates=CertificateStore(problem,request['target'])
                    payload=request.get('certificates')
                    if payload is not None:certificates.load_snapshot(payload)
                    learning['_certificates']=certificates
                if (request['options'].get('extended_regions') or request['options'].get('durable_regions') or request['options'].get('region_certificates')) and request.get('regional_cache_dir') is not None:
                    learning['_regional_cache_dir']=request['regional_cache_dir']
                if request.get('sat_service') is not None and '_sat_service' not in learning:learning['_sat_service']=request['sat_service']
                if request.get('reasoning_cache_dir') is not None:learning['_reasoning_cache_dir']=request['reasoning_cache_dir']
                if request.get('regional_cache_dir') is not None:learning['_regional_cache_dir']=request['regional_cache_dir']
                result=dfs_solve(problem,seconds=request['seconds'],target=request['target'],
                                 resume=request['checkpoint'],stop=cancel.is_set,progress=progress,
                                 learning_state=learning,options=request['options'])
                caches.remember(request,learning)
                if request['options'].get('shared_conflicts',False):
                    result['shared_conflicts']=shared_conflicts.snapshot()
                if mask_conflicts is not None:result['mask_conflicts']=mask_conflicts.snapshot()
                if certificates is not None:result['certificates']=certificates.snapshot()
                connection.send({'kind':'result','id':job_id,'pid':os.getpid(),'result':result,'worker_cache':dict(caches.stats)})
            except BaseException as exc:
                connection.send({'kind':'error','id':job_id,'pid':os.getpid(),
                                 'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(limit=8)})
                return
    except (EOFError,BrokenPipeError,OSError):
        return
    finally:
        caches.close()
        if shared_conflicts is not None:shared_conflicts.close()
        if mask_conflicts is not None:mask_conflicts.close()
        connection.close()


def _record_backend(result,backends,reasons):
    backend=result.get('native_backend')
    reason=result.get('native_fallback_reason')
    # DFS initializes this placeholder before its first cancellation check.
    # Its counters may be inherited from an earlier compiled slice, so neither
    # the placeholder label nor cumulative counters describe actual new work.
    if backend=='not_started' or backend=='python' and reason=='Propagation has not started.':
        return
    if isinstance(backend,str):backends.add(backend)
    if isinstance(reason,str):reasons.add(reason)


def _record_search_loop(result,backends,reasons):
    backend=result.get('search_loop_backend')
    reason=result.get('search_loop_fallback_reason')
    # A cancelled setup carries the Python placeholder and possibly cumulative
    # native counters from an earlier slice; it contributes no new loop usage.
    if (backend not in ('compiled','python','mixed')
            or result.get('native_backend')=='not_started'
            or result.get('native_fallback_reason')=='Propagation has not started.'):
        return
    backends.update(('compiled','python') if backend=='mixed' else (backend,))
    if isinstance(reason,str) and reason:reasons.add(reason)


def _backend_label(backends):
    return next(iter(backends)) if len(backends)==1 else 'mixed' if backends else 'not_started'


def _add_stats(destination,source,*,retiring=False):
    for key,value in source.items():
        if key in RESEARCH_MAX_GAUGES:
            destination[key]=max(destination.get(key,0),value)
            continue
        if retiring and (key in SHARED_GAUGES or key in MASK_GAUGES or key.endswith('_entries') or key in ('forced_bonds','forced_clusters','loop_witnesses','sat_assist_clauses',
                            'hierarchical_subtable_states','hierarchical_subtable_support_slots','hierarchical_cached_summaries','hierarchical_cached_support_slots','gold_oracle_template_bytes')):continue
        destination[key]=destination.get(key,0)+value


def _validate_job(problem,job,target,options):
    if not isinstance(job,dict) or set(job)!={'id','region','checkpoint'} or type(job['id']) is not int or job['id']<0:
        raise ValueError('Malformed parallel job.')
    if not isinstance(job['region'],list) or len(job['region'])!=problem.n or any(not isinstance(v,str) for v in job['region']):
        raise ValueError('Malformed parallel job region.')
    try: region=[int(value,16) for value in job['region']]
    except ValueError as exc: raise ValueError('Malformed parallel region bitset.') from exc
    initial=[sum(1<<code for code in set(row)) for row in problem.domains]
    if any(value<0 or value&~allowed for value,allowed in zip(region,initial)):
        raise ValueError('Parallel region contains invalid orientations.')
    _validate_dfs(problem,job['checkpoint'],target,options)
    for frame in job['checkpoint']['stack']:
        if any(int(value,16)&~allowed for value,allowed in zip(frame['domains'],region)):
            raise ValueError('A saved DFS frame escapes its parallel job region.')
    return region


def _solve_parallel(problem,*,seconds=10,seed=0,target='gold',hints=None,stop=None,progress=None,
          resume=None,workers=1,options=None,learning_state=None):
    """Run bounded parallel exact DFS; workers may change on every resume.

    Checkpoints remain incomplete on timeout, cancellation or worker failure.
    A legacy DFS checkpoint imports directly. Process crashes or replaying an
    intermediate snapshot can repeat work after its last committed job state;
    this is explicitly recorded and never treated as subtree exhaustion.
    """
    began=time.monotonic();workers=validate_workers(workers)
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:
        raise ValueError('seconds must be finite and nonnegative.')
    if type(seed) is not int or target not in ('gold','edges'):raise ValueError('Invalid seed or target.')
    if learning_state is not None and not isinstance(learning_state,dict):raise ValueError('learning_state must be a dictionary.')
    source_options=resume.get('search_options') if isinstance(resume,dict) else None
    options=normalize_options(source_options if options is None else options)
    if learning_state is None:learning_state={}
    knowledge=learning_state.get('_knowledge')
    if knowledge is not None and (not isinstance(knowledge,KnowledgeStore) or not knowledge.matches(problem,target)):
        raise ValueError('Parallel knowledge differs from the original puzzle, domains or target.')
    if knowledge is None and options.get('persistent_learning',True) and target=='gold':
        knowledge=KnowledgeStore(problem,target=target);learning_state['_knowledge']=knowledge
    shared_conflicts=learning_state.get('_shared_conflicts')
    if shared_conflicts is not None and (not isinstance(shared_conflicts,VerifiedConflictStore) or not shared_conflicts.matches(problem,target)):
        raise ValueError('Parallel shared conflicts differ from original puzzle, domains or target.')
    if shared_conflicts is None and options.get('shared_conflicts',False):
        shared_conflicts=VerifiedConflictStore(problem,target=target);learning_state['_shared_conflicts']=shared_conflicts
    mask_conflicts=learning_state.get('_mask_conflicts')
    if mask_conflicts is not None and (not isinstance(mask_conflicts,VerifiedMaskConflictStore) or not mask_conflicts.matches(problem,target)):
        raise ValueError('Parallel mask conflicts differ from original domains or target.')
    if mask_conflicts is None and options.get('smart_learning'):
        mask_conflicts=VerifiedMaskConflictStore(problem,target,proof_transport=options['proof_transport'],resumable_proof_checking=options.get('resumable_proof_checking',False),mixed_proof_dag=options.get('mixed_proof_dag',False),proof_indexing=options.get('proof_indexing',False));learning_state['_mask_conflicts']=mask_conflicts
    if mask_conflicts is not None:
        mask_conflicts.configure_policy(proof_transport=options['proof_transport'],
            resumable_proof_checking=options['resumable_proof_checking'],mixed_proof_dag=options['mixed_proof_dag'],proof_indexing=options['proof_indexing'])
    if mask_conflicts is not None:mask_conflicts.configure_isolated_producer(options['isolated_proof_producer'])
    certificates=learning_state.get('_certificates')
    if certificates is not None and (not isinstance(certificates,CertificateStore)
            or certificates.problem.fingerprint!=problem.fingerprint or certificates.target!=target
            or certificates.root!=tuple(sum(1<<c for c in set(row)) for row in problem.domains)):
        raise ValueError('Parallel certificates differ from original domains or target.')
    if certificates is None and options.get('proof_certificates'):
        certificates=CertificateStore(problem,target);learning_state['_certificates']=certificates
    preferred=[]
    for cell,row in enumerate(problem.domains):
        hint=(hints.get(cell) if isinstance(hints,dict) else hints[cell] if isinstance(hints,(list,tuple)) and cell<len(hints) else None)
        if isinstance(hint,(list,tuple)):hint=next((v for v in hint if v in row),None)
        preferred.append(hint if type(hint)is int and hint in row else 3*cell if 3*cell in row else min(row,default=0))
    coverage_scope=WorkScope.for_problem(problem,target) if all(problem.domains) else None
    coverage=None
    queue_proof_checks=queue_proof_retirements=queue_proof_splits=0
    queue_proof_seen={}
    deadline=began+seconds
    jobs={};queue=[];active={};slots=[];started_workers=0;error=None;terminal=None
    previous_elapsed=0.;retired_nodes=0;retired_stats={};next_id=0;completed=0;splits=0;replayed=0
    root_symmetry=None;native_backends=set();native_reasons=set();last_checkpoint=began
    search_loop_backends=set();search_loop_reasons=set()
    owners={};last_jobs={};worker_caches={};knowledge_sent={};conflicts_sent={};mask_sent={};certificates_sent={};affinity_hits=0;next_verification=began
    waiting_since={};policy_reorders=portfolio_assignments=discrepancy_assignments=0
    yield_events={};yield_pending=set();lease_started={};lease_budgets={};warm_slices={}
    slices_started=0;extended_slices=0;yield_requests=0;yielded_slices=0
    preparation_stalls={};unchanged_preparation=0;long_preparation_leases=0;frontier_advances=0;max_preparation_lease_ms=0
    def add(checkpoint,region=None):
        nonlocal next_id
        job={'id':next_id,'region':list(region if region is not None else checkpoint['stack'][0]['domains']),
             'checkpoint':copy.deepcopy(checkpoint)}
        jobs[next_id]=job;queue.append(next_id);waiting_since[next_id]=0;next_id+=1
    if isinstance(resume,dict) and resume.get('work_version')==1:
        from constraint_portable import dfs_roots
        for first in dfs_roots(problem,resume,target,seed,options):add(first)
    elif isinstance(resume,dict) and 'parallel_version' in resume:
        if (type(resume.get('parallel_version')) is not int or resume.get('parallel_version')!=VERSION or resume.get('fingerprint')!=problem.fingerprint
                or resume.get('target')!=target or resume.get('input_domains_sha256')!=_domain_hash(problem)):
            raise ValueError('Parallel checkpoint version, puzzle, domains or target differs.')
        if not isinstance(resume.get('digest'),str) or resume['digest']!=_digest(resume):
            raise ValueError('Parallel checkpoint integrity digest differs.')
        required={'jobs','next_job_id','retired_nodes','retired_stats','jobs_completed','splits','replayed_jobs','elapsed_seconds','inflight_jobs','seed','search_options'}
        if not required<=resume.keys() or not isinstance(resume['jobs'],list) or not resume['jobs']:
            raise ValueError('Parallel checkpoint is missing unfinished jobs.')
        for key in ('next_job_id','retired_nodes','jobs_completed','splits','replayed_jobs'):
            if type(resume[key]) is not int or resume[key]<0:raise ValueError('Malformed parallel counters.')
        if type(resume['seed']) is not int:raise ValueError('Malformed parallel seed.')
        if not isinstance(resume['retired_stats'],dict) or any(not isinstance(k,str) or type(v) is not int or v<0 for k,v in resume['retired_stats'].items()):
            raise ValueError('Malformed retired parallel counters.')
        previous_elapsed=resume['elapsed_seconds']
        if isinstance(previous_elapsed,bool) or not isinstance(previous_elapsed,(int,float)) or not math.isfinite(previous_elapsed) or previous_elapsed<0:
            raise ValueError('Malformed parallel elapsed time.')
        normalize_options(resume['search_options'])
        regions=[]
        for job in resume['jobs']:
            region=_validate_job(problem,job,target,options)
            if job['id'] in jobs:raise ValueError('Parallel job IDs repeat.')
            if any(all(a&b for a,b in zip(region,old)) for old in regions):
                raise ValueError('Parallel job regions overlap.')
            regions.append(region);jobs[job['id']]=copy.deepcopy(job);queue.append(job['id'])
        next_id=resume['next_job_id']
        if next_id<=max(jobs):raise ValueError('Parallel next job ID is invalid.')
        inflight=resume['inflight_jobs']
        if not isinstance(inflight,list) or any(type(i) is not int or i not in jobs for i in inflight) or len(set(inflight))!=len(inflight):
            raise ValueError('Malformed saved parallel worker leases.')
        retired_nodes=resume['retired_nodes'];retired_stats=resume['retired_stats'].copy()
        completed=resume['jobs_completed'];splits=resume['splits'];replayed=resume['replayed_jobs']+len(inflight)
        seed=resume['seed'];root_symmetry=copy.deepcopy(resume.get('root_symmetry'))
        preparation_stalls.update(_restore_preparation_stalls(resume.get('preparation_stalls',[]),jobs))
        if mask_conflicts is not None and resume.get('mask_conflicts') is not None:mask_conflicts.import_snapshot(resume['mask_conflicts'])
        if certificates is not None and resume.get('certificates') is not None:certificates.load_snapshot(resume['certificates'])
        if any(job['checkpoint'].get('root_symmetry')!=root_symmetry for job in jobs.values()):
            raise ValueError('Parallel root symmetry metadata differs across jobs.')
    else:
        first=dfs_solve(problem,seconds=0,seed=seed,target=target,hints=hints,resume=resume,options=options)['checkpoint']
        seed=first['seed'];root_symmetry=copy.deepcopy(first.get('root_symmetry'));previous_elapsed=first.get('elapsed_seconds',0.)
        # A standalone checkpoint may be an exported restricted subtree. Its
        # fingerprint identifies the puzzle, not ownership of the entire original
        # root. Previously consumed choices within its first frame were proved
        # closed by that DFS history; choices outside that frame were never owned.
        add(first,first['stack'][0]['domains'])
    if coverage_scope is not None and options.get('shared_coverage') and learning_state.get('_coverage_path'):
        coverage=CoverageLedger(learning_state['_coverage_path'],coverage_scope,symbolic=options.get('symbolic_coverage',False))
    def merge_knowledge(payload):
        if payload is not None and knowledge is not None and options.get('persistent_learning',True):
            knowledge.merge_snapshot(payload)
    def merge_shared(payload):
        if payload is not None and shared_conflicts is not None and options.get('shared_conflicts',False):
            # Even a worker-produced snapshot is only an untrusted claim here.
            # No raw worker payload is forwarded to another process or runner.
            shared_conflicts.import_snapshot(payload)
    def merge_auxiliary(payload):
        if mask_conflicts is not None and options.get('smart_learning') and payload.get('mask_conflicts') is not None:
            mask_conflicts.import_snapshot(payload['mask_conflicts'])
        if certificates is not None and options.get('proof_certificates') and payload.get('certificates') is not None:
            certificates.load_snapshot(payload['certificates'])
    class VerificationStopped(Exception):pass
    def verify_shared():
        nonlocal next_verification
        if options.get('mixed_proof_dag') and mask_conflicts is not None and knowledge is not None:mask_conflicts.add_gold_witnesses(knowledge.records())
        verifiers=[store for enabled,store in ((options.get('shared_conflicts'),shared_conflicts),
                   (options.get('smart_learning'),mask_conflicts)) if enabled and store is not None and store.pending_count]
        if not verifiers or time.monotonic()<next_verification:return
        remaining=deadline-time.monotonic()
        if remaining<=0 or _stopped(stop):return
        def check():
            if time.monotonic()>=deadline or _stopped(stop):raise VerificationStopped
        try:
            for store in verifiers:
                store.advance(check,seconds=min(.005,max(0,deadline-time.monotonic())),attempts=2,conflicts=128)
        except VerificationStopped:pass
        finally:next_verification=time.monotonic()+.1
    def totals():
        nodes=retired_nodes;stats=retired_stats.copy()
        for job in jobs.values():nodes+=job['checkpoint']['nodes'];_add_stats(stats,job['checkpoint']['stats'])
        stats.update(parallel_workers=workers,parallel_workers_started=started_workers,
                     parallel_jobs_queued=len(queue),parallel_jobs_active=len(active),
                     parallel_jobs_completed=completed,parallel_splits=splits,parallel_replayed_jobs=replayed,
                     parallel_affinity_hits=affinity_hits,parallel_slices_started=slices_started,
                     parallel_extended_slices=extended_slices,parallel_yield_requests=yield_requests,
                     parallel_yielded_slices=yielded_slices,
                     parallel_unchanged_preparation_slices=unchanged_preparation,parallel_preparation_long_leases=long_preparation_leases,
                     parallel_frontier_advances=frontier_advances,parallel_max_preparation_lease_ms=max_preparation_lease_ms,frontier_reorders=policy_reorders,portfolio_jobs=portfolio_assignments,discrepancy_jobs=discrepancy_assignments,shared_loop_witnesses=len(knowledge) if knowledge is not None else 0)
        if knowledge is not None and options.get('persistent_learning',True):stats['loop_witnesses']=len(knowledge)
        if shared_conflicts is not None and options.get('shared_conflicts',False):
            # These are coordinator verification counters and live gauges,
            # not sums of the same learned records seen by multiple jobs.
            stats.update(shared_conflicts.stats)
        if mask_conflicts is not None and options.get('smart_learning'):stats.update(mask_conflicts.stats)
        if certificates is not None and options.get('proof_certificates'):stats['portable_certificates']=len(certificates.records)
        stats.update(queue_proof_checks=queue_proof_checks,queue_proof_retirements=queue_proof_retirements,queue_proof_splits=queue_proof_splits)
        if coverage is not None:stats.update(coverage.stats)
        for key in ('job_cache_hits','job_cache_misses','job_cache_evictions','support_reuses','shared_sat_models','shared_sat_job_views'):
            stats['parallel_'+key]=sum(cache.get(key,0) for cache in worker_caches.values())
        if options.get('shared_sat_assistant'):
            stats['sat_assist_clauses']=sum(cache.get('shared_sat_clauses',0) for cache in worker_caches.values())
        return nodes,stats
    def parallel_info():
        return {'workers':workers,'pending_jobs':len(jobs),'queued_jobs':len(queue),'active_jobs':len(active),'completed_jobs':completed,'replayed_jobs':replayed}
    def snapshot():
        payload={'parallel_version':VERSION,'fingerprint':problem.fingerprint,'input_domains_sha256':_domain_hash(problem),
                 'target':target,'seed':seed,'jobs':copy.deepcopy(list(jobs.values())),'next_job_id':next_id,
                 'retired_nodes':retired_nodes,'retired_stats':retired_stats.copy(),'jobs_completed':completed,
                 'splits':splits,'replayed_jobs':replayed,'elapsed_seconds':previous_elapsed+time.monotonic()-began,
                 'inflight_jobs':list(active.values()),'search_options':options.copy(),'root_symmetry':copy.deepcopy(root_symmetry)}
        payload['preparation_stalls']=[[job,count] for job,count in sorted(preparation_stalls.items()) if job in jobs and count]
        if mask_conflicts is not None and options.get('smart_learning'):payload['mask_conflicts']=mask_conflicts.snapshot()
        if certificates is not None and options.get('proof_certificates'):payload['certificates']=certificates.snapshot()
        payload['digest']=_digest(payload);return payload
    def backend_fields():
        return {'native_backend':_backend_label(native_backends),
                'native_fallback_reason':'; '.join(sorted(native_reasons)) or None,
                'search_loop_backend':_backend_label(search_loop_backends),
                'search_loop_fallback_reason':'; '.join(sorted(search_loop_reasons)) or None}
    def notify(event,**extra):
        # Worker events expose a single job's current engine. Record that before
        # replacing it with the coordinator's whole-run summary for the UI.
        _record_backend(extra,native_backends,native_reasons)
        _record_search_loop(extra,search_loop_backends,search_loop_reasons)
        extra={**extra,**backend_fields()}
        if progress is not None:
            nodes,stats=totals();progress({'event':event,'nodes':nodes,'stats':stats,'elapsed_seconds':time.monotonic()-began,
                                         'search_options':options.copy(),'root_symmetry':copy.deepcopy(root_symmetry),'parallel':parallel_info(),
                                         **({'knowledge':knowledge.snapshot()} if knowledge is not None and options.get('persistent_learning',True) else {}),
                                         **({'shared_conflicts':shared_conflicts.snapshot()} if shared_conflicts is not None and options.get('shared_conflicts',False) else {}),
                                         **({'mask_conflicts':mask_conflicts.snapshot()} if mask_conflicts is not None and options.get('smart_learning') else {}),
                                         **({'certificates':certificates.snapshot()} if certificates is not None and options.get('proof_certificates') else {}),**extra})
    def retire(job_id,result):
        nonlocal retired_nodes
        retired_nodes+=result['nodes'];_add_stats(retired_stats,result['stats'],retiring=True);del jobs[job_id];owners.pop(job_id,None);preparation_stalls.pop(job_id,None)
    def accept(job_id,result):
        nonlocal terminal,completed,root_symmetry,unchanged_preparation,frontier_advances
        status=result.get('status')
        merge_knowledge(result.get('knowledge'));merge_shared(result.get('shared_conflicts'));merge_auxiliary(result)
        _record_backend(result,native_backends,native_reasons)
        _record_search_loop(result,search_loop_backends,search_loop_reasons)
        if status in ('solved','edge_perfect'):
            codes=result.get('codes');report=problem.validate(codes)
            domain_ok=isinstance(codes,(list,tuple)) and len(codes)==problem.n and all(code in problem.domains[cell] for cell,code in enumerate(codes))
            valid=report['valid'] or target=='edges' and report['unique_tiles'] and report['matched_edges']==len(problem.edges)
            if not valid or not domain_ok:raise ValueError('Worker completion failed independent validation.')
            result={**result,'validation':report};retire(job_id,result);terminal=result;completed+=1;frontier_advances+=1
        elif status=='infeasible':
            if not result.get('complete'):raise ValueError('Worker did not certify exhausted job.')
            if coverage is not None:
                coverage.record_completed(tuple(int(x,16) for x in jobs[job_id]['region']),status='infeasible',complete=True)
            retire(job_id,result);completed+=1;frontier_advances+=1
        elif status in ('timeout','stopped'):
            candidate={**jobs[job_id],'checkpoint':result.get('checkpoint')}
            _validate_job(problem,candidate,target,options)
            moved=_record_frontier_progress(preparation_stalls,job_id,jobs[job_id]['checkpoint'],candidate['checkpoint'])
            frontier_advances+=int(moved);unchanged_preparation+=int(not moved)
            jobs[job_id]=copy.deepcopy(candidate);queue.append(job_id)
            root_symmetry=copy.deepcopy(candidate['checkpoint'].get('root_symmetry'))
        else:raise ValueError('Worker returned an unknown or incomplete state.')
    def prune_queued():
        nonlocal completed,queue_proof_checks,queue_proof_retirements,queue_proof_splits
        exclusions=[]
        if coverage_scope is None:return
        if options.get('queue_pruning'):
            if mask_conflicts is not None:
                exclusions.extend(exclusion_box(record,coverage_scope.original) for record in mask_conflicts.records())
            if knowledge is not None:
                exclusions.extend(exclusion_box(tuple((c,1<<code) for c,code in record),coverage_scope.original) for record in knowledge.records())
        if coverage is not None:exclusions.extend(coverage.exclusions())
        if not exclusions:return
        revision=hash(tuple(exclusions))
        for job_id in list(queue):
            checkpoint=jobs[job_id]['checkpoint']
            if checkpoint.get('root_symmetry',{}).get('state')=='pending':continue
            stamp=(checkpoint['digest'],revision)
            if queue_proof_seen.get(job_id)==stamp:continue
            old=checkpoint_boxes(checkpoint);queue_proof_checks+=1
            def check():
                if _stopped(stop) or time.monotonic()>=deadline:raise VerificationStopped()
            try:remaining,complete=subtract_many(old,exclusions,max_boxes=2048,check=check)
            except VerificationStopped:return
            queue_proof_seen[job_id]=stamp
            if not complete or remaining==old:continue
            if coverage is not None and not remaining:
                coverage.record_completed(tuple(int(x,16) for x in jobs[job_id]['region']),status='infeasible',complete=True)
            queue.remove(job_id);retire(job_id,checkpoint);queue_proof_seen.pop(job_id,None)
            if not remaining:completed+=1;queue_proof_retirements+=1
            else:
                for domains in remaining:add(_root_copy(checkpoint,domains))
                queue_proof_splits+=1

    def distribute():
        nonlocal splits,retired_nodes
        if len(jobs)>=2*workers:return
        # Split at most one committed job per coordinator pass. Work in flight
        # is never modified. Source counters are retired exactly once.
        for job_id in sorted(queue,key=lambda job:_pending_work_hint(jobs[job]['checkpoint']),reverse=True):
            checkpoint=jobs[job_id]['checkpoint']
            if checkpoint.get('root_symmetry',{}).get('state')=='pending':continue
            children=partition_checkpoint(problem,checkpoint,parts=max(2,2*workers-len(jobs)+1),options=options)
            if len(children)==1 and children[0]['stack']==checkpoint['stack']:continue
            queue.remove(job_id);retire(job_id,checkpoint)
            for child in children:add(child)
            splits+=1
            return
    status='timeout'
    if _stopped(stop):status='stopped'
    elif time.monotonic()<deadline:
        # Symmetry setup has to finish before a narrowed root can be split.
        # This short serial phase preserves the exact checkpoint protocol.
        preparation_slice=.05
        for job_id in list(queue):
            cp=jobs[job_id]['checkpoint']
            while cp.get('root_symmetry',{}).get('state')=='pending' and time.monotonic()<deadline and not _stopped(stop):
                queue.remove(job_id)
                value=dfs_solve(problem,seconds=min(preparation_slice,max(0,deadline-time.monotonic())),target=target,resume=cp,
                                stop=lambda:_stopped(stop),progress=lambda event:notify(event['event'],worker_nodes=event['nodes'],**_progress_fields(event)),
                                learning_state={**({'_knowledge':knowledge} if knowledge is not None else {}),
                                    **({'_shared_conflicts':shared_conflicts} if shared_conflicts is not None else {}),
                                    **({'_mask_conflicts':mask_conflicts} if mask_conflicts is not None else {}),
                                    **({'_certificates':certificates} if certificates is not None else {}),
                                    **({'_regional_cache_dir':learning_state['_regional_cache_dir']} if '_regional_cache_dir' in learning_state else {}),
                                    **({'_reasoning_cache_dir':learning_state['_reasoning_cache_dir']} if '_reasoning_cache_dir' in learning_state else {}),
                                    **({'_sat_service':learning_state['_sat_service']} if '_sat_service' in learning_state else {})},options=options)
                accept(job_id,value)
                preparation_slice=min(5.,preparation_slice*2)
                if terminal or job_id not in jobs:break
                cp=jobs[job_id]['checkpoint']
            if terminal:break
    cancel=None;stopping_at=None
    try:
        if jobs and terminal is None and time.monotonic()<deadline and not _stopped(stop):
            prune_queued();distribute();context=mp.get_context('spawn');cancel=context.Event()
            for _ in range(min(workers,len(jobs))):
                parent,child=context.Pipe();yield_event=context.Event();yield_events[len(slots)]=yield_event
                signal=_WorkerSignal(cancel,yield_event)
                process=context.Process(target=_worker,args=(problem,child,signal),daemon=True)
                process.start();child.close();slots.append((process,parent));started_workers+=1
            while jobs and terminal is None:
                prune_queued()
                requested=_stopped(stop)
                if requested or time.monotonic()>=deadline or error is not None:
                    if stopping_at is None:stopping_at=time.monotonic();cancel.set()
                    status='error' if error else 'stopped' if requested else 'timeout'
                if stopping_at is None:
                    verify_shared()
                    distribute()
                    for index,(process,connection) in enumerate(slots):
                        if index not in active and queue:
                            priority=options.get('frontier_restarts') or options.get('limited_discrepancy')
                            if priority:
                                policy_reorders+=int(reorder_frontier(queue,jobs,preferred,discrepancy=options.get('limited_discrepancy',False),epoch=slices_started))
                                if slices_started%8==7:
                                    # Fairness is time-since-last-dispatch, not
                                    # smallest job ID (which can starve peers).
                                    oldest=min(queue,key=lambda job:(waiting_since.get(job,0),job))
                                    queue.remove(oldest);queue.insert(0,oldest)
                            job_id,hit=_choose_job(queue,index,owners,last_jobs,priority=priority);affinity_hits+=int(hit);active[index]=job_id
                            waiting_since[job_id]=slices_started+1
                            portfolio_assignments+=int(options.get('worker_portfolio',False))
                            discrepancy_assignments+=int(options.get('limited_discrepancy',False))
                            yield_events[index].clear();yield_pending.discard(index);lease_started[index]=time.monotonic()
                            budget=_slice_budget(warm_slices.get(index,0),slots=len(slots),active=len(active),queued=len(queue),
                                                 remaining=deadline-time.monotonic(),stalled=preparation_stalls.get(job_id,0))
                            lease_budgets[index]=budget
                            slices_started+=1;extended_slices+=int(budget>SLICE_SECONDS)
                            long_preparation_leases+=int(budget>MAX_SLICE_SECONDS)
                            max_preparation_lease_ms=max(max_preparation_lease_ms,int(budget*1000))
                            shared=None
                            if knowledge is not None and options.get('persistent_learning',True):
                                candidate=knowledge.snapshot()
                                if knowledge_sent.get(index)!=candidate['digest']:shared=candidate;knowledge_sent[index]=candidate['digest']
                            verified=None
                            if shared_conflicts is not None and options.get('shared_conflicts',False):
                                candidate=shared_conflicts.snapshot()
                                if conflicts_sent.get(index)!=candidate['digest']:
                                    verified=candidate;conflicts_sent[index]=candidate['digest']
                            mask_payload=certificate_payload=None
                            if mask_conflicts is not None and options.get('smart_learning'):
                                candidate=mask_conflicts.snapshot(include_pending=False)
                                if mask_sent.get(index)!=candidate['digest']:mask_payload=candidate;mask_sent[index]=candidate['digest']
                            if certificates is not None and options.get('proof_certificates'):
                                candidate=certificates.snapshot();digest=_digest(candidate)
                                if certificates_sent.get(index)!=digest:certificate_payload=candidate;certificates_sent[index]=digest
                            connection.send({'id':job_id,'checkpoint':jobs[job_id]['checkpoint'],'target':target,
                                             'region_key':hashlib.sha256(json.dumps(jobs[job_id]['region'],separators=(',',':')).encode()).hexdigest(),
                                             'knowledge':shared,'shared_conflicts':verified,'mask_conflicts':mask_payload,'certificates':certificate_payload,
                                             'regional_cache_dir':learning_state.get('_regional_cache_dir'),
                                             'reasoning_cache_dir':learning_state.get('_reasoning_cache_dir'),
                                             'sat_service':learning_state.get('_sat_service'),
                                             'options':worker_options(options,index),'seconds':budget})
                    # Never split a leased checkpoint: ask one busy worker to
                    # commit its current frontier first. At most one outstanding
                    # request prevents cancellation storms and repeated setup.
                    if not queue and len(active)<len(slots) and not yield_pending:
                        now=time.monotonic()
                        eligible=[index for index,job in active.items()
                                  if now-lease_started[index]>=_idle_yield_delay(preparation_stalls.get(job,0),lease_budgets.get(index,SLICE_SECONDS))
                                  and _pending_work_hint(jobs[job]['checkpoint'])>1]
                        if eligible:
                            index=max(eligible,key=lambda i:_pending_work_hint(jobs[active[i]]['checkpoint']))
                            yield_events[index].set();yield_pending.add(index);yield_requests+=1
                any_message=False
                for index,(process,connection) in enumerate(slots):
                    closed=False
                    while connection.poll():
                        any_message=True
                        try:message=connection.recv()
                        except (EOFError,OSError):closed=True;break
                        job_id=active.get(index)
                        if job_id is None or message.get('id')!=job_id:
                            error='Worker returned a result for a different job.';break
                        if message.get('kind')=='progress':
                            event=message['event'];merge_knowledge(event.get('knowledge'));merge_shared(event.get('shared_conflicts'));merge_auxiliary(event);notify(event['event'],job_id=job_id,worker_pid=message['pid'],worker_nodes=event['nodes'],
                                **_progress_fields(event))
                            if _stopped(stop) and stopping_at is None:
                                stopping_at=time.monotonic();cancel.set();status='stopped'
                        elif message.get('kind')=='result':
                            was_yielded=index in yield_pending
                            yield_pending.discard(index)
                            yielded_slices+=int(was_yielded and message['result'].get('status')=='stopped')
                            warm_slices[index]=(0 if was_yielded else min(2,warm_slices.get(index,0)+1)
                                                if message['result'].get('status')=='timeout' else 0)
                            del active[index]
                            if isinstance(message.get('worker_cache'),dict):worker_caches[index]=message['worker_cache']
                            try:accept(job_id,message['result'])
                            except Exception as exc:
                                if job_id in jobs and job_id not in queue:queue.append(job_id)
                                error=f'Invalid worker result: {exc}';replayed+=1
                            if jobs:
                                partial={}
                                if job_id in jobs:
                                    domains=[int(v,16) for v in jobs[job_id]['checkpoint']['stack'][-1]['domains']]
                                    codes=[value.bit_length()-1 if value.bit_count()==1 else -1 for value in domains]
                                    partial={'codes':codes,'assigned_tiles':sum(code>=0 for code in codes),'is_partial':True}
                                if time.monotonic()-last_checkpoint>=2.:
                                    notify('parallel_checkpoint',checkpoint=snapshot(),job_id=job_id,worker_pid=message['pid'],**partial);last_checkpoint=time.monotonic()
                                else:notify('parallel_progress',job_id=job_id,worker_pid=message['pid'],**partial)
                            break
                        else:
                            del active[index];queue.append(job_id);replayed+=1
                            error=message.get('error','Worker protocol failed.');break
                    if index in active and (closed or not process.is_alive() and not connection.poll()):
                        job_id=active.pop(index);queue.append(job_id);replayed+=1
                        error=f'Worker exited with code {process.exitcode}; last committed job retained.'
                if terminal is not None:cancel.set();break
                if stopping_at is not None and not active:break
                if stopping_at is not None and time.monotonic()-stopping_at>STOP_GRACE_SECONDS:
                    for index,job_id in list(active.items()):
                        queue.append(job_id);replayed+=1;del active[index]
                    error='A worker did not acknowledge cancellation; last committed jobs retained.';status='error';break
                if not jobs:break
                if not any_message:time.sleep(.005)
    except Exception as exc:
        error=f'Parallel coordinator error: {type(exc).__name__}: {exc}; last committed jobs retained.';status='error'
        if cancel is not None:cancel.set()
        for job_id in active.values():
            if job_id not in queue:queue.append(job_id)
            replayed+=1
        active.clear()
    finally:
        if cancel is not None:cancel.set()
        for process,connection in slots:
            try:connection.send(None)
            except (BrokenPipeError,EOFError,OSError):pass
        for process,connection in slots:
            process.join(timeout=.5)
            if process.is_alive():process.terminate();process.join(timeout=1)
            connection.close()
        if coverage is not None:coverage.close()
    nodes,stats=totals()
    if terminal is not None:
        value={**terminal,'nodes':nodes,'stats':stats}
        value.pop('checkpoint',None)
    elif not jobs:
        value={'status':'infeasible','codes':None,'nodes':nodes,'stats':stats,'complete':True}
    else:
        if error is not None:status='error'
        elif _stopped(stop):status='stopped'
        value={'status':status,'codes':None,'nodes':nodes,'stats':stats,'complete':False,'checkpoint':snapshot()}
    value.update(elapsed_seconds=time.monotonic()-began,method='parallel-dfs',search_options=options.copy(),
                 root_symmetry=copy.deepcopy(root_symmetry),parallel=parallel_info(),**backend_fields(),replay_scope='Only work after the last committed per-job checkpoint may replay after failure or intermediate-snapshot resume.')
    if knowledge is not None and options.get('persistent_learning',True):value['knowledge']=knowledge.snapshot()
    if shared_conflicts is not None and options.get('shared_conflicts',False):value['shared_conflicts']=shared_conflicts.snapshot()
    if mask_conflicts is not None and options.get('smart_learning'):value['mask_conflicts']=mask_conflicts.snapshot()
    if certificates is not None and options.get('proof_certificates'):value['certificates']=certificates.snapshot()
    if error is not None:value['error']=error
    return value


def solve(problem,*,seconds=10,seed=0,target='gold',hints=None,stop=None,progress=None,
          resume=None,workers=1,options=None,learning_state=None):
    from constraint_service_lifecycle_round26 import run
    normalized=normalize_options(resume.get('search_options') if options is None and isinstance(resume,dict) else options)
    return run(_solve_parallel,problem,dict(seconds=seconds,seed=seed,target=target,hints=hints,
        stop=stop,progress=progress,resume=resume,workers=workers,options=normalized,learning_state=learning_state))
