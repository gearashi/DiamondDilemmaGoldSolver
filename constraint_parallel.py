"""Spawn-based exact DFS over a durable queue of disjoint remaining regions.

Every committed job checkpoint owns its unfinished search tree. Splitting turns
pending sibling choices into disjoint domain boxes, without changing the puzzle
fingerprint. Worker caches are private to a job: a proof made under a restricted
job root must never escape into another job. Shared SAT conflicts use a separate
original-domain verifier in every process; serialized claims are re-proved before
application or rebroadcast. A process failure retains its last
committed checkpoint, so uncommitted work may replay but coverage is not lost.
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
from constraint_knowledge import KnowledgeStore
from constraint_shared_conflicts import VerifiedConflictStore, GAUGE_KEYS as SHARED_GAUGES

VERSION=1
SLICE_SECONDS=.75
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
    excluded={'event','nodes','stats','elapsed_seconds','search_options','root_symmetry','checkpoint','knowledge','shared_conflicts'}
    return {key:value for key,value in event.items() if key not in excluded}


def _validate_dfs(problem,checkpoint,target,options):
    # Zero budget validates structure/digest without consuming any branch.
    checked=dfs_solve(problem,seconds=0,target=target,resume=checkpoint,options=options)
    return checked['checkpoint']


def _root_copy(checkpoint,domains):
    result=copy.deepcopy(checkpoint)
    result.update(nodes=0,stats={key:0 for key in result['stats']},elapsed_seconds=0.0,
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
        self.problem,self.limit=problem,limit;self.jobs=OrderedDict();self.support=OrderedDict()
        self.binding=KnowledgeStore(problem,target='edges')
        self.stats={'job_cache_hits':0,'job_cache_misses':0,'job_cache_evictions':0,'support_reuses':0}
    def key(self,request):
        return (request['id'],request.get('region_key'),request['target'],
                json.dumps(request['options'],sort_keys=True,separators=(',',':')))
    def support_key(self,request):
        if not self.binding.matches(self.problem,'edges'):raise ValueError('Worker source domains changed.')
        return (self.problem.fingerprint,self.binding.domains_sha256,request['target'],
                json.dumps(request['checkpoint'].get('root_symmetry'),sort_keys=True,separators=(',',':')))
    def get(self,request):
        key=self.key(request)
        if not self.binding.matches(self.problem,'edges'):raise ValueError('Worker source domains changed.')
        if key in self.jobs:
            self.stats['job_cache_hits']+=1;state=self.jobs.pop(key);self.jobs[key]=state;return state
        self.stats['job_cache_misses']+=1;state={};support_key=self.support_key(request)
        if support_key in self.support:
            table=self.support.pop(support_key);self.support[support_key]=table
            state={'_nogoods':_NogoodStore(self.problem.fingerprint,request['target'],initial_domains=table.initial_domains),
                   '_support':table}
            self.stats['support_reuses']+=1
        self.jobs[key]=state
        while len(self.jobs)>self.limit:self.jobs.popitem(last=False);self.stats['job_cache_evictions']+=1
        return state
    def remember(self,request,state):
        table=state.get('_support')
        if table is not None:
            key=self.support_key(request);self.support[key]=table;self.support.move_to_end(key)
            while len(self.support)>self.limit:self.support.popitem(last=False)


def _choose_job(queue,slot,owners,last_jobs):
    """Prefer a warm job, but rotate after bounded consecutive slices."""
    previous,streak=last_jobs.get(slot,(None,0))
    if previous in queue and streak<AFFINITY_SLICES:
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


def _worker(problem,connection,cancel):
    caches=_WorkerCaches(problem);knowledge=None;shared_conflicts=None
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
                result=dfs_solve(problem,seconds=request['seconds'],target=request['target'],
                                 resume=request['checkpoint'],stop=cancel.is_set,progress=progress,
                                 learning_state=learning,options=request['options'])
                caches.remember(request,learning)
                if request['options'].get('shared_conflicts',False):
                    result['shared_conflicts']=shared_conflicts.snapshot()
                connection.send({'kind':'result','id':job_id,'pid':os.getpid(),'result':result,'worker_cache':dict(caches.stats)})
            except BaseException as exc:
                connection.send({'kind':'error','id':job_id,'pid':os.getpid(),
                                 'error':f'{type(exc).__name__}: {exc}','traceback':traceback.format_exc(limit=8)})
                return
    except (EOFError,BrokenPipeError,OSError):
        return
    finally:
        if shared_conflicts is not None:shared_conflicts.close()
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
        if retiring and (key in SHARED_GAUGES or key.endswith('_entries') or key in ('forced_bonds','forced_clusters','loop_witnesses','sat_assist_clauses',
                            'hierarchical_subtable_states','hierarchical_subtable_support_slots','hierarchical_cached_summaries','gold_oracle_template_bytes')):continue
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


def solve(problem,*,seconds=10,seed=0,target='gold',hints=None,stop=None,progress=None,
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
    deadline=began+seconds
    jobs={};queue=[];active={};slots=[];started_workers=0;error=None;terminal=None
    previous_elapsed=0.;retired_nodes=0;retired_stats={};next_id=0;completed=0;splits=0;replayed=0
    root_symmetry=None;native_backends=set();native_reasons=set();last_checkpoint=began
    search_loop_backends=set();search_loop_reasons=set()
    owners={};last_jobs={};worker_caches={};knowledge_sent={};conflicts_sent={};affinity_hits=0;next_verification=began
    def add(checkpoint,region=None):
        nonlocal next_id
        job={'id':next_id,'region':list(region if region is not None else checkpoint['stack'][0]['domains']),
             'checkpoint':copy.deepcopy(checkpoint)}
        jobs[next_id]=job;queue.append(next_id);next_id+=1
    if isinstance(resume,dict) and 'parallel_version' in resume:
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
        if any(job['checkpoint'].get('root_symmetry')!=root_symmetry for job in jobs.values()):
            raise ValueError('Parallel root symmetry metadata differs across jobs.')
    else:
        first=dfs_solve(problem,seconds=0,seed=seed,target=target,hints=hints,resume=resume,options=options)['checkpoint']
        seed=first['seed'];root_symmetry=copy.deepcopy(first.get('root_symmetry'));previous_elapsed=first.get('elapsed_seconds',0.)
        add(first,[hex(sum(1<<code for code in set(row))) for row in problem.domains])
    def merge_knowledge(payload):
        if payload is not None and knowledge is not None and options.get('persistent_learning',True):
            knowledge.merge_snapshot(payload)
    def merge_shared(payload):
        if payload is not None and shared_conflicts is not None and options.get('shared_conflicts',False):
            # Even a worker-produced snapshot is only an untrusted claim here.
            # No raw worker payload is forwarded to another process or runner.
            shared_conflicts.import_snapshot(payload)
    class VerificationStopped(Exception):pass
    def verify_shared():
        nonlocal next_verification
        if (shared_conflicts is None or not options.get('shared_conflicts',False)
                or not shared_conflicts.pending_count or time.monotonic()<next_verification):return
        remaining=deadline-time.monotonic()
        if remaining<=0 or _stopped(stop):return
        def check():
            if time.monotonic()>=deadline or _stopped(stop):raise VerificationStopped
        try:shared_conflicts.advance(check,seconds=min(.01,remaining),attempts=2,conflicts=128)
        except VerificationStopped:pass
        finally:next_verification=time.monotonic()+.1
    def totals():
        nodes=retired_nodes;stats=retired_stats.copy()
        for job in jobs.values():nodes+=job['checkpoint']['nodes'];_add_stats(stats,job['checkpoint']['stats'])
        stats.update(parallel_workers=workers,parallel_workers_started=started_workers,
                     parallel_jobs_queued=len(queue),parallel_jobs_active=len(active),
                     parallel_jobs_completed=completed,parallel_splits=splits,parallel_replayed_jobs=replayed,
                     parallel_affinity_hits=affinity_hits,shared_loop_witnesses=len(knowledge) if knowledge is not None else 0)
        if knowledge is not None and options.get('persistent_learning',True):stats['loop_witnesses']=len(knowledge)
        if shared_conflicts is not None and options.get('shared_conflicts',False):
            # These are coordinator verification counters and live gauges,
            # not sums of the same learned records seen by multiple jobs.
            stats.update(shared_conflicts.stats)
        for key in ('job_cache_hits','job_cache_misses','job_cache_evictions','support_reuses'):
            stats['parallel_'+key]=sum(cache.get(key,0) for cache in worker_caches.values())
        return nodes,stats
    def parallel_info():
        return {'workers':workers,'pending_jobs':len(jobs),'queued_jobs':len(queue),'active_jobs':len(active),'completed_jobs':completed,'replayed_jobs':replayed}
    def snapshot():
        payload={'parallel_version':VERSION,'fingerprint':problem.fingerprint,'input_domains_sha256':_domain_hash(problem),
                 'target':target,'seed':seed,'jobs':copy.deepcopy(list(jobs.values())),'next_job_id':next_id,
                 'retired_nodes':retired_nodes,'retired_stats':retired_stats.copy(),'jobs_completed':completed,
                 'splits':splits,'replayed_jobs':replayed,'elapsed_seconds':previous_elapsed+time.monotonic()-began,
                 'inflight_jobs':list(active.values()),'search_options':options.copy(),'root_symmetry':copy.deepcopy(root_symmetry)}
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
                                         **({'shared_conflicts':shared_conflicts.snapshot()} if shared_conflicts is not None and options.get('shared_conflicts',False) else {}),**extra})
    def retire(job_id,result):
        nonlocal retired_nodes
        retired_nodes+=result['nodes'];_add_stats(retired_stats,result['stats'],retiring=True);del jobs[job_id];owners.pop(job_id,None)
    def accept(job_id,result):
        nonlocal terminal,completed,root_symmetry
        status=result.get('status')
        merge_knowledge(result.get('knowledge'));merge_shared(result.get('shared_conflicts'))
        _record_backend(result,native_backends,native_reasons)
        _record_search_loop(result,search_loop_backends,search_loop_reasons)
        if status in ('solved','edge_perfect'):
            codes=result.get('codes');report=problem.validate(codes)
            domain_ok=isinstance(codes,(list,tuple)) and len(codes)==problem.n and all(code in problem.domains[cell] for cell,code in enumerate(codes))
            valid=report['valid'] or target=='edges' and report['unique_tiles'] and report['matched_edges']==len(problem.edges)
            if not valid or not domain_ok:raise ValueError('Worker completion failed independent validation.')
            result={**result,'validation':report};retire(job_id,result);terminal=result;completed+=1
        elif status=='infeasible':
            if not result.get('complete'):raise ValueError('Worker did not certify exhausted job.')
            retire(job_id,result);completed+=1
        elif status in ('timeout','stopped'):
            candidate={**jobs[job_id],'checkpoint':result.get('checkpoint')}
            _validate_job(problem,candidate,target,options)
            jobs[job_id]=copy.deepcopy(candidate);queue.append(job_id)
            root_symmetry=copy.deepcopy(candidate['checkpoint'].get('root_symmetry'))
        else:raise ValueError('Worker returned an unknown or incomplete state.')
    def distribute():
        nonlocal splits,retired_nodes
        if len(jobs)>=2*workers:return
        # Split at most one committed job per coordinator pass. Work in flight
        # is never modified. Source counters are retired exactly once.
        for job_id in list(queue):
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
                                    **({'_shared_conflicts':shared_conflicts} if shared_conflicts is not None else {})},options=options)
                accept(job_id,value)
                preparation_slice=min(5.,preparation_slice*2)
                if terminal or job_id not in jobs:break
                cp=jobs[job_id]['checkpoint']
            if terminal:break
    cancel=None;stopping_at=None
    try:
        if jobs and terminal is None and time.monotonic()<deadline and not _stopped(stop):
            distribute();context=mp.get_context('spawn');cancel=context.Event()
            for _ in range(min(workers,max(1,len(jobs)))):
                parent,child=context.Pipe();process=context.Process(target=_worker,args=(problem,child,cancel),daemon=True)
                process.start();child.close();slots.append((process,parent));started_workers+=1
            while jobs and terminal is None:
                requested=_stopped(stop)
                if requested or time.monotonic()>=deadline or error is not None:
                    if stopping_at is None:stopping_at=time.monotonic();cancel.set()
                    status='error' if error else 'stopped' if requested else 'timeout'
                if stopping_at is None:
                    verify_shared()
                    distribute()
                    for index,(process,connection) in enumerate(slots):
                        if index not in active and queue:
                            job_id,hit=_choose_job(queue,index,owners,last_jobs);affinity_hits+=int(hit);active[index]=job_id
                            shared=None
                            if knowledge is not None and options.get('persistent_learning',True):
                                candidate=knowledge.snapshot()
                                if knowledge_sent.get(index)!=candidate['digest']:shared=candidate;knowledge_sent[index]=candidate['digest']
                            verified=None
                            if shared_conflicts is not None and options.get('shared_conflicts',False):
                                candidate=shared_conflicts.snapshot()
                                if conflicts_sent.get(index)!=candidate['digest']:
                                    verified=candidate;conflicts_sent[index]=candidate['digest']
                            connection.send({'id':job_id,'checkpoint':jobs[job_id]['checkpoint'],'target':target,
                                             'region_key':hashlib.sha256(json.dumps(jobs[job_id]['region'],separators=(',',':')).encode()).hexdigest(),
                                             'knowledge':shared,'shared_conflicts':verified,'options':options,'seconds':min(SLICE_SECONDS,max(0,deadline-time.monotonic()))})
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
                            event=message['event'];merge_knowledge(event.get('knowledge'));merge_shared(event.get('shared_conflicts'));notify(event['event'],job_id=job_id,worker_pid=message['pid'],worker_nodes=event['nodes'],
                                **_progress_fields(event))
                            if _stopped(stop) and stopping_at is None:
                                stopping_at=time.monotonic();cancel.set();status='stopped'
                        elif message.get('kind')=='result':
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
    if error is not None:value['error']=error
    return value
