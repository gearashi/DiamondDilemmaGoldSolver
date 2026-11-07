"""One bounded SAT/witness owner shared by spawned worker processes.

A manager process owns each native instance; a lock serializes its calls. Query
results are semantic and root-bound, never opaque native checkpoints. Clients
poll cancellation while a bounded RPC completes; a cancelled request may finish
and populate reuse, but cannot publish an answer to that cancelled caller.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor,TimeoutError
from multiprocessing.managers import BaseManager
import hashlib,json,math,multiprocessing,threading,time
from pathlib import Path
from constraint_sat_assist import SatAssist,STAT_KEYS as SAT_STATS
from constraint_witness_round26 import SemanticWitnessPool,STAT_KEYS as WITNESS_STATS
from constraint_knowledge import KnowledgeStore

STAT_KEYS=('sat_service_requests','sat_service_batches','sat_service_cache_hits',
 'sat_service_models','sat_service_query_calls','sat_service_filter_calls','sat_service_unknown',
 'sat_service_witness_answers','sat_service_evictions')


class _Backend:
    def __init__(self,problem,target,directory=None,limit=256):
        if target not in ('gold','edges') or type(limit)is not int or not 1<=limit<=4096:raise ValueError('Invalid SAT service policy.')
        self.problem=problem;self.target=target;self.root=tuple(sum(1<<c for c in set(row)) for row in problem.domains)
        self.binding=(problem.fingerprint,target,self.root);self.limit=limit;self.cache=OrderedDict();self.lock=threading.RLock();self.closed=False
        self.helper=SatAssist(problem,self.root,target,witness_pool=True,support_cover=True)
        self.pool=SemanticWitnessPool(problem);self.helper.semantic_witnesses=self.pool
        self.knowledge=KnowledgeStore(problem,target=target);self.stats=dict.fromkeys(STAT_KEYS,0);self.stats['sat_service_models']=int(self.helper.solver is not None)
        self._reported={'sat':dict.fromkeys(SAT_STATS,0),'service':dict.fromkeys(STAT_KEYS,0),'witness':dict.fromkeys(WITNESS_STATS,0)}
        self.witness_file=None if directory is None else Path(directory)/'sat-service-witnesses'/(
            hashlib.sha256(json.dumps(self.binding,separators=(',',':')).encode()).hexdigest()+'.json')
        if self.witness_file is not None:
            try:self.pool.load(self.witness_file)
            except (OSError,ValueError,TypeError):pass

    def identity(self):return self.binding

    def _response(self,**payload):
        # Attribute each completed server operation to exactly one recipient.
        # Startup/load work and work preceding an RPC error remain unreported
        # until the next successfully returned response; owner totals are gauges,
        # not an additional contribution to worker totals.
        sources={'sat':self.helper.stats,'service':self.stats,'witness':self.pool.stats}
        for name,current in sources.items():
            payload[name+'_delta']={key:value-self._reported[name].get(key,0) for key,value in current.items()}
            self._reported[name]=dict(current)
        return {'binding':self.binding,'ready':self.helper.ready,'status':self.helper.status,
                'reason':self.helper.reason,'clauses':self.helper.clauses,**payload}

    def _domains(self,domains):
        values=tuple(domains)
        if len(values)!=len(self.root) or any(type(v)is not int or v<0 or v&r!=v for v,r in zip(values,self.root)):
            raise ValueError('Service request escapes original domains.')
        return values

    def request(self,requests,seconds=.1,conflicts=256):
        if type(requests)not in (tuple,list) or len(requests)>16 or type(conflicts)is not int or not 1<=conflicts<=100000:
            raise ValueError('Invalid SAT service batch.')
        if type(seconds)not in (int,float) or not math.isfinite(seconds) or not 0<=seconds<=.25:raise ValueError('Service lease must be at most .25 seconds.')
        normalized=[]
        for request in requests:
            if type(request)is not dict or set(request)!={'kind','domains','max_candidates'} or request['kind'] not in ('prepare','query','filter') or type(request['max_candidates'])is not int or not 0<=request['max_candidates']<=256:
                raise ValueError('Invalid SAT service request.')
            normalized.append((request['kind'],self._domains(request['domains']),request['max_candidates']))
        with self.lock:
            if self.closed:raise RuntimeError('SAT service is closed.')
            self.pool._check();deadline=time.monotonic()+seconds;answers=[];before=self.helper.stats.copy();self.stats['sat_service_batches']+=1
            ready=self.helper.prepare(seconds=max(0,deadline-time.monotonic()))
            self.stats['sat_service_models']=max(self.stats['sat_service_models'],int(self.helper.solver is not None))
            for kind,domains,maximum in normalized:
                self.stats['sat_service_requests']+=1;key=(kind,domains,self.helper._gold_revision)
                if kind=='prepare':answer={'outcome':'ready' if ready else self.helper.status}
                elif not all(domains):answer={'outcome':'unsat','core':tuple((c,m) for c,m in enumerate(domains) if m!=self.root[c]),'domains':domains}
                elif kind=='query' and key in self.cache:
                    self.cache.move_to_end(key);answer=dict(self.cache[key]);self.stats['sat_service_cache_hits']+=1
                elif not ready or time.monotonic()>=deadline:answer={'outcome':'unknown'};self.stats['sat_service_unknown']+=1
                elif kind=='filter':
                    self.stats['sat_service_filter_calls']+=1
                    result=self.helper.filter_domains(domains,seconds=max(0,deadline-time.monotonic()),conflicts=conflicts,max_candidates=maximum)
                    answer={'outcome':'filtered','domains':tuple(result)}
                else:
                    # A positive semantic witness is sufficient for the relaxed
                    # SAT query, never a Gold success claim.
                    supported=[0]*len(domains);self.pool.supports(domains,supported)
                    if all(supported):answer={'outcome':'sat'};self.stats['sat_service_witness_answers']+=1
                    else:
                        self.stats['sat_service_query_calls']+=1
                        literals=tuple((c,m) for c,m in enumerate(domains) if m!=self.root[c])
                        core=self.helper.query_masks(literals,seconds=max(0,deadline-time.monotonic()),conflicts=conflicts)
                        answer={'outcome':self.helper.last_outcome}
                        if core is not None:answer['core']=core
                        elif self.helper.last_outcome=='sat':
                            supports=self.helper._model_supports(domains,[],lambda:None)
                            if supports is not None:self.pool.expand_piece_symmetries(tuple(mask.bit_length()-1 for mask in supports),limit=4)
                    if answer['outcome'] in ('sat','unsat'):
                        self.cache[key]=dict(answer)
                        while len(self.cache)>self.limit:self.cache.popitem(last=False);self.stats['sat_service_evictions']+=1
                    else:self.stats['sat_service_unknown']+=1
                answers.append(answer)
            if self.witness_file is not None:
                try:self.witness_file.parent.mkdir(parents=True,exist_ok=True);self.pool.save(self.witness_file)
                except OSError:pass
            return self._response(answers=answers)

    def add_cuts(self,records):
        with self.lock:
            if self.closed:raise RuntimeError('SAT service is closed.')
            if self.target!='gold' or type(records)not in (tuple,list) or len(records)>256:raise ValueError('Invalid service Gold cuts.')
            # KnowledgeStore and helper both independently replay raw strands.
            for row in records:self.knowledge.add(row)
            added=self.helper.add_gold_witnesses(self.knowledge,max_records=256)
            self.pool.add_cuts(self.helper._gold_known)
            if added:self.cache.clear()
            return self._response(added=added)

    def counters(self):
        with self.lock:return {'service':dict(self.stats),'sat':dict(self.helper.stats),'witness':dict(self.pool.stats)}

    def close(self):
        with self.lock:
            if not self.closed:self.helper.close();self.closed=True


class _SATManager(BaseManager):pass
_SATManager.register('Backend',_Backend)


class SATServiceOwner:
    """Explicit owner lifecycle. Worker views must never shut down the server."""
    def __init__(self,problem,target='gold',*,directory=None,instances=1):
        if type(instances)is not int or not 1<=instances<=8:raise ValueError('SAT service supports one to eight affinity instances.')
        self.manager=_SATManager(ctx=multiprocessing.get_context('spawn'));self.manager.start();self.closed=False
        try:self.backends=[self.manager.Backend(problem,target,directory) for _ in range(instances)]
        except BaseException:self.manager.shutdown();raise
        self.binding=self.backends[0].identity()
    def client(self,initial=None):
        root=self.binding[2];initial=root if initial is None else tuple(initial)
        if len(initial)!=len(root) or any(type(v)is not int or v<0 or v&~r for v,r in zip(initial,root)):raise ValueError('SAT service view escapes original root.')
        # Stable scope affinity, independent of randomized Python hash seeds.
        slot=int(hashlib.sha256(json.dumps(initial).encode()).hexdigest(),16)%len(self.backends)
        return SATServiceClient(self.backends[slot],self.binding,initial)
    def counters(self):return [backend.counters() for backend in self.backends]
    def close(self):
        if self.closed:return
        self.closed=True
        try:
            for backend in self.backends:backend.close()
        finally:self.manager.shutdown()
    def __enter__(self):return self
    def __exit__(self,*args):self.close()


class SATServiceClient:
    def __init__(self,proxy,binding,initial):
        self.proxy=proxy;self.binding=binding;self.initial=tuple(initial);self.stats=dict.fromkeys(SAT_STATS,0)
        if len(self.initial)!=len(binding[2]) or any(type(v)is not int or v<0 or v&~r for v,r in zip(self.initial,binding[2])):raise ValueError('SAT service view escapes original root.')
        self.service_stats=dict.fromkeys(STAT_KEYS,0);self.witness_stats=dict.fromkeys(WITNESS_STATS,0)
        self.ready=False;self.status='building';self.reason=None;self.last_outcome='not_ready';self.witness_pool=True;self.support_cover=True
        self._executor=None;self._future=None;self._key=None;self._closed=False
    def __getstate__(self):
        if self._future is not None:raise ValueError('Cannot transfer an outstanding service RPC.')
        state=self.__dict__.copy();state['_executor']=None;return state
    def view(self,initial):
        initial=tuple(initial)
        if self._closed:raise RuntimeError('SAT service view is closed.')
        if len(initial)!=len(self.initial) or any(type(v)is not int or v<0 or v&~r for v,r in zip(initial,self.initial)):raise ValueError('SAT service subview escapes its parent root.')
        return SATServiceClient(self.proxy,self.binding,initial)
    def matches(self,problem,target,initial=None):
        expected=(problem.fingerprint,target,tuple(sum(1<<c for c in set(row)) for row in problem.domains))
        return (not self._closed and self.binding==expected and
                (initial is None or tuple(initial)==self.initial))
    def _call(self,method,args,key,check):
        if self._closed:raise RuntimeError('SAT service view is closed.')
        check()
        if self._executor is None:self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='sat-service-rpc')
        while self._future is not None:
            check()
            try:result=self._future.result(timeout=.01)
            except TimeoutError:
                # Future.result uses the same exception type for an unfinished
                # wait and for an RPC that itself failed with TimeoutError.
                # Re-read a completed future to distinguish that race exactly.
                if not self._future.done():continue
                try:result=self._future.result()
                except BaseException:
                    self._future=None;self._key=None;raise
            except BaseException:
                self._future=None;self._key=None;raise
            old=self._key
            # Stop at the publication boundary must preserve the completed
            # response and its unapplied deltas for the next identical request.
            if old==key:check()
            self._future=None;self._key=None
            self._receive(result)
            if old==key:return result
        self._key=key;self._future=self._executor.submit(getattr(self.proxy,method),*args)
        return self._call(method,args,key,check)
    def _rpc(self,requests,check,seconds,conflicts):
        if self._closed:raise RuntimeError('SAT service view is closed.')
        if type(seconds)not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid service client budget.')
        key=json.dumps(['request',requests,seconds,conflicts],sort_keys=True)
        return self._call('request',(requests,min(.25,max(0,seconds)),conflicts),key,check)['answers']
    def _receive(self,result):
        if result['binding']!=self.binding:raise ValueError('SAT service response scope changed.')
        for key,value in result['sat_delta'].items():self.stats[key]+=value
        for key,value in result.get('service_delta',{}).items():self.service_stats[key]+=value
        for key,value in result.get('witness_delta',{}).items():self.witness_stats[key]+=value
        self.stats['sat_assist_clauses']=result['clauses'];self.ready=result['ready'];self.status=result['status'];self.reason=result['reason']
    def _request(self,kind,domains,check,seconds,conflicts,maximum=16):
        domains=tuple(domains)
        if len(domains)!=len(self.initial) or any(type(v)is not int or v<0 or v&~r for v,r in zip(domains,self.initial)):raise ValueError('SAT view query escaped its job.')
        return self._rpc([{'kind':kind,'domains':domains,'max_candidates':maximum}],check,seconds,conflicts)[0]
    def prepare(self,check=lambda:None,*,seconds=.01):
        self._request('prepare',self.initial,check,seconds,256);return self.ready
    def filter_domains(self,domains,check=lambda:None,*,seconds=.01,conflicts=256,max_candidates=16):
        result=self._request('filter',domains,check,seconds,conflicts,max_candidates)
        return tuple(result.get('domains',domains))
    def query_masks(self,literals,check=lambda:None,*,seconds=.01,conflicts=256):
        current=list(self.initial);seen=set()
        for cell,mask in literals:
            if type(cell)is not int or not 0<=cell<len(current) or cell in seen or type(mask)is not int or mask<=0 or mask&~current[cell]:raise ValueError('Invalid service mask query.')
            current[cell]=mask;seen.add(cell)
        result=self._request('query',current,check,seconds,conflicts);self.last_outcome=result['outcome']
        core=result.get('core')
        return None if core is None else tuple((c,mask&self.initial[c]) for c,mask in core if self.initial[c]&~mask)
    def query(self,placements,check=lambda:None,**kwargs):
        if not isinstance(placements,(tuple,list)) or any(not isinstance(pair,(tuple,list)) or len(pair)!=2 or any(type(v)is not int for v in pair) or not 0<=pair[0]<len(self.initial) or not 0<=pair[1]<3*len(self.initial) for pair in placements):raise ValueError('Invalid service placements.')
        chosen=dict(placements)
        if len(chosen)!=len(placements):raise ValueError('Repeated service placement cell.')
        core=self.query_masks([(c,1<<v) for c,v in placements],check,**kwargs)
        if core is None:return None
        return tuple((c,chosen[c]) for c,mask in core if self.initial[c]&~mask)
    def drain_candidates(self):return []
    def drain_domain_candidates(self):return []
    def add_gold_witnesses(self,source,check=lambda:None,**kwargs):
        check()
        if not isinstance(source,KnowledgeStore) or source.fingerprint!=self.binding[0]:raise ValueError('Service requires bound Gold witnesses.')
        if self.binding[1]!='gold':raise ValueError('Gold cuts do not apply to an edge service.')
        records=source.records();key=json.dumps(['add_cuts',records],sort_keys=True)
        return self._call('add_cuts',(records,),key,check)['added']
    def close(self):
        self._closed=True
        if self._executor is not None:self._executor.shutdown(wait=False,cancel_futures=True)
