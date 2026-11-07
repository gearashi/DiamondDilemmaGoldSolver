"""Checked regional deductions; trusted continuation cursors are never proofs.

F11 carries a complete independently replayed DSU certificate. F12 reconstructs
its necessary Gold obstruction. Other regional stages use a fresh bounded DSU
certificate, independently checking every live and dead transition. A cap is
unknown; no completed flag, source hash or producer cursor establishes failure.
"""
from __future__ import annotations
from collections import OrderedDict
from copy import deepcopy
import json
from constraint_work import digest
from constraint_regions import HierarchicalRegionPruner
from constraint_region_incremental import source_binding
from constraint_region_proof_round26 import verify_core,build_core,STAT_KEYS as CORE_KEYS
from constraint_connectivity_round26 import verify_connectivity_reason
from constraint_region_certificate import CertificateError
from constraint_separator import _Budget

STAT_KEYS=tuple('regional_proof_'+name for name in ('checks','replays','nodes','rows','algebra_ops','unknown'))
QUEUE_KEYS=('regional_chain_queued','regional_chain_duplicates','regional_chain_evictions','regional_chain_attempts','regional_chain_emitted','regional_chain_caps','regional_chain_build_states','regional_chain_build_transitions')
MAX_BYTES=1024*1024
ORIGINS=('compiler','overlap','recursive','integer','connectivity','core')


def _scope(problem,domains):
    if not isinstance(domains,(list,tuple)) or len(domains)!=problem.n or any(type(v)is not int or v<=0 or v&~sum(1<<c for c in set(problem.domains[i])) for i,v in enumerate(domains)):raise ValueError('Regional proof scope escapes original domains.')
    return tuple(domains)


def verify_reason(problem,domains,data,check=lambda:None,*,stats=None,target='gold'):
    if target not in ('gold','edges'):raise ValueError('Invalid regional proof target.')
    counts={} if stats is None else stats
    for key in STAT_KEYS:counts.setdefault(key,0)
    counts['regional_proof_checks']+=1;check();domains=_scope(problem,domains)
    if type(data)is not dict or set(data)!={'kind','certificate'} or data['kind'] not in ('regional_core','regional_replay','connectivity'):raise ValueError('Malformed regional proof kind.')
    if len(json.dumps(data,separators=(',',':'),allow_nan=False).encode())>MAX_BYTES:raise ValueError('Regional proof exceeds the transport byte bound.')
    owner=HierarchicalRegionPruner(problem,target);before=source_binding(owner);record=data['certificate'];kind=data['kind']
    callback_error=[]
    def poll():
        try:check()
        except BaseException as exc:callback_error.append(exc);raise
    try:
        if kind=='connectivity':
            if target!='gold':raise ValueError('Gold connectivity cannot justify an edges-target conflict.')
            if type(record)is not dict:raise ValueError('Malformed connectivity certificate.')
            claimed=_scope(problem,record.get('domains'))
            if any(current&~old for current,old in zip(domains,claimed)):raise ValueError('Connectivity certificate requires narrower assumptions.')
            work={};counts['regional_proof_replays']+=1
            try:answer=verify_connectivity_reason(owner,record,poll,stats=work)
            finally:counts['regional_proof_algebra_ops']+=work.get('connectivity_reason_steps',0)
        else:
            if kind=='regional_replay':
                if type(record)is not dict or set(record)!={'origin','proof'} or record['origin'] not in ORIGINS:raise ValueError('Malformed independent regional replay.')
                record=record['proof']
            work=dict.fromkeys(CORE_KEYS,0);counts['regional_proof_replays']+=1
            try:assumptions=verify_core(owner,record,poll,max_states=50000,max_steps=1000000,stats=work)
            finally:
                counts['regional_proof_nodes']+=work['regional_core_verified_states']
                counts['regional_proof_rows']+=work['regional_core_verified_transitions']
                counts['regional_proof_algebra_ops']+=work['regional_core_terminal_checks']
            if any(domains[cell]&~mask for cell,mask in assumptions):raise ValueError('Regional certificate requires narrower assumptions.')
            answer=True
        poll()
        if before!=source_binding(owner):raise ValueError('Regional proof input changed during checking.')
        return answer is True
    except _Budget:
        if callback_error:raise
        counts['regional_proof_unknown']+=1;return False


class RegionalProofQueue:
    """Bounded optional claims; interruption preserves the current claim.

    Capture itself does no interruptible proof work after a stage mutates its
    domains. The DFS owner drains only after committing that stage's trail.
    Neither queue eviction nor a failed/capped replay removes search work.
    """
    def __init__(self,problem,target,*,limit=16,history=64):
        self.problem=problem;self.target=target;self.limit=limit;self.history=history
        self.owner=HierarchicalRegionPruner(problem,target);self.binding=source_binding(self.owner)
        self.pending=OrderedDict();self.attempted=OrderedDict();self.stats=dict.fromkeys((*QUEUE_KEYS,*STAT_KEYS),0)
    def _check(self):
        if self.binding!=source_binding(self.owner):raise ValueError('Regional proof producer scope changed.')
    def _put(self,domains,kind,certificate=None):
        self._check();domains=_scope(self.problem,domains)
        item={'domains':domains,'kind':kind,'certificate':deepcopy(certificate)};key=digest(item)
        if key in self.pending or key in self.attempted:self.stats['regional_chain_duplicates']+=1;return
        if len(json.dumps(item,separators=(',',':')).encode())>MAX_BYTES:self.stats['regional_chain_caps']+=1;return
        self.pending[key]=item;self.stats['regional_chain_queued']+=1
        while len(self.pending)>self.limit:self.pending.popitem(last=False);self.stats['regional_chain_evictions']+=1
    def capture(self,origin,before,after,*,cores=(),connectivity=()):
        self._check();before=_scope(self.problem,before)
        if origin not in ORIGINS or len(after)!=len(before) or any(type(v)is not int or v<0 or v&~old for old,v in zip(before,after)):raise ValueError('Invalid regional deduction capture.')
        if tuple(before)==tuple(after):return
        kept_core=False
        if origin=='core':
            for record in cores:
                assumptions=record.get('assumptions',())
                if all(not before[c]&~mask for c,mask in assumptions):self._put(before,'regional_core',record);kept_core=True
        if origin=='connectivity':
            for record in connectivity:
                # Some endpoint reasons follow sound possible-block pruning.
                # Their narrower scope remains explicit in the learned guard.
                if record.get('domains') and all(record['domains']):self._put(record['domains'],'connectivity',record)
        if kept_core:return
        fallback=origin
        if not all(after):self._put(before,fallback);return
        offered=0
        for cell,(old,new) in enumerate(zip(before,after)):
            removed=old&~new
            while removed and offered<self.limit:
                low=removed&-removed;removed^=low;trial=list(before);trial[cell]=low;self._put(trial,fallback);offered+=1
            if offered==self.limit:break
    def drain(self,check=lambda:None,*,limit=8,max_states=2048,max_steps=20000):
        self._check()
        if type(limit)is not int or not 0<=limit<=16 or type(max_states)is not int or not 1<=max_states<=50000 or type(max_steps)is not int or not 1<=max_steps<=1000000:raise ValueError('Invalid regional proof replay budget.')
        for key,item in list(self.pending.items())[:limit]:
            check();self.stats['regional_chain_attempts']+=1;domains=item['domains'];kind=item['kind'];data=None
            if item.get('ready') is not None:
                yield deepcopy(item['ready']);self.pending.pop(key);self.attempted[key]=True
                while len(self.attempted)>self.history:self.attempted.popitem(last=False)
                continue
            if item['certificate'] is not None:
                candidate={'kind':kind,'certificate':item['certificate']}
                if verify_reason(self.problem,domains,candidate,check,stats=self.stats,target=self.target):data=candidate
            else:
                cells=tuple(c for c,d in enumerate(domains) if d.bit_count()>1) or (0,)
                work=dict.fromkeys(CORE_KEYS,0)
                try:proof=build_core(self.owner,cells,domains,check,max_states=max_states,max_steps=max_steps,generalize=0,stats=work)
                finally:
                    self.stats['regional_chain_build_states']+=work['regional_core_build_states']
                    self.stats['regional_chain_build_transitions']+=work['regional_core_build_transitions']
                    self.stats['regional_proof_nodes']+=work['regional_core_verified_states']
                    self.stats['regional_proof_rows']+=work['regional_core_verified_transitions']
                    self.stats['regional_proof_algebra_ops']+=work['regional_core_terminal_checks']
                if proof is not None:data={'kind':'regional_replay','certificate':{'origin':kind,'proof':proof}}
            if data is not None:
                chain={'format':'diamond-deduction-chain-v1','fingerprint':self.problem.fingerprint,'target':self.target,
                  'original':[hex(sum(1<<v for v in set(row))) for row in self.problem.domains],
                  'root':[hex(v) for v in domains],'decisions':[],'events':[],
                  'terminal':{'rule':'regional','data':data}}
                if len(json.dumps(chain,separators=(',',':')).encode())<=MAX_BYTES:
                    self._check();item['ready']=deepcopy(chain);self.stats['regional_chain_emitted']+=1
                    check();yield deepcopy(chain)
                else:self.stats['regional_chain_caps']+=1
            else:self.stats['regional_chain_caps']+=1
            # Commit completed optional work only after all checks succeed.
            self.pending.pop(key);self.attempted[key]=True
            while len(self.attempted)>self.history:self.attempted.popitem(last=False)
