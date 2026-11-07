"""Bounded durable structural evidence and independently checked guard reduction.

Saved certificates remain untrusted until replayed in the current store. Cursors
skip completed optional minimization attempts; they never establish a deduction.
"""
from copy import deepcopy
import json
from constraint_work import digest
from constraint_deduction_chain_round26 import ChainError

MAX_BYTES=4*1024*1024
MAX_RECORDS=32
COUNTERS=('gpu_proof_cache_replays','gpu_proof_cache_evictions','gpu_proof_minimize_trials',
          'gpu_proof_minimize_successes','gpu_proof_minimize_unknown','gpu_proof_verifier_checks')

class ProofBudget(Exception):pass

def encoded_size(value):return len(json.dumps(value,sort_keys=True,indent=2).replace('\n','\r\n').encode('utf-8'))

class ProofCache:
    def __init__(self,problem,target,counters,resume=None):
        self.problem,self.target,self.counters=problem,target,counters;self.records=[];self.validated=set()
        if resume is not None:
            if (not isinstance(resume,dict) or set(resume)!={'version','records','digest'} or type(resume['version'])is not int or resume['version']!=1
                or resume['digest']!=digest({k:v for k,v in resume.items() if k!='digest'}) or encoded_size(resume)>MAX_BYTES):raise ValueError('Invalid structural proof cache.')
            rows=resume['records']
            if not isinstance(rows,list) or len(rows)>MAX_RECORDS:raise ValueError('Structural proof cache record cap.')
            seen=set()
            for row in rows:
                if not isinstance(row,dict) or set(row)!={'identity','certificate','cursor'}:raise ValueError('Malformed structural proof cache record.')
                key=row['identity'];certificate=row['certificate']
                if (type(key)is not str or len(key)!=64 or any(c not in '0123456789abcdef' for c in key) or key in seen
                    or type(row['cursor'])is not int or not 0<=row['cursor']<=2*problem.n+1
                    or not isinstance(certificate,dict) or certificate.get('fingerprint')!=problem.fingerprint or certificate.get('target')!=target):raise ValueError('Structural proof cache scope differs.')
                seen.add(key)
            self.records=deepcopy(rows)
    def snapshot(self):
        value={'version':1,'records':deepcopy(self.records)};value['digest']=digest(value)
        if encoded_size(value)>MAX_BYTES:raise ValueError('Structural proof cache exceeds byte cap.')
        return value
    def pending(self):return any(row['identity'] not in self.validated or row['cursor']<2*self.problem.n+1 for row in self.records)
    def unchecked(self):return any(row['identity'] not in self.validated for row in self.records)
    def offer(self,identity,certificate):
        if any(row['identity']==identity for row in self.records):return []
        row={'identity':identity,'certificate':deepcopy(certificate),'cursor':0}
        if encoded_size({'version':1,'records':[row],'digest':'0'*64})>MAX_BYTES:return [identity]
        self.records.append(row);evicted=[]
        while len(self.records)>MAX_RECORDS or encoded_size({'version':1,'records':self.records,'digest':'0'*64})>MAX_BYTES:
            key=self.records.pop(0)['identity'];self.validated.discard(key);evicted.append(key);self.counters['gpu_proof_cache_evictions']+=1
        return evicted
    def _admit(self,store,certificate,check,max_checks):
        stats={};left=max_checks;reused=store.stats.get('shared_conflicts_pending_reused',0)
        def bounded():
            nonlocal left
            check();self.counters['gpu_proof_verifier_checks']+=1;left-=1
            if left<0:raise ProofBudget
        try:
            added=store.admit_deduction_chain(certificate,bounded,stats=stats)
            self.counters['gpu_proof_admitted']+=added
        finally:
            self.counters['gpu_shared_pending_reused']+=max(0,store.stats.get('shared_conflicts_pending_reused',0)-reused)
            for key,value in stats.items():self.counters['gpu_proof_checked_'+key[6:]]+=value
    def advance(self,store,check=lambda:None,*,max_trials=4,max_checks=200000):
        if type(max_trials)is not int or not 1<=max_trials<=16 or type(max_checks)is not int or not 1<=max_checks<=2000000:raise ValueError('Invalid proof minimization budget.')
        row=next((r for r in self.records if r['identity'] not in self.validated),None)
        if row is None:row=next((r for r in self.records if r['cursor']<2*self.problem.n+1),None)
        if row is None:return
        check();key=row['identity']
        if key not in self.validated:
            self.counters['gpu_proof_cache_replays']+=1
            try:self._admit(store,row['certificate'],check,max_checks)
            except ChainError:
                self.counters['gpu_proof_invalid']+=1;self.records.remove(row);return
            except ProofBudget:
                self.counters['gpu_proof_cap_hits']+=1;self.records.remove(row);return
            self.validated.add(key)
        attempts=0;n=self.problem.n;original=[hex(sum(1<<v for v in d)) for d in self.problem.domains]
        while row['cursor']<2*n+1 and attempts<max_trials:
            check();cursor=row['cursor'];trial=deepcopy(row['certificate']);changed=False
            if cursor==0 and trial['terminal']['rule']=='loop':
                trial['root']=list(original);trial['events']=[]
                trial['decisions']=[[cell,hex(1<<code)] for cell,code in trial['terminal']['data']['placements']];changed=True
            elif 1<=cursor<=n:
                cell=cursor-1
                if trial['root'][cell]!=original[cell]:trial['root'][cell]=original[cell];changed=True
            elif cursor>n:
                cell=cursor-n-1;before=len(trial['decisions']);trial['decisions']=[item for item in trial['decisions'] if item[0]!=cell];changed=len(trial['decisions'])!=before
            if changed:
                attempts+=1;self.counters['gpu_proof_minimize_trials']+=1
                try:self._admit(store,trial,check,max_checks)
                except (ChainError,ProofBudget):self.counters['gpu_proof_minimize_unknown']+=1
                else:row['certificate']=trial;self.counters['gpu_proof_minimize_successes']+=1
            # Stop exceptions above preserve the previous checked candidate and
            # exact unfinished attempt; a failed bounded replay proves nothing.
            row['cursor']+=1
