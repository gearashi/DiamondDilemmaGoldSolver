"""Bounded original-root verification and optional cross-lane CUDA clause reuse.

GPU local clauses and serialized records are claims. Only the separate CPU
store's completed original-domain UNSAT answers may enter another lane.
"""
from __future__ import annotations
from constraint_shared_conflicts import VerifiedConflictStore,MAX_PENDING
from constraint_work import digest

COUNTERS=('gpu_claims_harvested','gpu_claims_offered','gpu_claims_oversize',
          'gpu_shared_imports','gpu_shared_comparisons','gpu_shared_verification_calls',
          'gpu_shared_verified','gpu_shared_rejected','gpu_shared_unknown','gpu_shared_pending_reused')

from gpu_deduction_round26 import COUNTERS as PROOF_COUNTERS
CHECKER_COUNTERS=('chain_events','chain_removed_values','chain_seam_placements','chain_hall_placements','chain_loop_checks')
from gpu_proof_cache_round26 import COUNTERS as PROOF_CACHE_COUNTERS,ProofCache
COUNTERS+=PROOF_CACHE_COUNTERS+PROOF_COUNTERS+tuple('gpu_proof_checked_'+key[6:] for key in CHECKER_COUNTERS)+('gpu_proof_admitted','gpu_proof_invalid','gpu_proof_duplicate_skips')

GAUGES=('gpu_shared_entries','gpu_shared_pending','gpu_shared_verifier_clauses')

class SharingStopped(Exception):pass

class GPUClauseExchange:
    def __init__(self,problem,target,*,store=None,resume=None):
        self.problem,self.target=problem,target;self.owned=store is None
        self.store=VerifiedConflictStore(problem,target) if store is None else store
        if not isinstance(self.store,VerifiedConflictStore) or not self.store.matches(problem,target):
            raise ValueError('Shared GPU verifier original root or objective differs.')
        self.proof_seen=[];self.proof_cursor=0
        self.counters=dict.fromkeys(COUNTERS,0);self.cursor=0;self._refresh_engine=None;self._refresh_signature=None
        try:
            if resume is not None:
                if (not isinstance(resume,dict) or set(resume)!=({'version','verified','pending','cursor','counters','digest'} if resume.get('version')==1 else ({'version','verified','pending','cursor','counters','digest','proof_seen','proof_cursor'} if resume.get('version')==2 else {'version','verified','pending','cursor','counters','digest','proof_seen','proof_cursor','proof_cache'}))
                        or resume['version'] not in (1,2,3) or type(resume['version'])is not int
                        or resume['digest']!=digest({k:v for k,v in resume.items() if k!='digest'})):
                    raise ValueError('Invalid GPU clause exchange checkpoint.')
                counters=resume['counters']
                if (not isinstance(counters,dict) or not set(counters)<=set(COUNTERS) or any(type(v)is not int or v<0 for v in counters.values())
                        or type(resume['cursor'])is not int or not 0<=resume['cursor']<512*32):raise ValueError('Invalid GPU sharing work counters.')
                if resume['version']>=2:
                    seen=resume['proof_seen'];cursor=resume['proof_cursor']
                    if (not isinstance(seen,list) or len(seen)>64
                        or any(type(v)is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v) for v in seen) or len(set(seen))!=len(seen)
                        or type(cursor)is not int or not 0<=cursor<512):raise ValueError('Invalid structural proof dedup state.')
                    self.proof_seen=list(seen);self.proof_cursor=cursor
                # Validate BOTH claim transports before changing a caller-owned
                # live store. The temporary validator never builds a SAT model.
                validator=VerifiedConflictStore(problem,target)
                try:validator.import_snapshot(resume['verified']);validator.import_pending(resume['pending'])
                finally:validator.close()
                self.store.import_snapshot(resume['verified']);self.store.import_pending(resume['pending'])
                self.counters.update(counters);self.cursor=resume['cursor']
            self.proof_cache=ProofCache(problem,target,self.counters,resume.get('proof_cache') if resume else None)
        except BaseException:
            if self.owned:self.store.close()
            raise
    def snapshot(self):
        payload={'version':3,'proof_cache':self.proof_cache.snapshot(),'proof_seen':list(self.proof_seen),'proof_cursor':self.proof_cursor,'verified':self.store.snapshot(),'pending':self.store.pending_snapshot(),
                 'cursor':self.cursor,'counters':dict(self.counters)}
        payload['digest']=digest(payload);return payload
    def verified_snapshot(self):return self.store.snapshot()
    def public_stats(self):
        return {**self.counters,'gpu_shared_entries':len(self.store),'gpu_shared_pending':self.store.pending_count,'gpu_shared_verifier_clauses':self.store.stats['shared_conflicts_clauses']}
    def _verify(self,check):
        before=dict(self.store.stats)
        try:self.store.advance(check,seconds=.01,attempts=1,conflicts=256)
        finally:
            for local,remote in [('gpu_shared_verification_calls','shared_conflicts_calls'),('gpu_shared_verified','shared_conflicts_verified'),
                                 ('gpu_shared_rejected','shared_conflicts_rejected'),('gpu_shared_unknown','shared_conflicts_unknown'),('gpu_shared_pending_reused','shared_conflicts_pending_reused')]:
                self.counters[local]+=max(0,self.store.stats.get(remote,0)-before.get(remote,0))
    def _structural(self,engine,check):
        if self.proof_cache.pending():
            self.proof_cache.advance(self.store,check);return self.proof_cache.unchecked()
        if engine is None or not getattr(engine,'proof_transport',False):return False
        trace=engine.deduction_input(self.proof_cursor)
        if trace is None:return False
        self.proof_cursor=trace['next'];identity=trace['identity']
        if identity in self.proof_seen:self.counters['gpu_proof_duplicate_skips']+=1;return False
        from gpu_deduction_round26 import produce
        charged={}
        try:
            certificate,_=produce(self.problem,self.target,trace['root'],trace['decisions'],trace['trail'],check=check,stats=charged)
            if certificate is not None:
                evicted=self.proof_cache.offer(identity,certificate)
                self.proof_seen=[key for key in self.proof_seen if key not in evicted]
                # Raw evidence is durable before replay. Cancellation cannot
                # make the next slice repeat native trace reconstruction.
                self.proof_seen.append(identity);self.proof_seen=self.proof_seen[-64:]
                self.proof_cache.advance(self.store,check)
            else:self.proof_seen.append(identity);self.proof_seen=self.proof_seen[-64:]
        finally:
            for key,value in charged.items():self.counters[key]+=value
        return self.proof_cache.unchecked()
    def sync(self,engine=None,check=lambda:None):
        check();verified_once=False
        proof_wait=self._structural(engine,check)
        if any(not row for row in self.store.records()):return True
        # Older pending claims always receive the first verification slice.
        if self.store.pending_count and not proof_wait:self._verify(check);verified_once=True
        if any(not row for row in self.store.records()):return True
        if engine is not None:
            if engine.delivered:raise ValueError('Enumeration exclusions cannot be shared as puzzle failures.')
            room=MAX_PENDING-self.store.pending_count
            if room>0:
                records,self.cursor,oversize=engine.learned_claims(self.cursor,lanes=4,limit=min(16,room))
                self.counters['gpu_claims_oversize']+=oversize;self.counters['gpu_claims_harvested']+=len(records)
                for row in records:self.counters['gpu_claims_offered']+=int(self.store.offer(row))
            if not verified_once and self.store.pending_count and not proof_wait:self._verify(check)
            if any(not row for row in self.store.records()):return True
            self.refresh(engine)
        return False
    def invalidate(self):self._refresh_engine=None;self._refresh_signature=None
    def needs_initial_refresh(self,engine):return self._refresh_engine is not engine
    def refresh(self,engine):
        records=tuple(tuple(tuple(pair) for pair in row) for row in sorted(self.store.records(),key=lambda row:(len(row),row))[:8])
        if any(not row for row in records):raise ValueError('Handle original-root contradiction before refreshing GPU clauses.')
        # Only local learning changes ring contents after a prior refresh. Cold
        # native restores may repeat comparisons, but never duplicate clauses.
        generations=engine.state[[15,24]].get().tobytes() if records else None
        signature=(records,generations)
        if records and (self._refresh_engine is not engine or self._refresh_signature!=signature):
            counts=engine.import_verified_clauses(self.store)
            self.counters['gpu_shared_imports']+=counts['inserted'];self.counters['gpu_shared_comparisons']+=counts['comparisons']
        self._refresh_engine=engine;self._refresh_signature=signature
    def close(self):
        self._refresh_engine=None
        if self.owned:self.store.close()
