"""Optional incremental SAT assistance for exact DFS.

The oracle encodes original domain, seam and physical-tile constraints. Optional
Gold clauses are independently re-proved from original segments before import.
UNSAT under placement assumptions proves a reusable forbidden conjunction.
SAT remains only possible feasibility; it is never accepted as a Gold solution.
Partial model construction, limits and interruption cannot prove UNSAT.
"""
from collections import OrderedDict,defaultdict,deque
import threading,time
from validator import reverse_mask
from constraint_knowledge import KnowledgeStore,MAX_RECORDS,revalidate_witness
STAT_KEYS=('sat_assist_build_steps','sat_assist_clauses','sat_assist_calls','sat_assist_unsat',
           'sat_assist_unknown','sat_assist_cache_hits','sat_assist_core_literals_saved','sat_assist_subset_hits',
           'sat_assist_gold_verified','sat_assist_gold_clauses','sat_assist_gold_duplicates',
           'sat_assist_gold_irrelevant','sat_assist_gold_cache_invalidations','sat_assist_gold_budget_skips',
           'sat_assist_domain_checks','sat_assist_domain_propagations','sat_assist_domain_calls',
           'sat_assist_domain_unsat','sat_assist_domain_unknown','sat_assist_domain_prunes',
           'sat_assist_domain_forced','sat_assist_mask_cores','sat_assist_mask_core_hits',
           'sat_assist_mask_core_literals_saved','sat_assist_domain_resumes',
           'sat_assist_domain_completed_skips','sat_assist_model_supports',
           'sat_assist_model_invalid','sat_assist_mask_calls','sat_assist_mask_unsat',
           'sat_assist_witness_pool_hits','sat_assist_witness_pool_checks',
           'sat_assist_support_cover_calls','sat_assist_support_cover_prunes')
class SatAssist:
    def __init__(self,problem,initial,target='gold',*,factory=None,witness_pool=False,
                 support_cover=False,capture_clauses=False):
        self.problem,self.initial,self.target=problem,tuple(initial),target
        self.witness_pool=bool(witness_pool);self.support_cover=bool(support_cover)
        self._witnesses=OrderedDict();self.semantic_witnesses=None;self._captured=[] if capture_clauses else None
        if len(self.initial)!=problem.n or any(type(value)is not int or value<0 or value>>(3*problem.n) for value in self.initial):
            raise ValueError('SAT initial domains must be valid orientation bitsets.')
        self._original_domains=tuple(tuple(sorted(set(row))) for row in problem.domains)
        self._gold_fingerprint=problem.fingerprint
        self._gold_known=set();self._gold_pending=deque();self._gold_limit=2048
        self.solver=None;self.ready=False;self.reason=None;self.status='building';self.variables=[];self._placement_decode={}
        self.ids=0;self.clauses=0;self.cache=OrderedDict();self.cores=OrderedDict();self.candidates=OrderedDict();self.last_outcome='not_ready';self.stats=dict.fromkeys(STAT_KEYS,0)
        self.domain_cores=OrderedDict();self.domain_candidates=OrderedDict()
        self.domain_sessions=OrderedDict();self._gold_revision=0
        if problem.n>160 or sum(value.bit_count() for value in self.initial)>80000:
            self.status='unsupported';self.reason='SAT assistance is capped at 160 cells and 80,000 placements.';self.pending=None
            return
        try:
            if factory is None:
                from pysat.solvers import Glucose4
                factory=Glucose4
            self.solver=factory();self.pending=self._build()
        except ImportError as exc:self.reason=str(exc);self.status='unavailable';self.pending=None
    def close(self):
        if self.solver is not None:self.solver.delete();self.solver=None
        self.ready=False
    def __del__(self):
        try:self.close()
        except Exception:pass
    def fresh(self):self.ids+=1;return self.ids
    def _exactly_one(self,literals):
        yield literals
        if len(literals)<2:return
        previous=self.fresh();yield [-literals[0],previous]
        for literal in literals[1:-1]:
            current=self.fresh();yield [-literal,current];yield [-previous,current];yield [-literal,-previous];previous=current
        yield [-literals[-1],-previous]
    def _build(self):
        from constraint_pruning import values
        for cell,domain in enumerate(self.initial):
            row={code:self.fresh() for code in values(domain)}
            self.variables.append(row)
            self._placement_decode.update((literal,(cell,code)) for code,literal in row.items())
            yield None
        for row in self.variables:yield from self._exactly_one(list(row.values()))
        by_tile=[[] for _ in range(self.problem.n)]
        for row in self.variables:
            for code,literal in row.items():by_tile[code//3].append(literal)
        for row in by_tile:yield from self._exactly_one(row)
        for a,sa,b,sb in self.problem.edges:
            left,right=defaultdict(list),defaultdict(list)
            for code,literal in self.variables[a].items():left[int(self.problem.masks[code,sa])].append(literal)
            for code,literal in self.variables[b].items():right[reverse_mask(int(self.problem.masks[code,sb]))].append(literal)
            common=left.keys()&right.keys()
            for mask in left.keys()-common:
                for lit in left[mask]:yield [-lit]
            for mask in right.keys()-common:
                for lit in right[mask]:yield [-lit]
            for mask in sorted(common):
                sig=self.fresh()
                for supports in (left[mask],right[mask]):
                    yield [-sig,*supports]
                    for lit in supports:yield [-lit,sig]
    def _post_gold(self,record):
        """Commit one completely verified clause without an intervening poll."""
        if self.clauses>=1000000:
            self.close();self.pending=None;self.status='unsupported'
            self.reason='SAT assistance exceeded its one-million-clause limit.'
            return False
        clause=[-self.variables[cell][code] for cell,code in record]
        self.solver.add_clause(clause)
        if self._captured is not None:self._captured.append(tuple(clause))
        self.clauses+=1;self.stats['sat_assist_clauses']=self.clauses
        self.stats['sat_assist_gold_clauses']+=1
        # Strengthening can invalidate SAT answers, but never a proved UNSAT.
        stale=[key for key,value in self.cache.items() if value is None]
        for key in stale:del self.cache[key]
        self.stats['sat_assist_gold_cache_invalidations']+=len(stale)
        self.domain_sessions.clear();self._gold_revision+=1
        self.last_outcome='unknown'
        return True

    def add_gold_witnesses(self,source,check=lambda:None,*,max_records=8):
        """Independently prove a bounded batch of Gold-loop exclusions.

        A matching KnowledgeStore supplies candidates, never proof authority.
        Novel records are revalidated against original tile segments. Exact
        previously imported records can reuse only this helper's own proof.
        Building models queue verified clauses; queries stay disabled until
        all queued clauses are installed. Each installed/queued record commits
        atomically, so Stop can retain a prefix of already completed proofs.
        """
        if type(max_records)is not int or not 0<=max_records<=MAX_RECORDS:
            raise ValueError('Gold SAT import budget must be between zero and 256 records.')
        if (self.target!='gold' or not isinstance(source,KnowledgeStore)
                or not source.matches(self.problem,'gold')
                or self.problem.fingerprint!=self._gold_fingerprint
                or tuple(tuple(sorted(set(row))) for row in self.problem.domains)!=self._original_domains):
            raise ValueError('Gold SAT witnesses require matching original puzzle domains and target.')
        check()
        if self.solver is None:return 0
        records=source.records()
        if len(records)>MAX_RECORDS:raise ValueError('Gold SAT source exceeds its record bound.')
        batch=[];seen=set();examined=0
        for record in records:
            # Strict types prevent bool aliases from reusing an integer proof.
            key=None
            if (isinstance(record,(list,tuple)) and 1<=len(record)<=self.problem.n
                    and all(isinstance(pair,(list,tuple)) and len(pair)==2
                            and type(pair[0])is int and type(pair[1])is int for pair in record)):
                key=tuple(sorted(tuple(pair) for pair in record))
            if key is not None and (key in self._gold_known or key in seen):
                self.stats['sat_assist_gold_duplicates']+=1;continue
            if examined>=max_records or len(self._gold_known)+len(batch)>=self._gold_limit:
                self.stats['sat_assist_gold_budget_skips']+=1;break
            check();examined+=1
            canonical=revalidate_witness(self.problem,record,target='gold')
            check();self.stats['sat_assist_gold_verified']+=1
            if canonical in self._gold_known or canonical in seen:
                self.stats['sat_assist_gold_duplicates']+=1;continue
            batch.append(canonical);seen.add(canonical)
        added=0
        for record in batch:
            check()
            if any(not self.initial[cell] & (1<<code) for cell,code in record):
                self.stats['sat_assist_gold_irrelevant']+=1
            elif self.ready:
                if not self._post_gold(record):return added
                added+=1
            else:
                self._gold_pending.append(record);added+=1
            self._gold_known.add(record)
        return added

    def prepare(self,check=lambda:None,*,seconds=.01):
        if self.solver is None:return False
        if self.ready:return True
        deadline=time.monotonic()+max(0,seconds)
        while time.monotonic()<deadline:
            check();self.stats['sat_assist_build_steps']+=1
            if self.pending is None:
                if self._gold_pending:
                    if not self._post_gold(self._gold_pending[0]):return False
                    self._gold_pending.popleft()
                    continue
                self.ready=True;self.status='ready';return True
            try:clause=next(self.pending)
            except StopIteration:self.pending=None;continue
            if clause is not None:
                if self.clauses>=1000000:
                    self.close();self.pending=None;self.status='unsupported';self.reason='SAT assistance exceeded its one-million-clause limit.'
                    return False
                self.solver.add_clause(clause)
                if self._captured is not None:self._captured.append(tuple(clause))
                self.clauses+=1;self.stats['sat_assist_clauses']=self.clauses
        return False
    def _domain_call(self,operation,check,deadline,*,conflicts):
        """Return a completed native answer or None; never accept interrupted work."""
        check()
        if time.monotonic()>=deadline:return None
        done=threading.Event();interrupted=[]
        def monitor():
            while not done.wait(.002):
                try:check()
                except BaseException as exc:interrupted.append(exc);self.solver.interrupt();return
                if time.monotonic()>=deadline:interrupted.append(None);self.solver.interrupt();return
        self.solver.clear_interrupt();self.solver.conf_budget(conflicts)
        watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
        try:answer=operation()
        finally:done.set();watcher.join();self.solver.clear_interrupt()
        if interrupted:
            if interrupted[0] is not None:raise interrupted[0]
            return None
        check()
        # A fast native call can finish before its monitor wakes. Its proof is
        # valid, but rejecting late answers keeps the resource rule explicit.
        return answer if time.monotonic()<deadline else None

    def _domain_assumptions(self,domains,check,deadline):
        assumptions=[]
        for cell,(mask,row) in enumerate(zip(domains,self.variables)):
            check()
            if time.monotonic()>=deadline:return None
            if mask.bit_count()==1:
                assumptions.append(row[mask.bit_length()-1]);continue
            excluded=self.initial[cell]&~mask;examined=0
            while excluded:
                if examined%64==0:
                    check()
                    if time.monotonic()>=deadline:return None
                bit=excluded&-excluded;excluded^=bit;examined+=1
                assumptions.append(-row[bit.bit_length()-1])
        return assumptions

    def _mask_core(self,assumptions):
        # A core containing excluded placement literals is a conjunction of
        # allowed MASKS, not a conjunction of positively selected placements.
        getter=getattr(self.solver,'get_core',None)
        raw=assumptions if getter is None else getter()
        raw=[] if raw is None else list(raw)
        supplied=set(assumptions);merged={}
        for literal in raw:
            if type(literal)is not int or literal not in supplied:
                raise RuntimeError('SAT returned an unrelated mask assumption core.')
            cell,code=self._placement_decode[abs(literal)]
            mask=(1<<code) if literal>0 else self.initial[cell]&~(1<<code)
            merged[cell]=merged.get(cell,self.initial[cell])&mask
        core=tuple(sorted((cell,mask) for cell,mask in merged.items() if mask!=self.initial[cell]))
        self.stats['sat_assist_mask_core_literals_saved']+=len(assumptions)-len(raw)
        return core

    def _remember_mask_core(self,core):
        if any(not mask for cell,mask in core):return  # Tautological conjunction.
        def subsumes(left,right):
            other=dict(right)
            return all(cell in other and not other[cell]&~mask for cell,mask in left)
        if any(subsumes(old,core) for old in self.domain_cores):return
        for old in list(self.domain_cores):
            if subsumes(core,old):del self.domain_cores[old]
        self.domain_cores[core]=None;self.domain_candidates[core]=None
        self.stats['sat_assist_mask_cores']+=1
        while len(self.domain_cores)>128:self.domain_cores.popitem(last=False)
        while len(self.domain_candidates)>128:self.domain_candidates.popitem(last=False)

    def _prune_mask_cores(self,domains,check):
        # One bounded pass is enough for soundness; the outer propagation loop
        # can ask again. Clauses remain valid when Gold clauses are added.
        for core in self.domain_cores:
            check();unknown=[]
            for cell,mask in core:
                if not domains[cell]&mask:break
                if domains[cell]&~mask:unknown.append((cell,mask))
                if len(unknown)>1:break
            else:
                self.stats['sat_assist_mask_core_hits']+=1
                if not unknown:domains[0]=0;return
                cell,mask=unknown[0];domains[cell]&=~mask

    def drain_domain_candidates(self):
        """Mask conflict CLAIMS bound to this helper's root; reverify before export."""
        result=list(self.domain_candidates);self.domain_candidates.clear();return result

    def query_masks(self,literals,check=lambda:None,*,seconds=.01,conflicts=256):
        """Prove a mask conjunction impossible in this exact effective root.

        Returned cores have the same semantics as LCG forbidden mask clauses.
        A restricted-root answer must be re-proved under original domains before
        cross-job sharing. SAT and unknown return None and are not Gold answers.
        """
        current=list(self.initial);seen=set()
        if not isinstance(literals,(list,tuple)) or len(literals)>self.problem.n:
            raise ValueError('Mask assumptions need a bounded literal list.')
        for pair in literals:
            if not isinstance(pair,(list,tuple)) or len(pair)!=2:raise ValueError('Invalid mask assumption.')
            cell,mask=pair
            if (type(cell)is not int or not 0<=cell<self.problem.n or cell in seen
                    or type(mask)is not int or mask<=0 or mask&~self.initial[cell]):
                raise ValueError('Mask assumption outside effective root.')
            seen.add(cell);current[cell]=mask
        check();self.last_outcome='not_ready'
        if not self.ready:return None
        self.last_outcome='unknown'
        for core in self.domain_cores:
            if all(not current[c]&~mask for c,mask in core):
                self.stats['sat_assist_mask_core_hits']+=1;self.last_outcome='unsat';return core
        if seconds<=0:return None
        deadline=time.monotonic()+seconds;assumptions=self._domain_assumptions(current,check,deadline)
        if assumptions is None:return None
        self.stats['sat_assist_mask_calls']+=1
        answer=self._domain_call(lambda:self.solver.solve_limited(assumptions=assumptions,expect_interrupt=True),
                                 check,deadline,conflicts=conflicts)
        if answer is False:
            core=self._mask_core(assumptions);self._remember_mask_core(core)
            self.stats['sat_assist_mask_unsat']+=1;self.last_outcome='unsat';return core
        if answer is True:self.last_outcome='sat'
        return None

    def _model_supports(self,domains,trial,check):
        getter=getattr(self.solver,'get_model',None)
        if getter is None:return None
        raw=getter()
        if raw is None:return None
        chosen=set(lit for lit in raw if type(lit)is int and lit>0)
        if any(lit not in chosen if lit>0 else -lit in chosen for lit in trial):
            self.stats['sat_assist_model_invalid']+=1;return None
        codes=[]
        for cell,row in enumerate(self.variables):
            if cell%32==0:check()
            found=[code for code,literal in row.items() if literal in chosen]
            if len(found)!=1 or not domains[cell]&(1<<found[0]):
                self.stats['sat_assist_model_invalid']+=1;return None
            codes.append(found[0])
        # Independently check the semantic SAT model before using it to skip
        # additional checks. It need not be a one-loop Gold arrangement.
        if (len({code//3 for code in codes})!=self.problem.n
                or any(int(self.problem.masks[codes[a],sa])!=reverse_mask(int(self.problem.masks[codes[b],sb]))
                       for a,sa,b,sb in self.problem.edges)
                or any(all(codes[c]==code for c,code in clause) for clause in self._gold_known)):
            self.stats['sat_assist_model_invalid']+=1;return None
        witness=tuple(codes)
        if self.semantic_witnesses is not None:self.semantic_witnesses.add(witness,check)
        if self.witness_pool:
            self._witnesses[witness]=None;self._witnesses.move_to_end(witness)
            while len(self._witnesses)>128:self._witnesses.popitem(last=False)
        return tuple(1<<code for code in codes)

    def proof_base(self):
        """Immutable deterministic CNF snapshot; temporary cover clauses are excluded.

        Capture must be explicitly enabled before construction.  The encoded
        domain/seam/inventory theory and independently verified Gold witnesses
        form an optimistic Gold model, so UNSAT is sufficient for Gold UNSAT.
        """
        if self._captured is None or not self.ready:
            raise ValueError('Proof capture requires a completely prepared captured model.')
        return {'clauses':tuple(self._captured),
                'variables':tuple(tuple(sorted(row.items())) for row in self.variables),
                'initial':self.initial,'target':self.target,
                'fingerprint':self.problem.fingerprint,
                'gold':tuple(sorted(self._gold_known))}

    def _pool_supports(self,domains,supported,check):
        if self.semantic_witnesses is not None:
            self.semantic_witnesses.add_cuts(self._gold_known,check)
            before=self.semantic_witnesses.stats['witness_index_hits']
            gained=self.semantic_witnesses.supports(domains,supported,check)
            self.stats['sat_assist_witness_pool_hits']+=self.semantic_witnesses.stats['witness_index_hits']-before
            self.stats['sat_assist_model_supports']+=gained
            return
        # Never trust context identity: each vector is checked against all
        # current domain restrictions and the current verified Gold clauses.
        for witness in tuple(self._witnesses):
            check();self.stats['sat_assist_witness_pool_checks']+=1
            if (any(not mask&(1<<code) for mask,code in zip(domains,witness))
                    or any(all(witness[c]==code for c,code in clause) for clause in self._gold_known)):
                continue
            gained=sum(not old&(1<<code) for old,code in zip(supported,witness))
            if gained:
                self.stats['sat_assist_witness_pool_hits']+=1
                self.stats['sat_assist_model_supports']+=gained
                for cell,code in enumerate(witness):supported[cell]|=1<<code

    def _cover_supports(self,current,session,assumptions,check,deadline,conflicts,maximum):
        """Cover all unsupported placements with model-producing SAT queries.

        Every temporary disjunction is guarded by a fresh activation variable.
        Its UNSAT answer applies to the ENTIRE disjunction, not just one value.
        A selector is permanently disabled in finally, including on Stop.
        """
        for _ in range(maximum):
            check()
            if time.monotonic()>=deadline:return
            uncovered=[(cell,code,lit) for cell,row in enumerate(self.variables)
                       for code,lit in row.items()
                       if current[cell].bit_count()>1 and current[cell]&(1<<code) and not session['supported'][cell]&(1<<code)]
            if not uncovered:return
            selector=self.fresh()
            self.solver.add_clause([-selector,*(lit for _,_,lit in uncovered)])
            self.stats['sat_assist_support_cover_calls']+=1
            self.stats['sat_assist_domain_calls']+=1
            try:
                trial=[*assumptions,selector]
                answer=self._domain_call(lambda:self.solver.solve_limited(assumptions=trial,expect_interrupt=True),
                                         check,deadline,conflicts=conflicts)
                if answer is False:
                    # The guard is an auxiliary literal and cannot be treated
                    # as a placement assumption by the mask-core decoder.
                    for cell,code,_ in uncovered:current[cell]&=~(1<<code)
                    self.stats['sat_assist_support_cover_prunes']+=len(uncovered)
                    self.stats['sat_assist_domain_unsat']+=1
                    return
                if answer is None:
                    self.stats['sat_assist_domain_unknown']+=1;return
                witness=self._model_supports(current,trial,check)
                if witness is None:return
                gained=sum((mask&~old).bit_count() for mask,old in zip(witness,session['supported']))
                if not gained:return  # Defensive against an invalid/incomplete backend model.
                self.stats['sat_assist_model_supports']+=gained
                for cell,mask in enumerate(witness):session['supported'][cell]|=mask
            finally:
                self.solver.add_clause([-selector])

    def filter_domains(self,domains,check=lambda:None,*,seconds=.01,conflicts=256,max_candidates=16):
        """Completed scoped consequences, resumable probes and witness reuse.

        Exact-context sessions remember completed reductions and SAT supports.
        Unknown work is retried; Stop never publishes partial reductions to the
        caller. Compact UNSAT cores also apply under other compatible contexts.
        Feasibility witnesses only skip redundant tests, never certify Gold.
        """
        original=tuple(domains)
        if (len(original)!=self.problem.n or any(type(mask)is not int or mask<0 or
                mask&~initial for mask,initial in zip(original,self.initial))):
            raise ValueError('SAT filtering domains must be subsets of original domains.')
        if type(max_candidates)is not int or not 0<=max_candidates<=256:
            raise ValueError('SAT candidate budget must be between zero and 256.')
        check();self.stats['sat_assist_domain_checks']+=1
        if not self.ready or seconds<=0 or not all(original):return original
        deadline=time.monotonic()+seconds
        session=self.domain_sessions.get(original)
        if session is None:
            session={'reduced':original,'supported':[0]*self.problem.n,'cursor':0}
            self.domain_sessions[original]=session
            while len(self.domain_sessions)>64:self.domain_sessions.popitem(last=False)
        else:
            self.domain_sessions.move_to_end(original);self.stats['sat_assist_domain_resumes']+=1
        current=list(session['reduced'])
        self._prune_mask_cores(current,check)
        def finish():
            check();result=tuple(current);session['reduced']=result
            self.stats['sat_assist_domain_prunes']+=sum((a&~b).bit_count() for a,b in zip(original,result))
            self.stats['sat_assist_domain_forced']+=sum(a.bit_count()>1 and b.bit_count()==1 for a,b in zip(original,result))
            return result
        if not all(current):return finish()
        if self.witness_pool:self._pool_supports(current,session['supported'],check)
        assumptions=self._domain_assumptions(current,check,deadline)
        if assumptions is None:return finish()
        propagate=getattr(self.solver,'propagate',None)
        if propagate is not None:
            self.stats['sat_assist_domain_propagations']+=1
            try:answer=self._domain_call(lambda:propagate(assumptions=assumptions),check,deadline,conflicts=conflicts)
            except NotImplementedError:answer=None
            if answer is None:self.stats['sat_assist_domain_unknown']+=1
            elif answer[0] is False:
                # Propagate does not promise a get_core() result; its complete
                # input context is the conservative conflict certificate here.
                self._remember_mask_core(tuple((c,mask) for c,mask in enumerate(current) if mask!=self.initial[c]))
                self.stats['sat_assist_domain_unsat']+=1;current[0]=0;return finish()
            elif answer[0] is True:
                for literal in answer[1]:
                    pair=self._placement_decode.get(abs(literal))
                    if pair is None:continue
                    cell,code=pair;current[cell]&=(1<<code) if literal>0 else ~(1<<code)
                if not all(current):
                    self.stats['sat_assist_domain_unsat']+=1;return finish()
        if self.support_cover:
            self._cover_supports(current,session,assumptions,check,deadline,conflicts,max_candidates)
            return finish()
        candidates=[(cell,code) for cell in sorted(range(self.problem.n),key=lambda c:(original[c].bit_count(),c))
                    for code in self.variables[cell] if original[cell]&(1<<code)]
        tested=visited=0;total=len(candidates)
        while total and visited<total and tested<max_candidates:
            check()
            if time.monotonic()>=deadline:return finish()
            index=session['cursor']%total;cell,code=candidates[index];bit=1<<code
            session['cursor']=(index+1)%total;visited+=1
            if not current[cell]&bit or current[cell].bit_count()<2 or session['supported'][cell]&bit:
                self.stats['sat_assist_domain_completed_skips']+=1;continue
            tested+=1;self.stats['sat_assist_domain_calls']+=1
            trial=[*assumptions,self.variables[cell][code]]
            answer=self._domain_call(lambda:self.solver.solve_limited(assumptions=trial,expect_interrupt=True),check,deadline,conflicts=conflicts)
            if answer is False:
                self.stats['sat_assist_domain_unsat']+=1;current[cell]&=~bit
                self._remember_mask_core(self._mask_core(trial))
            elif answer is True:
                session['supported'][cell]|=bit
                witness=self._model_supports(current,trial,check)
                if witness is not None:
                    self.stats['sat_assist_model_supports']+=sum((mask&~old).bit_count() for mask,old in zip(witness,session['supported']))
                    for c,mask in enumerate(witness):session['supported'][c]|=mask
            else:self.stats['sat_assist_domain_unknown']+=1
        return finish()

    def drain_candidates(self):
        """Return private proved cores as UNTRUSTED publication candidates.

        A caller must re-prove each core under original problem domains before
        exporting it outside this helper's possibly restricted initial domains.
        """
        result=list(self.candidates);self.candidates.clear();return result
    def query(self,placements,check=lambda:None,*,seconds=.01,conflicts=256):
        """Return None for SAT/unknown, or a proved core (possibly empty)."""
        self.last_outcome='not_ready'
        if not self.ready:return None
        decisions=tuple(sorted(set(tuple(x) for x in placements)))
        if len({c for c,_ in decisions})!=len(decisions):raise ValueError('Conflicting SAT placement assumptions.')
        for cell,code in decisions:
            if type(cell)is not int or type(code)is not int or not 0<=cell<self.problem.n or code not in self.variables[cell]:
                raise ValueError('SAT assumption outside original domains.')
        check()
        if decisions in self.cache:
            self.stats['sat_assist_cache_hits']+=1;core=self.cache[decisions]
            self.last_outcome='sat' if core is None else 'unsat';return core
        chosen=frozenset(decisions)
        for core in self.cores:
            if frozenset(core)<=chosen:
                self.stats['sat_assist_subset_hits']+=1;self.last_outcome='unsat';return core
        self.last_outcome='unknown';self.stats['sat_assist_calls']+=1
        if seconds<=0:self.stats['sat_assist_unknown']+=1;return None
        assumptions=[self.variables[cell][code] for cell,code in decisions]
        decode={literal:pair for literal,pair in zip(assumptions,decisions)}
        done=threading.Event();interrupted=[];deadline=time.monotonic()+seconds
        def monitor():
            while not done.wait(.002):
                try:check()
                except BaseException as exc:interrupted.append(exc);self.solver.interrupt();return
                if time.monotonic()>=deadline:interrupted.append(None);self.solver.interrupt();return
        self.solver.clear_interrupt();self.solver.conf_budget(conflicts)
        watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
        try:answer=self.solver.solve_limited(assumptions=assumptions,expect_interrupt=True)
        finally:done.set();watcher.join();self.solver.clear_interrupt()
        if interrupted:
            if interrupted[0] is not None:raise interrupted[0]
            self.stats['sat_assist_unknown']+=1;return None
        check()
        if answer is False:
            raw=self.solver.get_core() or []
            if any(lit not in decode for lit in raw):raise RuntimeError('SAT returned an unrelated assumption core.')
            core=tuple(sorted(decode[lit] for lit in raw));self.stats['sat_assist_unsat']+=1
            self.stats['sat_assist_core_literals_saved']+=len(decisions)-len(core)
            self.cache[decisions]=core;self.last_outcome='unsat'
            self.cores[core]=None;self.candidates[core]=None
            while len(self.cores)>64:self.cores.popitem(last=False)
            while len(self.candidates)>64:self.candidates.popitem(last=False)
        elif answer is True:self.cache[decisions]=None;core=None;self.last_outcome='sat'
        else:self.stats['sat_assist_unknown']+=1;return None
        while len(self.cache)>64:self.cache.popitem(last=False)
        return core
