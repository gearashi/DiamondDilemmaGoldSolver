"""Persistent plain-SAT session over an exact portable cube frontier.

Native clauses/heuristics survive warm slices. After process restart only the
remaining disjoint cubes, independently rechecked Gold cuts, and optionally a
verified native clause subset survive. An interrupted cube may repeat other
native internal work. Completed cubes never return.
"""
from __future__ import annotations
import copy,json,math,threading,time
from constraint_sat_assist import SatAssist
from constraint_work import WorkScope,cube,contains,frontier_snapshot,restore_frontier,digest,CoverageLedger,subtract_many


class SessionInterrupted(Exception):pass


class PersistentSatSession:
    def __init__(self,problem,scope,boxes=None,*,core_lifting=False,native_persistence=False,guard_reuse=False,guard_limit=8192,regional_compiler=False,simplification=False,theory=False):
        from constraint_sat_persistence import require_capability,STAT_KEYS as PERSIST_KEYS
        if type(native_persistence)is not bool:raise ValueError("Invalid native persistence option.")
        if native_persistence:require_capability()
        if not isinstance(scope,WorkScope) or not WorkScope.for_problem(problem,scope.target).accepts(scope):
            raise ValueError('SAT session puzzle/objective/domains differ.')
        if type(guard_reuse)is not bool:raise ValueError('Invalid SAT guard reuse policy.')
        if type(guard_limit)is not int or not 0<=guard_limit<=65536:raise ValueError('Invalid SAT guard lifetime bound.')
        if type(regional_compiler)is not bool:raise ValueError('Invalid regional SAT policy.')
        if type(theory)is not bool or theory and native_persistence:raise ValueError('Persistent theory cannot use the Glucose native proof logger.')
        self.theory=theory;self._external=None;self._theory_saved=None
        from constraint_sat_theory_round26 import STAT_KEYS as THEORY_KEYS,EXTERNAL_KEYS
        self._theory_prior=dict.fromkeys((*THEORY_KEYS,*EXTERNAL_KEYS),0)
        if type(simplification)is not bool:raise ValueError('Invalid SAT simplification policy.')
        self.simplification=simplification;self._simplified=None;self._simplification_key=None
        self.regional_cache_dir=None
        self.regional_compiler=regional_compiler;self._region_clauses=();self._region_next=0;self._region_ready=False
        self.guard_reuse=guard_reuse;self.guard_limit=guard_limit;self._guards=None
        self.problem,self.scope=problem,scope
        self.boxes=[cube(row,problem.n) for row in ((scope.root,) if boxes is None else boxes)]
        frontier_snapshot(scope,self.boxes)
        self.encoder=SatAssist(problem,scope.root,'edges',capture_clauses=True)
        self.solver=None;self.variables=None;self.next_id=0;self.selector=None
        self.core_lifting=bool(core_lifting);self._guard_domains={};self._base_clauses=();self._core_verifier=None
        self.gold_cuts=[];self.closed=[];self.last_codes=None;self._imported_masks=set()
        self.stats={'session_model_builds':0,'session_warm_slices':0,'session_native_calls':0,'session_cubes_closed':0,'session_gold_cuts':0,'session_decisions':0,'session_candidates':0,'session_imported_loops':0,'session_imported_masks':0}
        self.stats.update(session_core_candidates=0,session_core_verifications=0,session_core_model_builds=0,session_core_decisions=0,session_core_verified=0,session_core_unknown=0,session_core_boxes_removed=0,session_core_assignments_removed=0)
        self.stats.update(dict.fromkeys(('regional_dag_regions','regional_dag_arcs','regional_dag_constraints','regional_dag_compile_transitions','regional_dag_compile_states','regional_dag_cap_skips','regional_dag_expanded_histories','regional_dag_compile_reuses','regional_dag_compile_resumes','regional_dag_cursor_loads','regional_dag_cursor_saves','regional_dag_cursor_io_failures','regional_dag_compile_evictions','regional_dag_encoder_replays'),0))
        self.stats.update(dict.fromkeys(PERSIST_KEYS,0))
        from constraint_sat_guards_round26 import STAT_KEYS as GUARD_KEYS
        self.stats.update(dict.fromkeys(GUARD_KEYS,0))
        from constraint_sat_simplify_round26 import STAT_KEYS as SIMPLIFY_KEYS
        self.stats.update(dict.fromkeys(SIMPLIFY_KEYS,0));self.stats.update(self._theory_prior)
        self.native_persistence=native_persistence;self.persistence=None;self._saved_native_clauses=None;self._native_logging=False
        self._persist_prior=dict.fromkeys(PERSIST_KEYS,0)
        self._started=False;self._prior_decisions=0;self._prior_core_decisions=0;self.coverage=None

    def _binding(self):
        return WorkScope.for_problem(self.problem,self.scope.target).accepts(self.scope)

    def prepare(self,check,seconds):
        if not self._binding():raise ValueError('SAT session source domains changed.')
        if self.solver is not None:
            self._restore_theory(check)
            return True
        if not self.encoder.prepare(check,seconds=seconds):
            if self.encoder.status in ('unavailable','unsupported'):raise RuntimeError('Persistent SAT unavailable: '+str(self.encoder.reason))
            return False
        from pysat.solvers import Glucose4
        model=self.encoder.proof_base();self.variables=tuple(dict(row) for row in model['variables'])
        self._base_clauses=tuple(model['clauses'])
        self.next_id=max((abs(x) for row in model['clauses'] for x in row),default=0)
        self._prepare_regions(check)
        clauses=self._search_formula(check)
        if self.theory:
            from pysat.solvers import Cadical195
            from constraint_sat_theory_round26 import direct_propagator
            self.solver=Cadical195(bootstrap_with=clauses)
            self._external=direct_propagator(self.problem,self.variables,target=self.scope.target,check=check)
            self.solver.connect_propagator(self._external)
            for row in self.variables:
                for literal in row.values():self.solver.observe(literal)
            self._restore_theory(check)
        elif self.native_persistence or self._saved_native_clauses is not None:
            if self.persistence is None:
                from constraint_sat_persistence import VerifiedNativeClauses
                self.persistence=VerifiedNativeClauses(self._base_clauses,self.variables,
                    fingerprint=self.problem.fingerprint,target=self.scope.target,initial=self.scope.root)
                if self._saved_native_clauses is not None:self.persistence.queue_import(self._saved_native_clauses)
            if self._saved_native_clauses is not None:
                if not self.persistence.advance(check,ticks=4096):return False
            self._saved_native_clauses=None
            self._native_logging=self.native_persistence and self.persistence.disable_reason is None
            self.solver=self.persistence.create_solver(clauses,logging=self._native_logging)
        else:self.solver=Glucose4(bootstrap_with=clauses)
        self._post_regions(self.solver)
        for cut in self.gold_cuts:self._post_gold(cut)
        self.encoder.close();self.stats['session_model_builds']+=1
        return True

    def _restore_theory(self,check):
        if self._external is not None and self._theory_saved is not None:
            self._external.replay_check=check
            # Rechecking after an interrupted restore is charged. No unverified
            # clause enters the native model, and the full pending record stays.
            clauses=self._external.semantic_restore(self._theory_saved)
            for clause in clauses:self.solver.add_clause(list(clause))
            self._theory_saved=None

    def _theory_snapshot(self):
        live=self._external.semantic_snapshot() if self._external is not None else None
        if live is None:return copy.deepcopy(self._theory_saved)
        if self._theory_saved is not None:
            # Save checked live clauses plus pending input clauses. Restart
            # independently verifies both; neither serialization grants trust.
            rows=list(dict.fromkeys(tuple(row) for row in [*self._theory_saved['clauses'],*live['clauses']]))
            live['clauses']=[list(row) for row in rows[-256:]]
        return live

    def _sync_theory(self):
        if self._external is not None:
            for key,value in {**self._external.stats,**self._external.theory_stats}.items():self.stats[key]=self._theory_prior[key]+value

    def _retire_theory(self):
        if self._external is not None:
            self._sync_theory();self._theory_saved=self._theory_snapshot()
            self._theory_prior={key:self.stats[key] for key in self._theory_prior};self._external=None

    def configure_theory(self,enabled):
        if type(enabled)is not bool or enabled and self.native_persistence:raise ValueError('Persistent theory cannot use Glucose native proof logging.')
        if enabled==self.theory:return
        if self.solver is not None:
            self._prior_decisions+=int(self.solver.accum_stats().get('decisions',0))
            self._retire_theory();self.solver.delete();self.solver=None
            self.encoder=SatAssist(self.problem,self.scope.root,'edges',capture_clauses=True)
        # Original native auxiliary lemmas are optional and are not imported
        # into the different callback backend. Exact cube ownership is retained.
        if enabled:self.persistence=None;self._saved_native_clauses=None
        self.theory=enabled;self.selector=None;self._guard_domains={};self._reset_guards();self._native_logging=False

    def _search_formula(self,check=lambda:None):
        if not self.simplification:return self._base_clauses
        from constraint_sat_simplify_round26 import SimplifiedCNF
        protected=frozenset(lit for row in self.variables for lit in row.values())
        pin_eliminated=not(self.native_persistence or self.persistence is not None or self._saved_native_clauses is not None)
        key=(self._base_clauses,protected,pin_eliminated)
        if self._simplification_key==key:
            self.stats['simplify_cache_hits']+=1;return self._simplified.clauses
        result=SimplifiedCNF(self._base_clauses,protected,check,pin_eliminated=pin_eliminated,stats=self.stats)
        check();self._simplified=result;self._simplification_key=key
        return result.clauses

    def configure_simplification(self,enabled):
        if type(enabled)is not bool:raise ValueError('Invalid SAT simplification policy.')
        if enabled==self.simplification:return
        if self.solver is not None:
            self._prior_decisions+=int(self.solver.accum_stats().get('decisions',0))
            self._retire_theory();self.solver.delete();self.solver=None
            self.encoder=SatAssist(self.problem,self.scope.root,'edges',capture_clauses=True)
        self.simplification=enabled;self.selector=None;self._guard_domains={};self._reset_guards();self._native_logging=False

    def _prepare_regions(self,check):
        if not self.regional_compiler or self._region_ready:return
        from types import SimpleNamespace
        from constraint_region_models import add_regional_dags
        clauses=[];next_id=max((abs(x) for row in self._base_clauses for x in row),default=0)
        def fresh():
            nonlocal next_id
            next_id+=1;return next_id
        domains=[list(row) for row in self.variables]
        counters=add_regional_dags(self.problem,domains,backend='sat',
            solver=SimpleNamespace(add_clause=lambda row:clauses.append(tuple(row))),
            literal=lambda cell,code:self.variables[cell][code],new_variable=fresh,
            target=self.scope.target,check=check,stats_sink=self.stats,cache_dir=self.regional_cache_dir)
        check()
        self._region_clauses=tuple(clauses);self._region_next=next_id;self._region_ready=True
        # stats_sink already charges completed and interrupted compilation.

    def _post_regions(self,solver):
        for record in self._imported_masks:self._post_checked_mask(solver,record)
        if self.regional_compiler:
            for row in self._region_clauses:solver.add_clause(list(row))
            self.next_id=max(self.next_id,self._region_next)

    def import_checked(self,state,check=lambda:None):
        from constraint_knowledge import KnowledgeStore,revalidate_witness
        from constraint_shared_conflicts import VerifiedConflictStore
        from constraint_mask_conflicts import VerifiedMaskConflictStore
        knowledge=state.get('_knowledge')
        if knowledge is not None:
            if not isinstance(knowledge,KnowledgeStore) or not knowledge.matches(self.problem,self.scope.target):
                raise ValueError('SAT exchange Gold source differs.')
            for row in knowledge.records():
                check();record=revalidate_witness(self.problem,row,target=self.scope.target)
                if record not in self.gold_cuts:
                    self.gold_cuts.append(record);self.stats['session_imported_loops']+=1
                    if self.solver is not None:self._post_gold(record)
        for name,kind in (('_shared_conflicts',VerifiedConflictStore),('_mask_conflicts',VerifiedMaskConflictStore)):
            owner=state.get(name)
            if owner is None:continue
            if not isinstance(owner,kind) or not owner.matches(self.problem,self.scope.target):raise ValueError('SAT exchange conflict root differs.')
            owner.advance(check,seconds=.005,attempts=1)
            for row in owner.records():
                record=tuple((cell,1<<code) for cell,code in row) if name=='_shared_conflicts' else tuple(row)
                if record in self._imported_masks:continue
                check();self._imported_masks.add(record);self.stats['session_imported_masks']+=1
                if self.solver is not None:self._post_checked_mask(self.solver,record)
                # Bound optional reuse only; clauses already in a native model
                # remain valid, and discarded records can be checked again.
                if len(self._imported_masks)>256:self._imported_masks.remove(min(self._imported_masks))

    def _post_checked_mask(self,solver,record):
        # Negating a conjunction of domain memberships is an OR of placements
        # outside those memberships. Placement exactly-one axioms are retained.
        if any(not self.scope.root[cell]&mask for cell,mask in record):return
        solver.add_clause([lit for cell,mask in record for code,lit in self.variables[cell].items() if not mask>>code&1])

    def configure_regions(self,enabled):
        if type(enabled)is not bool:raise ValueError('Invalid regional SAT policy.')
        if enabled==self.regional_compiler:return
        if self.solver is not None:
            self._prior_decisions+=int(self.solver.accum_stats().get('decisions',0))
            self._retire_theory();self.solver.delete();self.solver=None
            self.encoder=SatAssist(self.problem,self.scope.root,'edges',capture_clauses=True)
        self.regional_compiler=enabled;self.selector=None;self._guard_domains={};self._reset_guards()
        self._native_logging=False
        # Verified clause admission and core verification keep the original
        # weaker edge base. Regional auxiliary clauses are regenerated locally,
        # never imported as proof authority or used to lift an unverified cut.

    def _sync_persistence(self):
        if self.persistence is None:return
        from constraint_sat_persistence import GAUGES
        for key,value in self.persistence.stats.items():
            self.stats[key]=(max(self._persist_prior[key],value) if key=='session_persist_log_peak_bytes'
                             else value if key in GAUGES else self._persist_prior[key]+value)

    def _rebuild_without_logging(self):
        if not self._native_logging:return
        replacement=self.persistence.create_solver(self._search_formula(),logging=False)
        try:
            for record in self.gold_cuts:
                if all(code in self.variables[cell] for cell,code in record):
                    replacement.add_clause([-self.variables[cell][code] for cell,code in record])
        except BaseException:replacement.delete();raise
        prior=self.solver
        self._prior_decisions+=int(prior.accum_stats().get('decisions',0))
        self.solver=replacement;self._native_logging=False;self.selector=None;self._guard_domains={};self._reset_guards()
        self.next_id=max((abs(lit) for row in self._base_clauses for lit in row),default=0)
        self.persistence.stats['session_persist_model_rebuilds']+=1;self.stats['session_model_builds']+=1
        self._post_regions(self.solver)
        prior.delete()

    def _maintain_persistence(self,check):
        if self.persistence is None:return
        if self._native_logging:self.persistence.harvest(self.solver,check)
        self.persistence.advance(check,ticks=4096)
        if self._native_logging and self.persistence.disable_reason is not None:self._rebuild_without_logging()

    def configure_persistence(self,enabled):
        if type(enabled)is not bool or enabled and self.theory:raise ValueError('Native proof logging cannot be combined with persistent theory.')
        if enabled:
            from constraint_sat_persistence import require_capability
            require_capability()
        if enabled==self.native_persistence:return
        if not enabled:self._rebuild_without_logging()
        elif self.solver is not None:
            # An explicit policy change rebuilds a fresh native session while
            # preserving the exact pending cubes and checked clause subset.
            from constraint_sat_persistence import VerifiedNativeClauses
            if self.persistence is None:self.persistence=VerifiedNativeClauses(self._base_clauses,self.variables,
                fingerprint=self.problem.fingerprint,target=self.scope.target,initial=self.scope.root)
            if self.persistence.disable_reason is not None:
                self.native_persistence=True;return
            replacement=self.persistence.create_solver(self._search_formula(),logging=True)
            try:
                for record in self.gold_cuts:
                    if all(code in self.variables[cell] for cell,code in record):replacement.add_clause([-self.variables[cell][code] for cell,code in record])
            except BaseException:replacement.delete();raise
            prior=self.solver;self._prior_decisions+=int(prior.accum_stats().get('decisions',0))
            self.solver=replacement;self._native_logging=True;self.persistence.offset=0;self.persistence.buffer=b'';self.persistence.discard_line=False
            self.selector=None;self._guard_domains={};self._reset_guards();self.next_id=max((abs(lit) for row in self._base_clauses for lit in row),default=0)
            self._post_regions(self.solver)
            self.stats['session_model_builds']+=1;prior.delete()
        self.native_persistence=enabled

    def _reset_guards(self):
        if self._guards is not None:self.stats['session_guard_pool_resets']+=1
        self._guards=None

    def configure_guards(self,enabled):
        if type(enabled)is not bool:raise ValueError('Invalid SAT guard reuse policy.')
        if enabled!=self.guard_reuse:
            self._deactivate();self.guard_reuse=enabled

    def _fresh_guard(self):
        self.next_id+=1;return self.next_id

    def _post_gold(self,record):
        if all(code in self.variables[cell] for cell,code in record):
            clause=[-self.variables[cell][code] for cell,code in record]
            self.solver.add_clause(clause)
            if self._core_verifier is not None:self._core_verifier.add_clause(clause)

    def _assumptions(self):
        return list(self.selector) if isinstance(self.selector,tuple) else [self.selector]

    def _deactivate(self):
        if self.selector is not None and not self.guard_reuse:
            for selector in self._assumptions():
                self.solver.add_clause([-selector])
                self.stats['session_guard_clauses']+=1;self.stats['session_guard_retirements']+=1
        self.selector=None;self._guard_domains={}

    def _activate(self):
        if self.selector is not None:return
        if self.guard_reuse:
            if self._guards is None:
                from constraint_sat_guards_round26 import DomainGuardPool
                self._guards=DomainGuardPool(self.scope.root,self.variables,self._fresh_guard,self.solver.add_clause,limit=self.guard_limit,stats=self.stats)
            self.selector,self._guard_domains=self._guards.assumptions(self.boxes[0])
            return
        if self.core_lifting:
            guards=[];self._guard_domains={}
            for cell,domain in enumerate(self.boxes[0]):
                if domain==self.scope.root[cell]:continue
                self.next_id+=1;selector=self.next_id;guards.append(selector)
                self._guard_domains[selector]=(cell,domain)
                self.solver.add_clause([-selector,*[literal for code,literal in self.variables[cell].items() if domain>>code&1]])
                self.stats['session_guard_clauses']+=1
            self.selector=tuple(guards)
        else:
            self.next_id+=1;self.selector=self.next_id
            for cell,domain in enumerate(self.boxes[0]):
                self.solver.add_clause([-self.selector,*[literal for code,literal in self.variables[cell].items() if domain>>code&1]])
                self.stats['session_guard_clauses']+=1

    def _lift_candidate(self):
        if not self.core_lifting:return None
        raw=self.solver.get_core() or []
        if any(type(lit)is not int or lit not in self._guard_domains for lit in raw):return None
        candidate=list(self.scope.root)
        for lit in raw:
            cell,mask=self._guard_domains[lit];candidate[cell]&=mask
        candidate=tuple(candidate)
        if candidate==self.boxes[0]:return None
        self.stats['session_core_candidates']+=1
        return candidate

    def _verify_lift(self,candidate,check,deadline):
        """Independent root-formula solve: no search selectors or native core trust."""
        if not contains(self.scope.root,candidate):raise ValueError('Lifted core escaped owned root.')
        check()
        if self._core_verifier is None:
            from pysat.solvers import Glucose4
            self._core_verifier=Glucose4(bootstrap_with=self._base_clauses)
            self.stats['session_core_model_builds']+=1
            for cut in self.gold_cuts:
                if all(code in self.variables[cell] for cell,code in cut):
                    self._core_verifier.add_clause([-self.variables[cell][code] for cell,code in cut])
        assumptions=[-literal for cell,row in enumerate(self.variables) for code,literal in row.items() if not candidate[cell]>>code&1]
        self.stats['session_core_verifications']+=1
        try:answer=self._native_call(self._core_verifier,assumptions,check,deadline)
        finally:self.stats['session_core_decisions']=self._prior_core_decisions+int(self._core_verifier.accum_stats().get('decisions',0))
        if answer is None:self.stats['session_core_unknown']+=1
        return answer is False if answer is not None else None

    def _subtract_lift(self,candidate,check):
        old=tuple(self.boxes)
        pending,complete=subtract_many(old,[candidate],check=check)
        if complete:
            self.boxes=list(pending)
            self.stats['session_core_boxes_removed']+=sum(contains(candidate,row) for row in old)
            self.stats['session_core_assignments_removed']+=sum(math.prod(mask.bit_count() for mask in row) for row in old)-sum(math.prod(mask.bit_count() for mask in row) for row in pending)
        if self.coverage is not None:
            self.coverage.record_completed(candidate,status='infeasible',complete=True,source_scope=self.scope,evidence={'kind':'independently_verified_sat_domain_core'})

    def _native_call(self,solver,assumptions,check,deadline):
        if solver is self.solver and self.theory:
            self._external.assert_semantic_source()
            self._external.replay_check=check
            if isinstance(self._external.error,SessionInterrupted):self._external.error=None
            solver.conf_budget(256);solver.dec_budget(256)
            answer=solver.solve_limited(assumptions=assumptions)
            self._sync_theory()
            if self._external.error is not None:raise self._external.error
            return answer
        logged=solver is self.solver and self._native_logging
        solver.clear_interrupt();solver.conf_budget(256 if logged else 2000);finished=threading.Event();errors=[]
        def monitor():
            while not finished.wait(.01):
                try:
                    check()
                    if logged and self.persistence.monitor_log(solver):solver.interrupt();return
                except BaseException as exc:errors.append(exc);solver.interrupt();return
                if time.monotonic()>=deadline:solver.interrupt();return
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:answer=solver.solve_limited(assumptions=assumptions,expect_interrupt=True)
        finally:finished.set();thread.join()
        # A completed UNSAT answer is valid even when Stop arrived just before
        # the native call returned. Preserve that proof; an interrupted/unknown
        # answer still raises, and unexpected callback errors are never hidden.
        if errors and (answer is not False or not isinstance(errors[0],SessionInterrupted)):raise errors[0]
        return answer

    def _solve_once(self,check,deadline):
        self.stats['session_native_calls']+=1
        try:return self._native_call(self.solver,self._assumptions(),check,deadline)
        finally:self.stats['session_decisions']=self._prior_decisions+int(self.solver.accum_stats().get('decisions',0))

    def advance(self,*,seconds=1,stop=None,progress=None):
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid SAT session budget.')
        started=time.monotonic();deadline=started+seconds;status='timeout';validation=None;last_progress=started
        requested=lambda:bool(stop() if callable(stop) else stop.is_set() if stop is not None else False)
        def check():
            nonlocal last_progress
            if progress and time.monotonic()-last_progress>=1:
                last_progress=time.monotonic();progress(self.result('searching',started))
            if requested():raise SessionInterrupted('stopped')
            if time.monotonic()>=deadline:raise SessionInterrupted('timeout')
        self.stats['session_warm_slices']+=int(self._started);self._started=True
        try:
            check()
            if self.coverage is not None:
                pending,complete=self.coverage.remaining(self.boxes,target_scope=self.scope,check=check)
                if complete and tuple(self.boxes)!=pending:
                    self._deactivate()
                    self.boxes=list(pending)
            if not self.boxes:return self.result('infeasible',started)
            if not self.prepare(check,max(0,deadline-time.monotonic())):raise SessionInterrupted('timeout')
            while self.boxes:
                check();self._maintain_persistence(check);self._activate();answer=self._solve_once(check,deadline)
                # Commit a completed native proof before polling cancellation.
                if answer is False:
                    candidate=self._lift_candidate()
                    closed=self.boxes.pop(0);self.closed.append(closed);self.stats['session_cubes_closed']+=1
                    if self.coverage is not None:self.coverage.record_completed(closed,status='infeasible',complete=True,source_scope=self.scope)
                    self._deactivate()
                    if candidate is not None and self.boxes and self._verify_lift(candidate,check,deadline) is True:
                        self.stats['session_core_verified']+=1
                        self._subtract_lift(candidate,check)
                    if progress:progress(self.result('searching',started))
                    continue
                if answer is None:continue
                model=self.solver.get_model()
                if self.simplification:model=self._simplified.reconstruct(model,check)
                selected={lit for lit in model if lit>0}
                codes=[next(code for code,lit in row.items() if lit in selected) for row in self.variables]
                if not all(mask>>code&1 for mask,code in zip(self.boxes[0],codes)):raise RuntimeError('Native SAT result escapes owned cube.')
                validation=self.problem.validate(codes);self.stats['session_candidates']+=1;self.last_codes=codes
                if not validation.get('unique_tiles') or validation.get('matched_edges')!=len(self.problem.edges):raise RuntimeError('Native SAT result failed seam/inventory validation.')
                if validation.get('valid') or self.scope.target=='edges':
                    status='solved' if validation.get('valid') else 'edge_perfect';break
                record=tuple(tuple(x) for x in self.problem.gold_nogood(codes))
                from constraint_knowledge import revalidate_witness
                checked=revalidate_witness(self.problem,record,target='gold')
                self.gold_cuts.append(checked);self._post_gold(checked);self.stats['session_gold_cuts']+=1
            else:status='infeasible'
        except SessionInterrupted as exc:
            status=str(exc)
            # Retain a bounded tail as UNVERIFIED claims. Stop does not run an
            # independent proof here; a resumed session must check every claim.
            if self._native_logging:self.persistence.harvest(self.solver)
        return self.result(status,started,validation)

    def result(self,status,started,validation=None):
        self._sync_persistence();self._sync_theory()
        stats=dict(self.stats,session_total_decisions=self.stats['session_decisions']+self.stats['session_core_decisions'],engine='sat-session',node_metric='SAT decisions including core verification',resumption_mode='warm_native_session_and_durable_disjoint_cubes')
        if self.coverage is not None:stats.update(self.coverage.stats)
        persistence_mode=('verifying_import' if self._saved_native_clauses is not None else
                          'logging' if self._native_logging else
                          'continuing_without_logger' if self.native_persistence and self.persistence is not None and self.persistence.disable_reason else
                          'preparing' if self.native_persistence else 'off')
        persistence_info={'enabled':self.native_persistence,'mode':persistence_mode,
                          'disable_reason':self.persistence.disable_reason if self.persistence is not None else None,
                          'proof_authority':'scoped_edge_cnf_only','import_variables':'exact_original_cnf_including_auxiliaries',
                          'full_native_state_saved':False}
        return {'status':status,'complete':status in ('solved','edge_perfect','infeasible'),
                'codes':self.last_codes if status in ('solved','edge_perfect') else None,'validation':validation,
                'nodes':stats['session_total_decisions'],'stats':stats,'elapsed_seconds':time.monotonic()-started,
                'checkpoint':self.snapshot(),'native_persistence':persistence_info}

    def snapshot(self):
        self._sync_persistence();self._sync_theory()
        out={'schema':'diamond-sat-session-v1','frontier':frontier_snapshot(self.scope,self.boxes),
             'gold_cuts':[[list(x) for x in row] for row in self.gold_cuts],'stats':dict(self.stats),'core_lifting':self.core_lifting,'guard_reuse':self.guard_reuse,'guard_limit':self.guard_limit,'regional_compiler':self.regional_compiler,'simplification':self.simplification,'theory':self.theory,
             'theory_clauses':self._theory_snapshot()}
        if self._saved_native_clauses is not None:out['native_clauses']=copy.deepcopy(self._saved_native_clauses)
        elif self.persistence is not None:out['native_clauses']=self.persistence.snapshot()
        out['digest']=digest(out);return out

    @classmethod
    def restore(cls,problem,payload,target='gold',*,native_persistence=False):
        if not isinstance(payload,dict) or payload.get('schema')!='diamond-sat-session-v1' or digest({k:v for k,v in payload.items() if k!='digest'})!=payload.get('digest'):
            raise ValueError('Malformed native-session checkpoint.')
        if target=='edges' and payload.get('gold_cuts'):raise ValueError('Gold-only cuts cannot enter an edge-target session.')
        expected=WorkScope.for_problem(problem,target);rows=restore_frontier(payload['frontier'],expected)
        scope=WorkScope.restore(payload['frontier']['scope'])
        policy=payload.get('core_lifting',False)
        if type(policy)is not bool:raise ValueError('Malformed native-session lifting policy.')
        obj=cls(problem,scope,rows,core_lifting=policy,native_persistence=native_persistence,guard_reuse=payload.get('guard_reuse',False),guard_limit=payload.get('guard_limit',8192),regional_compiler=payload.get('regional_compiler',False),simplification=payload.get('simplification',False),theory=payload.get('theory',False))
        try:
            from constraint_knowledge import revalidate_witness
            from constraint_sat_persistence import STAT_KEYS as PERSIST_KEYS,MAX_SNAPSHOT_BYTES
            for record in payload['gold_cuts']:obj.gold_cuts.append(revalidate_witness(problem,record,target='gold'))
            if any(k not in obj.stats or type(v)is not int or v<0 for k,v in payload['stats'].items()):raise ValueError('Malformed native-session counters.')
            saved=payload.get('native_clauses')
            if saved is not None:
                if not isinstance(saved,dict) or len(json.dumps(saved,separators=(',',':')).encode())>MAX_SNAPSHOT_BYTES:raise ValueError('Malformed or oversized saved native clauses.')
                obj._saved_native_clauses=copy.deepcopy(saved)
            obj._theory_saved=copy.deepcopy(payload.get('theory_clauses'));obj.stats.update(payload['stats']);obj._theory_prior={key:obj.stats[key] for key in obj._theory_prior};obj._prior_decisions=obj.stats['session_decisions'];obj._prior_core_decisions=obj.stats['session_core_decisions']
            obj._persist_prior={key:obj.stats[key] for key in PERSIST_KEYS}
            return obj
        except BaseException:obj.close();raise

    def close(self):
        if self.coverage is not None:self.coverage.close();self.coverage=None
        self.encoder.close()
        if self.solver is not None:self.solver.delete();self.solver=None
        if self._core_verifier is not None:self._core_verifier.delete();self._core_verifier=None


def solve(problem,*,seconds=10,seed=0,target='gold',hints=None,stop=None,progress=None,resume=None,learning_state=None,options=None):
    if target not in ('gold','edges'):raise ValueError('Invalid SAT target.')
    if any(not row for row in problem.domains):
        return {'status':'infeasible','complete':True,'codes':None,'nodes':0,'stats':{},'elapsed_seconds':0}
    enabled=bool(options and options.get('native_sat_persistence'))
    if enabled:
        from constraint_sat_persistence import require_capability
        require_capability()
    state={} if learning_state is None else learning_state
    if not isinstance(state,dict):raise ValueError('SAT learning state must be a trusted in-memory dict.')
    session=state.get('_sat_session')
    if session is None:
        if resume is not None and resume.get('work_version')==1:
            scope=WorkScope.for_problem(problem,target);session=PersistentSatSession(problem,scope,restore_frontier(resume,scope),native_persistence=enabled)
        elif resume is not None:session=PersistentSatSession.restore(problem,resume,target,native_persistence=enabled)
        else:
            scope=WorkScope.for_problem(problem,target);portable=state.get('_portable_frontier')
            rows=restore_frontier(portable,scope) if portable is not None else None
            session=PersistentSatSession(problem,scope,rows,native_persistence=enabled)
        state['_sat_session']=session
    elif not isinstance(session,PersistentSatSession) or not WorkScope.for_problem(problem,target).accepts(session.scope) or not session._binding():raise ValueError('Warm SAT session scope differs.')
    elif resume is not None and digest(session.snapshot())!=digest(resume):
        if isinstance(resume,dict) and resume.get('work_version')==1:
            scope=WorkScope.for_problem(problem,target)
            replacement=PersistentSatSession(problem,scope,restore_frontier(resume,scope),native_persistence=enabled)
        else:replacement=PersistentSatSession.restore(problem,resume,target,native_persistence=enabled)
        session.close();session=replacement;state['_sat_session']=session
    session.regional_cache_dir=state.get('_regional_cache_dir')
    session.configure_regions(bool(options and options.get('regional_compiler')))
    session.configure_simplification(bool(options and options.get('sat_simplification')))
    theory=bool(options and options.get('persistent_theory'))
    if not theory:session.configure_theory(False)
    session.configure_persistence(enabled)
    session.configure_theory(theory)
    session.configure_guards(bool(options and options.get('sat_guard_reuse')))
    session.import_checked(state)
    core_lifting=bool(options and options.get('sat_core_lifting'))
    if session.core_lifting!=core_lifting:session._deactivate();session.core_lifting=core_lifting
    if options and options.get('shared_coverage') and state.get('_coverage_path') and session.coverage is None:
        session.coverage=CoverageLedger(state['_coverage_path'],session.scope,symbolic=options.get('symbolic_coverage',False))
    if session.coverage is not None:session.coverage.symbolic=bool(options and options.get('symbolic_coverage'))
    try:
        callback=(lambda event:progress({'phase':'searching',**event})) if progress else None
        return session.advance(seconds=seconds,stop=stop,progress=callback)
    finally:
        if learning_state is None:session.close()
