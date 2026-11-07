"""Persistent plain-SAT session over an exact portable cube frontier.

Native clauses/heuristics survive warm slices. After process restart only the
remaining disjoint cubes and independently rechecked Gold cuts survive; an
interrupted cube may repeat native internal work. Completed cubes never return.
"""
from __future__ import annotations
import json,math,threading,time
from constraint_sat_assist import SatAssist
from constraint_work import WorkScope,cube,contains,frontier_snapshot,restore_frontier,digest,CoverageLedger


class SessionInterrupted(Exception):pass


class PersistentSatSession:
    def __init__(self,problem,scope,boxes=None):
        if not isinstance(scope,WorkScope) or not WorkScope.for_problem(problem,scope.target).accepts(scope):
            raise ValueError('SAT session puzzle/objective/domains differ.')
        self.problem,self.scope=problem,scope
        self.boxes=list((scope.root,) if boxes is None else boxes)
        frontier_snapshot(scope,self.boxes)
        self.encoder=SatAssist(problem,scope.root,'edges',capture_clauses=True)
        self.solver=None;self.variables=None;self.next_id=0;self.selector=None
        self.gold_cuts=[];self.closed=[];self.last_codes=None
        self.stats={'session_model_builds':0,'session_warm_slices':0,'session_native_calls':0,'session_cubes_closed':0,'session_gold_cuts':0,'session_decisions':0,'session_candidates':0}
        self._started=False;self._prior_decisions=0;self.coverage=None

    def _binding(self):
        return WorkScope.for_problem(self.problem,self.scope.target).accepts(self.scope)

    def prepare(self,check,seconds):
        if not self._binding():raise ValueError('SAT session source domains changed.')
        if self.solver is not None:return True
        if not self.encoder.prepare(check,seconds=seconds):
            if self.encoder.status in ('unavailable','unsupported'):raise RuntimeError('Persistent SAT unavailable: '+str(self.encoder.reason))
            return False
        from pysat.solvers import Glucose4
        model=self.encoder.proof_base();self.variables=tuple(dict(row) for row in model['variables'])
        self.next_id=max((abs(x) for row in model['clauses'] for x in row),default=0)
        self.solver=Glucose4(bootstrap_with=model['clauses'])
        for cut in self.gold_cuts:self._post_gold(cut)
        self.encoder.close();self.stats['session_model_builds']+=1
        return True

    def _post_gold(self,record):
        if all(code in self.variables[cell] for cell,code in record):
            self.solver.add_clause([-self.variables[cell][code] for cell,code in record])

    def _activate(self):
        if self.selector is not None:return
        self.next_id+=1;self.selector=self.next_id
        for cell,domain in enumerate(self.boxes[0]):
            self.solver.add_clause([-self.selector,*[literal for code,literal in self.variables[cell].items() if domain>>code&1]])

    def _solve_once(self,check,deadline):
        self.solver.clear_interrupt();self.solver.conf_budget(2000);finished=threading.Event();errors=[]
        def monitor():
            while not finished.wait(.01):
                try:check()
                except BaseException as exc:errors.append(exc);self.solver.interrupt();return
                if time.monotonic()>=deadline:self.solver.interrupt();return
        thread=threading.Thread(target=monitor,daemon=True);thread.start()
        try:answer=self.solver.solve_limited(assumptions=[self.selector],expect_interrupt=True)
        finally:finished.set();thread.join()
        # A completed UNSAT answer is valid even when Stop arrived just before
        # the native call returned. Preserve that proof; an interrupted/unknown
        # answer still raises, and unexpected callback errors are never hidden.
        if errors and (answer is not False or not isinstance(errors[0],SessionInterrupted)):raise errors[0]
        self.stats['session_native_calls']+=1
        self.stats['session_decisions']=self._prior_decisions+int(self.solver.accum_stats().get('decisions',0))
        return answer

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
                    if self.selector is not None:self.solver.add_clause([-self.selector]);self.selector=None
                    self.boxes=list(pending)
            if not self.boxes:return self.result('infeasible',started)
            if not self.prepare(check,max(0,deadline-time.monotonic())):raise SessionInterrupted('timeout')
            while self.boxes:
                check();self._activate();answer=self._solve_once(check,deadline)
                # Commit a completed native proof before polling cancellation.
                if answer is False:
                    closed=self.boxes.pop(0);self.closed.append(closed);self.stats['session_cubes_closed']+=1
                    if self.coverage is not None:self.coverage.record_completed(closed,status='infeasible',complete=True,source_scope=self.scope)
                    self.solver.add_clause([-self.selector]);self.selector=None
                    if progress:progress(self.result('searching',started))
                    continue
                if answer is None:continue
                selected={lit for lit in self.solver.get_model() if lit>0}
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
        except SessionInterrupted as exc:status=str(exc)
        return self.result(status,started,validation)

    def result(self,status,started,validation=None):
        stats=dict(self.stats,engine='sat-session',node_metric='SAT decisions',resumption_mode='warm_native_session_and_durable_disjoint_cubes')
        if self.coverage is not None:stats.update(self.coverage.stats)
        return {'status':status,'complete':status in ('solved','edge_perfect','infeasible'),
                'codes':self.last_codes if status in ('solved','edge_perfect') else None,'validation':validation,
                'nodes':self.stats['session_decisions'],'stats':stats,'elapsed_seconds':time.monotonic()-started,
                'checkpoint':self.snapshot()}

    def snapshot(self):
        out={'schema':'diamond-sat-session-v1','frontier':frontier_snapshot(self.scope,self.boxes),
             'gold_cuts':[[list(x) for x in row] for row in self.gold_cuts],'stats':dict(self.stats)}
        out['digest']=digest(out);return out

    @classmethod
    def restore(cls,problem,payload,target='gold'):
        if not isinstance(payload,dict) or payload.get('schema')!='diamond-sat-session-v1' or digest({k:v for k,v in payload.items() if k!='digest'})!=payload.get('digest'):
            raise ValueError('Malformed native-session checkpoint.')
        if target=='edges' and payload.get('gold_cuts'):raise ValueError('Gold-only cuts cannot enter an edge-target session.')
        expected=WorkScope.for_problem(problem,target);rows=restore_frontier(payload['frontier'],expected)
        scope=WorkScope.restore(payload['frontier']['scope']);obj=cls(problem,scope,rows)
        from constraint_knowledge import revalidate_witness
        for record in payload['gold_cuts']:obj.gold_cuts.append(revalidate_witness(problem,record,target='gold'))
        if any(k not in obj.stats or type(v)is not int or v<0 for k,v in payload['stats'].items()):raise ValueError('Malformed native-session counters.')
        obj.stats.update(payload['stats']);obj._prior_decisions=obj.stats['session_decisions']
        return obj

    def close(self):
        if self.coverage is not None:self.coverage.close();self.coverage=None
        self.encoder.close()
        if self.solver is not None:self.solver.delete();self.solver=None


def solve(problem,*,seconds=10,seed=0,target='gold',hints=None,stop=None,progress=None,resume=None,learning_state=None,options=None):
    if target not in ('gold','edges'):raise ValueError('Invalid SAT target.')
    if any(not row for row in problem.domains):
        return {'status':'infeasible','complete':True,'codes':None,'nodes':0,'stats':{},'elapsed_seconds':0}
    state={} if learning_state is None else learning_state
    if not isinstance(state,dict):raise ValueError('SAT learning state must be a trusted in-memory dict.')
    session=state.get('_sat_session')
    if session is None:
        if resume is not None and resume.get('work_version')==1:
            scope=WorkScope.for_problem(problem,target);session=PersistentSatSession(problem,scope,restore_frontier(resume,scope))
        elif resume is not None:session=PersistentSatSession.restore(problem,resume,target)
        else:
            scope=WorkScope.for_problem(problem,target);portable=state.get('_portable_frontier')
            rows=restore_frontier(portable,scope) if portable is not None else None
            session=PersistentSatSession(problem,scope,rows)
        state['_sat_session']=session
    elif not isinstance(session,PersistentSatSession) or not WorkScope.for_problem(problem,target).accepts(session.scope) or not session._binding():raise ValueError('Warm SAT session scope differs.')
    elif resume is not None and digest(session.snapshot())!=digest(resume):
        if isinstance(resume,dict) and resume.get('work_version')==1:
            scope=WorkScope.for_problem(problem,target)
            replacement=PersistentSatSession(problem,scope,restore_frontier(resume,scope))
        else:replacement=PersistentSatSession.restore(problem,resume,target)
        session.close();session=replacement;state['_sat_session']=session
    if options and options.get('shared_coverage') and state.get('_coverage_path') and session.coverage is None:
        session.coverage=CoverageLedger(state['_coverage_path'],session.scope)
    try:
        callback=(lambda event:progress({'phase':'searching',**event})) if progress else None
        return session.advance(seconds=seconds,stop=stop,progress=callback)
    finally:
        if learning_state is None:session.close()
