"""Bounded retained repair models; local query state is never global coverage.

CP-SAT model construction, guarded queries and completed semantic results are
reused. OR-Tools native search is restarted on every solve; no learned native
state persistence is claimed. Inactive query guards cannot restrict later work.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib,json,math,time
from pathlib import Path
from constraint_work import digest
from constraint_lns import score,_watch
from validator import reverse_mask

MAX_SNAPSHOT_BYTES=32*1024*1024
STAT_KEYS=('retained_model_caps','retained_model_build_attempts','retained_model_build_interruptions','retained_abandoned_variables','retained_abandoned_constraints','retained_model_builds','retained_model_reuses','retained_model_evictions',
 'retained_guard_queries','retained_selector_builds','retained_native_calls','retained_native_retries','retained_native_exceptions',
 'retained_native_branches','retained_native_conflicts','retained_unknown','retained_completed_hits',
 'retained_result_evictions','retained_bound_reuses','retained_upper_bound_reuses','retained_incumbent_hints','retained_incumbent_invalidations','retained_bound_hits','retained_gold_cuts','retained_cut_replays',
 'retained_restores','retained_rebuilt_variables','retained_rebuilt_constraints','retained_model_text_peak',
 'retained_policy_observations','retained_policy_selections','retained_policy_evictions','retained_policy_plans')


def _integer(value,low,high):return type(value)is int and low<=value<=high


class _ModelBudget(ValueError):pass


class RetainedRepairService:
    def __init__(self,problem,*,guard_limit=64,cache_limit=128,cut_limit=256,model_text_limit=64*1024*1024,resume=None):
        if not _integer(guard_limit,1,512) or not _integer(cache_limit,1,1024) or not _integer(cut_limit,1,1024) or not _integer(model_text_limit,1024,256*1024*1024):raise ValueError('Invalid retained repair limits.')
        self.problem=problem;self.policy={'guard_limit':guard_limit,'cache_limit':cache_limit,'cut_limit':cut_limit,'model_text_limit':model_text_limit}
        self.binding=self._identity();self.stats=dict.fromkeys(STAT_KEYS,0);self.results=[];self.pending=[];self.witnesses=[];self.outcomes={};self.turn=0
        self.model=None;self.solver=None;self.variables=[];self.selectors={};self.guards=0;self.guard_literals=[];self.closed=False;self.gold_service=None
        if resume is not None:self._restore(resume)
    def _identity(self):
        p=self.problem;root=Path(__file__).parent
        value={'fingerprint':p.fingerprint,'domains':[list(row) for row in p.domains],'fixed':sorted(p.fixed.items()),'tiles':p.tiles,'edges':p.edges,
                'masks':p.masks.tolist(),'neighbors':p.neighbors.tolist(),'sides':p.sides.tolist(),'board_edges':list(p.board.edges),'segment_counts':p._segment_counts,'segment_ports':p._segment_ports,'policy':self.policy,
                'sources':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('constraint_repair_service_round26.py','constraint_lns.py','constraint_repair_round24.py','constraint_repair_gold_round26.py','search_problem.py','validator.py')}}
        return json.loads(json.dumps(value,sort_keys=True))
    def _check(self):
        if self.closed:raise RuntimeError('Retained repair service is closed.')
        if self.binding!=self._identity():raise ValueError('Retained repair input or source changed.')
    def _drop_model(self):
        if self.model is not None:self.stats['retained_model_evictions']+=1
        self.model=self.solver=None;self.variables=[];self.selectors={};self.guards=0;self.guard_literals=[]
    def _size(self):
        size=len(str(self.model.proto).encode('utf-8'));self.stats['retained_model_text_peak']=max(size,self.stats['retained_model_text_peak']);return size
    def _build(self,cp,control):
        control.check();self.stats['retained_model_build_attempts']+=1;self._building=None
        try:self._assemble(cp,control)
        except BaseException:
            self.stats['retained_model_build_interruptions']+=1
            if self._building is not None:
                self.stats['retained_abandoned_variables']+=len(self._building.proto.variables)
                self.stats['retained_abandoned_constraints']+=len(self._building.proto.constraints)
            self._drop_model();raise
        finally:self._building=None
    def _assemble(self,cp,control):
        control.check();p=self.problem;model=cp.CpModel();self._building=model;variables=[];pieces=[];masks={}
        for cell,row in enumerate(p.domains):
            control.check();allowed=[code for code in row if cell not in p.fixed or code==p.fixed[cell]]
            if not allowed:allowed=[0];model.add_bool_or([])
            value=model.new_int_var_from_domain(cp.Domain.from_values(allowed),f'code_{cell}');variables.append(value)
            tile=model.new_int_var(0,p.n-1,f'tile_{cell}');model.add_allowed_assignments([value,tile],[(code,code//3) for code in allowed]);pieces.append(tile)
            for side in range(3):
                for reverse in (False,True):
                    rows=[(code,reverse_mask(int(p.masks[code,side])) if reverse else int(p.masks[code,side])) for code in allowed]
                    mask=model.new_int_var_from_domain(cp.Domain.from_values(sorted({v for _,v in rows})),f'm_{cell}_{side}_{int(reverse)}')
                    model.add_allowed_assignments([value,mask],rows);masks[cell,side,reverse]=mask
        model.add_all_different(pieces);matches=[]
        for index,(a,sa,b,sb) in enumerate(p.edges):
            control.check();match=model.new_bool_var(f'edge_{index}');left,right=masks[a,sa,False],masks[b,sb,True]
            model.add(left==right).only_enforce_if(match);model.add(left!=right).only_enforce_if(match.Not());matches.append(match)
        model.maximize(sum(matches));self.model=model;self.variables=variables;self.matches=matches;self.selectors={};self.guards=0;self.guard_literals=[]
        self.gold_guard=model.new_bool_var('gold_policy');self.solver=cp.CpSolver();self.solver.parameters.num_search_workers=1
        for witness in self.witnesses:self._cut(witness,record=False)
        self._check()
        if self._size()>self.policy['model_text_limit']:
            self._drop_model();raise _ModelBudget('Retained repair base model exceeds its configured text-size bound.')
        self.stats['retained_model_builds']+=1
        self.stats['retained_rebuilt_variables']+=len(model.proto.variables);self.stats['retained_rebuilt_constraints']+=len(model.proto.constraints)
    def _literal(self,cell,code):
        key=(cell,code)
        if key not in self.selectors:
            literal=self.model.new_bool_var(f'place_{cell}_{code}');self.model.add(self.variables[cell]==code).only_enforce_if(literal);self.model.add(self.variables[cell]!=code).only_enforce_if(literal.Not())
            self.selectors[key]=literal;self.stats['retained_selector_builds']+=1
        return self.selectors[key]
    def _allowed_by_cuts(self,context,candidate):
        if candidate is None:return False
        if context['target']!='gold' or not context['loop_aware']:return True
        return all(not all(candidate[cell]==code for cell,code in self.problem.gold_nogood(witness)) for witness in self.witnesses)
    def _invalidate_cut_incumbents(self):
        retained=[]
        for row in self.results:
            if row['candidate'] is not None and not self._allowed_by_cuts(row['context'],row['candidate']):
                self.stats['retained_incumbent_invalidations']+=1
                if not any(old['context']==row['context'] for old in self.pending):
                    self.pending.append({'context':deepcopy(row['context']),'candidate':None,'heuristic':deepcopy(row['heuristic']),'bound':row['bound'],'calls':0,'retry':False})
            else:retained.append(row)
        self.results=retained
        for row in self.pending:
            if row['candidate'] is not None and not self._allowed_by_cuts(row['context'],row['candidate']):
                row['candidate']=None;self.stats['retained_incumbent_invalidations']+=1
        while len(self.pending)>self.policy['cache_limit']:self.pending.pop(0);self.stats['retained_result_evictions']+=1
    def _cut(self,witness,*,record=True):
        checked=score(self.problem,witness);validation=self.problem.validate(witness)
        if checked['domain_violations'] or checked['matched_edges']!=len(self.problem.edges) or validation['valid']:raise ValueError('Retained Gold witness failed independent validation.')
        if record and witness in self.witnesses:return False
        if record and len(self.witnesses)>=self.policy['cut_limit']:return False
        placements=self.problem.gold_nogood(witness)
        if not placements:raise ValueError('Retained Gold cut has no independently established loop.')
        self.model.add_bool_or([self._literal(cell,code).Not() for cell,code in placements]).only_enforce_if(self.gold_guard)
        if record:
            self.witnesses.append(list(witness));self.stats['retained_gold_cuts']+=1;self._invalidate_cut_incumbents()
        else:self.stats['retained_cut_replays']+=1
        return True
    def _context(self,current,patch,target,loop_aware,minimum_edges):
        score(self.problem,current)
        if not isinstance(patch,(list,tuple,set,frozenset)) or not patch or len(patch)>self.problem.n or any(type(c)is not int or not 0<=c<self.problem.n for c in patch):raise ValueError('Invalid retained repair patch.')
        cells=sorted(patch)
        if len(set(cells))!=len(cells):raise ValueError('Invalid retained repair patch.')
        if target not in ('edges','gold') or type(loop_aware)is not bool or minimum_edges is not None and not _integer(minimum_edges,0,len(self.problem.edges)):raise ValueError('Invalid retained repair objective.')
        return {'board':list(current),'patch':cells,'target':target,'loop_aware':loop_aware,'minimum_edges':minimum_edges}
    def _candidate(self,context,candidate):
        checked=score(self.problem,candidate);cells=set(context['patch'])
        if checked['domain_violations'] or candidate==context['board'] or any(candidate[cell]!=context['board'][cell] for cell in range(self.problem.n) if cell not in cells):raise ValueError('Retained candidate failed independent patch/domain validation.')
        if context['minimum_edges'] is not None and checked['matched_edges']<context['minimum_edges']:raise ValueError('Retained candidate violates the requested bound.')
        return checked
    def _remember(self,row):
        self._check()
        key=digest(row['context']);self.results=[old for old in self.results if digest(old['context'])!=key]+[deepcopy(row)]
        if len(self.results)>self.policy['cache_limit']:self.results.pop(0);self.stats['retained_result_evictions']+=1
    def patch_query(self,current,patch,*,cp_model,control,seconds,seed,target,loop_aware=True,diagnostics=None,resume_state=None,minimum_edges=None):
        self._check();control.check();context=self._context(current,patch,target,loop_aware,minimum_edges);key=digest(context);details={} if diagnostics is None else diagnostics
        if type(seconds)not in (int,float) or not math.isfinite(seconds) or seconds<0 or type(seed)is not int:raise ValueError('Invalid retained query budget/seed.')
        saved=next((row for row in self.results if digest(row['context'])==key),None)
        if saved:
            if saved['candidate'] is not None:self._candidate(context,saved['candidate'])
            self.stats['retained_completed_hits']+=1;return deepcopy(saved['heuristic'] if saved['heuristic'] is not None else saved['candidate']),saved['status']
        # A completed edge objective is reusable across threshold changes with
        # the same exact excluded seed, patch and outside assignments. Gold
        # partial incumbents can predate later cuts, so this shortcut is
        # deliberately restricted to the edge-only/disabled-loop model.
        if target=='edges' or not loop_aware:
            for record in reversed(self.results):
                old=record['context']
                if any(old[name]!=context[name] for name in ('board','patch','target','loop_aware')):continue
                answer=None;status=None
                if record['status']=='OPTIMAL' and record['candidate'] is not None:
                    metrics=self._candidate(old,record['candidate'])
                    if record['bound']!=metrics['matched_edges']:continue
                    if minimum_edges is None or metrics['matched_edges']>=minimum_edges:answer=deepcopy(record['candidate']);status='OPTIMAL'
                    else:status='INFEASIBLE'
                elif record['status']=='INFEASIBLE' and record['candidate'] is None and (old['minimum_edges'] is None or minimum_edges is not None and minimum_edges>=old['minimum_edges']):status='INFEASIBLE'
                if status is not None:
                    self.stats['retained_bound_hits']+=1
                    self._remember({'context':context,'candidate':answer,'heuristic':deepcopy(answer),'status':status,'bound':record['bound']})
                    return answer,status
        deadline=min(control.deadline,time.monotonic()+seconds)
        if self.model is not None and self.guards>=self.policy['guard_limit']:self._drop_model()
        if self.model is None:
            details['model_build_attempted']=1
            try:self._build(cp_model,control)
            except _ModelBudget:
                self.stats['retained_model_caps']+=1;return None,'MODEL_BUDGET'
            details['model_built']=1
        else:self.stats['retained_model_reuses']+=1
        # Every query gets an explicit positive guard. No old guard is assumed
        # true; its optional constraints can always be disabled on later calls.
        guard=self.model.new_bool_var(f'query_{self.guards}');self.guards+=1
        previous_guards=list(self.guard_literals);self.guard_literals.append(guard)
        cells=context['patch'];self.model.add_forbidden_assignments([self.variables[cell] for cell in cells],[tuple(current[cell] for cell in cells)]).only_enforce_if(guard)
        if minimum_edges is not None:self.model.add(sum(self.matches)>=minimum_edges).only_enforce_if(guard)
        assumptions=[guard,self.gold_guard if target=='gold' and loop_aware else self.gold_guard.Not(),*(old.Not() for old in previous_guards)]
        assumptions.extend(self._literal(cell,current[cell]) for cell in range(self.problem.n) if cell not in set(cells))
        self.model.clear_assumptions();self.model.add_assumptions(assumptions);self.model.clear_hints()
        for cell,value in enumerate(self.variables):
            if current[cell] in self.problem.domains[cell]:self.model.add_hint(value,current[cell])
        if resume_state is not None:
            if not isinstance(resume_state,dict) or set(resume_state)!={'version','gold_cuts'} or type(resume_state['version'])is not int or resume_state['version']!=1 or not isinstance(resume_state['gold_cuts'],list) or len(resume_state['gold_cuts'])>512:raise ValueError('Invalid retained pending Gold witnesses.')
            for witness in resume_state['gold_cuts']:
                if target!='gold' or not loop_aware:raise ValueError('Gold witnesses conflict with query policy.')
                if witness not in self.witnesses:self._cut(witness)
        if target=='gold' and loop_aware:
            initial=score(self.problem,current)
            if not initial['domain_violations'] and initial['matched_edges']==len(self.problem.edges) and not self.problem.validate(current)['valid'] and self._cut(current):
                details['loop_cuts']=details.get('loop_cuts',0)+1
                if resume_state is not None and len(resume_state['gold_cuts'])<512:resume_state['gold_cuts'].append(list(current))
        self.stats['retained_guard_queries']+=1
        if self._size()>self.policy['model_text_limit']:
            self.stats['retained_model_caps']+=1;self._drop_model();return None,'MODEL_BUDGET'
        pending=next((row for row in self.pending if digest(row['context'])==key),None)
        if pending is None:
            pending={'context':deepcopy(context),'candidate':None,'heuristic':None,'bound':None,'calls':0,'retry':False};self.pending.append(pending)
            if len(self.pending)>self.policy['cache_limit']:self.pending.pop(0);self.stats['retained_result_evictions']+=1
        fallback=deepcopy(pending['candidate']);heuristic=deepcopy(pending['heuristic'])
        if pending['bound'] is not None:
            # Upper bounds survive added valid Gold cuts and a native rebuild,
            # but only in this exact guarded outside/patch/objective context.
            self.model.add(sum(self.matches)<=pending['bound']).only_enforce_if(guard)
            self.stats['retained_upper_bound_reuses']+=1
        if fallback is not None:
            previous=self._candidate(context,fallback);self.stats['retained_incumbent_hints']+=1
            if target=='edges' or not loop_aware or self.problem.validate(fallback)['valid']:
                self.stats['retained_bound_reuses']+=1
                self.model.add(sum(self.matches)>=previous['matched_edges']).only_enforce_if(guard)
            self.model.clear_hints()
            for cell,value in enumerate(self.variables):self.model.add_hint(value,fallback[cell])
        if self._size()>self.policy['model_text_limit']:
            self.stats['retained_model_caps']+=1;self._drop_model();return heuristic,'MODEL_BUDGET' if heuristic is None else 'UNKNOWN'
        while True:
            control.check()
            if time.monotonic()>=deadline:return heuristic,'MODEL_BUDGET' if heuristic is None else 'UNKNOWN'
            self.solver.parameters.max_time_in_seconds=max(.000001,deadline-time.monotonic());self.solver.parameters.random_seed=seed%2147483647
            if pending['retry']:self.stats['retained_native_retries']+=1
            pending['retry']=True;pending['calls']+=1;self.stats['retained_native_calls']+=1;details['native_calls']=details.get('native_calls',0)+1
            try:
                with _watch(control,deadline,self.solver):status=self.solver.solve(self.model)
            except BaseException:
                self.stats['retained_native_exceptions']+=1
                # A callback can raise after the native response arrived. Charge
                # any available response work, but never publish its result.
                try:
                    self.stats['retained_native_branches']+=int(self.solver.num_branches)
                    self.stats['retained_native_conflicts']+=int(self.solver.num_conflicts)
                except RuntimeError:pass
                self._drop_model();raise
            self.stats['retained_native_branches']+=int(self.solver.num_branches);self.stats['retained_native_conflicts']+=int(self.solver.num_conflicts)
            try:self._check()
            except BaseException:self._drop_model();raise
            name=self.solver.status_name(status)
            if status in (cp_model.OPTIMAL,cp_model.INFEASIBLE):pending['retry']=False
            if status==cp_model.MODEL_INVALID:raise RuntimeError('Invalid retained repair model: '+self.model.validate())
            if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
                candidate=[int(self.solver.value(value)) for value in self.variables];checked=self._candidate(context,candidate)
                if checked['matched_edges']!=round(self.solver.objective_value):raise RuntimeError('Retained objective failed independent seam scoring.')
                bound=float(self.solver.best_objective_bound)
                if not math.isfinite(bound) or bound<checked['matched_edges'] or bound>len(self.problem.edges):raise RuntimeError('Invalid retained native objective bound.')
                if heuristic is None or checked['objective']>score(self.problem,heuristic)['objective'] or target=='gold' and self.problem.validate(candidate)['valid']:heuristic=candidate;pending['heuristic']=candidate.copy()
                if fallback is None or checked['objective']>score(self.problem,fallback)['objective']:fallback=candidate;pending['candidate']=candidate.copy()
                pending['bound']=int(math.floor(bound))
                if target=='gold' and loop_aware and checked['matched_edges']==len(self.problem.edges) and not self.problem.validate(candidate)['valid']:
                    if self._cut(candidate):
                        fallback=deepcopy(pending['candidate'])
                        details['loop_cuts']=details.get('loop_cuts',0)+1
                        if resume_state is not None and len(resume_state['gold_cuts'])<512:resume_state['gold_cuts'].append(candidate.copy())
                        continue
                    self.stats['retained_unknown']+=1;return heuristic,'UNKNOWN'
            if status==cp_model.INFEASIBLE or status==cp_model.OPTIMAL:
                self._remember({'context':context,'candidate':fallback,'heuristic':heuristic,'status':name,'bound':pending['bound']})
                self.pending=[row for row in self.pending if row is not pending];return heuristic,name
            self.stats['retained_unknown']+=1;return heuristic,name
    def observe(self,name,improvement,*,complete):
        self._check()
        if not isinstance(name,str) or len(name)>64 or type(improvement)is not int or improvement<0 or type(complete)is not bool:raise ValueError('Invalid retained policy outcome.')
        if not complete:return
        row=self._policy_row(name);row['completed']+=1;row['gain']+=improvement;self.stats['retained_policy_observations']+=1
    def _policy_row(self,name):
        if name not in self.outcomes:
            if len(self.outcomes)>=64:self.outcomes.pop(next(iter(self.outcomes)));self.stats['retained_policy_evictions']+=1
            self.outcomes[name]={'completed':0,'gain':0,'selected':0}
        return self.outcomes[name]
    def select(self,names):
        self._check()
        if not isinstance(names,(list,tuple)) or not names or len(names)>64 or any(not isinstance(name,str) or len(name)>64 for name in names) or len(set(names))!=len(names):raise ValueError('Invalid retained policy choices.')
        for name in names:self._policy_row(name)
        # Every third choice explores the least selected option. Other choices
        # use completed gains only; UNKNOWN is never a useful completed proof.
        if self.turn%3==0:chosen=min(names,key=lambda name:(self.outcomes[name]['selected'],names.index(name)))
        else:chosen=max(names,key=lambda name:((self.outcomes[name]['gain']+1)/(self.outcomes[name]['completed']+1),-self.outcomes[name]['selected'],-names.index(name)))
        self.turn+=1;self.outcomes[chosen]['selected']+=1;self.stats['retained_policy_selections']+=1;return chosen
    def allocation(self,names,seconds):
        self._check()
        if not isinstance(names,(list,tuple)) or not names or len(names)>64 or len(set(names))!=len(names) or any(not isinstance(name,str) or len(name)>64 for name in names) or type(seconds)not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid retained policy allocation.')
        rows=[self._policy_row(name) for name in names]
        weights=[(row['gain']+1)/(row['completed']+1) for row in rows];total=sum(weights)
        self.stats['retained_policy_plans']+=1
        # Half the budget guarantees equal exploratory opportunity; the rest
        # follows completed gains. UNKNOWN never supplies a success or bound.
        return {name:seconds*(.5/len(names)+.5*weight/total) for name,weight in zip(names,weights)}
    def gold_pool(self):
        self._check()
        if self.gold_service is None:
            from constraint_repair_gold_round26 import GoldRepairPool
            self.gold_service=GoldRepairPool(self.problem)
        return self.gold_service
    def telemetry(self):
        self._check();values=dict(self.stats)
        if self.gold_service is not None:values.update(self.gold_service.stats)
        return values
    def snapshot(self):
        self._check();value={'retained_repair_version':1,'binding':deepcopy(self.binding),'results':deepcopy(self.results),'pending':deepcopy(self.pending),'witnesses':deepcopy(self.witnesses),'outcomes':deepcopy(self.outcomes),'turn':self.turn,'stats':dict(self.stats),'gold_pool':self.gold_service.snapshot() if self.gold_service is not None else None}
        value['digest']=digest(value)
        if len(json.dumps(value,indent=2).encode())>MAX_SNAPSHOT_BYTES:raise ValueError('Retained repair snapshot exceeds32MiB; no partial state published.')
        return value
    def _restore(self,value):
        if not isinstance(value,dict) or len(json.dumps(value,indent=2).encode())>MAX_SNAPSHOT_BYTES or set(value)!={'retained_repair_version','binding','results','pending','witnesses','outcomes','turn','stats','gold_pool','digest'} or type(value['retained_repair_version'])is not int or value['retained_repair_version']!=1 or value['digest']!=digest({k:v for k,v in value.items() if k!='digest'}) or value['binding']!=self.binding:raise ValueError('Retained repair snapshot binding/integrity mismatch.')
        for name,limit in (('results',self.policy['cache_limit']),('pending',self.policy['cache_limit']),('witnesses',self.policy['cut_limit'])):
            if not isinstance(value[name],list) or len(value[name])>limit:raise ValueError('Retained repair record cap exceeded.')
        for name in ('results','pending'):
            seen=set()
            for row in value[name]:
                expected={'context','candidate','heuristic','bound','status'} if name=='results' else {'context','candidate','heuristic','bound','calls','retry'}
                if not isinstance(row,dict) or set(row)!=expected:raise ValueError('Malformed retained query record.')
                c=row['context']
                if not isinstance(c,dict) or set(c)!={'board','patch','target','loop_aware','minimum_edges'} or self._context(c['board'],c['patch'],c['target'],c['loop_aware'],c['minimum_edges'])!=c:raise ValueError('Malformed retained query context.')
                key=digest(c)
                if key in seen:raise ValueError('Repeated retained query record.')
                seen.add(key)
                if row['candidate'] is not None:self._candidate(c,row['candidate'])
                if row['heuristic'] is not None:self._candidate(c,row['heuristic'])
                if row['bound'] is not None and not _integer(row['bound'],0,len(self.problem.edges)):raise ValueError('Invalid retained objective bound.')
                if name=='pending' and type(row['retry'])is not bool:raise ValueError('Invalid pending native retry marker.')
                if row['candidate'] is not None and row['bound'] is not None and row['bound']<score(self.problem,row['candidate'])['matched_edges']:raise ValueError('Retained bound is below its incumbent.')
                if name=='results' and row['status'] not in ('OPTIMAL','INFEASIBLE') or name=='pending' and not _integer(row['calls'],0,2**63-1):raise ValueError('Invalid retained query completion.')
        for witness in value['witnesses']:
            checked=score(self.problem,witness)
            if checked['domain_violations'] or checked['matched_edges']!=len(self.problem.edges) or self.problem.validate(witness)['valid'] or not self.problem.gold_nogood(witness):raise ValueError('Invalid retained Gold witness.')
        if not isinstance(value['stats'],dict) or set(value['stats'])!=set(STAT_KEYS) or any(not _integer(v,0,2**63-1) for v in value['stats'].values()) or not _integer(value['turn'],0,2**63-1):raise ValueError('Invalid retained work counts.')
        if not isinstance(value['outcomes'],dict) or len(value['outcomes'])>64 or any(not isinstance(k,str) or len(k)>64 or not isinstance(v,dict) or set(v)!={'completed','gain','selected'} or any(not _integer(x,0,2**63-1) for x in v.values()) for k,v in value['outcomes'].items()):raise ValueError('Invalid retained policy state.')
        for row in (*value['results'],*value['pending']):
            if row['candidate'] is not None and row['context']['target']=='gold' and row['context']['loop_aware'] and any(all(row['candidate'][cell]==code for cell,code in self.problem.gold_nogood(witness)) for witness in value['witnesses']):raise ValueError('Retained incumbent violates the saved Gold theory.')
        gold=None
        if value['gold_pool'] is not None:
            from constraint_repair_gold_round26 import GoldRepairPool
            gold=GoldRepairPool(self.problem,resume=value['gold_pool'])
        for name in ('results','pending','witnesses','outcomes','stats'):setattr(self,name,deepcopy(value[name]))
        self.gold_service=gold
        self.turn=value['turn'];self.stats['retained_restores']+=1
    def close(self):
        self._drop_model();self.closed=True
        if self.gold_service is not None:self.gold_service.close()
