"""Bounded trusted-local Gold repair model and optimization-master reuse.

Saved cores and optimal master bounds are local continuation records, not proof
certificates. Native CP-SAT search is never serialized; UNKNOWN calls retry and
are counted. The assumption-free Gold model is shared across repair modes.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time

MAX_BYTES = 32 * 1024 * 1024
STAT_KEYS = tuple('gold_pool_' + key for key in (
    'scope_checks', 'model_build_attempts', 'model_builds', 'model_reuses',
    'model_evictions', 'rebuilt_variables', 'rebuilt_constraints',
    'query_calls', 'native_calls', 'native_retries', 'native_unknown',
    'native_interrupted', 'native_branches', 'native_conflicts',
    'completed_hits', 'core_hits', 'core_clauses_added', 'query_hint_reuses', 'result_evictions', 'restores', 'witness_checks', 'master_validation_terms',
    'master_build_attempts', 'master_builds', 'master_build_interrupted', 'master_reuses', 'master_clauses_added',
    'master_calls', 'master_retries', 'master_unknown', 'master_interrupted',
    'master_branches', 'master_completed_hits', 'master_bound_reuses',
    'master_hint_reuses', 'master_evictions'))
ORACLE_KEYS = ('oracle_queries','oracle_branches','oracle_conflicts','oracle_unknown','oracle_core_cells')


def _integer(value, low, high): return type(value) is int and low <= value <= high

def _encoded(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')

def _digest(value): return hashlib.sha256(_encoded(value)).hexdigest()


class GoldRepairPool:
    def __init__(self, problem, *, model_limit=8, result_limit=256,
                 model_text_limit=64*1024*1024, resume=None):
        if not _integer(model_limit,1,64) or not _integer(result_limit,1,1024) or not _integer(model_text_limit,1024,256*1024*1024):
            raise ValueError('Invalid Gold pool limits.')
        self.problem=problem
        self.policy={'model_limit':model_limit,'result_limit':result_limit,'model_text_limit':model_text_limit}
        self.binding=self._identity(); self._stats=dict.fromkeys(STAT_KEYS,0)
        self.results=[]; self.failures=[]; self.pending=[]; self.masters=[]
        self._oracle_model=None; self._posted_cores=set(); self._native_masters={}; self.closed=False
        if resume is not None: self._restore(resume)

    def _identity(self):
        from constraint_repair_resume_round25 import _binding
        source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        from importlib.metadata import version, PackageNotFoundError
        try: dependency=version('ortools')
        except PackageNotFoundError: dependency=None
        return _digest({'semantic':_binding(self.problem,'gold_pool',{},self.policy),'source':source,'ortools':dependency})

    def _check(self):
        if self.closed: raise RuntimeError('Gold repair pool is closed.')
        if self.binding != self._identity(): raise ValueError('Gold pool source or problem changed.')
        self._stats['gold_pool_scope_checks']+=1

    @property
    def stats(self): return dict(self._stats)

    def _inc(self,key,value=1): self._stats['gold_pool_'+key]+=value

    def _assumptions(self,raw):
        if type(raw) not in (list,tuple): raw=list(raw)
        if len(raw)>self.problem.n or any(type(row) not in (list,tuple) or len(row)!=2 or
                not _integer(row[0],0,self.problem.n-1) or not _integer(row[1],0,3*self.problem.n-1) for row in raw):
            raise ValueError('Invalid Gold pool assumptions.')
        rows=sorted(tuple(row) for row in raw)
        if len({cell for cell,code in rows})!=len(rows): raise ValueError('Duplicate Gold pool assumption cell.')
        return tuple(rows)

    def _board(self,board,assumptions=()):
        from constraint_lns import score
        self._inc('witness_checks')
        score(self.problem,board)
        if any(code not in self.problem.domains[cell] for cell,code in enumerate(board)) or any(board[c]!=v for c,v in assumptions):
            raise ValueError('Gold pool witness violates its domains or assumptions.')
        validation=self.problem.validate(board)
        if not validation['valid']: raise ValueError('Gold pool witness is not an independent Gold solution.')
        return validation

    def _cores(self,rows):
        if type(rows) not in (list,tuple) or len(rows)>1024: raise ValueError('Invalid saved Gold cores.')
        return [self._assumptions(row) for row in rows]

    def _remember_core(self,core):
        if any(set(old)<=set(core) for old in self.failures): return
        self.failures=[old for old in self.failures if not set(core)<set(old)]+[core]
        while len(self.failures)>self.policy['result_limit']:
            self.failures.pop(0); self._inc('result_evictions')

    def _remember_result(self,assumptions,board):
        self.results=[row for row in self.results if tuple(map(tuple,row['assumptions']))!=assumptions]
        self.results.append({'assumptions':[list(row) for row in assumptions],'codes':list(board)})
        while len(self.results)>self.policy['result_limit']:
            self.results.pop(0); self._inc('result_evictions')

    def oracle(self,control,*,state_failures=()):
        self._check(); control.check(); cores=self._cores(state_failures)
        # state_failures are from the caller's already scope-bound local repair
        # checkpoint. This is deliberately not an external certificate API.
        for core in cores: self._remember_core(core)
        return _OracleView(self,control)

    def _native_oracle(self,control):
        pending=set(self.failures)-self._posted_cores
        if self._oracle_model is not None and len(self._posted_cores)+len(pending)>self.policy['result_limit']:
            self._oracle_model=None; self._posted_cores=set(); self._inc('model_evictions')
        if self._oracle_model is None:
            from constraint_repair_round24 import GoldRepairOracle
            self._inc('model_build_attempts')
            model=GoldRepairOracle(self.problem,control)
            self._check()
            text_size=len(str(model.model.model.proto).encode('utf-8'))
            self._inc('model_builds'); self._inc('rebuilt_variables',len(model.model.model.proto.variables))
            self._inc('rebuilt_constraints',len(model.model.model.proto.constraints))
            if text_size>self.policy['model_text_limit']:
                self._inc('model_evictions')
                from constraint_circuit import CircuitLimit
                raise CircuitLimit('Retained Gold model exceeds its text-size limit.')
            self._oracle_model=model
        else: self._inc('model_reuses')
        self._oracle_model.control=control
        self._oracle_model.failures=list(self.failures)
        native=self._oracle_model.model
        for core in self.failures:
            control.check()
            if core in self._posted_cores: continue
            # Each core proves this original-root placement combination cannot
            # satisfy Gold. Adding its negation is independent of the next query.
            if all(code in native.placements[cell] for cell,code in core):
                native.model.add_bool_or([native.placements[cell][code].Not() for cell,code in core])
                self._inc('core_clauses_added')
            self._posted_cores.add(core)
        if len(str(native.model.proto).encode('utf-8'))>self.policy['model_text_limit']:
            self._oracle_model=None; self._posted_cores=set(); self._inc('model_evictions')
            from constraint_circuit import CircuitLimit
            raise CircuitLimit('Retained Gold clauses exceed the model text-size limit.')
        native.model.clear_hints()
        if self.results:
            witness=self.results[-1]['codes']; self._board(witness)
            for variable,code in zip(native.codes,witness): native.model.add_hint(variable,code)
            self._inc('query_hint_reuses')
        return self._oracle_model

    def _query(self,view,assumptions,seconds):
        self._check(); view.control.check(); assumptions=self._assumptions(assumptions)
        if seconds is not None and (type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0):
            raise ValueError('Invalid Gold query budget.')
        self._inc('query_calls'); given=dict(assumptions)
        for core in self.failures:
            if all(given.get(cell)==code for cell,code in core):
                self._inc('core_hits'); return {'status':'infeasible','core':core,'cached':True,'binding':view.binding}
        for row in self.results:
            codes=row['codes']
            if all(codes[cell]==code for cell,code in assumptions):
                validation=self._board(codes,assumptions); self._check(); self._inc('completed_hits')
                return {'status':'solved','codes':list(codes),'validation':validation,'cached':True,'binding':view.binding}
        model=self._native_oracle(view.control); before=dict(model.stats); answer=None
        key=[list(row) for row in assumptions]; pending=next((row for row in self.pending if row==key),None)
        if pending is None:
            self.pending.append(key)
            if len(self.pending)>self.policy['result_limit']: self.pending.pop(0); self._inc('result_evictions')
        try: answer=model.query(assumptions,seconds=seconds)
        finally:
            for name in ORACLE_KEYS: view._stats[name]+=model.stats[name]-before[name]
            starts=model.stats['oracle_queries']-before['oracle_queries']; self._inc('native_calls',starts)
            if starts and pending is not None: self._inc('native_retries',starts)
            self._inc('native_branches',model.stats['oracle_branches']-before['oracle_branches'])
            self._inc('native_conflicts',model.stats['oracle_conflicts']-before['oracle_conflicts'])
            if answer is None: self._inc('native_interrupted',starts)
        self._check()
        if answer['status']=='solved':
            self._board(answer['codes'],assumptions); self._remember_result(assumptions,answer['codes'])
        elif answer['status']=='infeasible':
            core=self._assumptions(answer['core'])
            if not set(core)<=set(assumptions): raise RuntimeError('Native Gold core escaped its assumptions.')
            self._remember_core(core)
        else: self._inc('native_unknown'); return answer
        self.pending=[row for row in self.pending if row!=key]
        return answer

    def _fragment_context(self,fragments):
        if type(fragments) not in (list,tuple) or not 1<=len(fragments)<=4096: raise ValueError('Invalid retained fragment list.')
        rows=[]
        for fragment in fragments:
            if type(fragment) is not dict or set(fragment)!={'source','placements'} or not _integer(fragment['source'],0,31):
                raise ValueError('Invalid retained fragment.')
            places=self._assumptions(fragment['placements'])
            if not places or len({code//3 for cell,code in places})!=len(places): raise ValueError('Invalid fragment inventory.')
            rows.append({'source':fragment['source'],'placements':[list(pair) for pair in places]})
        return {'fragments':rows}

    def _constraints(self,rows,allowed):
        if type(rows) not in (list,tuple) or len(rows)>1024: raise ValueError('Invalid retained master rows.')
        result=[]
        for row in rows:
            if type(row) not in (list,tuple,set,frozenset) or any(type(item)is not int or item not in allowed for item in row):
                raise ValueError('Invalid retained master constraint.')
            converted=tuple(sorted(set(row)))
            if converted not in result: result.append(converted)
        return sorted(result)

    def _valid_choice(self,kind,context,constraints,chosen):
        if chosen is None: return False
        selected=set(chosen)
        self._inc('master_validation_terms',sum(map(len,constraints)))
        if kind=='hit': return all(selected.intersection(row) for row in constraints)
        if any(set(row)<=selected for row in constraints): return False
        placements=[pair for i in chosen for pair in context['fragments'][i]['placements']]
        self._inc('master_validation_terms',len(placements))
        return len({c for c,v in placements})==len(placements)==len({v//3 for c,v in placements})

    def _objective(self,kind,context,chosen):
        return len(chosen) if kind=='hit' else sum(len(context['fragments'][i]['placements']) for i in chosen)

    def minimum_hitting_set(self,cores,cells,control,cp_model,*,seed):
        from constraint_lns import score
        score(self.problem,seed)
        if type(cells) not in (list,tuple) or cells!=sorted(set(cells)) or any(not _integer(c,0,self.problem.n-1) or c in self.problem.fixed for c in cells):
            raise ValueError('Invalid retained hitting-set cells.')
        context={'cells':list(cells),'seed':list(seed)}
        constraints=self._constraints(cores,set(cells))
        chosen,work=self._master('hit',context,constraints,control,cp_model)
        return (set(chosen),work) if chosen is not None else (None,work)

    def fragment_master(self,fragments,conflicts,control,cp_model):
        context=self._fragment_context(fragments)
        rows=self._constraints(conflicts,set(range(len(fragments))))
        chosen,work=self._master('fragment',context,rows,control,cp_model)
        return chosen,work if type(work)is int else 0

    def _master(self,kind,context,constraints,control,cp):
        self._check(); control.check(); key=_digest({'kind':kind,'context':context})
        record=next((row for row in self.masters if row['key']==key),None)
        wanted=[list(row) for row in constraints]
        if record is not None and not set(map(tuple,record['constraints']))<=set(constraints):
            # Relaxation invalidates learned lower/upper bounds and exclusions.
            self.masters.remove(record); self._native_masters.pop(key,None); self._inc('master_evictions'); record=None
        if record is None:
            record={'key':key,'kind':kind,'context':deepcopy(context),'constraints':[], 'chosen':None,
                    'optimal':False,'infeasible':False,'retry':False,'bound':None}
            self.masters.append(record)
            while len(self.masters)>self.policy['model_limit']:
                old=self.masters.pop(0); self._native_masters.pop(old['key'],None); self._inc('master_evictions')
        same=record['constraints']==wanted
        if record['infeasible']:
            record['constraints']=wanted; self._inc('master_completed_hits'); return None,'infeasible'
        if record['optimal'] and self._valid_choice(kind,context,constraints,record['chosen']):
            record['constraints']=wanted; self._inc('master_completed_hits')
            if not same: self._inc('master_bound_reuses')
            return list(record['chosen']),0
        # Retain the old proved optimum as a valid bound under extra constraints.
        if record['optimal']:
            record['bound']=self._objective(kind,context,record['chosen']); record['optimal']=False
        record['constraints']=wanted
        native=self._native_masters.get(key)
        if native is None:
            model=cp.CpModel(); self._inc('master_build_attempts')
            try:
                control.check()
                names=context['cells'] if kind=='hit' else list(range(len(context['fragments'])))
                variables={i:model.new_bool_var('choice_'+str(i)) for i in names}
                if kind=='fragment':
                    by_cell={}; by_tile={}
                    for i,fragment in enumerate(context['fragments']):
                        if i%32==0: control.check()
                        for cell,code in fragment['placements']:
                            by_cell.setdefault(cell,[]).append(variables[i]); by_tile.setdefault(code//3,[]).append(variables[i])
                    for row in [*by_cell.values(),*by_tile.values()]: model.add_at_most_one(row)
                objective=sum(variables.values()) if kind=='hit' else sum(len(f['placements'])*variables[i] for i,f in enumerate(context['fragments']))
                if kind=='hit': model.minimize(objective)
                else: model.maximize(objective)
                native={'model':model,'variables':variables,'objective':objective,'constraints':set(),'bound':None}
                self._native_masters[key]=native; self._inc('master_builds')
            except BaseException:
                self._inc('master_build_interrupted'); raise
            finally:
                self._inc('rebuilt_variables',len(model.proto.variables)); self._inc('rebuilt_constraints',len(model.proto.constraints))
        else: self._inc('master_reuses')
        model=native['model']; variables=native['variables']
        for index,row in enumerate(constraints):
            if index%32==0: control.check()
            if row not in native['constraints']:
                model.add_bool_or([variables[i] if kind=='hit' else variables[i].Not() for i in row])
                native['constraints'].add(row); self._inc('master_clauses_added')
        if record['bound'] is not None and native['bound']!=record['bound']:
            model.add(native['objective']>=record['bound'] if kind=='hit' else native['objective']<=record['bound'])
            native['bound']=record['bound']; self._inc('master_bound_reuses')
        model.clear_hints()
        if self._valid_choice(kind,context,constraints,record['chosen']):
            selected=set(record['chosen'])
            for i,var in variables.items(): model.add_hint(var,int(i in selected))
            self._inc('master_hint_reuses')
        if len(str(model.proto).encode('utf-8'))>self.policy['model_text_limit']:
            self._native_masters.pop(key,None); self._inc('master_evictions'); return None,'unknown'
        control.check(); solver=cp.CpSolver(); solver.parameters.num_search_workers=1
        solver.parameters.max_time_in_seconds=max(.000001,control.deadline-time.monotonic())
        if record['retry']: self._inc('master_retries')
        record['retry']=True; self._inc('master_calls'); status=None
        from constraint_lns import _watch
        try:
            with _watch(control,control.deadline,solver): status=solver.solve(model)
        finally:
            if status is None: self._inc('master_interrupted')
        self._inc('master_branches',int(solver.num_branches)); self._check()
        if status==cp.MODEL_INVALID: raise RuntimeError('Invalid retained Gold master: '+model.validate())
        if status==cp.INFEASIBLE:
            record.update(infeasible=True,retry=False,optimal=False,chosen=None); return None,'infeasible'
        if status in (cp.FEASIBLE,cp.OPTIMAL):
            chosen=sorted(i for i,var in variables.items() if solver.value(var))
            if not self._valid_choice(kind,context,constraints,chosen): raise RuntimeError('Retained master candidate failed independent constraints.')
            objective=self._objective(kind,context,chosen)
            if objective!=round(solver.objective_value): raise RuntimeError('Retained master objective mismatch.')
            record['chosen']=chosen
            if status==cp.OPTIMAL: record.update(optimal=True,retry=False,bound=objective)
            else: self._inc('master_unknown')
            if kind=='fragment' or status==cp.OPTIMAL: return chosen,int(solver.num_branches)
        else: self._inc('master_unknown')
        return None,'unknown'

    def snapshot(self):
        self._check()
        value={'version':1,'binding':self.binding,'policy':dict(self.policy),'stats':dict(self._stats),
               'results':deepcopy(self.results),'failures':[[list(p) for p in row] for row in self.failures],
               'pending':deepcopy(self.pending),'masters':deepcopy(self.masters)}
        value['digest']=_digest(value)
        if len(_encoded(value))>MAX_BYTES: raise ValueError('Gold pool snapshot exceeds its byte limit.')
        return value

    def _restore(self,value):
        if type(value)is not dict or set(value)!={'version','binding','policy','stats','results','failures','pending','masters','digest'} or len(_encoded(value))>MAX_BYTES:
            raise ValueError('Invalid Gold pool checkpoint shape.')
        if type(value['version'])is not int or value['version']!=1 or value['binding']!=self.binding or value['policy']!=self.policy or type(value['policy'])is not dict or any(type(v)is not int for v in value['policy'].values()) or value['digest']!=_digest({k:v for k,v in value.items() if k!='digest'}):
            raise ValueError('Gold pool checkpoint binding mismatch.')
        stats=value['stats']
        if type(stats)is not dict or set(stats)!=set(STAT_KEYS) or any(not _integer(v,0,10**100) for v in stats.values()): raise ValueError('Invalid Gold pool counters.')
        for name,limit in (('results',self.policy['result_limit']),('failures',self.policy['result_limit']),('pending',self.policy['result_limit']),('masters',self.policy['model_limit'])):
            if type(value[name])is not list or len(value[name])>limit: raise ValueError('Gold pool saved record limit exceeded.')
        failures=self._cores(value['failures']); pending=[list(map(list,self._assumptions(row))) for row in value['pending']]
        for row in value['results']:
            if type(row)is not dict or set(row)!={'assumptions','codes'}: raise ValueError('Invalid saved Gold witness.')
            self._board(row['codes'],self._assumptions(row['assumptions']))
        seen=set()
        for row in value['masters']:
            if type(row)is not dict or set(row)!={'key','kind','context','constraints','chosen','optimal','infeasible','retry','bound'} or row['kind'] not in ('hit','fragment'):
                raise ValueError('Invalid saved Gold master.')
            kind=row['kind']; context=row['context']
            if type(context)is not dict: raise ValueError('Invalid saved master context.')
            if kind=='hit':
                if set(context)!={'cells','seed'} or type(context['cells'])is not list or context['cells']!=sorted(set(context['cells'])) or any(not _integer(c,0,self.problem.n-1) or c in self.problem.fixed for c in context['cells']): raise ValueError('Invalid saved release cells.')
                from constraint_lns import score
                score(self.problem,context['seed']); allowed=set(context['cells'])
            else:
                if set(context)!={'fragments'} or self._fragment_context(context['fragments'])!=context: raise ValueError('Invalid saved fragment scope.')
                allowed=set(range(len(context['fragments'])))
            constraints=self._constraints(row['constraints'],allowed)
            if row['constraints']!=[list(r) for r in constraints] or row['key']!=_digest({'kind':kind,'context':context}) or row['key'] in seen: raise ValueError('Invalid saved master identity.')
            seen.add(row['key'])
            if any(type(row[k])is not bool for k in ('optimal','infeasible','retry')): raise ValueError('Invalid saved master flags.')
            choice=row['chosen']
            if choice is not None and (type(choice)is not list or choice!=sorted(set(choice)) or any(type(i)is not int or i not in allowed for i in choice)): raise ValueError('Invalid saved master choice.')
            upper=self.problem.n if kind=='hit' else sum(len(f['placements']) for f in context['fragments'])
            if row['bound']is not None and not _integer(row['bound'],0,upper): raise ValueError('Invalid saved master bound.')
            if row['optimal'] and (row['infeasible'] or row['retry'] or not self._valid_choice(kind,context,constraints,choice) or row['bound']!=self._objective(kind,context,choice)): raise ValueError('Invalid saved optimum.')
            if row['infeasible'] and (choice is not None or row['retry']): raise ValueError('Invalid saved infeasibility.')
        # All validation completes before publishing any restored records.
        validation_cost=dict(self._stats)
        self._stats=dict(stats)
        for key,value_work in validation_cost.items(): self._stats[key]+=value_work
        self._inc('restores'); self.results=deepcopy(value['results'])
        self.failures=failures; self.pending=deepcopy(pending); self.masters=deepcopy(value['masters'])

    def close(self):
        self._oracle_model=None; self._posted_cores.clear(); self._native_masters.clear(); self.closed=True


class _OracleView:
    def __init__(self,pool,control):
        from ortools.sat.python import cp_model
        self.pool,self.problem,self.control,self.cp_model=pool,pool.problem,control,cp_model
        self.binding=(self.problem.fingerprint,tuple(tuple(map(int,row)) for row in self.problem.domains))
        self._stats=dict.fromkeys(ORACLE_KEYS,0)
    @property
    def stats(self): return dict(self._stats)
    @property
    def failures(self): return list(self.pool.failures)
    def query(self,assumptions=(),*,seconds=None): return self.pool._query(self,assumptions,seconds)
    def export_exclusions(self):
        self.pool._check()
        return {'kind':'gold_repair_assumption_cores','fingerprint':self.problem.fingerprint,
                'domains':[list(row) for row in self.problem.domains],
                'cores':[[list(pair) for pair in row] for row in self.pool.failures]}
