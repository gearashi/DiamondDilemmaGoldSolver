"""Durable local-repair controller; native CP-SAT queries may need retry.

Completed query identities include the exact outside board and patch. Nothing
in this cache is global puzzle coverage or an imported proof certificate.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib,json,math,random,time
from pathlib import Path
from constraint_work import digest
from constraint_lns import score,diverse_archive,_Control,_Cancelled,_select_patch,_solve_patch,_load_cp_model

EXPERIMENTS=('core_repair','loop_trades','fragment_crossover','repair_learning')
CONCLUSIVE=('OPTIMAL','INFEASIBLE','NO_LOCAL_DOMAIN')
MAX_SNAPSHOT_BYTES=64*1024*1024
def _sources():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('constraint_lns_resume_round25.py','constraint_lns.py','constraint_repair_round24.py','constraint_repair_resume_round25.py','search_problem.py','validator.py')}

def _tuples(value):return tuple(_tuples(x) for x in value) if isinstance(value,(list,tuple)) else value

def _board(problem,value):
    score(problem,value);return list(map(int,value))

class RepairSession:
    def __init__(self,problem,codes,*,phase_seconds,seed=0,target='gold',patch_sizes=(8,16,24,40),per_attempt_seconds=2,max_attempts=64,
                 candidate_boards=None,archive_size=8,adaptive=True,cache_attempts=True,loop_aware=True,dependency_chains=False,experiments=None,resume=None):
        if type(phase_seconds)not in (int,float) or not math.isfinite(phase_seconds) or phase_seconds<0 or type(seed)is not int or seed<0 or target not in ('gold','edges'):raise ValueError('Invalid durable repair budget or target.')
        if not patch_sizes or any(type(v)is not int or v<1 for v in patch_sizes) or len(patch_sizes)>32:raise ValueError('Invalid repair patch sizes.')
        if type(max_attempts)is not int or not 1<=max_attempts<=4096 or type(archive_size)is not int or not 1<=archive_size<=64:raise ValueError('Invalid bounded repair policy.')
        if type(per_attempt_seconds)not in (int,float) or not math.isfinite(per_attempt_seconds) or per_attempt_seconds<=0:raise ValueError('Invalid repair query budget.')
        if any(type(v)is not bool for v in (adaptive,cache_attempts,loop_aware,dependency_chains)):raise ValueError('Repair controls must be boolean.')
        experiments={} if experiments is None else experiments
        if not isinstance(experiments,dict) or set(experiments)-set(EXPERIMENTS) or any(type(v)is not bool for v in experiments.values()):raise ValueError('Invalid repair experiments.')
        self.problem=problem;initial=_board(problem,codes)
        self.config={'controller_version':1,'sources':_sources(),'fingerprint':problem.fingerprint,'domains':[list(row) for row in problem.domains],
            'fixed':[[int(c),int(v)] for c,v in sorted(problem.fixed.items())],'phase_seconds':float(phase_seconds),'seed':seed,'target':target,
            'patch_sizes':list(patch_sizes),'per_attempt_seconds':per_attempt_seconds,'max_attempts':max_attempts,'archive_size':archive_size,
            'adaptive':adaptive,'cache_attempts':cache_attempts,'loop_aware':loop_aware,'dependency_chains':dependency_chains,
            'experiments':[name for name in EXPERIMENTS if experiments.get(name)] if target=='gold' else []}
        if candidate_boards is not None and (not isinstance(candidate_boards,(list,tuple)) or len(candidate_boards)>64):raise ValueError('Repair input archive exceeds64 boards.')
        candidates=[initial]+[_board(problem,row) for row in ([] if candidate_boards is None else candidate_boards)]
        self.config['initial_candidates']=deepcopy(candidates)
        self.archive=diverse_archive([{'codes':row,**score(problem,row)} for row in candidates],capacity=archive_size)
        self.current=self.archive[0]['codes'].copy();self.best=self.current.copy();self.seen={tuple(self.current)};self.cache=[];self.pending=None
        self.index=0;self.size_index=0;self.stalls=0;self.seed_cursor=0;self.previous_size=None;self.rng=random.Random(seed)
        self.elapsed=0.;self.done=False;self.status='timeout';self.reason=None;self._active_since=None
        self.experiment_index=0;self.experiment_pending=None;self.experiment_elapsed=0.;self.experimental_reports=[]
        self.stats={'attempts':0,'accepted':0,'equal_alternatives':0,'improvements':0,'local_no_candidate':0,'candidates_checked':1,'revisited_boards':0,
            'workers':1,'native_statuses':{},'patch_sizes':list(patch_sizes),'dependency_chains':dependency_chains,'loop_cuts':0,'attempt_cache_hits':0,
            'conclusive_attempts':0,'patch_selections':0,'patch_size_counts':{},'size_changes':0,'seed_switches':0,
            'repair_resume_loads':0,'repair_native_retries':0,'repair_model_build_attempts':0,'repair_model_rebuilds':0,'repair_native_calls':0,'repair_cut_replays':0,'repair_completed_work_reused':0,'repair_seen_evictions':0}
        if resume is not None:self._restore(resume)
        for row in self.archive:
            if self._solved(row['codes']):self.best=row['codes'].copy();self.done=True;self.status='solved' if problem.validate(self.best)['valid'] else 'edge_perfect';break
    def _check_scope(self):
        if (self.config['fingerprint']!=self.problem.fingerprint or self.config['domains']!=[list(row) for row in self.problem.domains]
                or self.config['fixed']!=[[int(c),int(v)] for c,v in sorted(self.problem.fixed.items())] or self.config['sources']!=_sources()):
            raise ValueError('Repair input/source changed during the local continuation.')
    def _solved(self,board):
        metrics=score(self.problem,board);report=self.problem.validate(board)
        return not metrics['domain_violations'] and (report['valid'] or self.config['target']=='edges' and report['matched_edges']==len(self.problem.edges))
    def _account(self):
        if self._active_since is not None:
            now=time.monotonic();self.elapsed+=now-self._active_since;self._active_since=now
    def _key(self,board,patch):return digest({'board':board,'patch':patch,'scope':self.config})
    def _restore(self,value):
        if len(json.dumps(value,separators=(',',':')).encode())>MAX_SNAPSHOT_BYTES:raise ValueError('Repair checkpoint exceeds64MiB bound.')
        expected={'repair_session_version','config','current','best','archive','seen','cache','pending','index','size_index','stalls','seed_cursor','previous_size','rng',
            'elapsed','done','status','reason','experiment_index','experiment_pending','experiment_elapsed','experimental_reports','stats','digest'}
        if (not isinstance(value,dict) or set(value)!=expected or type(value['repair_session_version'])is not int or value['repair_session_version']!=1
                or value['digest']!=digest({k:v for k,v in value.items() if k!='digest'}) or value['config']!=self.config):raise ValueError('Durable repair input/policy differs.')
        maximum=self.config['max_attempts'];capacity=self.config['archive_size']
        if value['status'] not in ('timeout','stopped','solved','edge_perfect','inconclusive','skipped') or value['reason'] is not None and (not isinstance(value['reason'],str) or len(value['reason'])>4096):raise ValueError('Invalid repair status.')
        if value['previous_size'] is not None and (type(value['previous_size'])is not int or not 1<=value['previous_size']<=self.problem.n):raise ValueError('Invalid previous repair size.')
        for key in ('index','size_index','stalls','seed_cursor','experiment_index'):
            if type(value[key])is not int or value[key]<0:raise ValueError('Invalid repair controller cursor.')
        if value['index']>maximum or value['size_index']>=len(self.config['patch_sizes']) or value['experiment_index']>len(self.config['experiments']):raise ValueError('Repair controller cursor exceeds policy.')
        if type(value['done'])is not bool or any(type(value[k])not in (int,float) or not math.isfinite(value[k]) or value[k]<0 for k in ('elapsed','experiment_elapsed')):raise ValueError('Invalid repair timing state.')
        if not isinstance(value['archive'],list) or not 1<=len(value['archive'])<=capacity or not isinstance(value['seen'],list) or len(value['seen'])>maximum*4+1024:raise ValueError('Invalid repair board history bounds.')
        self.current=_board(self.problem,value['current']);self.best=_board(self.problem,value['best']);self.archive=[]
        for item in value['archive']:
            metrics=score(self.problem,item['codes'])
            if item!={'codes':item['codes'],**metrics}:raise ValueError('Repair archive failed independent scoring.')
            self.archive.append(deepcopy(item))
        self.seen={tuple(_board(self.problem,row)) for row in value['seen']}
        if not isinstance(value['cache'],list) or len(value['cache'])>maximum:raise ValueError('Repair completed-query cache exceeds bound.')
        self.cache=[]
        for item in value['cache']:
            if not isinstance(item,dict) or set(item)!={'board','patch','status','candidate','key'} or item['status'] not in CONCLUSIVE:raise ValueError('Invalid completed repair query.')
            self._validate_query(item)
            if item['candidate'] is not None:self._validate_candidate(item['candidate'],item['board'],item['patch'])
            self.cache.append(deepcopy(item))
        pending=value['pending']
        if pending is not None:
            if not isinstance(pending,dict) or set(pending)!={'board','patch','key','seed','calls','cuts'}:raise ValueError('Invalid pending repair query.')
            self._validate_query(pending)
            if type(pending['calls'])is not int or pending['calls']<0 or pending['seed']!=self.config['seed']+value['index']:raise ValueError('Pending repair seed/call counter differs.')
            cuts=pending['cuts']
            if not isinstance(cuts,dict) or cuts.get('version')!=1 or not isinstance(cuts.get('gold_cuts'),list) or len(cuts['gold_cuts'])>512:raise ValueError('Invalid pending local cut state.')
            for witness in cuts['gold_cuts']:self._validate_candidate(witness,pending['board'],pending['patch'])
        rng=value['rng']
        if not isinstance(rng,(list,tuple)) or len(rng)!=3 or rng[0]!=3 or not isinstance(rng[1],(list,tuple)) or len(rng[1])!=625 or rng[2] is not None:
            raise ValueError('Invalid bounded Python RNG state.')
        self.rng.setstate(_tuples(rng))
        stats=value['stats']
        if not isinstance(stats,dict) or set(stats)!=set(self.stats):raise ValueError('Invalid repair statistics schema.')
        for key,number in stats.items():
            if key in ('native_statuses','patch_size_counts'):
                if not isinstance(number,dict) or len(number)>64 or any(not isinstance(k,str) or len(k)>64 or type(v)is not int or v<0 for k,v in number.items()):raise ValueError('Invalid repair count map.')
            elif key=='patch_sizes':
                if number!=self.config['patch_sizes']:raise ValueError('Repair patch policy differs.')
            elif key=='dependency_chains':
                if number!=self.config['dependency_chains']:raise ValueError('Repair dependency policy differs.')
            elif type(number)is not int or number<0:raise ValueError('Invalid repair work counter.')
        for key in ('index','size_index','stalls','seed_cursor','previous_size','elapsed','done','status','reason','experiment_index','experiment_pending','experiment_elapsed','experimental_reports'):
            setattr(self,key,deepcopy(value[key]))
        if not isinstance(self.experimental_reports,list) or len(self.experimental_reports)>len(self.config['experiments']):raise ValueError('Invalid experimental repair reports.')
        if self.experiment_pending is not None:
            if not isinstance(self.experiment_pending,dict) or set(self.experiment_pending)!={'name','codes','boards','patch','checkpoint'} or self.experiment_index>=len(self.config['experiments']) or self.experiment_pending['name']!=self.config['experiments'][self.experiment_index]:raise ValueError('Invalid pending repair experiment.')
            _board(self.problem,self.experiment_pending['codes'])
            for board in self.experiment_pending['boards']:_board(self.problem,board)
        if self.status in ('solved','edge_perfect') and (not self.done or not self._solved(self.best) or (self.status=='solved')!=bool(self.problem.validate(self.best)['valid'])):raise ValueError('Saved repair terminal status lacks a valid target certificate.')
        self.pending=deepcopy(pending);self.stats=deepcopy(stats);self.stats['repair_resume_loads']+=1
    def _validate_query(self,item):
        _board(self.problem,item['board']);patch=item['patch']
        if not isinstance(patch,list) or patch!=sorted(set(patch)) or not patch or any(type(c)is not int or not 0<=c<self.problem.n or c in self.problem.fixed for c in patch):raise ValueError('Invalid repair query patch.')
        if item['key']!=self._key(item['board'],patch):raise ValueError('Repair query context differs.')
    def _validate_candidate(self,candidate,board,patch):
        checked=score(self.problem,candidate)
        if checked['domain_violations'] or any(candidate[c]!=board[c] for c in range(self.problem.n) if c not in patch) or {candidate[c]//3 for c in patch}!={board[c]//3 for c in patch}:
            raise ValueError('Repair result escaped its exact local query.')
    def _accept(self,candidate):
        metrics=score(self.problem,candidate);current=score(self.problem,self.current);best=score(self.problem,self.best);key=tuple(candidate)
        self.stats['candidates_checked']+=1
        if key in self.seen:self.stats['revisited_boards']+=1
        if key not in self.seen and len(self.seen)>=self.config['max_attempts']*4+1024:
            self.seen.remove(min(self.seen));self.stats['repair_seen_evictions']+=1
        self.seen.add(key);self.archive=diverse_archive([*self.archive,{'codes':list(candidate),**metrics}],capacity=self.config['archive_size'])
        improved=metrics['objective']>current['objective']
        if metrics['objective']>=current['objective']:
            self.current=list(candidate);self.stats['accepted']+=1;self.stats['equal_alternatives']+=int(not improved)
        if metrics['objective']>=best['objective'] or self._solved(candidate):
            self.best=list(candidate);self.stats['improvements']+=int(metrics['objective']>best['objective'])
        if self._solved(candidate):self.done=True;self.status='solved' if self.problem.validate(candidate)['valid'] else 'edge_perfect'
        return improved
    def _emit(self,progress,event,**extra):
        if progress is not None:progress({'event':event,'phase':'repair','codes':self.best.copy(),'stats':deepcopy(self.stats),**extra})
    def _experiment(self,control,progress):
        from constraint_repair_round24 import core_guided_repair,loop_merge_trade,fragment_crossover,scoped_repair_failure
        name=self.config['experiments'][self.experiment_index]
        if self.experiment_pending is None:
            metrics=score(self.problem,self.current);patch=sorted(_select_patch(self.problem,self.current,metrics,self.config['patch_sizes'][0],self.rng,self.config['target'],dependency_chains=self.config['dependency_chains']))
            self.experiment_pending={'name':name,'codes':self.current.copy(),'boards':[row['codes'].copy() for row in self.archive],'patch':patch,'checkpoint':None}
        pending=self.experiment_pending;budget=self.config['phase_seconds']/(len(self.config['experiments'])+1)
        seconds=min(max(0.,budget-self.experiment_elapsed),max(0.,control.deadline-time.monotonic()))
        if seconds<=0:
            self.experiment_index+=1;self.experiment_pending=None;self.experiment_elapsed=0.;return
        began=time.monotonic();kwargs={'seconds':seconds,'stop':control.stop,'durable':True,'resume':pending['checkpoint']}
        if name=='core_repair':trial=core_guided_repair(self.problem,pending['codes'],**kwargs)
        elif name=='fragment_crossover':trial=fragment_crossover(self.problem,pending['boards'],**kwargs)
        elif name=='repair_learning':trial=scoped_repair_failure(self.problem,pending['codes'],pending['patch'],**kwargs)
        elif score(self.problem,pending['codes'])['matched_edges']==len(self.problem.edges) and not score(self.problem,pending['codes'])['domain_violations']:
            trial=loop_merge_trade(self.problem,pending['codes'],pending['patch'],**kwargs)
        else:trial={'status':'inapplicable','continuation_complete':True}
        self.experiment_elapsed+=time.monotonic()-began;pending['checkpoint']=trial.get('checkpoint')
        report={'proposal':name,**{k:v for k,v in trial.items() if k!='checkpoint'}}
        if len(self.experimental_reports)>self.experiment_index:self.experimental_reports[self.experiment_index]=report
        else:self.experimental_reports.append(report)
        candidate=trial.get('best_board',trial.get('codes'))
        if candidate is not None:self._accept(candidate)
        if trial.get('continuation_complete') or self.experiment_elapsed>=budget:
            self.experiment_index+=1;self.experiment_pending=None;self.experiment_elapsed=0.
        self._emit(progress,'repair_progress')
    def advance(self,seconds,*,stop=None,progress=None,max_attempts_per_slice=None):
        self._check_scope()
        if type(seconds)not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid repair slice budget.')
        if max_attempts_per_slice is not None and (type(max_attempts_per_slice)is not int or max_attempts_per_slice<0):raise ValueError('Invalid repair slice attempt bound.')
        budget=min(seconds,max(0.,self.config['phase_seconds']-self.elapsed));control=_Control(budget,stop);completed=0;self._active_since=time.monotonic()
        try:
            if self.done or max_attempts_per_slice==0:return
            control.check()
            if any(cell in self.problem.fixed for cell in score(self.problem,self.current)['bad_cells']):self.done=True;self.status='skipped';self.reason='A fixed cell violates its legal domain.';return
            if len(self.problem.fixed)==self.problem.n:self.done=True;self.status='inconclusive';return
            try:cp_model=_load_cp_model()
            except ImportError:self.done=True;self.status='skipped';self.reason='OR-Tools unavailable; exact DFS can continue.';return
            while not self.done and self.experiment_index<len(self.config['experiments']):control.check();self._experiment(control,progress)
            while not self.done and self.index<self.config['max_attempts'] and (max_attempts_per_slice is None or completed<max_attempts_per_slice):
                control.check()
                if self.pending is None:
                    chosen=self.config['patch_sizes'][self.size_index if self.config['adaptive'] else self.index%len(self.config['patch_sizes'])]
                    patch=sorted(_select_patch(self.problem,self.current,score(self.problem,self.current),chosen,self.rng,self.config['target'],dependency_chains=self.config['dependency_chains']))
                    if not patch:self.done=True;self.status='inconclusive';break
                    size=len(patch);self.stats['patch_selections']+=1;self.stats['patch_size_counts'][str(size)]=self.stats['patch_size_counts'].get(str(size),0)+1
                    if self.previous_size is not None and size!=self.previous_size:self.stats['size_changes']+=1
                    self.previous_size=size
                    self.pending={'board':self.current.copy(),'patch':patch,'key':self._key(self.current,patch),'seed':self.config['seed']+self.index,'calls':0,'cuts':{'version':1,'gold_cuts':[]}}
                pending=self.pending;cached=next((row for row in self.cache if row['key']==pending['key']),None) if self.config['cache_attempts'] else None
                details={}
                if cached:
                    candidate=deepcopy(cached['candidate']);native_status=cached['status'];self.stats['attempt_cache_hits']+=1;self.stats['repair_completed_work_reused']+=1
                else:
                    self._emit(progress,'repair_attempt',patch_cells=pending['patch'])
                    if pending['calls']:self.stats['repair_native_retries']+=1
                    pending['calls']+=1;self.stats['attempts']+=1;self.stats['repair_model_build_attempts']+=1
                    try:
                        candidate,native_status=_solve_patch(self.problem,pending['board'],pending['patch'],cp_model=cp_model,control=control,
                            seconds=self.config['per_attempt_seconds'],seed=pending['seed'],target=self.config['target'],loop_aware=self.config['loop_aware'],diagnostics=details,resume_state=pending['cuts'])
                    finally:
                        self.stats['loop_cuts']+=details.get('loop_cuts',0);self.stats['repair_cut_replays']+=details.get('restored_loop_cuts',0);self.stats['repair_model_rebuilds']+=details.get('model_built',0);self.stats['repair_native_calls']+=details.get('native_calls',0)
                    self.stats['native_statuses'][native_status]=self.stats['native_statuses'].get(native_status,0)+1
                if candidate is not None:self._validate_candidate(candidate,pending['board'],pending['patch'])
                improved=self._accept(candidate) if candidate is not None else False
                if self.done:
                    self.pending=None;self.index+=1;self._emit(progress,'candidate');return
                if not cached and self.config['cache_attempts'] and native_status in CONCLUSIVE:
                    self.cache.append({k:deepcopy(pending[k]) for k in ('board','patch','key')}|{'status':native_status,'candidate':deepcopy(candidate)});self.stats['conclusive_attempts']+=1
                if control.cancelled() and native_status not in CONCLUSIVE:
                    self._emit(progress,'repair_progress');control.check() # Pending exact context stays available for retry.
                if candidate is None:self.stats['local_no_candidate']+=1
                self.stalls=0 if improved else self.stalls+1
                if self.config['adaptive']:
                    if self.stalls==0:self.size_index=0
                    elif native_status in ('UNKNOWN','MODEL_BUDGET','FEASIBLE'):self.size_index=max(0,self.size_index-1)
                    else:self.size_index=(self.size_index+1)%len(self.config['patch_sizes'])
                if self.stalls>=2*len(self.config['patch_sizes']) and len(self.archive)>1:
                    self.seed_cursor=(self.seed_cursor+1)%len(self.archive);alternate=self.archive[self.seed_cursor]['codes']
                    if alternate!=self.current:self.current=alternate.copy();self.stats['seed_switches']+=1
                    self.stalls=0
                self.pending=None;self.index+=1;completed+=1;self._emit(progress,'repair_progress')
            if self.index>=self.config['max_attempts'] and not self.done:self.done=True;self.status='inconclusive';self.reason='Configured local query limit reached.'
        except _Cancelled:self.status=control.reason or 'timeout'
        finally:self._account();self._active_since=None
    def finish_phase(self):
        self.done=True
        if not self._solved(self.best):self.status='inconclusive';self.reason='The allocated local-repair phase budget ended; global feasibility remains unknown.'
    def report(self):
        self._check_scope()
        metrics=score(self.problem,self.best)
        return {'status':self.status,'complete':self.status in ('solved','edge_perfect'),'scope':'local_repair_only','best_board':self.best.copy(),
            'best_score':metrics['matched_edges'],'best_objective':metrics['objective'],'domain_violations':metrics['domain_violations'],
            'validation':self.problem.validate(self.best),'elapsed_seconds':self.elapsed,'stats':deepcopy(self.stats),
            'candidate_archive':deepcopy(self.archive),'experimental_repair':deepcopy(self.experimental_reports),'continuation_complete':self.done,'reason':self.reason}
    def snapshot(self):
        self._check_scope();self._account();value={'repair_session_version':1,'config':deepcopy(self.config),'current':self.current.copy(),'best':self.best.copy(),'archive':deepcopy(self.archive),
            'seen':[list(row) for row in sorted(self.seen)],'cache':deepcopy(self.cache),'pending':deepcopy(self.pending),'rng':self.rng.getstate(),'stats':deepcopy(self.stats)}
        for key in ('index','size_index','stalls','seed_cursor','previous_size','elapsed','done','status','reason','experiment_index','experiment_pending','experiment_elapsed','experimental_reports'):value[key]=deepcopy(getattr(self,key))
        value['digest']=digest(value)
        if len(json.dumps(value,separators=(',',':')).encode())>MAX_SNAPSHOT_BYTES:raise ValueError('Repair checkpoint exceeds64MiB bound; pending state was retained in memory.')
        return value
