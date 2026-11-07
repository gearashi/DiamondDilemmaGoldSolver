"""Resumable heuristic sampler controller; no exclusion or coverage authority."""
from __future__ import annotations
from copy import deepcopy
import hashlib,math,time
from pathlib import Path
from constraint_work import digest

class SamplerSession:
    def __init__(self,problem,*,seed=0,replicas=128,backend='auto',target='gold',gold_aware=False,experiments=None,resume=None,allow_software=False):
        from gpu_sampling import _inputs,sampling_flags
        if type(seed)is not int or seed<0 or type(replicas)is not int or not 1<=replicas<=131072 or backend not in ('auto','cuda','webgpu'):
            raise ValueError('Invalid durable sampler configuration.')
        if target not in ('gold','edges') or type(gold_aware)is not bool:raise ValueError('Invalid durable sampling objective.')
        self.problem=problem;self.inputs=None if any(not row for row in problem.domains) else _inputs(problem,gold_aware=gold_aware);self.flags=sampling_flags(experiments)
        self.config={'fingerprint':problem.fingerprint,'domains':[list(row) for row in problem.domains],'fixed':[[int(c),int(v)] for c,v in sorted(problem.fixed.items())],
                     'replicas':replicas,'seed':seed,'target':target,'gold_aware':gold_aware,'flags':self.flags,'controller_version':1,
                     'sources':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ('gpu_sampling_resume_round25.py','gpu_sampling_state_round25.py','gpu_sampling.py','constraint_lns.py','search_problem.py','validator.py')}}
        self.experiments={} if experiments is None else dict(experiments);self.backend=backend;self.allow_software=allow_software
        self.crossover=None;self.crossover_parents=[]
        if self.experiments.get('cycle_crossover'):
            from sampling_crossover_round25 import CycleCrossover
            self.crossover=CycleCrossover(problem,gold_aware=gold_aware)
            self.config['controller_version']=2;self.config['cycle_crossover']=True
            self.config['sources']['sampling_crossover_round25.py']=hashlib.sha256((Path(__file__).parent/'sampling_crossover_round25.py').read_bytes()).hexdigest()
        self.engine=None;self.native=None;self.archive=[];self.launches=0;self.last_collected=-1;self.elapsed=0.;self.done=False;self.solution=None
        self.counts={'initial_boards':0,'move_proposals':0,'samples_considered':0,'archive_checks':0,'archive_repeated_observations':0,'sampler_restores':0}
        self.selected=None;self.device=None;self.last_experiments={};self._active_since=None;self.closed=False;self.poisoned=False
        if resume is not None:self._restore(resume)
    def _healthy(self):
        if self.poisoned or self.engine is not None and getattr(self.engine,'_poisoned',False):raise RuntimeError('Sampler operation failed; restore the last checkpoint into a fresh session.')
    def _check_scope(self):
        if (self.config['fingerprint']!=self.problem.fingerprint or self.config['domains']!=[list(row) for row in self.problem.domains]
                or self.config['fixed']!=[[int(c),int(v)] for c,v in sorted(self.problem.fixed.items())]
                or any(hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=sha for name,sha in self.config['sources'].items())):
            raise ValueError('Sampler input/source changed during the local continuation.')
    def _restore(self,value):
        expected={'sampler_session_version','config','selected','device','native','archive','launches','last_collected','elapsed','done','solution','counts','experiment_stats','digest'}
        if self.crossover is not None:expected.update(('crossover','crossover_parents'))
        if (not isinstance(value,dict) or set(value)!=expected or type(value['sampler_session_version'])is not int or value['sampler_session_version']!=(2 if self.crossover is not None else 1)
                or value['digest']!=digest({k:v for k,v in value.items() if k!='digest'}) or value['config']!=self.config):raise ValueError('Durable sampler input or policy differs.')
        if value['selected'] not in (None,'cuda','webgpu') or self.backend!='auto' and value['selected'] not in (None,self.backend):raise ValueError('Unfinished sampler requires the saved backend.')
        if (type(value['launches'])is not int or value['launches']<0 or type(value['last_collected'])is not int or not -1<=value['last_collected']<=value['launches']
                or type(value['done'])is not bool or type(value['elapsed'])not in (int,float) or not math.isfinite(value['elapsed']) or value['elapsed']<0):raise ValueError('Invalid sampler controller progress.')
        counts=value['counts']
        if not isinstance(counts,dict) or set(counts)!=set(self.counts) or any(type(v)is not int or v<0 for v in counts.values()):raise ValueError('Invalid sampler work counters.')
        r=self.config['replicas']
        if counts['initial_boards'] not in (0,r) or counts['move_proposals']!=value['launches']*r*64 or counts['samples_considered']!=counts['initial_boards']+counts['move_proposals']:
            raise ValueError('Sampler dispatch counters disagree.')
        if (value['native'] is None)!=(counts['initial_boards']==0):raise ValueError('Sampler native state is missing or unexpected.')
        from gpu_sampling import gold_score,score_board
        if not isinstance(value['archive'],list) or len(value['archive'])>8:raise ValueError('Invalid sampler archive.')
        archive=[]
        for item in value['archive']:
            if not isinstance(item,dict) or 'codes' not in item:raise ValueError('Malformed sampler archive board.')
            metrics=gold_score(self.problem,item['codes']) if self.config['gold_aware'] else score_board(self.problem,item['codes'])
            if item!={'codes':item['codes'],**metrics}:raise ValueError('Sampler archive failed independent rescoring.')
            archive.append(deepcopy(item))
        if len({tuple(row['codes']) for row in archive})!=len(archive):raise ValueError('Repeated sampler archive record.')
        solution=value['solution']
        if solution is not None:
            if not isinstance(solution,list):raise ValueError('Invalid saved sampling solution.')
            metrics=score_board(self.problem,solution);validation=self.problem.validate(solution)
            if metrics['domain_violations'] or not (validation['valid'] or self.config['target']=='edges' and metrics['matched_edges']==len(self.problem.edges)):
                raise ValueError('Saved sampling result is not a solution.')
        if not isinstance(value['experiment_stats'],dict) or any(type(v)is not int or v<0 for v in value['experiment_stats'].values()):raise ValueError('Invalid sampler experiment counts.')
        self.selected=value['selected'];self.device=value['device'];self.native=deepcopy(value['native']);self.archive=archive
        self.launches=value['launches'];self.last_collected=value['last_collected'];self.elapsed=float(value['elapsed']);self.done=value['done'];self.solution=deepcopy(solution)
        if self.crossover is not None:
            from sampling_crossover_round25 import CycleCrossover
            self.crossover=CycleCrossover(self.problem,gold_aware=self.config['gold_aware'],resume=value['crossover'])
            parents=value['crossover_parents']
            if not isinstance(parents,list) or len(parents)>8:raise ValueError('Invalid bounded crossover parent pool.')
            for row in parents:
                metrics=gold_score(self.problem,row['codes']) if self.config['gold_aware'] else score_board(self.problem,row['codes'])
                if row!={'codes':row['codes'],**metrics}:raise ValueError('Crossover parent pool failed independent rescoring.')
            if len({tuple(row['codes']) for row in parents})!=len(parents):raise ValueError('Repeated crossover parent pool board.')
            self.crossover_parents=deepcopy(parents)
        self.counts=dict(counts);self.last_experiments=dict(value['experiment_stats'])
    def _ensure_engine(self):
        if self.closed:raise RuntimeError("Sampler session is closed.")
        from gpu_backends import choose_backend
        from gpu_sampling import _make_sampler
        from gpu_sampling_state_round25 import restore
        if self.engine is not None:return
        selected=self.selected or choose_backend(self.backend)
        engine=_make_sampler(self.problem,*self.inputs,self.config['replicas'],self.config['seed'],selected,
            experiments=self.experiments,allow_software=self.allow_software,initialize=self.native is None)
        try:
            if self.native is not None:restore(self.problem,self.inputs[0],engine,self.native);self.counts['sampler_restores']+=1
            else:self.counts['initial_boards']=self.config['replicas'];self.counts['samples_considered']=self.config['replicas']
            self.engine=engine;self.selected=selected;self.device=engine.device
        except BaseException:self.poisoned=True;engine.close();raise
    def collect(self,*,inject=True):
        self._healthy()
        if self.engine is None or self.last_collected==self.launches:return
        from gpu_sampling import gold_score,score_board,_verification_view
        from constraint_lns import diverse_archive
        boards,scores=self.engine.top();verification=_verification_view(self.problem,len(boards));checked=[];observed={tuple(row['codes']) for row in self.archive}
        for codes,objective in zip(boards,scores):
            metrics=gold_score(self.problem,codes) if self.config['gold_aware'] else score_board(verification,codes)
            if metrics['gold_objective' if self.config['gold_aware'] else 'objective']!=int(objective):raise RuntimeError('Resumed GPU sample failed independent scoring.')
            values=list(map(int,codes));key=tuple(values);self.counts['archive_checks']+=1
            if key in observed:self.counts['archive_repeated_observations']+=1
            observed.add(key);checked.append({'codes':values,**metrics})
            if not metrics['domain_violations'] and metrics['matched_edges']==len(self.problem.edges):
                report=self.problem.validate(values)
                if report['valid'] or self.config['target']=='edges':self.solution=values;self.done=True
        self.archive=diverse_archive([*self.archive,*checked],capacity=8,rank_key='gold_objective' if self.config['gold_aware'] else 'objective')
        if not self.done and self.crossover is not None:
            from gpu_sampling import _crossover_step,_crossover_parent_pool
            try:
                self.crossover_parents=_crossover_parent_pool([*self.crossover_parents,*checked,*self.archive],gold_aware=self.config['gold_aware'])
                self.archive,solution=_crossover_step(self.problem,self.crossover,self.archive,self.engine,gold_aware=self.config['gold_aware'],target=self.config['target'],inject=inject,parents=self.crossover_parents)
                self.crossover_parents=_crossover_parent_pool([*self.crossover_parents,*self.archive],gold_aware=self.config['gold_aware'])
            except BaseException:self.poisoned=True;raise
            if solution is not None:self.solution=solution;self.done=True
        self.last_collected=self.launches
    def _account(self):
        if self._active_since is not None:
            now=time.monotonic();self.elapsed+=now-self._active_since;self._active_since=now
    def advance(self,seconds,*,stop=lambda:False,progress=None,max_dispatches=None):
        self._healthy()
        self._check_scope()
        if self.closed:raise RuntimeError("Sampler session is closed.")
        if type(seconds)not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid sampler slice budget.')
        if max_dispatches is not None and (type(max_dispatches)is not int or max_dispatches<0):raise ValueError('Invalid dispatch bound.')
        began=time.monotonic();last_progress=began;count=0;self._active_since=began
        try:
            if self.done or stop() or seconds==0 or max_dispatches==0:return
            if self.inputs is None:self.done=True;return
            self._ensure_engine()
            if not self.inputs[1]:self.collect();self.done=True;return
            while time.monotonic()-began<seconds and not stop() and not self.done and (max_dispatches is None or count<max_dispatches):
                self.engine.launch(64);self.launches+=1;count+=1
                if self.flags&4 and self.launches%8==0:self.engine.exchange((self.launches//8)%2)
                self.counts['move_proposals']+=self.config['replicas']*64;self.counts['samples_considered']+=self.config['replicas']*64
                if self.launches%8==0:self.collect()
                if progress is not None and time.monotonic()-last_progress>=1:
                    progress({'event':'sampling_progress',**self.report()});last_progress=time.monotonic()
        finally:self._account();self._active_since=None
    def finish_phase(self):
        self._healthy()
        if self.closed:raise RuntimeError("Sampler session is closed.")
        if self.engine is None and self.native is not None and self.last_collected!=self.launches:self._ensure_engine()
        if self.engine is not None:self.collect(inject=False)
        self.done=True
    def report(self):
        self._healthy()
        self._check_scope()
        experiments=self.engine.experiment_stats() if self.engine is not None else self.last_experiments
        top=self.archive[0] if self.archive else None
        return {'status':'skipped' if self.inputs is None else ('sampled' if self.done else 'timeout'),
            'reason':'Empty domain or incompatible fixed placements; exact search must decide feasibility.' if self.inputs is None else None,'backend':self.selected or self.backend,'device':self.device,'replicas':self.config['replicas'],
            **self.counts,'complete':False,'unique_boards':None,'duplicates_tracked':False,'elapsed_seconds':self.elapsed,
            'top_samples':deepcopy(self.archive),'top_board':list(top['codes']) if top else None,'best_score':top['matched_edges'] if top else None,
            'best_objective':top['objective'] if top else None,'domain_violations':top['domain_violations'] if top else None,
            'archive_capacity':8,'archive_unique_boards':len(self.archive),'gold_aware':self.config['gold_aware'],
            'sampling_experiments':{**self.experiments,**experiments,**(self.crossover.stats if self.crossover is not None else {})},'dispatches':self.launches,
            'hint_scope':'Heuristic samples only; no exclusions or coverage. Same-runtime native continuation.'}
    def snapshot(self):
        self._healthy()
        self._check_scope()
        if self.closed:raise RuntimeError("Sampler session is closed.")
        self._account()
        from gpu_sampling_state_round25 import encode
        native=encode(self.problem,self.inputs[0],self.engine) if self.engine is not None else deepcopy(self.native)
        experiments=self.engine.experiment_stats() if self.engine is not None else self.last_experiments
        value={'sampler_session_version':2 if self.crossover is not None else 1,'config':deepcopy(self.config),'selected':self.selected,'device':self.device,'native':native,
            'archive':deepcopy(self.archive),'launches':self.launches,'last_collected':self.last_collected,'elapsed':self.elapsed,'done':self.done,
            'solution':deepcopy(self.solution),'counts':dict(self.counts),'experiment_stats':dict(experiments)}
        if self.crossover is not None:value['crossover']=self.crossover.snapshot();value['crossover_parents']=deepcopy(self.crossover_parents)
        value['digest']=digest(value);return value
    def close(self):
        try:
            if self.engine is not None:
                try:
                    if not self.poisoned and not getattr(self.engine,'_poisoned',False):self.last_experiments=dict(self.engine.experiment_stats())
                finally:self.engine.close()
        finally:self.engine=None;self.closed=True
