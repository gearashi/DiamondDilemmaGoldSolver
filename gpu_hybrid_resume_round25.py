"""Durable hybrid scheduling; heuristics own no global pruning authority.

Version3 saves the sampler's native trajectory and local-repair controller.
Interrupted CP-SAT search restarts the exact pending query; it is not native
CP-SAT state persistence. Exact DFS retains its existing checkpoint contract.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib,math,time
from pathlib import Path
from constraint_work import digest
from constraint_lns import score,diverse_archive
from gpu_sampling_resume_round25 import SamplerSession
from constraint_lns_resume_round25 import RepairSession

SAMPLING=('ejection_chains','move_tabu','replica_exchange','obstruction_chains','cycle_crossover','lazy_gold_scoring')
REPAIR=('core_repair','loop_trades','fragment_crossover','repair_learning')
def _sources():
    root=Path(__file__).parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in (
        'gpu_hybrid_resume_round25.py','gpu_sampling.py','gpu_sampling_resume_round25.py','gpu_sampling_state_round25.py',
        'constraint_lns.py','constraint_lns_resume_round25.py','constraint_repair_round24.py','constraint_repair_resume_round25.py','search_problem.py','validator.py')}

def solve(problem,*,seconds=10,seed=0,target='gold',replicas=128,backend='auto',sampling_fraction=.2,repair_fraction=.3,
          hints=None,seed_board=None,stop=None,progress=None,resume=None,learning_state=None,options=None):
    from constraint_dfs import solve as exact_solve
    if type(seconds)not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid hybrid slice budget.')
    if target not in ('gold','edges') or type(seed)is not int or seed<0:raise ValueError('Invalid hybrid target/seed.')
    if any(type(x)not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in (sampling_fraction,repair_fraction)) or sampling_fraction+repair_fraction>1:raise ValueError('Invalid hybrid phase fractions.')
    if stop is None:stop=lambda:False
    elif not callable(stop):stop=stop.is_set
    options={} if options is None else options
    learning_state={} if learning_state is None else learning_state
    began=time.monotonic();last_save=began;sampler=None;repair=None
    sampling_policy={'seed':seed,'replicas':replicas,'gold_aware':bool(options.get('gpu_gold_sampling') and target=='gold'),
                     'experiments':{name:True for name in SAMPLING if options.get(name)}}
    repair_policy={'seed':seed,'dependency_chains':bool(options.get('dependency_repair')),
                   'experiments':{name:True for name in REPAIR if options.get(name)}}
    scope={'fingerprint':problem.fingerprint,'domains':[list(row) for row in problem.domains],
           'fixed':[[int(c),int(v)] for c,v in sorted(problem.fixed.items())],'target':target}
    total=learning_state.get('_hybrid_run_seconds',seconds)
    if type(total)not in (int,float) or not math.isfinite(total) or total<0:raise ValueError('Invalid whole-run hybrid budget.')
    budgets={'sampling':min(30.,total*sampling_fraction) if total else (30. if sampling_fraction else 0.),
             'repair':min(60.,total*repair_fraction) if total else (60. if repair_fraction else 0.)}
    state={'hybrid_version':3,'scope':scope,'sources':_sources(),'phase':'sampling','budgets':budgets,
        'sampling_policy':sampling_policy,'repair_policy':repair_policy,'sampler_state':None,'repair_state':None,
        'sampling':None,'repair':None,'dfs':None,'hints':deepcopy(hints),'candidates':[],
        'repair_inputs':None,'checked':0,'resumes':0,'transitions':0,'claims_offered':False}
    if resume is not None:
        if isinstance(resume,dict) and resume.get('hybrid_version')==3:
            expected=set(state)|{'digest'}
            if set(resume)!=expected or type(resume['hybrid_version'])is not int or resume['digest']!=digest({k:v for k,v in resume.items() if k!='digest'}) or resume['scope']!=scope:raise ValueError('Invalid durable hybrid checkpoint scope/digest.')
            if resume['phase'] not in ('sampling','repair','exact_dfs'):raise ValueError('Invalid saved hybrid phase.')
            if resume['phase']!='exact_dfs' and resume['sources']!=state['sources']:raise ValueError('Unfinished heuristic controller requires matching source semantics.')
            if resume['phase']=='sampling' and resume['sampling_policy']!=sampling_policy:raise ValueError('Unfinished sampler requires saved replicas/seed/policy.')
            if resume['phase'] in ('sampling','repair') and resume['repair_policy']!=repair_policy:raise ValueError('Unfinished hybrid requires the saved repair policy.')
            if (not isinstance(resume['budgets'],dict) or set(resume['budgets'])!={'sampling','repair'}
                or any(type(resume['budgets'][name])not in (int,float) or not math.isfinite(resume['budgets'][name]) or not 0<=resume['budgets'][name]<=cap for name,cap in (('sampling',30),('repair',60)))):raise ValueError('Invalid saved hybrid budget.')
            for key in ('checked','resumes','transitions'):
                if type(resume[key])is not int or resume[key]<0:raise ValueError('Invalid hybrid counter.')
            if type(resume['claims_offered'])is not bool or not isinstance(resume['candidates'],list) or len(resume['candidates'])>10:raise ValueError('Invalid hybrid archive.')
            state=deepcopy({k:v for k,v in resume.items() if k!='digest'});state['resumes']+=1
            for values in state['candidates']:score(problem,values)
            if state['phase']=='repair' and state['repair_inputs'] is None:raise ValueError('Missing saved repair input.')
            if state['repair_inputs'] is not None:
                if not isinstance(state['repair_inputs'],list) or not 1<=len(state['repair_inputs'])<=8:raise ValueError('Invalid repair input archive.')
                for values in state['repair_inputs']:score(problem,values)
            if state['phase']!='sampling' and state['sampler_state'] is not None or state['phase']!='repair' and state['repair_state'] is not None:raise ValueError('Completed hybrid phase retains unexpected native state.')
        else:
            old=resume
            if isinstance(old,dict) and 'hybrid_version' in old:
                if old['hybrid_version'] not in (1,2) or old.get('dfs') is None:raise ValueError('Invalid legacy hybrid checkpoint.')
                state['sampling'],state['repair']=deepcopy(old.get('sampling')),deepcopy(old.get('repair'));old=old['dfs']
            state['phase']='exact_dfs';state['dfs']=deepcopy(old);state['resumes']=1
    else:
        if seed_board is not None:score(problem,seed_board);state['candidates'].append(list(map(int,seed_board)))
        if hints is not None and not isinstance(hints,dict):
            try:score(problem,hints);state['candidates'].append(list(map(int,hints)))
            except (TypeError,ValueError):pass
    def remaining():return max(0.,seconds-(time.monotonic()-began))
    def snapshot():
        if sampler is not None:state['sampler_state']=sampler.snapshot();state['sampling']=sampler.report()
        if repair is not None:state['repair_state']=repair.snapshot();state['repair']=repair.report()
        value=deepcopy(state);value['digest']=digest(value);return value
    def telemetry():return {'gpu_sampling':state['sampling'],'local_repair':state['repair'],'cpu_repair':state['repair'],'phase':state['phase'],
        'durable_resume':True,'resume_scope':'sampling_repair_and_exact_dfs','hybrid_resumes':state['resumes'],'hybrid_phase_transitions':state['transitions']}
    def emit(event):
        nonlocal last_save
        if progress is None:return
        if sampler is not None:state['sampling']=sampler.report()
        if repair is not None:state['repair']=repair.report()
        phase=state['phase'];item={**event,'phase':phase,'phase_elapsed_seconds':event.get('elapsed_seconds',0),'elapsed_seconds':time.monotonic()-began,**telemetry()}
        if item.get('event')=='checkpoint':
            native=item.get('payload',item.get('checkpoint'))
            if native is None:raise ValueError('Exact checkpoint event lacks a payload.')
            state['dfs']=deepcopy(native);item.pop('checkpoint',None);item['payload']=snapshot()
        progress(item)
        if phase!='exact_dfs' and time.monotonic()-last_save>=30:
            progress({'event':'checkpoint','payload':snapshot(),**telemetry()});last_save=time.monotonic()
    def transition(phase):
        state['phase']=phase;state['transitions']+=1
        if progress is not None:
            progress({'event':'checkpoint','payload':snapshot(),**telemetry()})
            progress({'event':'phase',**telemetry(),'elapsed_seconds':time.monotonic()-began})
    def certificate(values,phase):
        metrics=score(problem,values)
        if metrics['domain_violations'] or metrics['matched_edges']!=len(problem.edges):return None
        validation=problem.validate(values);state['checked']+=1
        if not (validation['valid'] or target=='edges' and validation['unique_tiles']):return None
        if progress is not None:progress({'event':'candidate','phase':phase,'codes':list(values),'validation':validation,'nodes':0,'elapsed_seconds':time.monotonic()-began})
        return {'status':'solved' if validation['valid'] else 'edge_perfect','codes':list(values),'validation':validation,'complete':True,'nodes':0,
                'stats':{'gpu_candidates_checked':state['checked']},'elapsed_seconds':time.monotonic()-began,'dfs_elapsed_seconds':0.,'method':'gpu_sampling_dfs',**telemetry()}
    def interrupted():
        checkpoint=snapshot()
        return {'status':'stopped' if stop() else 'timeout','complete':False,'nodes':0,'stats':{'gpu_candidates_checked':state['checked']},
                'elapsed_seconds':time.monotonic()-began,'dfs_elapsed_seconds':0.,'checkpoint':checkpoint,'method':'gpu_sampling_dfs',**telemetry()}
    def offer_claims():
        if not state['repair'] or target!='gold' or not options.get('smart_learning'):return
        from constraint_mask_conflicts import VerifiedMaskConflictStore
        store=learning_state.get('_mask_conflicts')
        if store is None:
            store=VerifiedMaskConflictStore(problem,target,proof_transport=options.get('proof_transport',False),resumable_proof_checking=options.get('resumable_proof_checking',False),mixed_proof_dag=options.get('mixed_proof_dag',False));learning_state['_mask_conflicts']=store
        claim_identity=digest([state['scope'],state['repair'].get('experimental_repair',[])])
        previous=learning_state.get('_hybrid_offered_claims')
        if isinstance(previous,tuple) and len(previous)==2 and previous[0] is store and previous[1]==claim_identity:return
        store.configure_policy(proof_transport=options.get('proof_transport',False),resumable_proof_checking=options.get('resumable_proof_checking',False),mixed_proof_dag=options.get('mixed_proof_dag',False))
        offered=0
        for trial in state['repair'].get('experimental_repair',[]):
            payload=trial.get('exclusions',{})
            if payload.get('kind')!='gold_repair_assumption_cores' or payload.get('fingerprint')!=problem.fingerprint or payload.get('domains')!=[list(row) for row in problem.domains]:continue
            for core in payload.get('cores',[]):
                if all(type(cell)is int and 0<=cell<problem.n and code in problem.domains[cell] for cell,code in core):offered+=bool(store.offer(tuple((cell,1<<code) for cell,code in core)))
        state['repair']['claims_queued_for_independent_proof']=offered;state['claims_offered']=True;learning_state['_hybrid_offered_claims']=(store,claim_identity)
    try:
        if stop() or not remaining():return interrupted()
        if state['phase']=='sampling':
            for values in state['candidates']:
                result=certificate(values,'saved_hint')
                if result:return result
            sampler=SamplerSession(problem,backend=backend,target=target,resume=state['sampler_state'],**state['sampling_policy'])
            emit({'event':'phase','budget_seconds':max(0.,state['budgets']['sampling']-sampler.elapsed)})
            sampler.advance(min(remaining(),max(0.,state['budgets']['sampling']-sampler.elapsed)),stop=stop,progress=emit)
            state['sampling']=sampler.report()
            if sampler.solution is not None:
                result=certificate(sampler.solution,'sampling')
                if result:return result
            if not sampler.done and sampler.elapsed<state['budgets']['sampling']:return interrupted()
            if stop():return interrupted()
            sampler.finish_phase();state['sampling']=sampler.report()
            if sampler.solution is not None:
                result=certificate(sampler.solution,'sampling')
                if result:return result
            state['candidates']=[row['codes'] for row in diverse_archive([{'codes':v,**score(problem,v)} for v in state['candidates']]+[{'codes':row['codes'],**score(problem,row['codes'])} for row in sampler.archive],capacity=8)]
            sampler.close();sampler=None;state['sampler_state']=None
            if state['candidates']:
                state['repair_inputs']=deepcopy(state['candidates']);state['hints']=state['candidates'][0].copy();transition('repair')
            else:transition('exact_dfs')
        if state['phase']=='repair':
            inputs=state['repair_inputs']
            repair=RepairSession(problem,inputs[0],phase_seconds=state['budgets']['repair'],target=target,candidate_boards=inputs[1:],resume=state['repair_state'],**state['repair_policy'])
            if stop() or not remaining():return interrupted()
            emit({'event':'phase','budget_seconds':max(0.,state['budgets']['repair']-repair.elapsed)})
            repair.advance(min(remaining(),max(0.,state['budgets']['repair']-repair.elapsed)),stop=stop,progress=emit)
            state['repair']=repair.report()
            result=certificate(repair.best,'repair')
            if result:return result
            if not repair.done and repair.elapsed<state['budgets']['repair']:return interrupted()
            if stop():return interrupted()
            repair.finish_phase();state['repair']=repair.report()
            if score(problem,repair.best)['objective']>=score(problem,state['hints'])['objective']:state['hints']=repair.best.copy()
            repair=None;state['repair_state']=None;offer_claims();transition('exact_dfs')
        if stop() or not remaining():return interrupted()
        offer_claims() # A fresh process must requeue saved claims before exact verification.
        result=exact_solve(problem,seconds=remaining(),seed=seed,target=target,hints=state['hints'],stop=stop,progress=emit,
                           resume=state['dfs'],options=options,learning_state=learning_state)
        result['stats']['gpu_candidates_checked']=state['checked'];result['dfs_elapsed_seconds']=result['elapsed_seconds']
        result['elapsed_seconds']=time.monotonic()-began;result.update(telemetry(),method='gpu_sampling_dfs')
        if 'checkpoint' in result:state['dfs']=result['checkpoint'];result['checkpoint']=snapshot()
        return result
    finally:
        if sampler is not None:sampler.close()
