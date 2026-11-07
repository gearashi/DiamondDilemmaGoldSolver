"""Fixed-dispatch sampler comparisons, with real fresh-process continuation.

All configurations use the same resumable controller, including the baseline.
This compares heuristic proposals and observed archive boards, never exhausted
search space. Per-dispatch costs differ; proposal counts are not CPU/GPU ops.
"""
from __future__ import annotations
import argparse,copy,datetime,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
from benchmark_provenance import source_snapshot,runtime_snapshot,repository_snapshot,require_assertions
from benchmark_round25 import require_fresh,write_artifact,terminate_tree

ROOT=Path(__file__).resolve().parent
PROFILES={'baseline':{},'ejection':{'ejection_chains':True},'tabu':{'move_tabu':True},
          'exchange':{'replica_exchange':True},'lazy':{'ejection_chains':True,'lazy_gold_scoring':True},'crossover':{'cycle_crossover':True},'guided':{'ejection_chains':True,'obstruction_chains':True},
          'legacy-combined':{'ejection_chains':True,'move_tabu':True,'replica_exchange':True}}


def continuation_signature(value):
    """Exclude elapsed wall time and explicit restore bookkeeping only."""
    from constraint_work import digest
    value=copy.deepcopy(value)
    for key in ('digest','elapsed'):value.pop(key,None)
    value['counts'].pop('sampler_restores',None)
    return digest(value)


def independently_check(problem,report,snapshot):
    from gpu_sampling import gold_score,score_board
    for item in report['top_samples']:
        expected=gold_score(problem,item['codes']) if report['gold_aware'] else score_board(problem,item['codes'])
        if item!={'codes':item['codes'],**expected}:raise ValueError('Sampler archive failed independent scoring.')
    solution=snapshot.get('solution')
    validation=problem.validate(solution) if solution is not None else None
    if validation is not None and not validation['valid']:raise ValueError('Sampler returned an invalid Gold solution.')
    return validation


def worker(args):
    require_assertions()
    from search_problem import SearchProblem
    from gpu_sampling_resume_round25 import SamplerSession
    output=Path(args.output);state_dir=output.with_suffix('.state')
    if output.suffix.lower()!='.json':raise ValueError('Worker output must be a fresh .json file.')
    require_fresh((output,state_dir))
    raw=Path(args.data).read_bytes();problem=SearchProblem(json.loads(raw.decode('utf-8-sig')))
    resume=None;resume_artifact=None
    if args.resume is not None:
        saved=Path(args.resume).read_bytes();resume=json.loads(saved)
        resume_artifact={'path':str(Path(args.resume).resolve()),'sha256':hashlib.sha256(saved).hexdigest(),'bytes':len(saved)}
    output.parent.mkdir(parents=True,exist_ok=True);state_dir.mkdir()
    started=time.monotonic();session=SamplerSession(problem,seed=args.seed,replicas=args.replicas,backend=args.backend,
        target='gold',gold_aware=args.objective=='gold',experiments=PROFILES[args.profile],resume=resume)
    try:
        initial=session.report();initial_dispatches=initial['dispatches']
        tick=time.monotonic();session.advance(args.limit_seconds,max_dispatches=args.dispatches)
        if not session.done and session.launches-initial_dispatches!=args.dispatches:
            raise RuntimeError('Fixed-work sampler hit its time bound before completing the requested dispatches.')
        if args.finish:session.finish_phase()
        search_seconds=time.monotonic()-tick
        snapshot=session.snapshot();report=session.report()
        validation=independently_check(problem,report,snapshot)
        artifact=write_artifact(state_dir/'sampler.json',snapshot)
        result={'status':'solved' if validation is not None else 'sampled' if args.finish else 'paused',
            'case':'original','profile':args.profile,'experiments':PROFILES[args.profile],
            'objective':args.objective,'seed':args.seed,'replicas':args.replicas,'requested_dispatches':args.dispatches,
            'initial_dispatches':initial_dispatches,'completed_dispatches':report['dispatches']-initial_dispatches,
            'cumulative_dispatches':report['dispatches'],'finish_phase':args.finish,'report':report,
            'independent_validation':validation,'continuation_signature':continuation_signature(snapshot),
            'runtime_binding':snapshot['native']['binding'] if snapshot['native'] is not None else None,
            'snapshot_artifact':artifact,'resume_artifact':resume_artifact,
            'input_sha256':hashlib.sha256(raw).hexdigest(),'fingerprint':problem.fingerprint,
            'search_seconds':search_seconds,'worker_wall_seconds':time.monotonic()-started,
            'scope':'Fixed 64-move dispatches using the durable sampler controller; archive repeats are observations, not all proposals or exhaustive coverage.'}
        write_artifact(output,result)
        print(json.dumps({key:result[key] for key in ('status','profile','objective','replicas','completed_dispatches','search_seconds')}),flush=True)
    finally:session.close()


def matrix_configurations(profiles,objectives,replicas,seeds):
    configs=[(name,objective,count) for objective in dict.fromkeys(objectives) for count in dict.fromkeys(replicas) for name in dict.fromkeys(profiles)]
    return [(seed,*config) for i,seed in enumerate(dict.fromkeys(seeds)) for config in (configs if i%2==0 else list(reversed(configs)))]


def validate_pair(uninterrupted,prefix,resumed):
    if uninterrupted['continuation_signature']!=resumed['continuation_signature']:
        raise ValueError('Fresh-process resume changed the final native/controller/archive state.')
    expected=prefix['snapshot_artifact']
    if resumed['resume_artifact']!=expected:raise ValueError('Resume did not consume the exact saved sampler artifact.')
    if resumed['initial_dispatches']!=prefix['cumulative_dispatches']:
        raise ValueError('Resume repeated or lost completed dispatches.')
    if prefix['completed_dispatches']+resumed['completed_dispatches']!=uninterrupted['completed_dispatches']:
        raise ValueError('Resume dispatch accounting differs from uninterrupted work.')
    if resumed['report']['sampler_restores']!=1:
        # A solved prefix requires no native continuation at all.
        if prefix['status']!='solved' or resumed['report']['sampler_restores']!=0:
            raise ValueError('Unexpected native restore accounting.')
    return {'same_final_state':True,'same_completed_dispatches':True,'completed_dispatches_replayed':0,
            'resume_artifact_sha256':expected['sha256'],
            'continuation_signature':resumed['continuation_signature'],
            'scope':'Same device, runtime, input and policy; elapsed times and restore counters differ.'}


def launch(args,path,*,profile,objective,seed,replicas,dispatches,finish,resume=None):
    require_fresh((path,path.with_suffix('.state'),path.with_suffix('.stdout.txt'),path.with_suffix('.stderr.txt')))
    command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--output',str(path.resolve()),
        '--data',str(args.data.resolve()),'--backend',args.backend,'--profile',profile,'--objective',objective,
        '--seed',str(seed),'--replicas',str(replicas),'--dispatches',str(dispatches),'--limit-seconds',str(args.limit_seconds)]
    if finish:command.append('--finish')
    if resume is not None:command.extend(('--resume',str(Path(resume).resolve())))
    began=time.monotonic();utc=datetime.datetime.now(datetime.timezone.utc).isoformat();child=None;cleanup_attempted=False
    try:
        child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),start_new_session=os.name!='nt',
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        try:stdout,stderr=child.communicate(timeout=args.limit_seconds+120)
        except subprocess.TimeoutExpired:
            cleanup_attempted=True;cleanup=terminate_tree(child);stdout,stderr=child.communicate(timeout=30)
            path.with_suffix('.stdout.txt').write_text(stdout,encoding='utf-8');path.with_suffix('.stderr.txt').write_text(stderr,encoding='utf-8')
            raise RuntimeError('Sampler benchmark timed out; no further cases started. '+cleanup)
        path.with_suffix('.stdout.txt').write_text(stdout,encoding='utf-8');path.with_suffix('.stderr.txt').write_text(stderr,encoding='utf-8')
        if child.returncode or not path.is_file():
            raise RuntimeError('Sampler benchmark child failed; no further cases started. '+stderr[-1500:])
        result=json.loads(path.read_text(encoding='utf-8'))
        result.update(started_at=utc,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            process_exit_code=child.returncode,coordinator_wall_seconds=time.monotonic()-began)
        return result
    except BaseException as error:
        # A POSIX child group can outlive an exited coordinator. Attempt cleanup
        # even when JSON parsing or result validation fails after return code0.
        if child is not None and not cleanup_attempted:
            cleanup_attempted=True
            try:error.add_note(terminate_tree(child))
            except Exception as cleanup_error:error.add_note('Owned child cleanup failed: '+str(cleanup_error))
        raise
    finally:
        if child is not None and not cleanup_attempted and child.poll() is None:terminate_tree(child)


def main():
    require_assertions()
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--data',type=Path,default=ROOT/'data/tiles.json');p.add_argument('--backend',choices=('cuda','webgpu'),default='cuda')
    p.add_argument('--dispatches',type=int,default=18,help='64moves per dispatch; default18 splits after9, exercising an uncollected archive tail.');p.add_argument('--limit-seconds',type=float,default=600)
    p.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES));p.add_argument('--objectives',nargs='+',choices=('edges','gold'),default=['edges','gold'])
    p.add_argument('--replica-counts',nargs='+',type=int,default=[128,4096]);p.add_argument('--seeds',nargs='+',type=int,default=[71,197])
    p.add_argument('--resume-comparison',action='store_true');p.add_argument('--evidence-stage',choices=('development','frozen'),default='development')
    p.add_argument('--worker',action='store_true');p.add_argument('--profile',choices=PROFILES,default='baseline');p.add_argument('--objective',choices=('edges','gold'),default='edges')
    p.add_argument('--seed',type=int,default=71);p.add_argument('--replicas',type=int,default=128);p.add_argument('--resume',type=Path);p.add_argument('--finish',action='store_true')
    args=p.parse_args()
    if not 1<=args.dispatches<=1_000_000:p.error('dispatches must be within1..1000000.')
    if not math.isfinite(args.limit_seconds) or not 0<args.limit_seconds<=3600:p.error('limit-seconds must be positive and at most3600.')
    if any(not 1<=value<=131072 for value in [args.replicas,*args.replica_counts]):p.error('replicas must be within1..131072.')
    if any(value<0 for value in [args.seed,*args.seeds]):p.error('seeds must be nonnegative.')
    if args.worker:return worker(args)
    if args.resume_comparison and args.dispatches<2:p.error('resume comparison needs at least2dispatches.')
    require_fresh((args.output/'report.json',args.output/'results.json'))
    before=source_snapshot(ROOT);runtime_before=runtime_snapshot();repository=repository_snapshot(ROOT)
    if args.evidence_stage=='frozen' and (not repository.get('commit') or repository.get('dirty') is not False):
        raise RuntimeError('Frozen benchmark evidence requires a clean committed checkout.')
    raw=args.data.read_bytes();input_hash=hashlib.sha256(raw).hexdigest();args.output.mkdir(parents=True,exist_ok=True)
    configs=matrix_configurations(args.profiles,args.objectives,args.replica_counts,args.seeds)
    for number,config in enumerate(configs):
        for label in ('uninterrupted','prefix','resumed') if args.resume_comparison else ('uninterrupted',):
            path=args.output/f'{number:03d}-{label}.json'
            require_fresh((path,path.with_suffix('.state'),path.with_suffix('.stdout.txt'),path.with_suffix('.stderr.txt')))
    rows=[]
    for number,(seed,profile,objective,replicas) in enumerate(configs):
        kwargs=dict(profile=profile,objective=objective,seed=seed,replicas=replicas)
        full=launch(args,args.output/f'{number:03d}-uninterrupted.json',dispatches=args.dispatches,finish=True,**kwargs)
        row={'configuration':kwargs,'uninterrupted':full}
        if args.resume_comparison:
            first=args.dispatches//2
            prefix=launch(args,args.output/f'{number:03d}-prefix.json',dispatches=first,finish=False,**kwargs)
            restored=launch(args,args.output/f'{number:03d}-resumed.json',dispatches=args.dispatches-first,finish=True,
                resume=prefix['snapshot_artifact']['path'],**kwargs)
            row.update(prefix=prefix,resumed=restored,comparison=validate_pair(full,prefix,restored))
        for record in [row[key] for key in ('uninterrupted','prefix','resumed') if key in row]:
            if record['input_sha256']!=input_hash:raise RuntimeError('Sampler benchmark input changed between workers.')
        rows.append(row);(args.output/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({**kwargs,'best_score':full['report']['best_score'],'proposals':full['report']['move_proposals'],
            'resume_equivalent':row.get('comparison',{}).get('same_final_state')}),flush=True)
    after=source_snapshot(ROOT);runtime_after=runtime_snapshot();repository_after=repository_snapshot(ROOT)
    if before!=after or runtime_before!=runtime_after:raise RuntimeError('Source or runtime changed; no accepted benchmark report.')
    if args.evidence_stage=='frozen' and repository!=repository_after:raise RuntimeError('Frozen Git identity changed; no accepted benchmark report.')
    if hashlib.sha256(args.data.read_bytes()).hexdigest()!=input_hash:raise RuntimeError('Input changed; no accepted benchmark report.')
    report={'evidence_stage':args.evidence_stage,'repository':repository,'repository_after':repository_after,
        'source_before':before,'source_after':after,'runtime_before':runtime_before,'runtime_after':runtime_after,
        'input_sha256':input_hash,'backend':args.backend,'profiles':{name:PROFILES[name] for name in dict.fromkeys(args.profiles)},
        'dispatches':args.dispatches,'resume_comparison':args.resume_comparison,'planned_cases':len(configs),'completed_cases':len(rows),
        'results':rows,'scope':'Original-input heuristic sampling using the same durable controller in every profile. Samples and archive repeats are not unique-position counts or exact search coverage.',
        'timing_scope':'Fresh worker processes include GPU setup/compilation; no wall-time speedup claim without separate warmup-controlled measurements.',
        'non_claim':'Bounded stochastic comparisons do not identify a universally best algorithm or prove Gold infeasibility.'}
    write_artifact(args.output/'report.json',report)

if __name__=='__main__':main()
