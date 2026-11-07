"""R25 serial finite/full-board comparisons with scoped work accounting.

CPU/GPU-domain/SAT routes validate unfinished Cartesian volume. The separate
hybrid pipeline uses DFS symmetry and never reports that volume metric.
Budgets are diagnostic only. Unknown/time-limited searches never prove the Gold
puzzle infeasible. Engine node counters have different units and are not ranked.
A child failure stops the matrix immediately. Cleanup targets the owned POSIX
process group or a live Windows task tree; after a Windows coordinator exits,
this harness cannot establish that every descendant is gone. It never claims a
universal orphan guarantee or starts another case after that ownership failure.
"""
from __future__ import annotations
import argparse,datetime,hashlib,json,math,os,platform,signal,subprocess,sys,time
from pathlib import Path
from benchmark_provenance import source_snapshot, runtime_snapshot, repository_snapshot, require_assertions
from benchmark_gpu_profiles import GPU_PROFILES,warmup_domain
from benchmark_checkpoint_publication import CheckpointPublisher
from benchmark_pipeline_profiles import HYBRID_PROFILES,DEFAULT_HYBRID_PROFILES,HYBRID_SLICE_SECONDS
ROOT=Path(__file__).resolve().parent
PROFILES={
 'baseline':{},
 'reuse':dict(resumable_reasoning=True,incremental_hall=True,hall_rematch=True,reversible_gold_graph=True,shared_coverage=True,symbolic_coverage=True,queue_pruning=True),
 'sat-learning':dict(resumable_reasoning=True,sat_assist=True,witness_pool=True,retained_relations=True,shared_sat_assistant=True,higher_order_relations=True),
 'arithmetic':dict(joint_conservation=True,affine_reasoning=True,incremental_affine=True,pb_learning=True,conflict_pb=True,integer_lattice=True),
 'regional-v24':dict(symbolic_regions=True,symbolic_inventory=True,elimination=True,mini_buckets=True,relaxed_diagrams=True,representative_connectivity=True,subset_joins=True,lazy_regions=True,certified_relaxation=True,labelled_routing=True),
 'regional':dict(incremental_regions=True,durable_regions=True,compact_factors=True,gold_elimination=True,separator_search=True,selective_refinement=True,joint_symbolic_regions=True,algebraic_connectivity=True,residual_isomorphism=True,gold_pricing=True,multichain_routing=True,region_certificates=True),
}
PROFILES['combined']={**PROFILES['reuse'],**PROFILES['sat-learning'],**PROFILES['arithmetic'],**PROFILES['regional'],
                      'native_explanations':True,'mixed_proof_dag':True}

SAT_PROFILES={
 'baseline':dict(persistent_sat=True,native_sat_persistence=False),
 'lifted':dict(persistent_sat=True,native_sat_persistence=False,sat_core_lifting=True,shared_coverage=True,symbolic_coverage=True),
}
SAT_PARTITIONS=32
SAT_SLICE_SECONDS=30


def source_hashes():return source_snapshot(ROOT)
def volume(rows):return sum(math.prod(x.bit_count() for x in row) for row in rows)
def pending_frontier(scope,result,engine,*,checkpoint_dir=None):
 """Validate committed ownership before counting any removed assignments."""
 from constraint_work import WorkScope,checkpoint_boxes,frontier_snapshot,restore_frontier
 if result.get('status') in ('solved','edge_perfect'):return None
 saved=result.get('checkpoint')
 if saved is None:
  if result.get('status')=='infeasible' and result.get('complete') is True:return ()
  raise ValueError('An incomplete benchmark result has no committed remaining frontier.')
 if engine in ('gpu','sat'):
  if engine=='gpu':
   from gpu_checkpoint_round25 import checkpoint_frontier,is_reference
   if is_reference(saved) and checkpoint_dir is None:
    raise ValueError('A referenced GPU checkpoint needs its exact local store.')
   frontier=checkpoint_frontier(saved,checkpoint_dir)
  else:frontier=saved['frontier']
  if not isinstance(frontier,dict):raise ValueError('Benchmark checkpoint has no portable remaining frontier.')
  if WorkScope.restore(frontier['scope'])!=scope:
   raise ValueError('Benchmark remaining frontier changed its original owned scope.')
  rows=restore_frontier(frontier,scope)
 else:rows=tuple(row for job in saved['jobs'] for row in checkpoint_boxes(job['checkpoint']))
 if result.get('status')=='infeasible':
  if result.get('complete') is not True or rows:
   raise ValueError('Infeasibility requires complete exhaustion of the committed frontier.')
 elif not rows:
  raise ValueError('An incomplete benchmark result has an empty remaining frontier.')
 # All CPU cases start from the same original portable root, without adding a
 # canonical symmetry restriction. Scope/disjointness errors invalidate metrics.
 return restore_frontier(frontier_snapshot(scope,rows),scope)


def initial_frontier(scope,engine):
 """Partition the same full original space; native SAT needs several cubes."""
 from constraint_work import frontier_snapshot,restore_frontier
 rows=(scope.root,)
 if engine=='sat':
  from gpu_domain_runner import split_cubes
  rows=tuple(split_cubes(rows,SAT_PARTITIONS))  # Pure host-side exact partition.
 payload=frontier_snapshot(scope,rows)
 checked=restore_frontier(payload,scope)
 if volume(checked)!=volume((scope.root,)):
  raise ValueError('Initial benchmark partition does not cover the original root.')
 return payload


def require_fresh(paths):
 for value in paths:
  path=Path(value)
  if path.exists() or path.is_symlink():
   raise FileExistsError('Benchmark output/state already exists; choose a fresh path: '+str(path))


def write_artifact(path,payload):
 """Exclusive artifact creation never replaces source, input, or prior work."""
 data=(json.dumps(payload,indent=2,allow_nan=False)+'\n').encode('utf-8')
 with Path(path).open('xb') as stream:stream.write(data)
 return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}


def run_sat(problem,*,seconds,seed,initial,options,learning_state):
 """Retain native state between cooperative slices, then always release it."""
 from constraint_sat_session import solve
 started=time.monotonic();remaining=seconds;resume=initial;slices=0
 try:
  while True:
   result=solve(problem,seconds=min(SAT_SLICE_SECONDS,remaining),seed=seed,target='gold',
       resume=resume,options=options,learning_state=learning_state)
   slices+=1
   if result.get('status')!='timeout':break
   remaining=max(0.0,seconds-(time.monotonic()-started))
   if remaining<=0:break
   resume=result.get('checkpoint')
   if resume is None:raise ValueError('A timed SAT slice lost its remaining frontier.')
  result['benchmark_sat_slices']=slices
  return result
 finally:
  session=learning_state.get('_sat_session')
  if session is not None:session.close()



def _hybrid_checkpoint(problem,payload):
 """Check the local controller envelope, without claiming portable coverage."""
 from constraint_work import digest
 expected={'fingerprint':problem.fingerprint,'domains':[list(row) for row in problem.domains],
     'fixed':[[int(cell),int(code)] for cell,code in sorted(problem.fixed.items())],'target':'gold'}
 if (not isinstance(payload,dict) or type(payload.get('hybrid_version'))is not int or payload['hybrid_version']!=3
     or payload.get('scope')!=expected or payload.get('phase') not in ('sampling','repair','exact_dfs')
     or payload.get('digest')!=digest({key:value for key,value in payload.items() if key!='digest'})):
  raise ValueError('Hybrid benchmark lost its bound version3 continuation.')
 return payload


def run_hybrid(problem,*,seconds,seed,replicas,backend,options,learning_state,publication):
 """Run bounded app-like slices; retain controller/native state and publish I/O."""
 from gpu_sampling import solve
 if type(seconds)not in (int,float) or not math.isfinite(seconds) or not 0<seconds<=3600:
  raise ValueError('Hybrid benchmark requires a positive budget of at most one hour.')
 if type(replicas)is not int or not 1<=replicas<=131072 or backend not in ('auto','cuda','webgpu'):
  raise ValueError('Invalid hybrid benchmark GPU policy.')
 if options.get('durable_hybrid') is not True:raise ValueError('All hybrid profiles require durable_hybrid.')
 if options.get('adaptive_temperatures') and replicas<8:raise ValueError('Adaptive temperatures require at least8 replicas.')
 learning_state['_hybrid_run_seconds']=seconds
 started=time.monotonic();remaining=seconds;resume=None;history=[]
 try:
  while True:
   result=solve(problem,seconds=min(HYBRID_SLICE_SECONDS,remaining),seed=seed,target='gold',replicas=replicas,
       backend=backend,resume=resume,options=options,learning_state=learning_state,
       progress=publication.progress,stop=publication.failed)
   status=result.get('status');saved=result.get('checkpoint')
   if status in ('timeout','stopped'):
    _hybrid_checkpoint(problem,saved)
   elif status=='infeasible':
    if result.get('complete') is not True or result.get('phase')!='exact_dfs':
     raise ValueError('Hybrid infeasibility requires completed exact DFS, not a heuristic timeout.')
   elif status not in ('solved','edge_perfect'):
    raise ValueError('Hybrid benchmark returned an unsuccessful status: '+str(status))
   if saved is not None:
    _hybrid_checkpoint(problem,saved);publication.publish(saved)
   history.append({'status':status,'phase':result.get('phase'),
       'gpu_dispatches':(result.get('gpu_sampling') or {}).get('dispatches',0),
       'repair_native_calls':((result.get('local_repair') or {}).get('stats') or {}).get('repair_native_calls',0),
       'exact_nodes':result.get('nodes',0)})
   if status!='timeout' or publication.failed():break
   remaining=max(0.,seconds-(time.monotonic()-started))
   if remaining<=0:break
   resume=saved
  result['benchmark_hybrid_slices']=len(history);result['hybrid_slice_history']=history
  result['hybrid_phase_policy']={'whole_run_seconds':seconds,'slice_seconds':HYBRID_SLICE_SECONDS,
      'sampling_fraction':.2,'repair_fraction':.3,'sampling_cap_seconds':30.,'repair_cap_seconds':60.,
      'durable_hybrid':True,'budget_allocation':'Allocated once; version3 checkpoints retain consumed phase budgets.'}
  return result
 finally:
  # Retain helpers across slices, release every known native owner on success
  # and failure. Several names may reference the same verifier.
  runtime=learning_state.get('_runtime');owners=[learning_state.get(key) for key in
      ('_sat_session','_shared_sat','_shared_conflicts','_mask_conflicts')]
  if runtime is not None:owners.extend(getattr(runtime,key,None) for key in ('sat','higher','gold_oracle','mask_conflicts','local_mask_conflicts'))
  seen=set();errors=[]
  for owner in owners:
   if owner is None or id(owner) in seen or not callable(getattr(owner,'close',None)):continue
   seen.add(id(owner))
   try:owner.close()
   except BaseException as error:errors.append(error)
  if errors:
   active=sys.exc_info()[1]
   if active is not None:active.add_note('Hybrid helper cleanup also failed: '+repr(errors[0]))
   else:raise RuntimeError('Hybrid helper cleanup failed.') from errors[0]


def hybrid_work(result):
 """Preserve distinct cumulative units; never sum heterogeneous search work."""
 sampling=result.get('gpu_sampling') or {};repair=result.get('local_repair') or {}
 return {'gpu_sampling':{key:sampling.get(key,0) for key in
      ('initial_boards','move_proposals','dispatches','samples_considered','archive_checks','archive_repeated_observations','sampler_restores')},
     'sampling_experiments':sampling.get('sampling_experiments',{}),
     'local_repair':{'stats':repair.get('stats',{}),'experimental_repair':repair.get('experimental_repair',[])},
     'exact_dfs':{'nodes':result.get('nodes',0),'stats':{key:value for key,value in result.get('stats',{}).items() if key!='gpu_candidates_checked'}},
     'independent_gold_candidates_checked':result.get('stats',{}).get('gpu_candidates_checked',0),
     'interpretation':'Cumulative GPU move proposals, local logical/native queries, and DFS nodes are separate units; no total-position or closed-root-volume metric.'}


def terminate_tree(child):
 """Request owned-group cleanup and reap the coordinator, with explicit limits."""
 if os.name=='nt':
  if child.poll() is not None:return 'Windows coordinator already exited; descendant cleanup cannot be confirmed without a retained tree handle.'
  try:
   stopped=subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],capture_output=True,
       timeout=30,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
  except subprocess.TimeoutExpired as exc:
   raise RuntimeError('Benchmark worker-tree cleanup timed out; no further case may run.') from exc
  if stopped.returncode:
   raise RuntimeError('Benchmark worker-tree cleanup failed; no further case may run.')
 else:
  # The coordinator may have exited while children still hold its session's
  # process-group ID. Attempt that owned group even after poll() reports exit.
  try:os.killpg(child.pid,signal.SIGKILL)
  except ProcessLookupError:
   if child.poll() is None:raise RuntimeError('Benchmark process group disappeared before cleanup was confirmed.')
   return 'Owned POSIX process group no longer exists.'
 try:child.wait(timeout=30)
 except subprocess.TimeoutExpired as exc:
  raise RuntimeError('Benchmark child remained alive after tree cleanup; no further case may run.') from exc
 return 'Termination requested for owned group/tree; coordinator reaped. Escaped descendants are not tracked.'


def worker(args):
 require_assertions()
 from search_options import normalize_options,validate_algorithm_options
 algorithm={'cpu':'parallel-dfs','gpu':'gpu-domains','sat':'sat','hybrid':'hybrid'}.get(args.engine)
 if algorithm is None:raise ValueError('Unknown benchmark engine.')
 profiles=HYBRID_PROFILES if args.engine=='hybrid' else SAT_PROFILES if args.engine=='sat' else PROFILES if args.engine=='cpu' else GPU_PROFILES
 if args.profile not in profiles:raise ValueError('Profile is unavailable for benchmark engine: '+args.engine)
 overrides=profiles[args.profile]
 options=normalize_options({'native':'compiled',**overrides} if args.engine in ('cpu','hybrid') else overrides)
 validate_algorithm_options(algorithm,options)
 args.output=Path(args.output)
 if args.output.suffix.lower()!='.json':raise ValueError('Benchmark worker output must be a fresh .json file.')
 state_dir=args.output.with_suffix('.state')
 require_fresh((args.output,state_dir))
 from search_problem import SearchProblem
 from constraint_work import WorkScope,digest
 provenance={'kind':'original_puzzle_file' if args.case=='original' else 'synthetic_planted_fixture'}
 if args.case=='original':
  source=args.data.read_bytes();provenance['input_sha256']=hashlib.sha256(source).hexdigest()
  problem=SearchProblem(json.loads(source.decode('utf-8-sig')))
 else:
  from search_fixtures import planted_gold
  data,board,reference=planted_gold(args.seed,size=1 if args.case=='small' else 4,extra_edges=3 if args.case=='small' else 12)
  if args.case in ('small','residual'):
   # Remove unique endpoint fingerprints while retaining the exact planted loop.
   for tile in data['tiles']:
    points={side:sorted({p for segment in tile['segments'] for s,p in segment if s==side}) for side in range(3)}
    tile['segments']=[[[s,(2,10)[points[s].index(p)]] for s,p in segment] for segment in tile['segments']]
  fixed=None
  if args.case=='residual':
   import random
   cells=list(range(len(reference)));random.Random(args.seed+9101).shuffle(cells)
   fixed={cell:reference[cell] for cell in cells[args.free_cells:]}
  problem=SearchProblem(data,board,fixed=fixed)
  if args.case in ('small','residual'):
   import random
   rng=random.Random(args.seed+900)
   problem.domains=[[code for code in row if code==reference[cell] or rng.random()<.7] for cell,row in enumerate(problem.domains)]
  assert problem.validate(reference)['valid']
  assert all(code in problem.domains[cell] for cell,code in enumerate(reference))
  provenance.update(generator='search_fixtures.planted_gold',fixture_seed=args.seed,board_size=1 if args.case=='small' else 4,
      extra_edges=3 if args.case=='small' else 12,collapsed_endpoint_pairs=args.case in ('small','residual'),
      nonreference_retention_probability=.7 if args.case in ('small','residual') else None,
      requested_free_cells=args.free_cells if args.case=='residual' else None,
      planted_reference_sha256=digest(reference),planted_reference_independently_validated=True,
      tile_data_sha256=digest(data),fixed_placements_sha256=digest(sorted(problem.fixed.items())))
 scope=WorkScope.for_problem(problem,'gold');initial=None if args.engine=='hybrid' else initial_frontier(scope,args.engine);began=time.monotonic()
 args.output.parent.mkdir(parents=True,exist_ok=True)
 state_dir.mkdir()  # Exclusive case ownership also rejects concurrent reuse.
 initial_info=None;pipeline_input=None
 if args.engine=='hybrid':
  pipeline_input=write_artifact(state_dir/'pipeline-input.json',{'fingerprint':problem.fingerprint,'target':'gold',
      'domains':[list(row) for row in problem.domains],'fixed':sorted(problem.fixed.items()),
      'search_contract':'Fresh heuristic phases followed by DFS using its own symmetry rules; no portable root is consumed and no raw-volume comparison is claimed.'})
 else:
  initial_artifact=write_artifact(state_dir/'initial-frontier.json',initial)
  initial_info={**initial_artifact,'digest':initial['digest'],'scope':scope.payload(),
      'requested_boxes':SAT_PARTITIONS if args.engine=='sat' else 1,'actual_boxes':len(initial['boxes']),
      'cartesian_volume':str(volume((scope.root,))),'same_full_original_root':True,
      'partition_method':'gpu_domain_runner.split_cubes' if args.engine=='sat' else 'single original root'}
 coverage_path=state_dir/'coverage.sqlite';regional_path=state_dir/'regional';regional_path.mkdir()
 if args.engine=='cpu':
  from constraint_parallel import solve
  result=solve(problem,seconds=args.seconds,seed=args.seed,target='gold',workers=args.workers,resume=initial,
      options=options,learning_state={'_coverage_path':coverage_path,'_regional_cache_dir':regional_path})
 elif args.engine=='hybrid':
  solver_began=time.monotonic()
  with CheckpointPublisher(state_dir/'published-checkpoint.json') as publication:
   result=run_hybrid(problem,seconds=args.seconds,seed=args.seed,replicas=args.replicas,backend=getattr(args,'hybrid_backend','auto'),
       options=options,learning_state={'_coverage_path':coverage_path,'_regional_cache_dir':regional_path},publication=publication)
  result['checkpoint_publication']=publication.report();result['solver_wall_seconds']=time.monotonic()-solver_began
  result['pipeline_work']=hybrid_work(result)
 elif args.engine=='gpu':
  from gpu_domain_runner import solve
  warmup=warmup_domain(problem,options)
  solver_began=time.monotonic()
  with CheckpointPublisher(state_dir/'published-checkpoint.json') as publication:
   result=solve(problem,seconds=args.seconds,seed=args.seed,target='gold',replicas=args.replicas,work_frontier=initial,backend='cuda',
       options=options,learning_state={'_coverage_path':coverage_path,'_gpu_checkpoint_dir':state_dir/'gpu-checkpoints'},
       progress=publication.progress,stop=publication.failed)
   publication.publish(result.get('checkpoint'))
  result['checkpoint_publication']=publication.report()
  result['gpu_warmup']=warmup
  result['solver_wall_seconds']=time.monotonic()-solver_began
 else:
  result=run_sat(problem,seconds=args.seconds,seed=args.seed,initial=initial,options=options,
      learning_state={'_coverage_path':coverage_path})
 status=result['status']
 if status in ('solved','edge_perfect') and result.get('codes') is None:
  raise ValueError('A solved benchmark result has no independently checkable witness.')
 if result.get('codes') is not None:
  codes=result['codes']
  if len(codes)!=problem.n or any(type(code)is not int or code not in problem.domains[cell] for cell,code in enumerate(codes)):
   raise ValueError('Benchmark witness escapes original domains.')
  check=problem.validate(codes)
  if not check['valid']:raise ValueError('Benchmark witness failed independent Gold validation.')
  result['independent_validation']=check
 pending=None if args.engine=='hybrid' else pending_frontier(scope,result,args.engine,checkpoint_dir=state_dir/'gpu-checkpoints')
 # A solved result is a witness, not an enumeration completion: no space claim.
 if args.engine!='hybrid' and status not in ('solved','edge_perfect'):
  remaining=volume(pending);total=volume([scope.root])
  if not 0<=remaining<=total:raise ValueError('Invalid committed remaining volume.')
  result['raw_space']={'initial':str(total),'remaining':str(remaining),'closed':str(total-remaining),'closed_log10':math.log10(total-remaining) if total>remaining else None,'definition':'Cartesian orientation assignments absent from the validated committed frontier of the same original unsymmetrized root; not a count of legal boards.'}
 if result.get('checkpoint') is not None:
  result['checkpoint_artifact']=write_artifact(state_dir/'checkpoint.json',result['checkpoint'])
  if args.engine=='gpu':
   from benchmark_checkpoint_artifacts import retained_store_snapshot,reopen_gpu_checkpoint
   result['checkpoint_reopen']=reopen_gpu_checkpoint(result['checkpoint_artifact'],state_dir/'gpu-checkpoints',scope)
   dependencies=retained_store_snapshot(state_dir/'gpu-checkpoints')
   if dependencies is not None:result['checkpoint_dependencies_artifact']=write_artifact(state_dir/'checkpoint-dependencies.json',dependencies)
 for key in ('checkpoint','knowledge','shared_conflicts','mask_conflicts','local_mask_conflicts','certificates'):result.pop(key,None)
 if args.engine=='sat':
  stats=result.get('stats',{})
  result['sat_work']={key:stats.get(key,0) for key in ('session_decisions','session_core_decisions','session_total_decisions','session_model_builds','session_core_model_builds','session_native_calls','session_cubes_closed','session_core_candidates','session_core_verifications','session_core_verified','session_core_unknown','session_core_boxes_removed','session_core_assignments_removed')}
  result['sat_work']['interpretation']='Native search and independent core-verification decisions are separate; nodes includes both. Closed Cartesian assignments are not legal-board counts.'
 result.update(case=args.case,engine=args.engine,solver_engine=algorithm,
     profile=args.profile,seed=args.seed,budget_seconds=args.seconds,workers=args.workers if args.engine=='cpu' else 1 if args.engine in ('sat','hybrid') else 0,replicas=args.replicas if args.engine in ('gpu','hybrid') else 0,
     cells=problem.n,fixed_cells=len(problem.fixed),free_cells=args.free_cells if args.case=='residual' else None,
     fingerprint=problem.fingerprint,domains_sha256=digest(scope.original),
     provenance=provenance,options=options,profile_overrides=overrides,
     solver_seed_effect='not used; repeated measurements on original input' if args.engine in ('gpu','sat') else 'seeded search ordering',
     fixture_seed_effect='none on original input' if args.case=='original' else 'seed changes the generated finite fixture',
     node_interpretation={
      'cpu':'Search-node evaluations in the shared Python/native controller, including native fast rejections; not a count of all CPU operations.',
      'hybrid':'Exact DFS search-node evaluations only; GPU proposals and repair work are reported separately.',
      'gpu':'Native GPU-domain branch decisions; not comparable directly with CPU evaluation counts.',
      'sat':'Native SAT search and independent core-verification decisions; see the separate SAT work counters.'}[args.engine],
     evidence_scope='Worker result only; frozen claims require its unchanged-source coordinator report.',
     benchmark_state={'directory':str(state_dir),'coverage_enabled':bool(options.get('shared_coverage')),
                      'coverage_file_created':coverage_path.exists(),'regional_directory_supplied':args.engine in ('cpu','hybrid')},
     wall_seconds=time.monotonic()-began)
 if args.engine=='hybrid':
  result['pipeline_input']=pipeline_input;result['hybrid_backend_requested']=getattr(args,'hybrid_backend','auto')
  result['solver_seed_effect']='Seeded GPU proposals, repair choices and exact DFS ordering; equal wall budgets need not reach equal phases.'
 else:result['initial_frontier']=initial_info
 write_artifact(args.output,result)
 print(json.dumps({k:result.get(k) for k in ('case','engine','profile','seed','status','nodes','wall_seconds')}),flush=True)

def main():
 require_assertions()
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data',type=Path,default=ROOT/'data/tiles.json')
 parser.add_argument('--worker',action='store_true');parser.add_argument('--engine',choices=['cpu','gpu','sat','hybrid'],default='cpu');parser.add_argument('--case',choices=['original','planted','small','residual'],default='original');parser.add_argument('--profile',choices=tuple(dict.fromkeys((*PROFILES,*SAT_PROFILES,*GPU_PROFILES,*HYBRID_PROFILES))),default='baseline')
 parser.add_argument('--seconds',type=float,default=60);parser.add_argument('--seed',type=int,default=71);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--replicas',type=int,default=512);parser.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES));parser.add_argument('--include-gpu',action='store_true')
 parser.add_argument('--suite',choices=['finite','original','gpu','sat','hybrid','all'],default='all');parser.add_argument('--free-cells',type=int,default=60)
 parser.add_argument('--finite-cases',nargs='+',choices=('small','residual'),default=['small','residual'],help='Finite cases; small runs once per profile/seed, independently of residual depths.')
 parser.add_argument('--finite-seconds',type=float,default=120,help='Bounded per-case finite search budget.')
 parser.add_argument('--free-cell-counts',nargs='+',type=int,help='Serial residual depths; defaults to --free-cells.')
 parser.add_argument('--worker-counts',nargs='+',type=int,help='Serial original-puzzle comparisons at each CPU worker count.')
 parser.add_argument('--gpu-profiles',nargs='+',choices=GPU_PROFILES,default=list(GPU_PROFILES))
 parser.add_argument('--gpu-cases',nargs='+',choices=('small','residual','original'),default=['original'])
 parser.add_argument('--replica-counts',nargs='+',type=int,help='Serial GPU-domain lane counts; defaults to --replicas, within1..512.')
 parser.add_argument('--sat-profiles',nargs='+',choices=SAT_PROFILES,default=list(SAT_PROFILES))
 parser.add_argument('--sat-cases',nargs='+',choices=('small','residual','original'),default=['small','residual','original'],help='Dedicated SAT suite, the same deterministic partition of up to32 boxes for both profiles.')
 parser.add_argument('--hybrid-profiles',nargs='+',choices=HYBRID_PROFILES,default=list(DEFAULT_HYBRID_PROFILES))
 parser.add_argument('--hybrid-cases',nargs='+',choices=('small','residual','original'),default=['small','residual','original'],help='Dedicated opt-in full hybrid pipeline; never included in --suite all.')
 parser.add_argument('--hybrid-replica-counts',nargs='+',type=int,help='Serial hybrid sampling replica counts,1..131072; defaults to --replicas.')
 parser.add_argument('--hybrid-backend',choices=('auto','cuda','webgpu'),default='auto')
 parser.add_argument('--evidence-stage',choices=['development','frozen'],default='development')
 args=parser.parse_args()
 if not 0<args.seconds<=3600:parser.error('Use a bounded positive per-case budget of at most one hour.')
 if not 0<args.finite_seconds<=3600:parser.error('finite-seconds must be positive and at most one hour.')
 if not 1<=args.free_cells<=160 or any(not 1<=n<=160 for n in args.free_cell_counts or []):parser.error('free-cell counts must be1..160.')
 if not 1<=args.workers<=64 or any(not 1<=n<=64 for n in args.worker_counts or []):parser.error('CPU worker counts must be1..64.')
 replica_limit=131072 if args.engine=='hybrid' and args.worker or args.suite=='hybrid' else 512
 if not 1<=args.replicas<=replica_limit or any(not 1<=n<=512 for n in args.replica_counts or []):parser.error('GPU domain replica counts must be1..512; hybrid sampling allows1..131072.')
 if any(not 1<=n<=131072 for n in args.hybrid_replica_counts or []):parser.error('Hybrid sampling replica counts must be1..131072.')
 if args.worker and args.engine=='hybrid':
  if HYBRID_PROFILES.get(args.profile,{}).get('adaptive_temperatures') and args.replicas<8:parser.error('Adaptive temperatures require at least8 replicas.')
 elif args.suite=='hybrid':
  if any(HYBRID_PROFILES[name].get('adaptive_temperatures') for name in args.hybrid_profiles) and any(n<8 for n in args.hybrid_replica_counts or [args.replicas]):parser.error('Adaptive temperatures require at least8 replicas.')
 if args.worker:return worker(args)
 require_fresh((args.output/'report.json',args.output/'results.json'))
 args.output.mkdir(parents=True,exist_ok=True);before=source_hashes();runtime_before=runtime_snapshot();repository=repository_snapshot(ROOT);records=[]
 if args.evidence_stage=='frozen' and (not repository.get('commit') or repository.get('dirty') is not False):
  raise RuntimeError('Frozen benchmark evidence requires a clean, readable committed checkout.')
 worker_counts=list(dict.fromkeys(args.worker_counts or [args.workers]))
 free_cell_counts=list(dict.fromkeys(args.free_cell_counts or [args.free_cells]))
 finite_cases=list(dict.fromkeys(args.finite_cases));args.profiles=list(dict.fromkeys(args.profiles))
 args.sat_profiles=list(dict.fromkeys(args.sat_profiles));sat_cases=list(dict.fromkeys(args.sat_cases))
 args.gpu_profiles=list(dict.fromkeys(args.gpu_profiles));gpu_cases=list(dict.fromkeys(args.gpu_cases));replica_counts=list(dict.fromkeys(args.replica_counts or [args.replicas]))
 args.hybrid_profiles=list(dict.fromkeys(args.hybrid_profiles));hybrid_cases=list(dict.fromkeys(args.hybrid_cases));hybrid_replica_counts=list(dict.fromkeys(args.hybrid_replica_counts or [args.replicas]))
 jobs=[]
 if args.suite in ('finite','all'):
  configurations=[(case,depth,profile) for case in finite_cases for depth in (free_cell_counts if case=='residual' else [None]) for profile in args.profiles]
  jobs += [('cpu',case,profile,seed,args.finite_seconds,1,depth,args.replicas) for seed in (817,291) for case,depth,profile in (configurations if seed==817 else list(reversed(configurations)))]
 if args.suite in ('original','all'):
  jobs += [('cpu','original',profile,seed,args.seconds,workers,None,args.replicas) for seed in (71,197) for workers in (worker_counts if seed==71 else list(reversed(worker_counts))) for profile in (args.profiles if seed==71 else list(reversed(args.profiles)))]
 if args.suite=='gpu' or (args.suite=='all' and args.include_gpu):
  gpu_finite=[(case,depth,profile,replicas) for case in gpu_cases if case!='original' for depth in (free_cell_counts if case=='residual' else [None]) for replicas in replica_counts for profile in args.gpu_profiles]
  jobs += [('gpu',case,profile,seed,args.finite_seconds,args.workers,depth,replicas) for seed in (817,291) for case,depth,profile,replicas in (gpu_finite if seed==817 else list(reversed(gpu_finite)))]
  if 'original' in gpu_cases:
   gpu_original=[(profile,replicas) for replicas in replica_counts for profile in args.gpu_profiles]
   jobs += [('gpu','original',profile,seed,args.seconds,args.workers,None,replicas) for seed in (71,197) for profile,replicas in (gpu_original if seed==71 else list(reversed(gpu_original)))]
 if args.suite=='sat':
  finite_configs=[(case,depth,profile) for case in sat_cases if case!='original' for depth in (free_cell_counts if case=='residual' else [None]) for profile in args.sat_profiles]
  jobs += [('sat',case,profile,seed,args.finite_seconds,1,depth,args.replicas) for seed in (817,291) for case,depth,profile in (finite_configs if seed==817 else list(reversed(finite_configs)))]
  if 'original' in sat_cases:
   jobs += [('sat','original',profile,seed,args.seconds,1,None,args.replicas) for seed in (71,197) for profile in (args.sat_profiles if seed==71 else list(reversed(args.sat_profiles)))]
 if args.suite=='hybrid':
  finite_configs=[(case,depth,profile,replicas) for case in hybrid_cases if case!='original' for depth in (free_cell_counts if case=='residual' else [None]) for replicas in hybrid_replica_counts for profile in args.hybrid_profiles]
  jobs += [('hybrid',case,profile,seed,args.finite_seconds,1,depth,replicas) for seed in (817,291) for case,depth,profile,replicas in (finite_configs if seed==817 else list(reversed(finite_configs)))]
  if 'original' in hybrid_cases:
   configs=[(profile,replicas) for replicas in hybrid_replica_counts for profile in args.hybrid_profiles]
   jobs += [('hybrid','original',profile,seed,args.seconds,1,None,replicas) for seed in (71,197) for profile,replicas in (configs if seed==71 else list(reversed(configs)))]
 # Reject every planned collision before launching any child or overwriting logs.
 for number,(engine,case,profile,seed,budget,workers,free_cells,replicas) in enumerate(jobs):
  label=case+(f'-free{free_cells}' if case=='residual' else '')
  planned=args.output/f'{number:02d}-{engine}-{label}-{profile}-{seed}.json'
  require_fresh((planned,planned.with_suffix('.state'),planned.with_suffix('.stdout.txt'),planned.with_suffix('.stderr.txt')))
 input_before=hashlib.sha256(args.data.read_bytes()).hexdigest() if any(job[1]=='original' for job in jobs) else None
 for number,(engine,case,profile,seed,budget,workers,free_cells,replicas) in enumerate(jobs):
  case_label=case+(f'-free{free_cells}' if case=='residual' else '')
  path=args.output/f'{number:02d}-{engine}-{case_label}-{profile}-{seed}.json'
  command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--output',str(path.resolve()),'--data',str(args.data.resolve()),'--engine',engine,'--case',case,'--profile',profile,'--seed',str(seed),'--seconds',str(budget),'--workers',str(workers),'--replicas',str(replicas),'--free-cells',str(free_cells if free_cells is not None else args.free_cells)]
  if engine=='hybrid':command.extend(['--hybrid-backend',args.hybrid_backend])
  started_at=datetime.datetime.now(datetime.timezone.utc).isoformat()
  tick=time.monotonic();child=None;cleanup_result=None;cleanup_attempted=False
  try:
   child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),start_new_session=os.name!='nt',env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
   try:stdout,stderr=child.communicate(timeout=budget+90)
   except subprocess.TimeoutExpired:
    cleanup_attempted=True;cleanup_result=terminate_tree(child)
    stdout,stderr=child.communicate(timeout=30)
    raise subprocess.TimeoutExpired(command,budget+90,output=stdout,stderr=stderr)
   (path.with_suffix('.stdout.txt')).write_text(stdout,encoding='utf-8');(path.with_suffix('.stderr.txt')).write_text(stderr,encoding='utf-8')
   record=json.loads(path.read_text(encoding='utf-8')) if child.returncode==0 and path.exists() else {'status':'process_error','error':stderr[-3000:]}
   record['process_exit_code']=child.returncode
  except subprocess.TimeoutExpired as exc:
   (path.with_suffix('.stdout.txt')).write_text(exc.output or '',encoding='utf-8');(path.with_suffix('.stderr.txt')).write_text(exc.stderr or '',encoding='utf-8')
   record={'status':'hard_timeout','process_exit_code':child.returncode if child is not None else None}
  except BaseException as error:
   # JSON decoding, storage errors and interrupts can happen after the group
   # leader exits. Retire the owned group before propagating the original error.
   if child is not None and not cleanup_attempted:
    cleanup_attempted=True
    try:cleanup_result=terminate_tree(child)
    except BaseException as cleanup_error:error.add_note('Benchmark cleanup also failed: '+repr(cleanup_error))
   raise
  finally:
   if child is not None and child.poll() is None and not cleanup_attempted:
    cleanup_attempted=True;cleanup_result=terminate_tree(child)
  if cleanup_result is not None:record['cleanup']=cleanup_result
  record.update(case=case,engine=engine,profile=profile,seed=seed,budget_seconds=budget,free_cells=free_cells,replicas=replicas if engine in ('gpu','hybrid') else 0,started_at=started_at,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),total_wall_seconds=time.monotonic()-tick);records.append(record)
  print(json.dumps({key:record.get(key) for key in ('engine','case','profile','seed','status','nodes','total_wall_seconds')}),flush=True)
  (args.output/'results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
  # Do not launch another case after an unexpected coordinator exit or any
  # failed search. Even an exited coordinator may have left descendants.
  if record['status'] in ('process_error','hard_timeout','error'):break
 after=source_hashes();runtime_after=runtime_snapshot();repository_after=repository_snapshot(ROOT);input_after=hashlib.sha256(args.data.read_bytes()).hexdigest() if input_before is not None else None
 input_unchanged=input_before==input_after and all(row.get('provenance',{}).get('input_sha256')==input_before for row in records if row['case']=='original' and row['status'] not in ('process_error','hard_timeout','error'))
 report={'repository':repository,'repository_after':repository_after,'evidence_stage':args.evidence_stage,'cpu_worker_counts':worker_counts,'python':sys.version,'platform':platform.platform(),'runtime_before':runtime_before,'runtime_after':runtime_after,'runtime_unchanged':runtime_before==runtime_after,'source_before':before,'source_after':after,'source_unchanged':before==after,'input_sha256':input_before,'input_after_sha256':input_after,'input_unchanged':input_unchanged,'profiles':{name:PROFILES[name] for name in args.profiles},'gpu_profiles':{name:GPU_PROFILES[name] for name in args.gpu_profiles},'gpu_cases':gpu_cases,'gpu_replica_counts':replica_counts,'sat_profiles':{name:SAT_PROFILES[name] for name in args.sat_profiles},'sat_cases':sat_cases,'sat_partition_boxes':SAT_PARTITIONS,'hybrid_profiles':{name:HYBRID_PROFILES[name] for name in args.hybrid_profiles},'hybrid_cases':hybrid_cases,'hybrid_replica_counts':hybrid_replica_counts,'hybrid_backend':args.hybrid_backend,'hybrid_comparison_scope':'Full heuristic-to-exact pipeline with durable budgets; DFS uses its symmetry rules, so no portable-root volume metric is reported.','seed_interpretation':'GPU-domain and persistent-SAT original cases are repeated measurements, not randomized search seeds. Finite fixture seeds change generated instances; CPU search uses its seed.','suite':args.suite,'planned_cases':len(jobs),'completed_cases':len(records),
         'aborted_after_child_failure':any(row['status'] in ('process_error','hard_timeout','error') for row in records),
         'cleanup_scope':'Owned POSIX group or live Windows task tree only. An exited Windows coordinator has no retained tree handle; escaped/orphan descendants are not universally tracked. No further case starts after a failure.',
         'free_cells':args.free_cells,'free_cell_counts':free_cell_counts,'finite_cases':finite_cases,'finite_seconds':args.finite_seconds,'source_convention':'Raw local hashes plus canonical LF hashes; Git normalizes line endings.','cpu_workers':args.workers,'replicas':args.replicas,'results':records,'non_claim':'Bounded original searches are incomplete unless independently validated. These metrics do not identify a universally fastest solver.'}
 (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 if report['aborted_after_child_failure']:
  # Preserve logs and the partial report BEFORE attempting best-effort cleanup
  # of an already-exited coordinator's remaining group.
  if 'cleanup' not in records[-1]:
   try:records[-1]['cleanup']=terminate_tree(child) if child is not None else 'No child was created.'
   except Exception as exc:records[-1]['cleanup']='Cleanup failed: '+str(exc)
   (args.output/'results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
   (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
  raise RuntimeError('A benchmark child failed; the remaining matrix was not started.')
 if args.evidence_stage=='frozen' and repository_after!=repository:raise RuntimeError('The committed checkout changed during frozen benchmarking.')
 if not input_unchanged:raise RuntimeError('Original puzzle inputs changed during the benchmark; results must not be used as final evidence.')
 if runtime_before!=runtime_after:raise RuntimeError('Runtime dependencies changed during the benchmark; results must not be used as final evidence.')
 if before!=after:raise RuntimeError('Sources changed during the benchmark; results must not be used as final evidence.')
if __name__=='__main__':main()
