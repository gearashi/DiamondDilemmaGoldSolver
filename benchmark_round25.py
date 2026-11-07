"""R25 serial finite/full-board comparisons with exact unfinished-volume accounting.

Budgets are diagnostic only. Unknown/time-limited searches never prove the Gold
puzzle infeasible. Engine node counters have different units and are not ranked.
A child failure stops the matrix immediately. Cleanup targets the owned POSIX
process group or a live Windows task tree; after a Windows coordinator exits,
this harness cannot establish that every descendant is gone. It never claims a
universal orphan guarantee or starts another case after that ownership failure.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,platform,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROFILES={
 'baseline':{},
 'reuse':dict(resumable_reasoning=True,incremental_hall=True,hall_rematch=True,reversible_gold_graph=True,shared_coverage=True,symbolic_coverage=True,queue_pruning=True),
 'sat-learning':dict(resumable_reasoning=True,sat_assist=True,witness_pool=True,retained_relations=True,shared_sat_assistant=True,higher_order_relations=True),
 'arithmetic':dict(joint_conservation=True,affine_reasoning=True,incremental_affine=True,pb_learning=True,conflict_pb=True,integer_lattice=True),
 'regional':dict(symbolic_regions=True,symbolic_inventory=True,elimination=True,mini_buckets=True,relaxed_diagrams=True,representative_connectivity=True,subset_joins=True,lazy_regions=True,certified_relaxation=True,labelled_routing=True),
}

def source_hashes():
 return {p.name:{'raw_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'lf_sha256':hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()} for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.c','.cu','.wgsl','.html')}
def volume(rows):return sum(math.prod(x.bit_count() for x in row) for row in rows)
def pending_frontier(scope,result,engine):
 """Validate committed ownership before counting any removed assignments."""
 from constraint_work import checkpoint_boxes,frontier_snapshot,restore_frontier
 if result.get('status') in ('solved','edge_perfect'):return None
 saved=result.get('checkpoint')
 if saved is None:
  if result.get('status')=='infeasible' and result.get('complete') is True:return ()
  raise ValueError('An incomplete benchmark result has no committed remaining frontier.')
 if engine=='gpu':return restore_frontier(saved['frontier'],scope)
 rows=tuple(row for job in saved['jobs'] for row in checkpoint_boxes(job['checkpoint']))
 if not rows and result.get('complete') is not True:
  raise ValueError('An incomplete benchmark result has an empty remaining frontier.')
 # All CPU cases start from the same original portable root, without adding a
 # canonical symmetry restriction. Scope/disjointness errors invalidate metrics.
 return restore_frontier(frontier_snapshot(scope,rows),scope)


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
 from search_problem import SearchProblem
 from constraint_work import WorkScope,frontier_snapshot,digest
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
 scope=WorkScope.for_problem(problem,'gold');initial=frontier_snapshot(scope,[scope.root]);began=time.monotonic()
 args.output.parent.mkdir(parents=True,exist_ok=True)
 state_dir=args.output.with_suffix('.state')
 if state_dir.exists():raise FileExistsError('Benchmark state already exists; use a fresh per-case output: '+str(state_dir))
 state_dir.mkdir()
 coverage_path=state_dir/'coverage.sqlite';regional_path=state_dir/'regional';regional_path.mkdir()
 options={'native':'compiled',**PROFILES[args.profile]}
 if args.engine=='cpu':
  from constraint_parallel import solve
  result=solve(problem,seconds=args.seconds,seed=args.seed,target='gold',workers=args.workers,resume=initial,
      options=options,learning_state={'_coverage_path':coverage_path,'_regional_cache_dir':regional_path})
 elif args.engine=='gpu':
  from gpu_domain_runner import solve
  result=solve(problem,seconds=args.seconds,seed=args.seed,target='gold',replicas=args.replicas,work_frontier=initial,backend='cuda')
 else:raise ValueError(args.engine)
 if result.get('codes') is not None:
  check=problem.validate(result['codes']);assert check['valid']
  assert all(code in problem.domains[cell] for cell,code in enumerate(result['codes']))
  result['independent_validation']=check
 pending=pending_frontier(scope,result,args.engine);status=result['status']
 # A solved result is a witness, not an enumeration completion: no space claim.
 if status not in ('solved','edge_perfect'):
  remaining=volume(pending);total=volume([scope.root]);assert 0<=remaining<=total
  result['raw_space']={'initial':str(total),'remaining':str(remaining),'closed':str(total-remaining),'closed_log10':math.log10(total-remaining) if total>remaining else None,'definition':'Cartesian orientation assignments absent from the validated committed frontier of the same original unsymmetrized root; not a count of legal boards.'}
 for key in ('checkpoint','knowledge','shared_conflicts','mask_conflicts','local_mask_conflicts','certificates'):result.pop(key,None)
 result.update(case=args.case,engine=args.engine,solver_engine='parallel-dfs' if args.engine=='cpu' else 'gpu-domains',
     profile=args.profile,seed=args.seed,budget_seconds=args.seconds,workers=args.workers if args.engine=='cpu' else 0,replicas=args.replicas,
     cells=problem.n,fixed_cells=len(problem.fixed),fingerprint=problem.fingerprint,domains_sha256=digest(scope.original),
     provenance=provenance,options=options if args.engine=='cpu' else {},
     benchmark_state={'directory':str(state_dir),'coverage_enabled':args.engine=='cpu' and bool(options.get('shared_coverage')),
                      'coverage_file_created':coverage_path.exists(),'regional_directory_supplied':args.engine=='cpu'},
     wall_seconds=time.monotonic()-began)
 args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:result.get(k) for k in ('case','engine','profile','seed','status','nodes','wall_seconds')}),flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data',type=Path,default=Path('C:/DiamondDilemmaGoldSolver/data/tiles.json'))
 parser.add_argument('--worker',action='store_true');parser.add_argument('--engine',choices=['cpu','gpu'],default='cpu');parser.add_argument('--case',choices=['original','planted','small','residual'],default='original');parser.add_argument('--profile',choices=PROFILES,default='baseline')
 parser.add_argument('--seconds',type=float,default=60);parser.add_argument('--seed',type=int,default=71);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--replicas',type=int,default=512);parser.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES));parser.add_argument('--include-gpu',action='store_true')
 parser.add_argument('--suite',choices=['finite','original','gpu','all'],default='all');parser.add_argument('--free-cells',type=int,default=60)
 args=parser.parse_args()
 if not 0<args.seconds<=3600:parser.error('Use a bounded positive per-case budget of at most one hour.')
 if not 1<=args.free_cells<=160:parser.error('free-cells must be1..160.')
 if args.worker:return worker(args)
 args.output.mkdir(parents=True,exist_ok=True);before=source_hashes();records=[]
 jobs=[]
 if args.suite in ('finite','all'):
  jobs += [('cpu',case,profile,seed,120,1) for case in ('small','residual') for seed in (817,291) for profile in (args.profiles if seed==817 else list(reversed(args.profiles)))]
 if args.suite in ('original','all'):
  jobs += [('cpu','original',profile,seed,args.seconds,args.workers) for seed in (71,197) for profile in (args.profiles if seed==71 else list(reversed(args.profiles)))]
 if args.suite=='gpu' or (args.suite=='all' and args.include_gpu):
  jobs += [('gpu','original','baseline',seed,args.seconds,args.workers) for seed in (71,197)]
 input_before=hashlib.sha256(args.data.read_bytes()).hexdigest() if any(job[1]=='original' for job in jobs) else None
 for number,(engine,case,profile,seed,budget,workers) in enumerate(jobs):
  path=args.output/f'{number:02d}-{engine}-{case}-{profile}-{seed}.json'
  command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--output',str(path.resolve()),'--data',str(args.data.resolve()),'--engine',engine,'--case',case,'--profile',profile,'--seed',str(seed),'--seconds',str(budget),'--workers',str(workers),'--replicas',str(args.replicas),'--free-cells',str(args.free_cells)]
  tick=time.monotonic();child=None;cleanup_result=None
  try:
   child=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),start_new_session=os.name!='nt',env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
   try:stdout,stderr=child.communicate(timeout=budget+90)
   except subprocess.TimeoutExpired:
    cleanup_result=terminate_tree(child)
    stdout,stderr=child.communicate(timeout=30)
    raise subprocess.TimeoutExpired(command,budget+90,output=stdout,stderr=stderr)
   (path.with_suffix('.stdout.txt')).write_text(stdout,encoding='utf-8');(path.with_suffix('.stderr.txt')).write_text(stderr,encoding='utf-8')
   record=json.loads(path.read_text(encoding='utf-8')) if child.returncode==0 and path.exists() else {'status':'process_error','error':stderr[-3000:]}
   record['process_exit_code']=child.returncode
  except subprocess.TimeoutExpired as exc:
   (path.with_suffix('.stdout.txt')).write_text(exc.output or '',encoding='utf-8');(path.with_suffix('.stderr.txt')).write_text(exc.stderr or '',encoding='utf-8')
   record={'status':'hard_timeout','process_exit_code':child.returncode if child is not None else None}
  finally:
   if child is not None and child.poll() is None:cleanup_result=terminate_tree(child)
  if cleanup_result is not None:record['cleanup']=cleanup_result
  record.update(case=case,engine=engine,profile=profile,seed=seed,total_wall_seconds=time.monotonic()-tick);records.append(record)
  print(json.dumps({key:record.get(key) for key in ('engine','case','profile','seed','status','nodes','total_wall_seconds')}),flush=True)
  (args.output/'results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
  # Do not launch another case after an unexpected coordinator exit or any
  # failed search. Even an exited coordinator may have left descendants.
  if record['status'] in ('process_error','hard_timeout','error'):break
 after=source_hashes();input_after=hashlib.sha256(args.data.read_bytes()).hexdigest() if input_before is not None else None
 input_unchanged=input_before==input_after and all(row.get('provenance',{}).get('input_sha256')==input_before for row in records if row['case']=='original' and row['status'] not in ('process_error','hard_timeout','error'))
 report={'python':sys.version,'platform':platform.platform(),'source_before':before,'source_after':after,'source_unchanged':before==after,'input_sha256':input_before,'input_after_sha256':input_after,'input_unchanged':input_unchanged,'profiles':{name:PROFILES[name] for name in args.profiles},'suite':args.suite,'planned_cases':len(jobs),'completed_cases':len(records),
         'aborted_after_child_failure':any(row['status'] in ('process_error','hard_timeout','error') for row in records),
         'cleanup_scope':'Owned POSIX group or live Windows task tree only. An exited Windows coordinator has no retained tree handle; escaped/orphan descendants are not universally tracked. No further case starts after a failure.',
         'free_cells':args.free_cells,'source_convention':'Raw local hashes plus canonical LF hashes; Git normalizes line endings.','cpu_workers':args.workers,'replicas':args.replicas,'results':records,'non_claim':'Bounded original searches are incomplete unless independently validated. These metrics do not identify a universally fastest solver.'}
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
 if not input_unchanged:raise RuntimeError('Original puzzle inputs changed during the benchmark; results must not be used as final evidence.')
 if before!=after:raise RuntimeError('Sources changed during the benchmark; results must not be used as final evidence.')
if __name__=='__main__':main()
