"""R24 serial whole-board comparisons with exact unfinished-volume accounting.

Budgets are diagnostic only. Unknown/time-limited searches never prove the Gold
puzzle infeasible. Engine node counters have different units and are not ranked.
"""
from __future__ import annotations
import argparse,hashlib,json,math,os,platform,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PROFILES={
 'prior':{},
 'reuse':dict(resumable_reasoning=True,incremental_hall=True,residual_reuse=True),
 'sat-witness':dict(sat_assist=True,witness_pool=True,retained_relations=True),
 'algebra':dict(joint_conservation=True,affine_reasoning=True,pb_learning=True,obstruction_rules=True,interface_formulas=True),
 'regional':dict(symbolic_regions=True,symbolic_inventory=True,elimination=True,mini_buckets=True,relaxed_diagrams=True,representative_connectivity=True,subset_joins=True,lazy_regions=True,certified_relaxation=True,labelled_routing=True),
 'branching':dict(dual_piece_branching=True,strand_branching=True,completion_ordering=True),
}
def source_hashes():
 return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.py','.c','.cu','.wgsl')}
def volume(rows):return sum(math.prod(x.bit_count() for x in row) for row in rows)
def worker(args):
 from search_problem import SearchProblem
 from constraint_work import WorkScope,frontier_snapshot,checkpoint_boxes
 if args.case=='original':problem=SearchProblem(json.loads(args.data.read_text(encoding='utf-8-sig')))
 else:
  from search_fixtures import planted_gold
  data,board,_=planted_gold(args.seed,size=4);problem=SearchProblem(data,board)
 scope=WorkScope.for_problem(problem,'gold');initial=frontier_snapshot(scope,[scope.root]);began=time.monotonic()
 if args.engine=='cpu':
  from constraint_parallel import solve
  result=solve(problem,seconds=args.seconds,seed=args.seed,target='gold',workers=args.workers,resume=initial,options={'native':'compiled',**PROFILES[args.profile]})
  cp=result.get('checkpoint');pending=[row for job in cp['jobs'] for row in checkpoint_boxes(job['checkpoint'])] if cp else []
 elif args.engine=='gpu':
  from gpu_domain_runner import solve
  result=solve(problem,seconds=args.seconds,seed=args.seed,target='gold',replicas=args.replicas,work_frontier=initial,backend='cuda')
  cp=result.get('checkpoint');pending=[tuple(int(x,16) for x in row) for row in cp['frontier']['boxes']] if cp else []
 else:raise ValueError(args.engine)
 if result.get('codes') is not None:
  check=problem.validate(result['codes']);assert check['valid'];result['independent_validation']=check
 complete=result.get('complete',False);status=result['status']
 # A solved result is a witness, not an enumeration completion: no space claim.
 if status not in ('solved','edge_perfect'):
  remaining=volume(pending);total=volume([scope.root]);assert 0<=remaining<=total
  result['raw_space']={'initial':str(total),'remaining':str(remaining),'closed':str(total-remaining),'closed_log10':math.log10(total-remaining) if total>remaining else None,'definition':'Cartesian orientation assignments removed by completed deductions/search; not a count of legal boards.'}
 for key in ('checkpoint','knowledge','shared_conflicts','mask_conflicts','local_mask_conflicts','certificates'):result.pop(key,None)
 result.update(case=args.case,engine=args.engine,profile=args.profile,seed=args.seed,budget_seconds=args.seconds,workers=args.workers,replicas=args.replicas,wall_seconds=time.monotonic()-began)
 args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
 print(json.dumps({k:result.get(k) for k in ('case','engine','profile','seed','status','nodes','wall_seconds')}),flush=True)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--data',type=Path,default=Path('C:/DiamondDilemmaGoldSolver/data/tiles.json'))
 parser.add_argument('--worker',action='store_true');parser.add_argument('--engine',choices=['cpu','gpu'],default='cpu');parser.add_argument('--case',choices=['original','planted'],default='original');parser.add_argument('--profile',choices=PROFILES,default='prior')
 parser.add_argument('--seconds',type=float,default=60);parser.add_argument('--seed',type=int,default=71);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--replicas',type=int,default=512);parser.add_argument('--profiles',nargs='+',choices=PROFILES,default=list(PROFILES));parser.add_argument('--include-gpu',action='store_true')
 args=parser.parse_args()
 if args.worker:return worker(args)
 args.output.mkdir(parents=True,exist_ok=True);before=source_hashes();records=[]
 jobs=[('cpu','planted',profile,817,30) for profile in args.profiles]
 jobs += [('cpu','original',profile,seed,args.seconds) for seed in (71,197) for profile in (args.profiles if seed==71 else list(reversed(args.profiles)))]
 if args.include_gpu:jobs += [('gpu','original','prior',seed,args.seconds) for seed in (71,197)]
 for number,(engine,case,profile,seed,budget) in enumerate(jobs):
  path=args.output/f'{number:02d}-{engine}-{case}-{profile}-{seed}.json'
  command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker','--output',str(path.resolve()),'--data',str(args.data.resolve()),'--engine',engine,'--case',case,'--profile',profile,'--seed',str(seed),'--seconds',str(budget),'--workers',str(args.workers),'--replicas',str(args.replicas)]
  tick=time.monotonic()
  try:
   child=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=budget+90,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0),env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
   (path.with_suffix('.stdout.txt')).write_text(child.stdout,encoding='utf-8');(path.with_suffix('.stderr.txt')).write_text(child.stderr,encoding='utf-8')
   record=json.loads(path.read_text()) if child.returncode==0 and path.exists() else {'status':'process_error','error':child.stderr[-3000:]}
   record['process_exit_code']=child.returncode
  except subprocess.TimeoutExpired:record={'status':'hard_timeout','process_exit_code':None}
  record.update(case=case,engine=engine,profile=profile,seed=seed,total_wall_seconds=time.monotonic()-tick);records.append(record)
  print(json.dumps({key:record.get(key) for key in ('engine','case','profile','seed','status','nodes','total_wall_seconds')}),flush=True)
  (args.output/'results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
 after=source_hashes();report={'python':sys.version,'platform':platform.platform(),'source_before':before,'source_after':after,'source_unchanged':before==after,'input_sha256':hashlib.sha256(args.data.read_bytes()).hexdigest(),'results':records,'non_claim':'Bounded original searches are incomplete unless independently validated. These metrics do not identify a universally fastest solver.'}
 (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
 if before!=after:raise RuntimeError('Sources changed during the benchmark; results must not be used as final evidence.')
 if any(row['status'] in ('process_error','hard_timeout','error') for row in records):raise RuntimeError('A benchmark child failed.')
if __name__=='__main__':main()
