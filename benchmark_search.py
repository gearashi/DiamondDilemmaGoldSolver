"""Bounded, serial, fresh-process comparison on identical Gold objectives."""
from __future__ import annotations
import argparse, datetime, hashlib, importlib.metadata, json, os, platform, subprocess, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
METHODS=('gpu-dfs','dfs','cp','cp-sat','sat','hybrid')

def gpu_baseline(problem,seconds,seed,replicas=128,backend='auto'):
    from dfs_gpu import DFS,fill_order
    from gpu_backends import choose_backend
    from exact_frontier import plan_frontier
    from systematic_jobs import JobLedger
    began=time.monotonic();deadline=began+seconds
    if problem.n!=160 or problem.fixed:raise ValueError('GPU baseline comparison requires a full 160-cell problem.')
    order=fill_order(problem.neighbors)
    frontier=plan_frontier(problem.masks,problem.neighbors,problem.sides,order,replicas,stop_requested=lambda:time.monotonic()>=deadline)
    engine=DFS
    if choose_backend(backend)=='webgpu':
        from dfs_webgpu import WebGPUDFS
        engine=WebGPUDFS
    gpu=engine(problem.masks,problem.neighbors,problem.sides,n=replicas,seed=seed,order=order,prefixes=[])
    ledger=JobLedger.empty(len(frontier.prefixes),replicas)
    checked=0;codes=None;status='timeout'
    try:
        ledger.refill(gpu,frontier.prefixes,frontier.lengths)
        while time.monotonic()<deadline:
            gpu.step(32)
            for lane in gpu.pending():
                candidate=gpu.boards[:,int(lane)].get().astype(int).tolist()
                report=problem.validate(candidate);checked+=1
                if report['valid']:codes=candidate;status='solved';break
                if report['matched_edges']!=len(problem.edges):raise AssertionError('GPU edge validation failed')
                gpu.acknowledge([int(lane)])
            if codes is not None:break
            ledger.refill(gpu,frontier.prefixes,frontier.lengths)
            if ledger.completed==ledger.total:status='infeasible';break
        return {'status':status,'codes':codes,'elapsed_seconds':time.monotonic()-began,
                'nodes':int(ledger.counter_totals(gpu)[0]),'complete':status in ('solved','infeasible'),
                'stats':{'device':gpu.device,'candidates_checked':checked,'max_depth':int(gpu.maxdepths.get().max())}}
    finally:
        if hasattr(gpu,'close'):gpu.close()

def solve_method(problem,method,seconds,seed,replicas=128,backend='auto'):
    if method=='gpu-dfs':return gpu_baseline(problem,seconds,seed,replicas,backend)
    if method=='dfs':
        from constraint_dfs import solve
        return solve(problem,seconds=seconds,seed=seed,target='gold')
    if method=='hybrid':
        from gpu_sampling import solve
        return solve(problem,seconds=seconds,seed=seed,target='gold',replicas=replicas,backend=backend)
    from constraint_models import solve
    return solve(problem,engine=method,seconds=seconds,seed=seed,target='gold')

def worker(args):
    from search_problem import SearchProblem
    body=json.loads(args.case_file.read_text(encoding='utf-8'))
    problem=SearchProblem(body['data'])
    result=solve_method(problem,args.worker,args.seconds,args.seed,args.replicas,args.backend)
    codes=result.get('codes')
    report=problem.validate(codes) if codes is not None else None
    if result.get('status')=='solved' and not (report and report['valid']):raise AssertionError('Invalid claimed Gold solution')
    result.pop('checkpoint',None)
    result['independent_validation']=report
    args.result_file.write_text(json.dumps(result,indent=2),encoding='utf-8')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def cpu_description():
    if sys.platform=='win32':
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return str(winreg.QueryValueEx(key,'ProcessorNameString')[0]).strip()
        except OSError:pass
    return platform.processor()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=ROOT/'data/tiles.json')
    parser.add_argument('--output',type=Path,default=ROOT/'audit/search-comparison')
    parser.add_argument('--seconds',type=float,default=10)
    parser.add_argument('--seeds',type=int,nargs='+',default=[71,197])
    parser.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    parser.add_argument('--replicas',type=int,default=128)
    parser.add_argument('--backend',choices=('auto','cuda','webgpu'),default='auto')
    parser.add_argument('--worker',choices=METHODS)
    parser.add_argument('--case-file',type=Path)
    parser.add_argument('--result-file',type=Path)
    parser.add_argument('--seed',type=int,default=71)
    args=parser.parse_args()
    args.output=args.output.resolve()
    if not 0<args.seconds<=300:parser.error('Use a bounded per-case budget of 0..300 seconds.')
    if args.worker:return worker(args)
    from search_fixtures import planted_gold
    source_paths=[p for p in ROOT.iterdir() if p.name in {'benchmark_search.py','search_fixtures.py','search_problem.py','constraint_dfs.py','constraint_models.py','gpu_sampling.py','sampling_kernels.wgsl','dfs_gpu.py','dfs_webgpu.py','dfs_kernels.cu','dfs_kernels.wgsl','gpu_engine.py','gpu_backends.py','exact_frontier.py','systematic_jobs.py','validator.py','geometry.py'}]
    source_hashes={str(p):sha(p) for p in source_paths}
    began=time.monotonic();started=datetime.datetime.now(datetime.timezone.utc).isoformat()
    args.output.mkdir(parents=True,exist_ok=True)
    cases=[]
    for seed in [817,291]:
        data,board,reference=planted_gold(seed)
        cases.append({'name':f'planted-{seed}','data':data,'known_satisfiable':True})
    cases.append({'name':'real-gold','data':json.loads(args.data.read_text(encoding='utf-8-sig')),'known_satisfiable':None})
    paths=[]
    for case in cases:
        path=args.output/(case['name']+'.json');path.write_text(json.dumps(case),encoding='utf-8');paths.append(path)
    records=[]
    # Rotate method order between seeds; each run is an isolated interpreter.
    for seed_index,seed in enumerate(args.seeds):
        methods=list(args.methods);methods=methods[seed_index:]+methods[:seed_index]
        for case,path in zip(cases,paths):
            for method in methods:
                result_file=args.output/f'{case["name"]}-{seed}-{method}.json'
                command=[sys.executable,'-B',str(Path(__file__).resolve()),'--worker',method,'--seconds',str(args.seconds),
                         '--case-file',str(path.resolve()),'--result-file',str(result_file.resolve()),'--seed',str(seed),
                         '--replicas',str(args.replicas),'--backend',args.backend]
                tick=time.monotonic();environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                try:
                    process=subprocess.run(command,cwd=ROOT,env=environment,capture_output=True,text=True,timeout=args.seconds+30,
                                           creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                    if process.returncode==0:result=json.loads(result_file.read_text(encoding='utf-8'))
                    else:result={'status':'error','error':(process.stdout+process.stderr)[-5000:]}
                except subprocess.TimeoutExpired:
                    result={'status':'hard_timeout','error':'Worker exceeded search budget plus 30-second setup/shutdown guard.'}
                elapsed=time.monotonic()-tick
                result.update(method=method,case=case['name'],seed=seed,budget_seconds=args.seconds,wall_seconds=elapsed,
                              input_sha256=sha(path),known_satisfiable=case['known_satisfiable'])
                records.append(result)
                result_file.write_text(json.dumps(result,indent=2),encoding='utf-8')
                print(f'{case["name"]} seed={seed} {method}: {result["status"]} {elapsed:.3f}s',flush=True)
                (args.output/'results.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    summary={}
    for method in args.methods:
        rows=[r for r in records if r['method']==method]
        solved=[r for r in rows if r['status']=='solved']
        summary[method]={'solved':len(solved),'runs':len(rows),'real_solved':sum(r['case']=='real-gold' for r in solved),
                         'errors':sum(r['status'] in ('error','hard_timeout') for r in rows),
                         'solved_wall_seconds':sum(r['wall_seconds'] for r in solved),
                         'timeouts':sum(r['status']=='timeout' for r in rows)}
    ranking=sorted(args.methods,key=lambda m:(-summary[m]['real_solved'],-summary[m]['solved'],summary[m]['errors'],summary[m]['solved_wall_seconds']))
    output={'summary':summary,'ranking_on_tested_cases':ranking if any(summary[m]['solved'] for m in ranking) else [],'validated_solve_winner':ranking[0] if summary[ranking[0]]['solved'] and all((summary[ranking[0]]['real_solved'],summary[ranking[0]]['solved'])!=(summary[m]['real_solved'],summary[m]['solved']) for m in ranking[1:]) else None,'non_claim':'Synthetic timings do not establish the fastest solver for the full Gold puzzle; a timeout is not infeasibility.'}
    (args.output/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    versions=[]
    for package in ['numpy','ortools','python-sat','cupy-cuda12x','wgpu']:
        try:versions.append(package+'=='+importlib.metadata.version(package))
        except importlib.metadata.PackageNotFoundError:pass
    if any(sha(p)!=source_hashes[str(p)] for p in source_paths):raise RuntimeError('Load-bearing source changed during the comparison; results cannot be attributed to one version.')
    manifest={'schema_version':1,'claim_id':'bounded-gold-search-comparison','repository':{'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT))},
      'command':subprocess.list2cmdline([sys.executable,*sys.argv]),'environment':{'software':[platform.python_version(),*versions],'hardware':platform.platform()+' '+cpu_description()+'; GPU devices: '+', '.join(sorted({str((r.get('gpu_sampling') or {}).get('device') or r.get('stats',{}).get('device')) for r in records if (r.get('gpu_sampling') or {}).get('device') or r.get('stats',{}).get('device')}))},
      'mathematics':{'assertion_tested':'Compare independently validated single-loop Gold solve times on two planted160-tile instances and the encoded real160-tile instance.','coefficient_domain':'Exact integer endpoint masks and graph connectivity','conventions':'code=3*tile+rotation; seam reverses endpoint p to12-p; only declared segment endpoints connect',
      'inputs':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p)} for p in [*paths,*source_paths]],'bounds':{'seconds_per_run':args.seconds,'external_guard_seconds':30,'seeds':args.seeds,'replicas':args.replicas,'cpu_workers':1,'serial_runs':True},
      'non_claims':['No universal fastest algorithm claim','No Gold impossibility claim from timeouts','No source-diagram transcription proof','GPU sampling is heuristic and not exhaustive coverage']},
      'randomness':{'used':True,'generator':'Per-engine seeded random generator; fixtures Python random.Random','seed':args.seeds[0]},
      'run':{'started_at':started,'runtime_seconds':time.monotonic()-began,'exit_status':0},
      'outputs':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':sha(p)} for p in [args.output/'results.json',args.output/'summary.json']],
      'checks':['Every claimed solution passed independent validator.py','Planted references validated before use','Fresh subprocess per method/case/seed','Load-bearing source hashes unchanged between start and finish','Includes interpreter/model/GPU initialization in wall time'],
      'result':'Bounded timings recorded; '+str(sum(r['status']=='solved' for r in records))+' validated solutions across '+str(len(records))+' runs.',
      'residual_risks':['Small benchmark set; planted cases differ from real puzzle','GPU clock/driver cache and operating-system scheduling affect timing','No enforceable Windows process memory cap; runtime guard is enforced']}
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))
if __name__=='__main__':main()
