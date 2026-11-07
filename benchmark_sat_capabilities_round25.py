"""Isolated native SAT persistence capability checks; never used by the solver.

A printed proof is not a successful backend result unless the child also exits
zero. Unsupported proof logging and native cleanup failures remain diagnostic
records. No solver is enabled or installed by this script.
"""
import argparse,hashlib,json,platform,subprocess,sys,time
from pathlib import Path
from benchmark_provenance import require_assertions,source_snapshot,runtime_snapshot,repository_snapshot

ROOT=Path(__file__).resolve().parent

CHILD = r"""
from benchmark_provenance import require_assertions
require_assertions()
import ctypes,json,os,sys
if os.name=='nt':ctypes.windll.kernel32.SetErrorMode(0x0001|0x0002|0x8000)
from pysat.solvers import Solver,NoSuchSolverError,SolverNames
name=getattr(SolverNames,sys.argv[1])[0];proof=sys.argv[2]=='1'
try:
    solver=Solver(name=name,bootstrap_with=((1,2),(1,-2),(-1,2),(-1,-2)),with_proof=proof)
except (ImportError,NotImplementedError,AttributeError,AssertionError,ValueError,NoSuchSolverError) as exc:
    print(json.dumps({'status':'unsupported','error':type(exc).__name__+': '+str(exc)}),flush=True)
    sys.exit(0)
methods=[name for name in dir(solver.solver) if any(word in name.lower() for word in ('learn','snapshot','serial','export','save','restore','dump'))]
answer=solver.solve()
trace=solver.get_proof() if proof else None
print(json.dumps({'status':'solved_before_cleanup','answer':answer,'proof':trace,'candidate_persistence_methods':methods}),flush=True)
solver.delete()
print(json.dumps({'status':'cleanup_returned'}),flush=True)
"""


def probe(name,proof,*,timeout=5):
    require_assertions()
    started=time.monotonic()
    try:
        child=subprocess.run([sys.executable,'-B','-c',CHILD,name,'1' if proof else '0'],capture_output=True,text=True,timeout=timeout,cwd=ROOT,
                             creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        records=[]
        for line in child.stdout.splitlines():
            try:records.append(json.loads(line))
            except ValueError:pass
        answer={'backend':name,'proof_logging':proof,'exit_code':child.returncode,'records':records,'stderr':child.stderr[-2000:]}
        candidate=next((row for row in records if row.get('status')=='solved_before_cleanup'),None)
        if proof and child.returncode==0 and candidate and candidate.get('proof'):
            from constraint_proof import RUPChecker
            try:answer['independently_checked']=RUPChecker(((1,2),(1,-2),(-1,2),(-1,-2))).verify(candidate['proof'])
            except (ValueError,TypeError):answer['independently_checked']=False
        answer['safe_completed']=child.returncode==0 and candidate is not None and candidate.get('answer') is False
    except subprocess.TimeoutExpired:
        answer={'backend':name,'proof_logging':proof,'status':'timeout','safe_completed':False}
    answer['elapsed_seconds']=time.monotonic()-started
    return answer


def run_capability_checks():
    require_assertions()
    import pysat,pysolvers,pysat.solvers
    names=[key for key,value in vars(pysat.solvers.SolverNames).items() if not key.startswith('_') and isinstance(value,tuple)]
    rows=[]
    for name in names:
        for proof in (False,True):
            row=probe(name,proof);rows.append(row)
            print(json.dumps({'backend':name,'proof_logging':proof,'exit_code':row.get('exit_code'),'safe_completed':row['safe_completed']}),flush=True)
    inputs=[Path(__file__).resolve(),Path(pysat.solvers.__file__),Path(pysolvers.__file__)]
    return {'proposal':5,'python':platform.python_version(),'platform':platform.platform(),'python_sat':pysat.__version__,
            'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},'results':rows,
            'safe_proof_backends':[row['backend'] for row in rows if row['proof_logging'] and row.get('safe_completed') and row.get('independently_checked')],
            'non_claims':['This installed platform/build does not establish capabilities of every OS or future backend build.','No native proof logger or restart-clause importer is enabled in the solver.']}


def main():
    # Guard before any optional/native import or capability subprocess.
    require_assertions()
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args()
    if args.output.suffix.lower()!='.json' or args.output.exists() or args.output.is_symlink():
        raise ValueError('SAT capability output must be a fresh .json path.')
    before=source_snapshot(ROOT);runtime_before=runtime_snapshot();repository_before=repository_snapshot(ROOT)
    if args.evidence_stage=='frozen' and (not repository_before.get('commit') or repository_before.get('dirty') is not False):
        raise RuntimeError('Frozen SAT capability evidence requires a clean, readable committed checkout.')
    result=run_capability_checks()
    after=source_snapshot(ROOT);runtime_after=runtime_snapshot();repository_after=repository_snapshot(ROOT)
    if before!=after or runtime_before!=runtime_after:
        raise RuntimeError('Sources or runtime changed during SAT capability evaluation.')
    if repository_before.get('commit')!=repository_after.get('commit'):
        raise RuntimeError('Git HEAD changed during SAT capability evaluation.')
    if args.evidence_stage=='frozen' and repository_before!=repository_after:
        raise RuntimeError('Git identity changed during frozen SAT capability evaluation.')
    result.update(evidence_stage=args.evidence_stage,evidence_status='completed_diagnostic',
        source_before=before,source_after=after,runtime_before=runtime_before,runtime_after=runtime_after,
        repository_before=repository_before,repository_after=repository_after,source_unchanged=True,runtime_unchanged=True,
        child_evidence_scope='Isolated capability children are bounded by this parent source/runtime/Git envelope; individual unsupported or crashed backends are diagnostic failures, not accepted capabilities.')
    encoded=json.dumps(result,indent=2,allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:stream.write(encoded)
    print(json.dumps({'python':result['python'],'python_sat':result['python_sat'],'platform':result['platform'],
        'safe_proof_backends':result['safe_proof_backends'],'evidence_stage':args.evidence_stage}),flush=True)
if __name__=='__main__':main()
