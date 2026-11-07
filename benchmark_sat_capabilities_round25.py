"""Isolated native SAT persistence capability checks; never used by the solver.

A printed proof is not a successful backend result unless the child also exits
zero. Unsupported proof logging and native cleanup failures remain diagnostic
records. No solver is enabled or installed by this script.
"""
import argparse,hashlib,json,platform,subprocess,sys,time
from pathlib import Path

CHILD = r"""
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
    started=time.monotonic()
    try:
        child=subprocess.run([sys.executable,'-B','-c',CHILD,name,'1' if proof else '0'],capture_output=True,text=True,timeout=timeout,
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


def main():
    import pysat,pysolvers,pysat.solvers
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    names=[key for key,value in vars(pysat.solvers.SolverNames).items() if not key.startswith('_') and isinstance(value,tuple)]
    rows=[]
    for name in names:
        for proof in (False,True):
            row=probe(name,proof);rows.append(row)
            print(json.dumps({'backend':name,'proof_logging':proof,'exit_code':row.get('exit_code'),'safe_completed':row['safe_completed']}),flush=True)
    inputs=[Path(__file__).resolve(),Path(pysat.solvers.__file__),Path(pysolvers.__file__)]
    result={'proposal':5,'python':platform.python_version(),'platform':platform.platform(),'python_sat':pysat.__version__,
            'source_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},'results':rows,
            'safe_proof_backends':[row['backend'] for row in rows if row['proof_logging'] and row.get('safe_completed') and row.get('independently_checked')],
            'non_claims':['This installed platform/build does not establish capabilities of every OS or future backend build.','No native proof logger or restart-clause importer is enabled in the solver.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'python':result['python'],'python_sat':result['python_sat'],'platform':result['platform'],'safe_proof_backends':result['safe_proof_backends']}),flush=True)
if __name__=='__main__':main()
