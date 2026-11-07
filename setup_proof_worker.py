"""Prepare the optional isolated proof producer; never installs or starts WSL.

Windows requires an existing distribution with python3 and python3-venv.
Linux/macOS use this interpreter and its normal pinned SAT package.
"""
import argparse,json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent

def linux_path(path):
    path=path.absolute()
    if not path.drive or path.drive.startswith('\\'):raise ValueError('Use a project folder on a local Windows drive.')
    return '/mnt/'+path.drive[0].lower()+path.as_posix()[2:]

def run(argv,*,capture=False,input=None,timeout=None):
    result=subprocess.run(argv,cwd=ROOT,input=input,text=True,timeout=timeout,stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if result.returncode:raise RuntimeError('Proof-worker setup command failed: '+str(result.returncode)+' '+(result.stderr or '').strip())
    return result.stdout

def prepare(distro='Ubuntu',*,install=True):
    if sys.platform=='win32':
        prefix=['wsl.exe','-d',distro,'--'];folder=ROOT/'.venv-wsl-proof'
        interpreter=[*prefix,linux_path(folder/'bin'/'python3')]
        if install:
            try:run([*prefix,'python3','-m','venv',linux_path(folder)])
            except RuntimeError as exc:raise RuntimeError('The selected WSL distribution needs python3 and python3-venv. Install those inside that distribution, then retry. '+str(exc)) from exc
            run([*interpreter,'-m','pip','install','python-sat==1.9.dev15'])
        worker=linux_path(ROOT/'constraint_proof_worker_round26.py')
    elif sys.platform in ('linux','darwin'):
        interpreter=[sys.executable];worker=str(ROOT/'constraint_proof_worker_round26.py')
        if install:run([*interpreter,'-m','pip','install','python-sat==1.9.dev15'])
    else:raise RuntimeError('The isolated native producer supports Linux, macOS, or an existing WSL distribution.')
    from constraint_proof_worker_round26 import identity
    request={'version':1,'clauses':[[1,2],[1,-2],[-1,2],[-1,-2]],'backend':'g4','seconds':1,'conflicts':10000}
    request['identity']=identity(request)
    result=json.loads(run([*interpreter,'-B',worker,'--one'],capture=True,input=json.dumps(request),timeout=45))
    from constraint_proof_indexed_round26 import IndexedRUPChecker
    if result.get('identity')!=request['identity'] or result.get('python_sat')!='1.9.dev15' or not isinstance(result.get('proof'),list):raise RuntimeError('Native proof-worker identity or package check failed.')
    checker=IndexedRUPChecker(request['clauses'],result['proof'])
    while not checker.finished:checker.advance()
    if not checker.result:raise RuntimeError('The independent host checker rejected the setup proof.')
    if sys.platform=='win32':(folder/'diamond-config.json').write_text(json.dumps({'distro':distro}),encoding='utf-8')
    return {'verified':True,'platform':sys.platform,'distro':distro if sys.platform=='win32' else None,'python_sat':result['python_sat']}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--distro',default=os.environ.get('DIAMOND_PROOF_WSL_DISTRO','Ubuntu'));parser.add_argument('--verify-only',action='store_true');args=parser.parse_args()
    try:print(json.dumps(prepare(args.distro,install=not args.verify_only)))
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as exc:parser.exit(1,str(exc)+'\n')
