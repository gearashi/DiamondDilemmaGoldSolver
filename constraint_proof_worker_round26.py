"""Isolated RUP producer protocol. Run only in a compatible Linux/macOS Python.

The supervisor stays warm. Every native query runs in a separate child whose
clean exit is required before its bytes are returned. The caller independently
checks all proof lines against its own exact formula. No native state is imported.
"""
import hashlib,json,math,os,subprocess,sys,threading,time
from pathlib import Path

MAX_INPUT=16*1024*1024
MAX_OUTPUT=512*1024
BACKENDS=('g3','g4','g42','lgl','mcb')

def encoded(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode('ascii')
def identity(value):return hashlib.sha256(encoded(value)).hexdigest()

def validate(request):
    if not isinstance(request,dict) or set(request)!={'version','clauses','backend','seconds','conflicts','identity'} or type(request['version'])is not int or request['version']!=1:
        raise ValueError('Invalid proof producer request.')
    if identity({k:v for k,v in request.items() if k!='identity'})!=request['identity']:raise ValueError('Proof request digest differs.')
    if request['backend'] not in BACKENDS:raise ValueError('Unknown proof backend.')
    if type(request['seconds']) not in (int,float) or not math.isfinite(request['seconds']) or not 0<request['seconds']<=5:raise ValueError('Invalid proof time cap.')
    if type(request['conflicts']) is not int or not 1<=request['conflicts']<=1000000:raise ValueError('Invalid proof conflict cap.')
    clauses=request['clauses']
    if not isinstance(clauses,list) or len(clauses)>250000 or any(not isinstance(row,list) or len(row)>4096 or any(type(v)is not int or not 0<abs(v)<=1000000 for v in row) for row in clauses):raise ValueError('Invalid proof formula.')
    return request

def native(request):
    validate(request)
    if sys.platform not in ('linux','darwin'):raise ValueError('Native proof logging is disabled on this platform.')
    from pysat.solvers import Solver
    import pysat
    solver=Solver(name=request['backend'],bootstrap_with=request['clauses'],with_proof=True)
    done=threading.Event();limited=threading.Event()
    def monitor():
        until=time.monotonic()+request['seconds']
        while not done.wait(.01):
            log=getattr(solver.solver,'prfile',None)
            if time.monotonic()>=until or log is not None and os.fstat(log.fileno()).st_size>256*1024:
                limited.set()
                try:solver.interrupt()
                except NotImplementedError:pass
                return
    thread=threading.Thread(target=monitor,daemon=True);thread.start()
    try:
        try:
            solver.conf_budget(request['conflicts'])
            answer=solver.solve_limited(expect_interrupt=True)
        except NotImplementedError:
            answer=solver.solve()
        proof=solver.get_proof() if answer is False and not limited.is_set() else None
        if proof is not None and (len(proof)>4096 or any(type(line)is not str or len(line)>16384 for line in proof) or sum(map(len,proof))>262144):proof=None
        if proof is not None and not any(line.strip()=='0' for line in proof):proof.append('0')
        result={'version':1,'identity':request['identity'],'proof':proof,'python_sat':pysat.__version__,
                'source':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'decisions':int(solver.accum_stats().get('decisions',0))}
    finally:
        done.set();thread.join();solver.delete()
    return result

def supervisor():
    print(json.dumps({'ready':1}),flush=True)
    while True:
        line=sys.stdin.buffer.readline(MAX_INPUT+1)
        if not line:return
        if len(line)>MAX_INPUT or not line.endswith(b'\n'):return
        request=None
        try:
            request=validate(json.loads(line))
            child=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--one'],input=line,
                stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=request['seconds']+3)
            result=json.loads(child.stdout) if child.returncode==0 and len(child.stdout)<=MAX_OUTPUT else None
        except (ValueError,OSError,subprocess.TimeoutExpired):result=None
        print(json.dumps({'identity':request.get('identity') if isinstance(request,dict) else None,'result':result}),flush=True)

if __name__=='__main__':
    if '--one' in sys.argv:
        raw=sys.stdin.buffer.read(MAX_INPUT+1)
        if len(raw)>MAX_INPUT:raise SystemExit(2)
        print(json.dumps(native(json.loads(raw))),flush=True)
    else:supervisor()
