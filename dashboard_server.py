"""Loopback-only live Diamond Dilemma dashboard and solver controls."""
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path
import argparse,ctypes,datetime,json,os,secrets,subprocess,sys,threading
from urllib.parse import urlsplit
from io_utils import read_json_shared
from gpu_backends import BACKENDS
from search_algorithms import ALGORITHMS,LABELS
ROOT=Path(__file__).resolve().parent
RUNTIME=ROOT/'runtime';RUNTIME.mkdir(exist_ok=True)
TOKEN=secrets.token_urlsafe(32)
CONTROL_LOCK=threading.Lock()
CHILD=None
LAUNCH_CONFIG={}

def read_json(name,default=None):
    try:return read_json_shared(RUNTIME/name)
    except (OSError,ValueError):return default
def pid_alive(pid):
    if type(pid)is not int or pid<=0:return False
    if os.name=='nt':
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x100000|0x1000,False,pid)
        if not handle:return False
        try:return kernel.WaitForSingleObject(ctypes.c_void_p(handle),0)==258
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))
    try:os.kill(pid,0);return True
    except PermissionError:return True  # The process exists but belongs to another user.
    except ProcessLookupError:return False
    except OSError:return False
def live_pid():
    global CHILD
    # Reap an exited owned child before a POSIX PID probe: kill(pid, 0) also
    # succeeds for zombies, which would otherwise keep a stale lock alive.
    child=CHILD
    child_alive=child is not None and child.poll() is None
    try:
        pid=int((RUNTIME/'run.lock').read_text())
        if child is not None and not child_alive and pid==child.pid:return None
        if pid_alive(pid):return pid
    except (OSError,ValueError):pass
    # Windows venv launchers can have a different PID from the real interpreter.
    if child_alive:return child.pid
    return None
def start_solver(body):
    global CHILD,LAUNCH_CONFIG
    if live_pid():raise ValueError('A solver is already running.')
    seconds=body.get('seconds',3600);replicas=body.get('replicas',131072);seed=body.get('seed',20261007)
    method=body.get('method','systematic')
    if method not in ('systematic','stochastic','constraint'):raise ValueError('Unknown search method.')
    algorithm=body.get('algorithm','gpu-dfs')
    if algorithm not in ALGORITHMS:raise ValueError('Unknown search algorithm.')
    if method=='stochastic' and 'algorithm' in body:raise ValueError('Choose a search algorithm or legacy stochastic search, not both.')
    if method=='constraint' and 'algorithm' not in body:raise ValueError('Constraint search requires an explicit algorithm.')
    constraint=algorithm!='gpu-dfs'
    if constraint:method='constraint'
    elif method!='stochastic':method='systematic'
    cpu=constraint and algorithm!='hybrid'
    target='gold' if constraint else 'edges'
    if 'target' in body and body['target']!=target:raise ValueError(f'This algorithm searches the {target} target.')
    backend=body.get('backend','auto')
    if backend not in BACKENDS:raise ValueError('Choose GPU backend auto, cuda, or webgpu.')
    if not cpu and sys.platform=='darwin' and (backend=='cuda' or method=='stochastic'):raise ValueError('CUDA is unavailable on macOS. Choose a CPU algorithm or auto/webgpu for Metal.')
    if method=='stochastic' and backend=='webgpu':raise ValueError('The stochastic engine requires CUDA; choose another algorithm for other GPUs.')
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or type(replicas)is not int or type(seed)is not int:
        raise ValueError('Duration must be numeric; replicas and seed must be integers.')
    if (seconds!=0 and not 1<=seconds<=86400) or not 128<=replicas<=131072 or not 0<=seed<2**32:raise ValueError('Choose Unlimited (0) or 1..86400 seconds, 128..131072 replicas, and a nonnegative 32-bit seed.')
    seconds=float(seconds)
    # Only remove a lock after its recorded process has ended.
    (RUNTIME/'run.lock').unlink(missing_ok=True)
    # Clear a stale request before spawn; the runner must not erase a new request.
    (RUNTIME/'stop.request').unlink(missing_ok=True)
    script='constraint_search.py' if constraint else 'systematic_search.py' if method=='systematic' else 'solve.py'
    cmd=[sys.executable,str(ROOT/script),'--resume','--stop-file-initialized','--seconds',str(seconds),'--replicas',str(replicas),'--seed',str(seed)]
    if method!='stochastic':cmd.extend(['--backend',backend])
    if constraint:cmd.extend(['--algorithm',algorithm])
    with (RUNTIME/'solver.log').open('a',encoding='utf-8')as log:
        log.write('\nStarted '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n');log.flush()
        CHILD=subprocess.Popen(cmd,cwd=str(ROOT),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt'else 0)
    actual_algorithm=algorithm if method!='stochastic' else 'stochastic'
    actual_backend='cpu' if cpu else backend
    actual_replicas=0 if cpu else replicas
    LAUNCH_CONFIG={'method':method,'algorithm':actual_algorithm,'target':target,'backend':actual_backend,
                  'launcher_pid':CHILD.pid,'replicas':actual_replicas,'requested_replicas':replicas,
                  'time_limit_seconds':seconds,'seed':seed,'resume':True,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    return {'started':True,'pid':CHILD.pid,'replicas':actual_replicas,'backend':actual_backend,
            'algorithm':actual_algorithm,'method':method,'target':target}

def request_stop():
    pid=live_pid()
    if pid is None:return {'stop_requested':False,'process_running':False}
    (RUNTIME/'stop.request').write_text('stop from dashboard',encoding='utf-8')
    return {'stop_requested':True,'process_running':True,'display_state':'stopping'}

def startup_status(pid):
    """Use only configuration belonging to the active process, never an old run."""
    status={'state':'starting','pid':pid}
    config=read_json('run-config.json',{})
    if isinstance(config,dict) and config.get('pid')==pid:
        for field in ('method','algorithm','target','backend','replicas','seed','started_at','run_id'):
            if field in config:status[field]=config[field]
        if 'seconds' in config:status['time_limit_seconds']=config['seconds']
    child_active=CHILD is not None and callable(getattr(CHILD,'poll',None)) and CHILD.poll() is None
    if child_active or LAUNCH_CONFIG.get('launcher_pid')==pid:
        for field in ('method','algorithm','target','backend','replicas','time_limit_seconds','seed','started_at'):
            if field in LAUNCH_CONFIG:status[field]=LAUNCH_CONFIG[field]
    args=getattr(CHILD,'args',None) if child_active else None
    if isinstance(args,(tuple,list)):
        for flag,field,cast in (('--replicas','replicas',int),('--seconds','time_limit_seconds',float),('--seed','seed',int),('--backend','backend',str),('--algorithm','algorithm',str)):
            try:status[field]=cast(args[args.index(flag)+1])
            except (ValueError,IndexError,TypeError):pass
        scripts={Path(str(value)).name for value in args}
        if 'constraint_search.py' in scripts:status['method']='constraint'
        elif 'systematic_search.py' in scripts:status['method']='systematic'
        elif 'solve.py' in scripts:status['method']='stochastic'
    algorithm=status.get('algorithm')
    if status.get('method')=='constraint' or algorithm in ALGORITHMS[1:]:
        status.update(method='constraint',target='gold')
        cpu=algorithm!='hybrid'
        if cpu:status.update(replicas=0,backend='cpu')
        restoring=algorithm in ALGORITHMS[1:] and (RUNTIME/'searches'/(algorithm+'-gold')/'checkpoint.json').exists()
        status['startup_phase']='restoring_checkpoint' if restoring else 'building_model' if cpu else 'initializing_gpu'
        if algorithm in ('dfs','hybrid'):
            status['phase_message']='Restoring exact Gold search branches.' if restoring else ('Preparing CPU constraints and exact DFS.' if cpu else 'Preparing GPU sampling and constrained DFS.')
        else:
            status['phase_message']='Loading saved Gold constraints and rebuilding the CPU model.' if restoring else 'Building the CPU Gold constraint model.'
    else:
        systematic=status.get('method')=='systematic'
        if systematic:status.setdefault('algorithm','gpu-dfs')
        elif status.get('method')=='stochastic':status.setdefault('algorithm','stochastic')
        status.setdefault('target','edges')
        restoring=(RUNTIME/('systematic/checkpoint.npz' if systematic else 'checkpoint.npz')).exists()
        status['startup_phase']='restoring_checkpoint' if restoring else 'initializing_gpu'
        status['phase_message']=('Restoring disjoint branches and saved cursors.' if restoring else 'Preparing disjoint GPU search branches.') if systematic else ('Restoring saved search and validating its cache.' if restoring else 'Preparing the GPU search.')
    return status

def state_payload():
    pid=live_pid();status=read_json('status.json',{})
    if not isinstance(status,dict):status={}
    if pid and status.get('pid')!=pid:status=startup_status(pid)
    stop_requested=bool(pid) and (RUNTIME/'stop.request').exists()
    # Preserve the runner's actual phase, including checkpoint telemetry.
    display_state=status.get('state','idle')
    if stop_requested and display_state!='saving_checkpoint':display_state='stopping'
    return {'service':'diamond-dilemma','installation_root':str(ROOT.resolve()),'status':status,'display_state':display_state,
        'stop_requested':stop_requested,'best':read_json('best.json'),
        'live':read_json('live.json'),'history':read_json('history.json',[]),
        'process_running':bool(pid),'control_token':TOKEN}

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*_):pass
    def allowed_host(self):
        return self.headers.get('Host','')in{f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
    def send(self,status,payload,kind='application/json; charset=utf-8'):
        data=json.dumps(payload).encode()if kind.startswith('application/json')else payload
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(len(data)));self.send_header('Cache-Control','no-store');self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers();self.wfile.write(data)
    def do_GET(self):
        if not self.allowed_host():return self.send(403,{'error':'Loopback host required'})
        path=urlsplit(self.path).path
        if path=='/':
            return self.send(200,(ROOT/'Dashboard.html').read_bytes(),'text/html; charset=utf-8')
        if path=='/api/puzzle':
            from geometry import build_board
            raw=json.loads((ROOT/'data'/'tiles.json').read_text(encoding='utf-8-sig'))
            return self.send(200,{'tiles':raw['tiles'],'geometry':build_board().as_dict(),'source_url':raw.get('source_url')})
        if path=='/api/state':
            return self.send(200,state_payload())
        if path=='/favicon.ico':return self.send(204,b'','image/x-icon')
        return self.send(404,{'error':'Not found'})
    def do_POST(self):
        if not self.allowed_host():return self.send(403,{'error':'Loopback host required'})
        origin=self.headers.get('Origin')
        valid_origins={f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'}
        if origin is not None and origin not in valid_origins:return self.send(403,{'error':'Origin rejected'})
        if not secrets.compare_digest(self.headers.get('X-Diamond-Control',''),TOKEN):return self.send(403,{'error':'Control token rejected'})
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<=size<=4096:raise ValueError('Request too large')
            if not self.headers.get('Content-Type','').startswith('application/json'):raise ValueError('JSON required')
            body=json.loads(self.rfile.read(size)or b'{}')
            if not isinstance(body,dict):raise ValueError('JSON object required')
            with CONTROL_LOCK:
                path=urlsplit(self.path).path
                if path=='/api/start':return self.send(200,start_solver(body))
                if path=='/api/stop':
                    return self.send(200,request_stop())
                return self.send(404,{'error':'Not found'})
        except (ValueError,OSError)as exc:return self.send(400,{'error':str(exc)})
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--port',type=int,default=8766);args=parser.parse_args()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'Diamond Dilemma dashboard: http://127.0.0.1:{args.port}/',flush=True)
    server.serve_forever()
