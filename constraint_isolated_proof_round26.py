"""Optional bounded asynchronous native proof producer behind independent RUP.

The exact input identity includes all clauses, backend and limits. A pending or
failed request is unknown. Startup and native work happen in an owned helper;
callers can stop promptly without abandoning their exact search frontier.
"""
import atexit,hashlib,json,math,os,queue,subprocess,sys,threading,time
from collections import OrderedDict
from pathlib import Path
from constraint_proof_worker_round26 import encoded,identity,validate,MAX_INPUT,MAX_OUTPUT,BACKENDS

ROOT=Path(__file__).resolve().parent

def command():
    worker=ROOT/'constraint_proof_worker_round26.py'
    if sys.platform in ('linux','darwin'):return [sys.executable,'-B',str(worker)]
    if sys.platform!='win32':return None
    python=ROOT/'.venv-wsl-proof'/'bin'/'python3'
    if not (ROOT/'.venv-wsl-proof'/'pyvenv.cfg').is_file():return None
    def linux(path):
        absolute=path.absolute()
        if not absolute.drive or absolute.drive.startswith('\\'):raise ValueError('WSL proof paths require a local drive.')
        return '/mnt/'+absolute.drive[0].lower()+absolute.as_posix()[2:]
    try:configured=json.loads((ROOT/'.venv-wsl-proof'/'diamond-config.json').read_text(encoding='utf-8')).get('distro','Ubuntu')
    except (OSError,ValueError,AttributeError):configured='Ubuntu'
    distro=os.environ.get('DIAMOND_PROOF_WSL_DISTRO',configured)
    if type(distro)is not str or not distro or len(distro)>128:return None
    return ['wsl.exe','-d',distro,'--',linux(python),'-B',linux(worker)]

class IsolatedRUPProducer:
    def __init__(self,argv=None,*,startup_timeout=30,reply_grace=5):
        if any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in (startup_timeout,reply_grace)):raise ValueError('Invalid proof worker timeout.')
        self.startup_timeout=startup_timeout;self.reply_grace=reply_grace
        self.argv=command() if argv is None else list(argv)
        self.records=OrderedDict();self.pending=None;self.closed=False;self.failed=False;self.process=None
        self.lock=threading.Lock();self.requests=queue.Queue(maxsize=1);self.thread=None
        self.stats=dict.fromkeys(('requests','ready_hits','pending_hits','rejected','native_decisions','starts'),0)
        self.source=hashlib.sha256((ROOT/'constraint_proof_worker_round26.py').read_bytes()).hexdigest()

    def _start(self):
        if self.thread is None:
            self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start()

    def _readline(self,timeout):
        # A wedged supervisor cannot retain an owned process indefinitely.
        process=self.process
        def expire():
            if process is not None and process.poll() is None:
                try:process.kill()
                except OSError:pass
        timer=threading.Timer(timeout,expire);timer.daemon=True;timer.start()
        try:return process.stdout.readline(MAX_OUTPUT+1)
        finally:timer.cancel()

    def _run(self):
        try:
            self.process=subprocess.Popen(self.argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            self.stats['starts']+=1
            if self.closed:return
            ready=self._readline(self.startup_timeout)
            if json.loads(ready)!={'ready':1}:return
            while not self.closed:
                request=self.requests.get()
                if request is None:return
                self.process.stdin.write(encoded(request)+b'\n');self.process.stdin.flush()
                line=self._readline(request['seconds']+self.reply_grace)
                envelope=json.loads(line) if len(line)<=MAX_OUTPUT else None
                result=envelope.get('result') if isinstance(envelope,dict) and envelope.get('identity')==request['identity'] else None
                valid=(isinstance(result,dict) and result.get('identity')==request['identity'] and result.get('version')==1
                       and result.get('source')==self.source and result.get('python_sat')=='1.9.dev15')
                proof=result.get('proof') if valid else None
                if proof is not None and (not isinstance(proof,list) or len(proof)>4096 or any(type(x)is not str or len(x)>16384 for x in proof) or sum(map(len,proof))>262144):proof=None
                with self.lock:
                    self.records[request['identity']]=proof
                    while len(self.records)>8:self.records.popitem(last=False)
                    self.stats['native_decisions']+=max(0,int(result.get('decisions',0))) if valid else 0
                    self.stats['rejected']+=int(proof is None)
                    self.pending=None
        except (OSError,ValueError,TypeError,BrokenPipeError):
            with self.lock:
                if self.pending is not None:self.records[self.pending]=None
                self.pending=None;self.stats['rejected']+=1
        finally:
            self.failed=True
            with self.lock:
                if self.pending is not None:self.records[self.pending]=None;self.pending=None;self.stats['rejected']+=1
            if self.process is not None:
                try:self.process.stdin.close()
                except (OSError,ValueError):pass
                try:self.process.wait(timeout=8)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait()
                self.process.stdout.close()

    def produce_rup(self,clauses,check=lambda:None,*,seconds=.1,conflicts=10000,backend='g4'):
        if self.closed or self.failed or self.argv is None:return None
        if backend not in BACKENDS:raise ValueError('Unknown isolated proof backend.')
        if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('Invalid proof query duration.')
        if not seconds:return None
        request={'version':1,'clauses':[list(row) for row in clauses],'backend':backend,
                 'seconds':.25,'conflicts':conflicts}
        request['identity']=identity(request);validate(request)
        if len(encoded(request))>MAX_INPUT:return None
        check();until=time.monotonic()+seconds
        self._start()
        with self.lock:
            if request['identity'] in self.records:
                self.stats['ready_hits']+=1;value=self.records[request['identity']]
                return None if value is None else list(value)
            if self.pending is None:
                self.pending=request['identity'];self.stats['requests']+=1;self.requests.put_nowait(request)
            elif self.pending==request['identity']:self.stats['pending_hits']+=1
            else:return None
        while time.monotonic()<until:
            check()
            with self.lock:
                if request['identity'] in self.records:
                    value=self.records[request['identity']];return None if value is None else list(value)
            time.sleep(min(.005,max(0,until-time.monotonic())))
        return None

    def close(self):
        self.closed=True
        try:self.requests.put_nowait(None)
        except queue.Full:pass
        if self.process is not None:
            # This helper owns only optional proof work, never search progress.
            try:self.process.kill()
            except OSError:pass
        if self.thread is not None:self.thread.join(timeout=.05)

_PROVIDER=None

def provider():
    global _PROVIDER
    if _PROVIDER is None or _PROVIDER.closed:
        _PROVIDER=IsolatedRUPProducer();atexit.register(_PROVIDER.close)
    return _PROVIDER
