"""Quiescent native SAT worker for reconnecting local controllers.

Only the worker owns native state. The local socket transports
bounded JSON and accepts the per-process authentication key. Reload an exact
frontier after losing the worker. No native CDCL ABI is serialized to disk.
"""
import json
import os
import secrets
import subprocess
import sys
from multiprocessing.connection import Client,Listener
from pathlib import Path
from threading import Timer

LIMIT = 16*1024*1024


def _encode(value):
    return json.dumps(value,separators=(',',':')).encode('ascii')


def _send(conn,value):
    conn.send_bytes(_encode(value))


def _receive(conn):
    return json.loads(conn.recv_bytes(LIMIT))


def _slice(solver,milliseconds):
    timer = Timer(milliseconds/1000,solver.interrupt)
    timer.start()
    try:
        result = solver.solve_limited(expect_interrupt=True)
    finally:
        timer.cancel()
        timer.join()
        solver.clear_interrupt()
    return result


def serve(clauses):
    from pysat.solvers import Glucose4
    key = secrets.token_bytes(32)
    # Bind only local interfaces. Neither endpoint uses pickle.
    family = 'AF_INET' if os.name=='nt' else 'AF_UNIX'
    address = ('127.0.0.1',0) if family=='AF_INET' else None
    with Listener(address=address,family=family,authkey=key) as listener, Glucose4(bootstrap_with=clauses) as solver:
        print(json.dumps(dict(address=listener.address,auth=key.hex(),family=family,pid=os.getpid())),flush=True)
        answer = None
        calls = 0
        while True:
            with listener.accept() as conn:
                request = _receive(conn)
                op = request.get('op')
                if op=='slice':
                    ms = request.get('milliseconds')
                    if type(ms) is not int or not 1<=ms<=60000:
                        _send(conn,{'error':'invalid time slice'})
                        continue
                    if answer is None:
                        answer = _slice(solver,ms)
                        calls += 1
                elif op not in ('status','close'):
                    _send(conn,{'error':'invalid operation'})
                    continue
                _send(conn,dict(pid=os.getpid(),answer=answer,calls=calls,stats=solver.accum_stats(),
                                model=solver.get_model() if answer else None))
                if op=='close':
                    return


def request(endpoint,command):
    address = tuple(endpoint['address']) if endpoint['family']=='AF_INET' else endpoint['address']
    with Client(address,family=endpoint['family'],authkey=bytes.fromhex(endpoint['auth'])) as conn:
        _send(conn,command)
        return _receive(conn)


def session_probe(clauses,*,milliseconds=25,max_slices=32):
    """Two separate controller processes reconnect to the same paused worker."""
    if type(max_slices) is not int or max_slices<1 or type(milliseconds) is not int or not 1<=milliseconds<=60000:
        raise ValueError('Invalid retained process budget.')
    path = str(Path(__file__).resolve())
    command = [sys.executable,'-B',path]
    hidden = subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
    payload = _encode(clauses)
    if len(payload)+1>LIMIT:
        raise ValueError('Formula exceeds the protocol limit.')
    with subprocess.Popen([*command,'worker'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                          creationflags=hidden) as worker:
        try:
            worker.stdin.write(payload+b'\n')
            worker.stdin.close()
            descriptor = worker.stdout.readline(4096)
            if not descriptor:
                raise RuntimeError('Worker failed to start.')
            endpoint = json.loads(descriptor)
            def controller(op):
                # Transfer credentials over input, never an argument or a persisted record.
                reply = subprocess.run([*command,'controller'],input=_encode([endpoint,op]),capture_output=True,
                                       timeout=10+milliseconds/1000,check=True,creationflags=hidden)
                return json.loads(reply.stdout)
            first = controller(dict(op='slice',milliseconds=milliseconds))
            second = controller(dict(op='status'))
            if first!=second or first['pid']!=endpoint['pid']:
                raise RuntimeError('Quiescent reconnect changed native state or process.')
            final = second
            while final['answer'] is None and final['calls']<max_slices:
                final = request(endpoint,dict(op='slice',milliseconds=milliseconds))
            if final['answer'] is True:
                witness = set(final['model'])
                if any(not any(lit in witness for lit in row) for row in clauses):
                    raise RuntimeError('Retained worker returned an invalid model.')
            request(endpoint,dict(op='close'))
            worker.wait(timeout=5)
            if worker.returncode:
                raise RuntimeError('Retained worker did not exit cleanly.')
            return dict(reconnected=True,first_complete=first['answer'] is not None,answer=final['answer'],
                        slices=final['calls'],decisions=final['stats']['decisions'],conflicts=final['stats']['conflicts'])
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.wait()


if __name__=='__main__':
    if sys.argv[1:] == ['worker']:
        serve(json.loads(sys.stdin.buffer.readline(LIMIT)))
    elif sys.argv[1:] == ['controller']:
        endpoint,operation = json.loads(sys.stdin.buffer.read(65536))
        print(json.dumps(request(endpoint,operation)))
    else:
        raise SystemExit('Use the solver CLI process-session method.')
