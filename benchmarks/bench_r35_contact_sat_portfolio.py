"""Compare installed SAT backends on an identical authenticated contact-loss formula."""
import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic
from threading import Timer


def run(cnf,solver_name,seconds,out,hint,proof,conflicts=0):
    from pysat.solvers import Solver
    from pysat.formula import CNF
    src=CNF(from_file=str(cnf))
    report=dict(cnf=cnf.name,sha256=hashlib.sha256(cnf.read_bytes()).hexdigest(),backend=solver_name,seconds_limit=None if conflicts else seconds)
    start=monotonic()
    with Solver(name=solver_name,bootstrap_with=src.clauses,with_proof=bool(proof)) as solver:
        if hint:solver.set_phases([i+1 if (hint>>i)&1 else -i-1 for i in range(160)])
        if conflicts:
            solver.conf_budget(conflicts);answer=solver.solve_limited()
            report['requested_conflicts']=conflicts
        elif solver_name in ('cadical153','cadical195'):
            answer=None
            deadline=start+seconds
            while answer is None and monotonic()<deadline:
                solver.conf_budget(128)
                answer=solver.solve_limited()
        else:
            timer=Timer(seconds,solver.interrupt);timer.daemon=True;timer.start()
            try:answer=solver.solve_limited(expect_interrupt=True)
            finally:timer.cancel();timer.join()
        report.update(answer=answer,seconds=monotonic()-start,**solver.accum_stats())
        if answer:
            report['phase']=hex(sum(1<<(v-1) for v in solver.get_model() if 0<v<=160))
        elif answer is False and proof:
            import os,shutil
            if os.name=='nt':
                import ctypes
                ctypes.CDLL('ucrtbase').fflush(None)
                ctypes.CDLL('msvcrt').fflush(None)
            with proof.open('wb') as f:
                solver.solver.prfile.seek(0);shutil.copyfileobj(solver.solver.prfile,f)
            report['proof_bytes']=proof.stat().st_size
    out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    c=argparse.ArgumentParser(description=__doc__)
    for x in ('cnf','out'):c.add_argument('--'+x,type=Path,required=True)
    c.add_argument('--solver-name',default='glucose4');c.add_argument('--seconds',type=float,default=120)
    c.add_argument('--hint',default=0,type=lambda x:int(x,0));c.add_argument('--proof',type=Path);c.add_argument('--conflicts',type=int,default=0)
    a=c.parse_args();run(**vars(a))
