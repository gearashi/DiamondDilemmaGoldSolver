"""Export and independently check the sharpened reference-loss boundary proof."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from time import monotonic
from threading import Timer

from bench_r35_loss_proof import original
from bench_r35_loss_tight import problem_scope,verify_21,tight_clauses,InventorySAT
from r32_gold_loss import digest
from r35_loss_bound import CHECKED_LOWER_BOUND, REFERENCE_SCOPE


def export(tiles,groups_count,destination):
    p,ref,g,req=problem_scope(tiles);_,_,packed,_,outside=original(tiles)
    assert digest([3,p.fingerprint,sorted(ref),sorted(req),[sorted(j) for j in packed]])==REFERENCE_SCOPE and CHECKED_LOWER_BOUND==20
    if groups_count==13:g=g[:13]
    elif groups_count in (11,12):
        extra=sorted((row for row in g[:13] if row not in packed),key=len)
        g=[*packed,*extra[:groups_count-10]]
    elif groups_count==10:g=list(packed)
    elif groups_count!=len(g):raise ValueError('Expected the checked 10–13 or 344 groups.')
    sat=InventorySAT(p,packed,req,outside,20,native=None,excess=True,projected=True,additional_groups=g)
    ext,stats=tight_clauses(p,packed,req,outside,sat.physical_losses,max(abs(v) for c in sat.clauses for v in c))
    clauses=sat.clauses+ext;highest=max(abs(v) for c in clauses for v in c)
    destination.mkdir(parents=True,exist_ok=True)
    dest=destination/f'cap20-{len(g)}.cnf'
    with dest.open('w',encoding='ascii',newline='\n') as f:
        f.write(f'p cnf {highest} {len(clauses)}\n')
        for c in clauses:f.write(' '.join(map(str,c))+' 0\n')
    ph,_,verified=verify_21(p,ref,g,req) if groups_count!=10 else (0,None,None)
    return dict(groups=len(g),clauses=len(clauses),base_clauses=len(sat.clauses),extra_clauses=len(ext),variables=highest,
                cnf_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),phase=hex(ph),upper=verified,ownership=stats)

def prove(tiles,group_count,destination,seconds):
    import os,tempfile
    from pysat.solvers import Glucose4
    from pysat.formula import CNF
    evidence=export(tiles,group_count,destination);base=destination/f'cap20-{group_count}';start=monotonic()
    cnf=CNF(from_file=str(base)+'.cnf');hint=int(evidence['phase'],16)
    before=evidence['base_clauses']
    # Preserve the model construction and native decision state of the uninstrumented trial.
    tempfile.tempdir=str(destination.resolve())
    with Glucose4(bootstrap_with=cnf.clauses[:before],with_proof=True) as sat:
        sat.append_formula(cnf.clauses[before:])
        sat.set_phases([i if hint>>(i-1)&1 else -i for i in range(1,161)])
        timer=Timer(seconds,sat.interrupt);timer.daemon=True;timer.start()
        try:answer=sat.solve_limited(expect_interrupt=True)
        finally:timer.cancel();timer.join()
        evidence.update(status=str(answer),work=sat.accum_stats(),seconds=monotonic()-start)
        if answer is False:
            if os.name=='nt':
                # Glucose writes via the C runtime while PySAT reads via Python.
                # Flush the native stream before reading its final empty clause.
                import ctypes
                ctypes.CDLL('ucrtbase').fflush(None)
                ctypes.CDLL('msvcrt').fflush(None)
            path=Path(str(base)+'.drat');sat.prfile.seek(0)
            with path.open('wb') as f:shutil.copyfileobj(sat.prfile,f)
            evidence['proof_bytes']=path.stat().st_size
            with path.open('rb') as f:
                evidence['proof_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
                f.seek(-3,2)
                if f.read()!=b'\n0\n':raise IOError('Incomplete native proof trace.')
        if answer is True:evidence['solution']=hex(sum(1<<(v-1) for v in sat.get_model() if 0<v<=160))
    Path(str(base)+'-proof.json').write_text(json.dumps(evidence,indent=2)+'\n',encoding='utf8')
    return evidence

if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('action',choices=('export','prove'))
    cli.add_argument('--tiles',required=True);cli.add_argument('--groups',type=int,default=344);cli.add_argument('--directory',type=Path,required=True)
    cli.add_argument('--seconds',type=float,default=1100);args=cli.parse_args()
    if args.action=='prove':row=prove(args.tiles,args.groups,args.directory,args.seconds)
    else:row=export(args.tiles,args.groups,args.directory)
    print(json.dumps(row,indent=2))
