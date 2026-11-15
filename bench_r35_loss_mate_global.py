"""Compare fixed-phase direction/conditional Hall support in an unpinned signed loss query."""
import argparse
import hashlib
from pathlib import Path
from time import monotonic
from bench_r35_loss_tight import original,problem_scope,InventorySAT,physical_price,read_json
from bench_r35_loss_euler import minimal_named,euler_clause
from r31_gold_connectivity import PhaseInterface
from r32_gold_bonds import inventory
from r35_signed_mate_support import hall_clauses,rejects
from io_utils import atomic_json


def build(tiles,cap,level,sphere):
    p,ref,g,req=problem_scope(tiles);initial=g[:13]
    _,_,packed,_,outside=original(tiles)
    model=InventorySAT(p,packed,req,outside,cap,excess=True,projected=True,additional_groups=initial,native=None)
    rows=list(model.clauses)
    minimal,_=minimal_named(p,initial,req,model.physical_losses,max(abs(v) for c in rows for v in c));rows+=minimal
    if sphere:
        ec,_=euler_clause(PhaseInterface(p),max(abs(v) for c in rows for v in c));rows+=ec
    zero,hall,work=hall_clauses(p.tiles,allowed=inventory(p).possible if level=='hall' else None,
                                 max_atoms=16 if level=='hall' else 0,max_sides=18)
    if level!='off':rows+=zero
    if level=='hall':rows+=hall
    return p,ref,g,req,rows,dict(zero=len(zero) if level!='off' else 0,hall=len(hall) if level=='hall' else 0,compilation=work)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tiles',required=True);parser.add_argument('--cap',type=int,default=21)
    parser.add_argument('--level',choices=('off','raw','hall'),default='raw');parser.add_argument('--sphere',action='store_true')
    parser.add_argument('--seconds',type=float,default=420);parser.add_argument('--export',action='store_true');parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();p,ref,g,req,rows,report=build(a.tiles,a.cap,a.level,a.sphere)
    dest=a.output.with_suffix('.cnf');top=max(abs(v) for c in rows for v in c)
    with dest.open('w',encoding='ascii',newline='\n') as f:
        f.write(f'p cnf {top} {len(rows)}\n')
        for c in rows:f.write(' '.join(map(str,c))+' 0\n')
    report.update(cap=a.cap,level=a.level,sphere=a.sphere,variables=top,clauses=len(rows),cnf_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    if not a.export:
        from pysat.solvers import Solver
        check=InventorySAT.__new__(InventorySAT);check.solver=Solver(name='g4',bootstrap_with=rows);check.n=p.n
        try:
            t=monotonic();ans=check.decide(a.seconds);report.update(answer=ans,work=check.solver.accum_stats(),seconds=monotonic()-t)
            if ans:
                phi=check.phase();cost,_=physical_price(p.tiles,ref,g,req,phi);assert cost<=a.cap
                report['witness']=dict(phase=hex(phi),cost=cost)
        finally:check.close()
    atomic_json(a.output,report);print(report)
