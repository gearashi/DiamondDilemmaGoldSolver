"""Partition the Euler cap-21 decision by the attained ten-group phase minimum."""
import argparse
from itertools import product
from math import inf
from pathlib import Path
from time import monotonic
from bench_r35_loss_euler import euler_clause,minimal_named
from bench_r35_loss_tight import InventorySAT,original,problem_scope,tight_clauses,read_json,physical_price
from r31_gold_connectivity import PhaseInterface
from io_utils import atomic_json


def lifted(base,n,top):
    shift=top-n
    return [[(1 if v>0 else -1)*(abs(v)+shift) if abs(v)>n else v for v in clause] for clause in base]


def run(tiles,case,seconds,output):
    from pysat.solvers import Solver
    p,ref,g,req=problem_scope(tiles);g=g[:13];_,_,pack,_,outside=original(tiles);data=PhaseInterface(p)
    cnf=InventorySAT(p,pack,req,outside,21,excess=True,projected=True,additional_groups=g,native=None)
    def highest(rows):return max(abs(v) for c in rows for v in c)
    balance,_=minimal_named(p,g,req,cnf.physical_losses,highest(cnf.clauses))
    euler,_=euler_clause(data,max(highest(balance),highest(cnf.clauses)))
    rows=cnf.clauses+balance+euler;top=highest(rows)
    if case=='ten20':
        ten=InventorySAT(p,pack,req,outside,20,excess=True,projected=True,native=None)
        rows+=lifted(ten.clauses,p.n,top)
    elif case=='ten21':
        extra,_=tight_clauses(p,pack,req,outside,cnf.physical_losses,top);rows+=extra
    else:raise ValueError('Expected a disjoint-minimum case.')
    hint=int(read_json('docs/R35_OVERLAP_BOUNDARY.json')['disjoint20']['phase'],16)
    enc=InventorySAT.__new__(InventorySAT);enc.solver=Solver(name='g4',bootstrap_with=rows)
    enc.n=p.n;enc.solver.set_phases([i if hint>>(i-1)&1 else -i for i in range(1,p.n+1)])
    start=monotonic()
    try:
        verdict=enc.decide(seconds);report=dict(case=case,answer=verdict,clauses=len(rows),seconds=monotonic()-start,work=enc.solver.accum_stats())
        if verdict:
            w=enc.phase();price,_=physical_price(p.tiles,ref,g,req,w);weight=sum(r[w>>i&1][-1] for i,r in enumerate(data.rows))
            assert price<=21 and weight in data.euler;report['witness']=dict(phase=hex(w),price=price,euler=weight)
    finally:enc.close()
    atomic_json(output,report);return report

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--tiles',required=True);a.add_argument('--case',choices=('ten20','ten21'),required=True)
    a.add_argument('--seconds',type=float,default=480);a.add_argument('--output',type=Path,required=True);v=a.parse_args()
    print(run(v.tiles,v.case,v.seconds,v.output))
