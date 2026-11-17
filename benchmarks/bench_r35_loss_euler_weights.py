"""Partition the cap21 query by absolute Euler with the original first-tile gauge."""
import argparse
from itertools import product
from pathlib import Path
from time import monotonic
from bench_r35_loss_tight import InventorySAT,original,problem_scope,physical_price
from bench_r35_loss_euler import minimal_named,euler_clause,PhaseInterface
from io_utils import atomic_json


def run(tiles,seconds,out):
    p,ref,g,req=problem_scope(tiles);g=g[:13];_,_,pack,_,outside=original(tiles);data=PhaseInterface(p)
    cnf=InventorySAT(p,pack,req,outside,21,excess=True,projected=True,additional_groups=g)
    strongest,_=minimal_named(p,g,req,cnf.physical_losses,cnf.solver.nof_vars());cnf.solver.append_formula(strongest)
    ec,shape=euler_clause(data,cnf.solver.nof_vars());cnf.solver.append_formula(ec)
    report=[]
    try:
        for mag in (3,1,5):
            trigger=cnf.solver.nof_vars()+1
            for vals in product((0,1),repeat=len(shape['bits'])):
                value=shape['offset']+shape['scale']*sum(x<<b for x,(b,_) in zip(vals,shape['bits']))
                if abs(value)!=mag:
                    cnf.solver.add_clause([-trigger,*[-lit if x else lit for x,(_,lit) in zip(vals,shape['bits'])]])
            t=monotonic();before=cnf.solver.accum_stats();v=cnf.decide(seconds,[trigger]);after=cnf.solver.accum_stats()
            step=dict(abs_euler=mag,signs=[-mag,mag],gauge='phase[0] == 0',seconds=monotonic()-t,
                      answer=v,work={k:after[k]-before[k] for k in after})
            if v:
                w=cnf.phase();cost,_=physical_price(p.tiles,ref,g,req,w);value=sum(row[w>>i&1][-1] for i,row in enumerate(data.rows))
                assert cost<=21 and abs(value)==mag;step['witness']=dict(phase=hex(w),cost=cost,euler=value)
            report.append(step);atomic_json(out,report);print(step,flush=True);cnf.solver.add_clause([-trigger])
    finally:cnf.close()


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--tiles',required=True);a.add_argument('--seconds',type=float,default=150)
    a.add_argument('--output',type=Path,required=True);args=a.parse_args();run(args.tiles,args.seconds,args.output)
