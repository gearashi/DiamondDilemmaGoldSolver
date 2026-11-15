"""Price the native formula with small-denominator covers and independently checked combinations."""
import argparse
import json
from pathlib import Path

from bench_r35_contact_arity import extra_cnf
from bench_r35_loss_boundary344 import problem_scope
from r32_gold_syndrome import OutsideSyndromes
from r35_contact_certificate import verify
from r35_integer_sat import InventorySAT


def build(tiles,geo,paths,proofs,cap,selection):
    p,ref,gs,req=problem_scope(tiles)
    packed,_,outside=OutsideSyndromes.layout(ref,gs[:13],req)
    read=lambda f,k:[frozenset(map(tuple,g)) for g in json.loads(f.read_bytes())[k]]
    groups=[*gs[:13],*(g-req for g in read(geo,'geographic_cores')),*(g-req for g in read(paths,'edge_perfect_cores') if len(g)<=13)]
    m=InventorySAT(p,packed,req,outside,cap,native=None,additional_groups=[g for g in groups if g not in packed])
    pool=m.clauses[:]
    meta=[]
    for index in selection:
        proof=proofs[index]
        bound=verify(p,ref,req,gs[:13],proof,charged=[g-req for g in groups])
        if not bound:raise ValueError('The selected weights are not covered by the physical source.')
        top=max(abs(v) for c in pool for v in c)
        extra=extra_cnf(proof,req,ref-req,m.physical_losses,cap,'exact',top)
        pool.extend(extra);meta.append(len(extra))
    return pool,meta


def run(tiles,geo,paths,weights,caps,out,budget,export,selection):
    from pysat.solvers import Solver
    proofs=json.loads(weights.read_bytes())['certificates'];results=[]
    for cap in caps:
        for label in selection:
            idx=[int(i) for i in label.split(',')]
            clauses,sizes=build(tiles,geo,paths,proofs,cap,idx)
            if export:
                with Path(str(export).format(cap=cap,ids=label.replace(',','-'))).open('w',encoding='ascii',newline='\n') as f:
                    f.write(f'p cnf {max(abs(v) for c in clauses for v in c)} {len(clauses)}\n')
                    for c in clauses:f.write(' '.join(map(str,c))+' 0\n')
            with Solver(name='cadical153',bootstrap_with=clauses) as s:
                s.conf_budget(budget);ans=s.solve_limited()
                row=dict(cap=cap,selection=idx,extra=sizes,answer=ans,**s.accum_stats())
                if ans:row['phase']=hex(sum(1<<(x-1) for x in s.get_model() if 0<x<=160))
                results.append(row)
            out.write_text(json.dumps(results,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for arg in ('tiles','geo','paths','weights','out'):cli.add_argument('--'+arg,required=True,type=Path)
    cli.add_argument('--caps',type=int,nargs='+',default=[29,30]);cli.add_argument('--budget',type=int,default=15000)
    cli.add_argument('--export',type=Path);cli.add_argument('--selection',nargs='+',default=['0','1','0,1','0,1,2'])
    a=cli.parse_args();run(**vars(a))
