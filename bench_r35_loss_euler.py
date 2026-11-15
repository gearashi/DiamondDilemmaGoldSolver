"""Spherical Euler and minimal named departures under all original physical groups."""
import argparse
import json
from itertools import product
from pathlib import Path
from time import monotonic
from bench_r35_loss_boundary344 import build,problem_scope,verify_21
from bench_r35_loss_proof import original,parity_conflict
from r29_gold_phase import EndpointCuts
from r31_gold_connectivity import PhaseInterface
from r32_gold_loss import read_json
from r35_integer_sat import InventorySAT
from r35_loss_witness import physical_price,raw_sides,cover_missing
from io_utils import atomic_json


from r35_signed_conditions import euler_clauses as euler_clause


def minimal_named(p,groups,required,losses,top):
    # Choose an inclusion-minimal representative for an upper-cap decision.
    # A voluntary loss then uniquely hits some compulsory departure group.
    sides=EndpointCuts(p).sides;raw=raw_sides(p.tiles)
    cleanable=[set(g)-required for g in groups if not parity_conflict(raw,set(g)|required)]
    clauses=[];aux=0
    for pair,lost in sorted(losses.items()):
        a,b=pair;x,y=a//3+1,b//3+1;rhs=sides[a][1]^sides[b][1]^(sides[a][0].bit_count()%2)
        alternatives=[]
        for group in cleanable:
            if pair not in group:continue
            top+=1;aux+=1;alternatives.append(top)
            for other in sorted(group-{pair}):clauses.append([-top,-losses[other]])
        t=-y if rhs else y
        clauses.extend([[-lost,x,t,*alternatives],[-lost,-x,-t,*alternatives]])
    return clauses,dict(auxiliary_private_groups=aux,cleanable_sizes=sorted(map(len,cleanable)))


def status(tiles,cap,arm,seconds,output,workers):
    p,ref,all_groups,req=problem_scope(tiles);g=all_groups[:13];_,_,pack,_,outside=original(tiles);iface=PhaseInterface(p)
    hint=int(read_json('docs/R35_OVERLAP_BOUNDARY.json')['disjoint20']['phase'],16)
    def euler(w):return sum(row[w>>i&1][-1] for i,row in enumerate(iface.rows))
    pure,w21,_=verify_21(p,ref,all_groups,req)
    price,fail=physical_price(p.tiles,ref,all_groups,req,hint);lost=(fail&ref)|cover_missing(all_groups,req,fail)
    assert price==len(lost)==22 and euler(hint) in iface.euler and euler(pure) not in iface.euler
    result=dict(cap=cap,arm=arm,groups=len(g),upper22=dict(phase=hex(hint),euler=euler(hint),lost=sorted(lost)),
                excluded21=dict(phase=hex(pure),euler=euler(pure)),allowed=sorted(iface.euler));start=monotonic()
    if arm=='cp':
        from ortools.sat.python import cp_model
        m,phase,loss=build(p,ref,g,req,cap,hint,lost)
        expr=sum(low[-1]+(high[-1]-low[-1])*v for v,(low,high) in zip(phase,iface.rows))
        m.add_linear_expression_in_domain(expr,cp_model.Domain.from_values(sorted(iface.euler)))
        s=cp_model.CpSolver();s.parameters.max_time_in_seconds=seconds;s.parameters.num_search_workers=workers
        s.parameters.random_seed=3577;s.parameters.linearization_level=2;res=s.solve(m)
        answer=True if res in (cp_model.OPTIMAL,cp_model.FEASIBLE) else False if res==cp_model.INFEASIBLE else None
        result['work']=dict(branches=s.num_branches,conflicts=s.num_conflicts)
        if answer:w=sum(s.value(v)<<i for i,v in enumerate(phase))
    else:
        native=InventorySAT(p,pack,req,outside,cap,excess=True,projected=True,additional_groups=g,native=None if arm=='export' else 'g4')
        top=max(abs(v) for c in native.clauses for v in c)
        own,detail=minimal_named(p,g,req,native.physical_losses,top)
        if arm=='basic':own=[]
        ec,shape=euler_clause(iface,max([top]+[abs(v) for c in own for v in c]))
        result.update(ownership=detail,euler_encoding=shape,clauses=len(native.clauses)+len(own)+len(ec))
        if arm=='export':
            rows=native.clauses+own+ec
            dest=output.with_suffix('.cnf')
            with dest.open('w',encoding='ascii',newline='\n') as f:
                f.write(f'p cnf {max(abs(v) for c in rows for v in c)} {len(rows)}\n')
                for c in rows:f.write(' '.join(map(str,c))+' 0\n')
            result['cnf']=dest.name;atomic_json(output,result);return result
        try:
            native.solver.append_formula(own+ec);native.solver.set_phases([i if hint>>(i-1)&1 else -i for i in range(1,p.n+1)])
            answer=native.decide(seconds);result['work']=native.solver.accum_stats()
            if answer:w=native.phase()
        finally:native.close()
    result.update(answer=answer,seconds=monotonic()-start)
    if answer:
        c,_=physical_price(p.tiles,ref,all_groups,req,w);assert c<=cap and euler(w) in iface.euler
        result['witness']=dict(phase=hex(w),cost=c,euler=euler(w))
    atomic_json(output,result);return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',required=True);cli.add_argument('--cap',type=int,default=21)
    cli.add_argument('--arm',choices=('basic','minimal','cp','export'),required=True);cli.add_argument('--seconds',type=float,default=600)
    cli.add_argument('--workers',type=int,default=3);cli.add_argument('--output',type=Path,required=True);a=cli.parse_args()
    r=status(a.tiles,a.cap,a.arm,a.seconds,a.output,a.workers);print(r.get('answer'),r.get('work'),r.get('witness'),flush=True)
