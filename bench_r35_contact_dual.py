"""Find and verify fractional and integer lower bounds on omitted original bonds."""
import argparse
from fractions import Fraction
from functools import reduce
import json
from math import lcm
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope


def scope(problem, groups, required, geometric, contact, length, physical_count, include_geo):
    rows=[]
    for kind, family in [('physical',groups[:physical_count]),('geographic',geometric if include_geo else []),
                         ('edge_perfect',[g for g in contact if len(g)<=length])]:
        for g in family:
            option=frozenset(g)-required
            if option:rows.append((kind,frozenset(g),option))
    return rows


def dual(rows):
    from ortools.linear_solver import pywraplp
    solver=pywraplp.Solver.CreateSolver('GLOP')
    vars={p:solver.NumVar(0,solver.infinity(),'') for p in sorted(set.union(set(),*(g for _,_,g in rows)))}
    constraints=[solver.Add(sum(vars[p] for p in sorted(g))>=1) for _,_,g in rows]
    solver.Minimize(sum(vars.values()));assert solver.Solve()==solver.OPTIMAL
    frac=[max(Fraction(0),Fraction(c.dual_value()).limit_denominator(100000)) for c in constraints]
    denominator=reduce(lcm,(v.denominator for v in frac),1)
    weights=[int(v*denominator) for v in frac]
    maximum=max(sum(v for v,(_,_,g) in zip(weights,rows) if p in g) for p in vars)
    bound=(sum(weights)+maximum-1)//maximum
    proof=[dict(kind=kind,bonds=sorted(orig),weight=w) for w,(kind,orig,_) in zip(weights,rows) if w]
    return dict(lp=solver.Objective().Value(),lower=bound,weight_sum=sum(weights),maximum=maximum,weighted_cores=proof)


def integer(rows,seconds):
    from ortools.sat.python import cp_model
    model=cp_model.CpModel()
    est={p:model.new_bool_var('') for p in sorted(set.union(set(),*(g for _,_,g in rows)))}
    for _,_,g in rows:model.add_bool_or([est[p] for p in sorted(g)])
    model.minimize(sum(est.values()))
    engine=cp_model.CpSolver();engine.parameters.max_time_in_seconds=seconds;engine.parameters.num_search_workers=2
    status=engine.solve(model);verdict=engine.status_name(status)
    out=dict(status=verdict,bound=engine.best_objective_bound,branches=engine.num_branches,conflicts=engine.num_conflicts)
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        lost={x for x,v in est.items() if engine.value(v)}
        if not all(g&lost for _,_,g in rows):raise ArithmeticError('Invalid integral hitting assignment.')
        out.update(upper=len(lost),lost=sorted(lost))
    return out


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True)
    cli.add_argument('--claims',type=Path,required=True)
    cli.add_argument('--geographic',type=Path)
    cli.add_argument('--physical-count',type=int,choices=[0,13,344],default=13)
    cli.add_argument('--lengths',type=int,nargs='+',default=[5,9,13,17])
    cli.add_argument('--integer-seconds',type=float,default=0)
    cli.add_argument('--output',required=True,type=Path)
    a=cli.parse_args();p,ref,physical,req=problem_scope(a.tiles)
    read=lambda path,kind:[frozenset(map(tuple,g)) for g in json.loads(path.read_bytes())[kind]]
    geo=read(a.geographic,'geographic_cores') if a.geographic else []
    contact=read(a.claims,'edge_perfect_cores')
    reports=[]
    for length in a.lengths:
        rows=scope(p,physical,req,geo,contact,length,a.physical_count,bool(a.geographic))
        value=dual(rows)
        if a.integer_seconds:value['integer']=integer(rows,a.integer_seconds)
        value.update(length=length,total_groups=len(rows))
        reports.append(value)
        out=dict(fingerprint=p.fingerprint,named_reference=len(ref),required=sorted(req),reports=reports)
        a.output.write_text(json.dumps(out,separators=(',',':'))+'\n',encoding='utf8')
        print(length,value['lp'],value['lower'],len(value['weighted_cores']),value.get('integer',{}).get('bound'),flush=True)
