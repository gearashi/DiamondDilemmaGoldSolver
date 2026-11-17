"""Construct and optimize a complete physical signed-inventory witness with a global loss cutoff."""
import argparse
from collections import Counter
import json
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope,build
from r35_contact_certificate import CERTIFICATE
from r35_loss_witness import cover_missing,physical_price


def run(tiles,geo,path,out,upper,lower,seconds,seed,workers):
    from ortools.sat.python import cp_model
    p,ref,groups,req=problem_scope(tiles)
    read=lambda f,k:[frozenset(map(tuple,g)) for g in json.loads(f.read_bytes())[k]]
    gs=[*groups[:13],*(g-req for g in read(geo,'geographic_cores')),
        *(g-req for g in read(path,'edge_perfect_cores') if len(g)<=13)]
    phase=0xce55aaa6e89109fca073547e8c580ba32e0e5fa2
    _,failed=physical_price(p.tiles,ref,gs,req,phase);hint=(ref&failed)|cover_missing(gs,req,failed)
    m,ph,loss=build(p,ref,gs,req,upper,phase,hint)
    if upper<lower:raise ValueError('The requested attainment cap is below the independently certified floor.')
    # At an attained cap, any cheaper unsigned voluntary departure assignment
    # can be padded up to it. The raw signed repricer recovers its actual cost.
    cert=json.loads(CERTIFICATE.read_bytes())
    degree=Counter();addons=[]
    for r in cert['weighted_cores']:
        bonds=frozenset(map(tuple,r['bonds']))-req;w=r['weight']
        for b in bonds:degree[b]+=w
        n=m.new_int_var(0,len(bonds)-1,'');m.add(n==sum(loss[b] for b in bonds)-1);addons.append((n,w))
    slack=upper*cert['maximum']-cert['weight_sum']
    m.add(sum((cert['maximum']-degree[b])*v for b,v in loss.items())+sum(w*n for n,w in addons)<=slack)
    result=dict(upper=upper,lower=lower,seed=seed,workers=workers,native='CP-SAT',witnesses=[])
    class Record(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            phase=sum(self.value(v)<<i for i,v in enumerate(ph))
            chosen={b for b,v in loss.items() if self.value(v)}
            raw=physical_price(p.tiles,ref,gs,req,phase)
            if raw is None:raise ArithmeticError('Native phase mismatches raw source inventory.')
            actual=raw[1]&ref|cover_missing(gs,req,raw[1])
            if len(actual)!=raw[0] or len(actual)>len(chosen):raise ArithmeticError('Invalid native loss assignment.')
            result['witnesses'].append(dict(phase=hex(phase),cost=len(actual),lost=sorted(actual),seconds=self.wall_time))
            out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    e=cp_model.CpSolver();e.parameters.num_search_workers=workers;e.parameters.max_time_in_seconds=seconds
    e.parameters.random_seed=seed;e.parameters.linearization_level=2
    answer=e.solve(m,Record());result.update(status=e.status_name(answer),branches=e.num_branches,conflicts=e.num_conflicts,bound=e.best_objective_bound)
    out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for x in ('tiles','geo','path','out'):cli.add_argument('--'+x,required=True,type=Path)
    cli.add_argument('--lower',type=int,default=29);cli.add_argument('--upper',type=int,default=32)
    cli.add_argument('--seconds',type=float,default=180);cli.add_argument('--seed',type=int,default=3581);cli.add_argument('--workers',type=int,default=3)
    a=cli.parse_args();run(**vars(a))
