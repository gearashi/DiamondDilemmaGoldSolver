"""Separate a global near-reference loss assignment against arbitrary retained components."""
import argparse
import json
from pathlib import Path
from time import monotonic

from bench_r35_geographic_components import parts
from bench_r35_geographic_loss_cores import minimize_obstruction
from bench_r35_induced_contacts import assess,incidence
from bench_r35_loss_boundary344 import problem_scope
from r28_geo_sewing import PosePacking
from r35_induced_contacts import InducedContacts
from r35_geographic_packing_check import check_packing


def next_cut(p,ref,req,lost,checker,packing):
    retained=(ref-lost)|req;table=incidence(p)
    for component in sorted(parts(retained),key=len,reverse=True):
        geographic,matched=checker.count(component)
        if not geographic:
            core=minimize_obstruction(packing,component)
            if core is None or check_packing(p,core)['feasible'] is not False:raise ArithmeticError('Incomplete geometric core.')
            return 'geographic_cores',core
        if matched:continue
        # A smaller disconnected residue proves the rule when one of its
        # remaining connected components is itself an edge-perfect conflict.
        current=component;width=max(1,len(current)//2)
        while width:
            ordered=sorted(current);i=0
            while i<len(ordered):
                subset=set(ordered[:i]+ordered[i+width:])
                candidate=next((c for c in sorted(parts(subset),key=len) if checker.proves(c)),None)
                if candidate is None:i+=width;continue
                current=candidate;ordered=sorted(current);i=0
            width//=2
        audited=assess(p,table,current)
        if not audited['poses'] or audited['compatible']:raise ArithmeticError('Unconfirmed physical contact.')
        return 'edge_perfect_cores',current
    if check_packing(p,retained)['feasible'] is False:
        core=minimize_obstruction(packing,retained)
        if core and check_packing(p,core)['feasible'] is False:return 'geographic_cores',core
    return None,None


def run(tiles,geo,paths,seed,output,seconds,limit):
    from ortools.sat.python import cp_model
    p,ref,gs,req=problem_scope(tiles)
    groups=[frozenset(g)-req for g in gs];frozen={}
    for f,k in [(geo,'geographic_cores'),(paths,'edge_perfect_cores')]:
        frozen[k]=[frozenset(map(tuple,g)) for g in json.loads(f.read_bytes())[k]]
        groups.extend(g-req for g in frozen[k])
    original=json.loads(seed.read_bytes())['reports'][-1]['integer']
    lost=set(map(tuple,original['lost']));cap=len(lost)
    assert all(g&lost for g in groups)
    checker=InducedContacts(p);pack=PosePacking(p)
    deadline=monotonic()+seconds
    report=dict(fingerprint=p.fingerprint,cap=cap,rounds=[],geographic_cores=[],edge_perfect_cores=[])
    model=cp_model.CpModel();v={b:model.new_bool_var('') for b in sorted(ref-req)}
    for g in groups:model.add_bool_or([v[b] for b in sorted(g)])
    model.add(sum(v.values())==cap)
    solver=cp_model.CpSolver();solver.parameters.num_search_workers=2
    def save():output.write_text(json.dumps(report,separators=(',',':'))+'\n',encoding='utf8')
    while len(report['rounds'])<limit:
        kind,core=next_cut(p,ref,req,lost,checker,pack)
        if core is None:
            report['status']='all connected sets allow matching interior contacts';report['witness']=sorted(lost);save();return
        if core in frozen[kind]:raise ArithmeticError('Revisited a previously excluded retention set.')
        frozen[kind].append(core);report[kind].append(sorted(core))
        row=dict(kind=kind,core_size=len(core),lost=sorted(lost));report['rounds'].append(row);save()
        model.add_bool_or([v[b] for b in sorted(core-req)])
        model.clear_hints()
        for b,var in v.items():model.add_hint(var,int(b in lost))
        if monotonic()>=deadline:break
        solver.parameters.max_time_in_seconds=max(.05,deadline-monotonic())
        status=solver.solve(model);row['next']=solver.status_name(status);row['branches']=solver.num_branches
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            report['status']=row['next'];save();return
        lost={b for b,x in v.items() if solver.value(x)}
        if len(lost)!=cap or not all(g&lost for g in groups):raise ArithmeticError('Invalid returned donor.')
    report['status']='bounded screen';save()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for f in ('tiles','geo','paths','seed','output'):cli.add_argument('--'+f,required=True,type=Path)
    cli.add_argument('--seconds',type=float,default=120);cli.add_argument('--limit',type=int,default=32)
    a=cli.parse_args();run(**vars(a))
