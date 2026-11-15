"""Discriminate the signed cap-20 boundary under 344 checked original departure groups."""
import argparse
import hashlib
import json
from pathlib import Path
from time import monotonic

from r28_exchange import read_json
from r29_gold_phase import EndpointCuts
from r31_gold_connectivity import PhaseInterface
from r32_gold_bonds import inventory
from r32_gold_loss import reference_groups
from r35_loss_witness import physical_price,raw_sides,reciprocal
from search_problem import SearchProblem


def problem_scope(tiles):
    root=Path(__file__).resolve().parent
    p=SearchProblem(read_json(tiles));ref,_,initial=reference_groups(p,read_json(root/'docs/R29_INCUMBENT.json')['codes'])
    additions=read_json(root/'docs/R35_INTEGER_DEPARTURES.json')['groups']
    groups=[set(g) for g in initial]+[set(map(tuple,row)) for row in additions]
    return p,set(ref),groups,set(inventory(p).required)


def verify_21(p,ref,groups,required):
    root=Path(__file__).resolve().parent
    witness=read_json(root/'docs/R35_OVERLAP_BOUNDARY.json')['all21_witness']
    phase=int(witness['phase'],16)
    price,failed=physical_price(p.tiles,ref,groups,required,phase)
    colors=raw_sides(p.tiles)
    labels=[(slots,color^(phase>>(d//3)&1)) for d,(slots,color) in enumerate(colors)]
    failures={pair for pair in ref if labels[pair[0]]!=reciprocal(labels[pair[1]])}
    lost=failures|set(map(tuple,witness['additional']))
    assert failures==ref&failed and len(lost)==price==21 and not required&lost and all(g&lost for g in groups)
    return phase,lost,dict(phase=hex(phase),cost=price,mismatched=len(failures),covered_groups=len(groups),
                         remaining_before_additional=sum(not g&failures for g in groups),explicit_departures=sorted(lost))


def build(p,ref,groups,required,cap,hint,loss_hint):
    from ortools.sat.python import cp_model
    model=cp_model.CpModel();phase=[model.new_bool_var('p'+str(v)) for v in range(p.n)]
    model.add(phase[0]==0)
    iface=PhaseInterface(p);sides=EndpointCuts(p).sides
    for j,mod in enumerate(iface.mod[:-1]):
        base=sum(lo[j] for lo,_ in iface.rows)
        coef=[hi[j]-lo[j] for lo,hi in iface.rows]
        if mod:
            active=[phase[i] for i,k in enumerate(coef) if k%2]
            if not base%2:active.append(model.new_constant(1))
            model.add_bool_xor(active)
        else:model.add(sum(k*v for v,k in zip(phase,coef) if k)==-base)
    loss={pair:model.new_bool_var('l'+str(i)) for i,pair in enumerate(sorted(ref-required))}
    for pair in sorted(ref|required):
        a,b=pair;x,y=phase[a//3],phase[b//3]
        rhs=sides[a][1]^sides[b][1]^(sides[a][0].bit_count()%2)
        marker=loss.get(pair,0)
        if rhs:
            model.add(x+y+marker>=1);model.add(x+y-marker<=1)
        else:
            model.add(x-y<=marker);model.add(y-x<=marker)
    for group in groups:model.add_bool_or([loss[pair] for pair in sorted(group-required)])
    model.add(sum(loss.values())==cap)
    for i,v in enumerate(phase):model.add_hint(v,(hint>>i)&1)
    for e,v in loss.items():model.add_hint(v,int(e in loss_hint))
    return model,phase,loss


def main():
    from ortools.sat.python import cp_model
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',required=True);cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--seconds',type=float,default=480);cli.add_argument('--workers',type=int,default=4);cli.add_argument('--cap',type=int,default=20)
    args=cli.parse_args();p,ref,g,req=problem_scope(args.tiles);hint,lost,verified=verify_21(p,ref,g,req)
    m,phase,loss=build(p,ref,g,req,args.cap,hint,lost)
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=args.seconds;solver.parameters.num_search_workers=args.workers
    solver.parameters.random_seed=3544;solver.parameters.linearization_level=2
    t=monotonic();status=solver.solve(m);elapsed=monotonic()-t
    report=dict(verified21=verified,scope=dict(puzzle=p.fingerprint,reference=len(ref),required=len(req),groups=len(g)),
                cap=args.cap,status=solver.status_name(status),seconds=elapsed,workers=args.workers,branches=solver.num_branches,
                conflicts=solver.num_conflicts,stats=solver.response_stats(),model=m.model_stats())
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        w=sum(solver.value(v)<<i for i,v in enumerate(phase));price,failed=physical_price(p.tiles,ref,g,req,w)
        chosen={e for e,v in loss.items() if solver.value(v)}
        assert price<=args.cap and len(chosen)==args.cap and not (ref&failed)-chosen and all(row&chosen for row in g)
        report['witness']=dict(phase=hex(w),verified_cost=price,chosen=sorted(chosen))
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(report['status'],report['branches'],elapsed,flush=True)


if __name__=='__main__':main()
