"""Loss ownership at a certified disjoint lower bound, including overlapping departures."""
import argparse
import json
from pathlib import Path
from time import monotonic

from bench_r35_loss_boundary344 import build, problem_scope, verify_21
from bench_r35_loss_proof import original, parity_conflict
from r29_gold_phase import EndpointCuts
from r35_integer_sat import InventorySAT
from r35_loss_bound import CHECKED_LOWER_BOUND,REFERENCE_SCOPE
from r32_gold_loss import digest,read_json
from r35_loss_witness import physical_price,raw_sides


def classify(p, groups, required):
    sides=raw_sides(p.tiles)
    covered=set()
    fixed=set()
    flexible=[]
    for g in groups:
        movable=set(g)-required
        if covered&movable:
            raise ValueError("Disjoint lower-bound obligations overlap.")
        covered|=movable
        if parity_conflict(sides, sorted(set(g)|required)):
            fixed|=movable
        else:
            flexible.append(tuple(sorted(movable)))
    return covered,fixed,flexible


def tighten(p, groups, required, outside, phase, losses, model):
    """At equality each disjoint group pays max(1, number of phase errors)."""
    covered,fixed,flex=classify(p,groups,required)
    sides=EndpointCuts(p).sides
    errors={}
    for pair,lost in losses.items():
        a,b=pair;x,y=phase[a//3],phase[b//3]
        rhs=sides[a][1]^sides[b][1]^(sides[a][0].bit_count()%2)
        error=lost if pair in fixed or pair not in covered else model.new_bool_var('error_'+str(pair))
        model.add_bool_xor([x,y,error if rhs else error.Not()])
        errors[pair]=error
    for group in flex:
        local=[errors[pair] for pair in group]
        clean=model.new_bool_var('clean_'+str(group[0]))
        for error in local:model.add_implication(clean,error.Not())
        model.add_bool_or([clean,*local])
        model.add(sum(losses[e] for e in group)==sum(local)+clean)
    return dict(rigid=len(fixed)+len(outside),flexible_groups=[len(g) for g in flex])


def tight_clauses(p, groups, required, outside, losses, highest):
    covered,fixed,flex=classify(p,groups,required)
    sides=EndpointCuts(p).sides
    clause=[]
    def allocate():
        nonlocal highest
        highest+=1
        return highest
    def reverse(pair,marker,clean=None):
        a,b=pair;x,y=a//3+1,b//3+1
        rhs=sides[a][1]^sides[b][1]^(sides[a][0].bit_count()%2)
        t=-y if rhs else y
        tail=[-marker] if clean is None else [-marker,clean]
        clause.extend([[x,t,*tail],[-x,-t,*tail]])
    for pair in sorted(set(outside)|fixed):reverse(pair,losses[pair])
    for group in flex:
        clean=allocate()
        for i,pair in enumerate(group):
            a,b=pair;x,y=a//3+1,b//3+1
            rhs=sides[a][1]^sides[b][1]^(sides[a][0].bit_count()%2)
            t=y if rhs else -y
            # A clean group permits one voluntary loss. A dirty group permits errors only.
            clause.extend([[x,t,-clean],[-x,-t,-clean]])
            reverse(pair,losses[pair],clean)
            for other in group[:i]:clause.append([-clean,-losses[pair],-losses[other]])
    return clause,dict(rigid=len(fixed)+len(outside),flexible_groups=[len(g) for g in flex])


def run(tiles, seconds, workers, arm, only_disjoint=False):
    p,ref,g,req=problem_scope(tiles);_,_,pack,required,outside=original(tiles)
    req=set(req)
    identity=digest([3,p.fingerprint,sorted(ref),sorted(required),[sorted(j) for j in pack]])
    assert identity==REFERENCE_SCOPE and CHECKED_LOWER_BOUND==20
    hint,lost,verified=verify_21(p,ref,g,req)
    if only_disjoint:
        g=list(map(set,pack));hint=int(read_json('docs/R35_OVERLAP_BOUNDARY.json')['disjoint20']['phase'],16)
        colors=raw_sides(p.tiles)
        price,failed=physical_price(p.tiles,ref,g,req,hint)
        lost=failed&ref
        for grp in g:
            if not grp&lost:lost.add(min(grp-req))
        assert len(lost)==price==20
    start=monotonic()
    report=dict(arm=arm,scope=identity,packed=len(pack),group_count=len(g),verified_upper=verified)
    if arm.startswith('cp'):
        from ortools.sat.python import cp_model
        model,phase,loss=build(p,ref,g,req,20,hint,lost)
        if arm=='cp-tight':report['ownership']=tighten(p,pack,req,outside,phase,loss,model)
        solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=seconds;solver.parameters.num_search_workers=workers
        solver.parameters.linearization_level=2;solver.parameters.random_seed=3544
        status=solver.solve(model);answer=True if status in (cp_model.OPTIMAL,cp_model.FEASIBLE) else False if status==cp_model.INFEASIBLE else None
        report.update(work=dict(decisions=solver.num_branches,conflicts=solver.num_conflicts),model=model.model_stats())
        if answer:
            witness=sum(solver.value(v)<<i for i,v in enumerate(phase));chosen={e for e,v in loss.items() if solver.value(v)}
    else:
        native=InventorySAT(p,pack,req,outside,20,native='g4',excess=True,projected=True,additional_groups=g)
        try:
            if arm=='sat-tight':
                extra,report['ownership']=tight_clauses(p,pack,req,outside,native.physical_losses,max(abs(v) for c in native.clauses for v in c))
                native.solver.append_formula(extra);report['extra_clauses']=len(extra)
            native.solver.set_phases([i if hint>>(i-1)&1 else -i for i in range(1,p.n+1)])
            answer=native.decide(seconds)
            if answer:
                witness=native.phase();lits=set(native.solver.get_model());chosen={e for e,v in native.physical_losses.items() if v in lits}
            report['work']=native.solver.accum_stats()
        finally:native.close()
    report.update(answer=answer,seconds=monotonic()-start)
    if answer:
        price,failed=physical_price(p.tiles,ref,g,req,witness)
        assert price==len(chosen)==20 and not (ref&failed)-chosen and not chosen&req and all(grp&chosen for grp in g)
        report['witness']=dict(phase=hex(witness),losses=sorted(chosen),cost=price)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tiles',required=True);parser.add_argument('--seconds',type=float,default=420)
    parser.add_argument('--workers',type=int,default=4);parser.add_argument('--arm',choices=('cp-tight','cp','sat-tight','sat'),required=True)
    parser.add_argument('--disjoint',action='store_true');parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    result=run(a.tiles,a.seconds,a.workers,a.arm,a.disjoint);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result['answer'],result['work'],result.get('ownership'))
