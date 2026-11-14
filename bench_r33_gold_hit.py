"""Replay a projected loss witness and lift its exact outside phase errors to integer inventory."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import types

from ortools.sat.python import cp_model
from r28_geo_color import signed_phase_model
from r29_gold_phase import EndpointCuts, _reciprocal
from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem

RELEASE = '41f1e9c980fc441c3151ff21b5f73ae12ca3d9d7'
ORIGINAL = 'e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853'
OBSERVED = ((16,240),(205,216),(294,427),(380,475))


def released_engine(path):
    source = (Path(path).read_bytes() if path else
              subprocess.check_output(['git','show',RELEASE+':r32_gold_syndrome.py'],cwd=Path(__file__).parent))
    source = source.replace(b'\r\n',b'\n')
    if hashlib.sha256(source).hexdigest()!='b7eb53a4c78a85ef63841c82c823fc706399ef568f3c04773a2b61490a6d3fbf':
        raise ValueError('Expected the released v1.13.0 syndrome engine')
    old = types.ModuleType('r32_v113_syndrome')
    exec(compile(source,'r32_v113_syndrome.py','exec'),old.__dict__)
    return old.OutsideSyndromes


def check_integer(problem,reference,query,errors,seconds,euler,budget):
    # Pairings on compatible outside phases are still free to break physically.
    required = inventory(problem).required
    model,phases=signed_phase_model(problem,euler=euler,pins={},bonds=required)
    sides=EndpointCuts(problem).sides
    def equation(pair,flipped=False):
        a,b=pair
        mask,start=sides[a]
        parity=start^sides[b][1]^(mask.bit_count()&1)^int(flipped)
        return phases[a//3]!=phases[b//3] if parity else phases[a//3]==phases[b//3]
    for pair in query.outside:
        model.add(equation(pair,pair in errors))
    if budget is not None:
        losses={pair:model.new_bool_var('physical_loss') for pair in sorted(reference)}
        for pair,lost in losses.items():
            model.add(equation(pair)).only_enforce_if(~lost)
            if pair in required:
                model.add(lost==0)
        for group in query.groups:
            model.add(sum(losses[pair] for pair in group)>=1)
        model.add(sum(losses.values())<=budget)
    solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=seconds
    status=solver.solve(model)
    if status==cp_model.MODEL_INVALID:
        raise ValueError(solver.solution_info())
    result=dict(status=solver.status_name(status),integer_inventory=True,euler=euler,loss_budget=budget,
                branches=solver.num_branches,conflicts=solver.num_conflicts,seconds=round(solver.wall_time,5))
    if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
        colors=[solver.value(x) for x in phases]
        labels=[(m,c^colors[i//3]) for i,(m,c) in enumerate(sides)]
        def fails(bond):
            a,b=bond
            return labels[b] != _reciprocal(*labels[a])
        realized={pair for pair in query.outside if fails(pair)}
        counts=Counter(labels)
        assert all((value==counts[_reciprocal(*label)] if label!=_reciprocal(*label) else value%2==0)
                   for label,value in counts.items())
        assert realized==errors and not any(fails(pair) for pair in required)
        result['phases']=colors
        if budget is not None:
            dropped={p for p,v in losses.items() if solver.value(v)}
            assert len(dropped)<=budget and all(dropped & group for group in query.groups)
            assert all(not fails(pair) for pair in reference-dropped)
            result['relaxed_physical_drops']=sorted(dropped)
    return result


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--legacy-source',help='v1.13.0 r32_gold_syndrome.py if release commit is unavailable')
    cli.add_argument('--seconds',type=float,default=30)
    cli.add_argument('--output',required=True)
    opts=cli.parse_args()
    p=SearchProblem(read_json(opts.tiles))
    if p.fingerprint!=ORIGINAL:
        raise ValueError('This witness belongs to the released original puzzle')
    ref,_,groups=reference_groups(p,read_json(Path(__file__).parent/'docs/R29_INCUMBENT.json')['codes'])
    args=(p,ref,groups,inventory(p).required)
    query=OutsideSyndromes(*args)
    replay={}
    for name,engine in (('v113',released_engine(opts.legacy_source)(*args)),('current',query)):
        row=engine.indexed(17,4,opts.seconds,start=874,stop=875)
        assert row['status']=='PHASE_RELAXATION_FEASIBLE' and row['next_rank']==874
        assert tuple(map(tuple,row['outside_errors']))==OBSERVED
        replay[name]=row
    trials=[check_integer(p,ref,query,set(OBSERVED),opts.seconds,euler,limit)
            for euler,limit in ((False,None),(True,None),(False,17),(True,17))]
    report=dict(fingerprint=p.fingerprint,release_control=RELEASE,reference_bonds=len(ref),outside_equations=len(query.outside),
                prefix_rank=874,outside_errors=OBSERVED,replay=replay,integer_trials=trials,
                scope='One exact outside phase-error set. A phase-compatible outside pair may still break physically.')
    Path(opts.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    main()
