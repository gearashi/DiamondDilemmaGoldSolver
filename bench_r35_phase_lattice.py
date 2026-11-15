"""Replay actual integer arrivals and their partial assumptions against native signed inventory."""
import argparse
from collections import Counter
import json
from pathlib import Path

from r28_exchange import read_json
from r28_geo_color import signed_phase_model
from r29_gold_phase import SignedParity
from r31_gold_connectivity import PhaseInterface
from r32_gold_bonds import inventory
from r32_gold_loss import reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r35_phase_lattice import PhaseColumns,price
from search_problem import SearchProblem


def original_cases(p):
    reference,_,groups=reference_groups(p,read_json('docs/R29_INCUMBENT.json')['codes'])
    _,required,outside=OutsideSyndromes.layout(reference,groups,inventory(p).required)
    raw=read_json('docs/R34_LOSS_MEASUREMENTS.json')
    arrivals={tuple(v) for row in raw['cases'] for v in row['arrivals']}
    hit=read_json('docs/R33_HIT_CORE.json')
    arrivals.add(tuple(sorted(outside.index(tuple(e)) for e in hit['outside_errors'])))
    data=SignedParity(p)
    def relation(pair,error):
        a,b=pair
        ma,ca=data.sides[a];_,cb=data.sides[b]
        return a//3,b//3,ca^cb^(ma.bit_count()&1)^error
    fixed=[relation(e,0) for e in sorted(required)]
    for errors in sorted(arrivals):
        signed=[relation(pair,int(i in errors)) for i,pair in enumerate(outside)]
        for count in (0,24,60,90,len(outside)):
            yield dict(errors=errors,retained_outside=count),fixed+signed[:count]
    core=[relation(tuple(pair),1) for pair in hit['disagree']]
    core.extend(relation(tuple(pair),0) for pair in hit['agree'])
    yield dict(historical_minimal_core=True),core
    for i in range(len(core)):
        yield dict(historical_core_delete=i),core[:i]+core[i+1:]


def native(p,relations,euler):
    from ortools.sat.python import cp_model
    m,x=signed_phase_model(p,euler=euler,pins={})
    for u,v,flip in relations:m.add(x[u]!=x[v] if flip else x[u]==x[v])
    solver=cp_model.CpSolver();solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=3
    result=solver.solve(m)
    return solver.status_name(result)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',default='data/tiles.json');ap.add_argument('--output',type=Path)
    args=ap.parse_args();p=SearchProblem(read_json(args.tiles))
    original=PhaseInterface(p)
    out=[]
    for name,relations in original_cases(p):
        parity=SignedParity(p)
        for u,v,z in relations:parity.base.add((1<<u)^(1<<v),z,())
        result=dict(scope=name,gf2_pass=parity.base.conflict is None)
        for euler in (False,True):
            cols=PhaseColumns(original,relations,euler=euler)
            fit=price(cols,max_operations=1000000)
            oracle=native(p,relations,euler)
            assert fit['status'] not in ('INFEASIBLE','XOR_CONFLICT') or oracle=='INFEASIBLE'
            result['euler' if euler else 'balance']=dict(lattice=fit['status'],native=oracle,
                free_components=len(cols.columns),proofs=fit.get('proofs',[]) if fit['status']=='INFEASIBLE' else [])
        out.append(result)
    text=json.dumps(out,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':main()
