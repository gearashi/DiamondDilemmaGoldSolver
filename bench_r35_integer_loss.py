"""Check original integer arrivals against direct phases, retained SAT and native CP-SAT."""
from itertools import combinations
from math import inf
from pathlib import Path
from time import monotonic,perf_counter
import argparse
import json

from r35_integer_sat import InventorySAT
from r32_gold_bonds import inventory
from r32_gold_loss import read_json,reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r33_gold_refine import IntegerPatternFilter,make_integer_filter
from search_problem import SearchProblem


def compare(tiles,conflicts):
    p=SearchProblem(read_json(tiles))
    docs=Path(__file__).parent/'docs'
    ref,_,groups=reference_groups(p,read_json(docs/'R29_INCUMBENT.json')['codes'])
    q=OutsideSyndromes(p,ref,groups,inventory(p).required)
    patterns=read_json(docs/'R34_INTEGER_CORES_FINAL.json')['arrivals']
    native=IntegerPatternFilter(p,ref,q.groups,q.required,q.outside,17)
    direct=make_integer_filter(p,ref,q.groups,q.required,q.outside,17)
    sat=InventorySAT(p,q.groups,q.required,q.outside,17)
    full=(1<<len(q.outside))-1
    records=[]
    try:
        for width in ('full','one-in-four','unrestricted'):
            for idx in patterns if width!='unrestricted' else [patterns[0]]:
                number=sum(1<<i for i in idx)
                mask=full if width=='full' else (sum(1<<i for i in range(0,len(q.outside),4)) if width=='one-in-four' else 0)
                t=perf_counter()
                verdict=native._decide(mask,number,monotonic()+1.5)
                native_s=perf_counter()-t
                expected=False if verdict==native.cp.INFEASIBLE else (True if verdict in (native.cp.OPTIMAL,native.cp.FEASIBLE) else None)
                base=sat.solver.accum_stats()
                answer=sat.solve(mask,number,conflicts)
                step={k:v-base[k] for k,v in sat.solver.accum_stats().items()}
                cold=InventorySAT(p,q.groups,q.required,q.outside,17)
                try:
                    other=cold.solve(mask,number,conflicts)
                    standalone=cold.solver.accum_stats()
                finally:
                    cold.close()
                known=[x for x in (expected,answer,other) if x is not None]
                if len(set(known))>1:
                    raise AssertionError((width,idx,expected,answer,other))
                row=dict(scope=width,indices=idx,native=expected,native_seconds=round(native_s,6),
                         native_branches=native.solver.num_branches,sat=answer,sat_work=step,cold=other,cold_work=standalone)
                if width=='full':
                    cost,ph,support=direct.image.minimum(idx,inf)
                    got=direct(idx,inf)
                    if got or (known and known[0]!=got):
                        raise AssertionError((idx,got,known))
                    row.update(exact=got,unconditional_inventory=ph is None,core_mask=hex(support) if support is not None else None,
                               core_value=hex(number&support) if support is not None else None)
                records.append(row)
                print(width,len(records),expected,answer,other,step.get('decisions'),flush=True)
    finally:
        sat.close()
    return dict(fingerprint=p.fingerprint,rows=records,phase_assignments_per_pattern=len(direct.image.images),
                exact_work=dict(direct.work),sat_clauses=sat.clause_count,conflict_budget=conflicts)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',required=True)
    parser.add_argument('--conflicts',type=int,default=4096)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare(args.tiles,args.conflicts),indent=2)+'\n',encoding='utf8')
