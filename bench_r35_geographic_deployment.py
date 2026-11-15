"""Price conditional geometric cuts on a shared SAT encoding and authentic complete donors."""
import argparse
from collections import Counter
import json
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from r32_gold_syndrome import OutsideSyndromes
from r35_geographic_packing_check import check_packing
from r35_geographic_replay import geographic_departures
from r35_integer_sat import InventorySAT
from r35_loss_witness import physical_price


def run(tiles, receipt, tours, out, capset, budget):
    from pysat.solvers import Solver
    p,ref,groups,req=problem_scope(tiles)
    packed,_,outside=OutsideSyndromes.layout(ref,groups[:13],req)
    doc=json.loads(receipt.read_bytes())
    full=[frozenset(map(tuple,cs)) for cs in doc['geographic_cores']]
    geo=geographic_departures(p,ref,req,full)
    if any(not any(core<=candidate-req for core in geo) for candidate in full):
        raise ArithmeticError('Source-independent replay did not reproduce the complete claim family.')
    initial=json.loads((Path(__file__).parent/'docs/R35_GEOGRAPHIC_LOSS_CORES.json').read_bytes())
    baseline=[frozenset(map(tuple,group)) for group in initial['geographic_cores']]
    base={group-req for group in baseline}
    assert all(check_packing(p,group)['feasible'] is False for group in baseline)
    alternatives=sorted(set(geo)|base,key=lambda g:(len(g),sorted(g)))
    target=0xd237aba7e91101e88041413e8c589902021e5126
    fixed=[i+1 if target>>i&1 else -i-1 for i in range(1,p.n)]
    report=dict(puzzle=p.fingerprint,raw_cores=len(full),projected=len(geo),native='cadical153',queries=[],donors=[])
    for f in tours:
        donor=json.loads(f.read_bytes())
        assert donor['fingerprint']==p.fingerprint
        pairs=set(map(tuple,donor['pairs'])); assert req<=pairs
        kept=pairs&ref
        hit=[i for i,c in enumerate(full) if c<=kept|req]
        alone=check_packing(p,kept); compulsory=check_packing(p,kept|req)
        assert alone['feasible'] is True and compulsory['feasible'] is False
        assert set(pairs)-kept and req-ref
        report['donors'].append(dict(file=f.name,retained=len(kept),applicable=hit,
                                      retained_only=alone,with_required=compulsory,
                                      smallest_applicable_bond_count=min((len(full[i]) for i in hit),default=None)))
    for cap in capset:
        model=InventorySAT(p,packed,req,outside,cap,native=None,
                           additional_groups=[g for g in [*groups[:13],*alternatives] if g not in packed])
        def clause(g):
            return tuple(model.physical_losses[pair] for pair in sorted(g-req))
        for label,pins in [('unfixed',[]),('fixed',fixed)]:
            for which,wanted in [('physical',set()),('first_two',base),('expanded',set(geo))]:
                removed=Counter(clause(g) for g in alternatives if g not in wanted)
                cnf=[]
                for line in model.clauses:
                    key=tuple(line)
                    if removed[key]:
                        removed[key]-=1
                    else: cnf.append(line)
                assert not any(removed.values())
                with Solver(name='cadical153',bootstrap_with=cnf) as solver:
                    solver.conf_budget(budget)
                    verdict=solver.solve_limited(assumptions=pins)
                    stats=solver.accum_stats()
                    assert stats['conflicts']<=budget+max(16,budget//50)
                    row=dict(cap=cap,scope=label,variant=which,requested_conflicts=budget,verdict=verdict,**stats)
                    if verdict:
                        phase=sum(1<<(v-1) for v in solver.get_model() if 0<v<=p.n)
                        price=physical_price(p.tiles,ref,[*groups[:13],*wanted],req,phase)
                        assert price and price[0]<=cap
                        row.update(phase=hex(phase),loss=price[0])
                    report['queries'].append(row)
        out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True);cli.add_argument('--claims',type=Path,required=True)
    cli.add_argument('--tours',type=Path,nargs='*',default=[]);cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--caps',nargs='+',type=int,default=[21,22,23,24,25]);cli.add_argument('--budget',type=int,default=15000)
    args=cli.parse_args()
    run(args.tiles,args.claims,args.tours,args.output,args.caps,args.budget)
