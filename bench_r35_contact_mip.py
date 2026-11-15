"""Probe the final signed cutoff in the original physical variables with exact integer inventory equations."""
import argparse
import json
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from r35_contact_certificate import xor_conflict
from r35_loss_witness import physical_price, cover_missing


def run(tiles, geo, paths, parity, output, cap, seconds, export_lp=None):
    from ortools.linear_solver import pywraplp
    problem, ref, physical, required = problem_scope(tiles)
    read = lambda path, key: [frozenset(map(tuple, row)) for row in json.loads(path.read_bytes())[key]]
    groups = [*physical[:13], *read(geo, 'geographic_cores'),
              *(row for row in read(paths, 'edge_perfect_cores') if len(row) <= 13)]
    extra = read(parity, 'required_xor') if parity else []
    if not all(xor_conflict(problem, row | required) for row in extra):
        raise ArithmeticError('A projected odd-cycle inequality is invalid for this scope.')
    from r35_contact_mip import build_model
    engine, phase, loss = build_model(problem, ref, groups, required, cap, extra)
    engine.SetTimeLimit(int(seconds*1000))
    start = 0xce77aac6e89509e8867b457e8c580a233a5e5f22
    cost, forced = physical_price(problem.tiles, ref, groups, required, start)
    omitted = (ref & forced) | cover_missing(groups, required, forced)
    engine.SetHint([*phase,*loss.values()],[(start>>i)&1 for i in range(problem.n)]+[int(b in omitted) for b in loss])
    if export_lp is not None:
        export_lp.write_text(engine.ExportModelAsLpFormat(False), encoding='ascii',newline='\n')
        return
    status = engine.Solve()
    result = dict(status={engine.INFEASIBLE:'INFEASIBLE', engine.OPTIMAL:'OPTIMAL', engine.FEASIBLE:'FEASIBLE', engine.NOT_SOLVED:'UNKNOWN'}.get(status,str(status)), cap=cap, seconds=engine.wall_time()/1000, nodes=engine.nodes(),
                  lp_iterations=engine.iterations(), lower=engine.Objective().BestBound(), implied_cycles=len(extra))
    if status in (engine.OPTIMAL, engine.FEASIBLE):
        answer = sum(int(v.solution_value()>.5)<<i for i,v in enumerate(phase))
        actual, failed = physical_price(problem.tiles, ref, groups, required, answer)
        omitted = (ref & failed) | cover_missing(groups, required, failed)
        if actual != len(omitted) or actual > cap:
            raise ArithmeticError('Native positive failed raw source replay.')
        result['witness'] = dict(phase=hex(answer), cost=actual, lost=sorted(omitted))
    output.write_text(json.dumps(result,indent=2)+'\n', encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('tiles', 'geo', 'paths', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--parity', type=Path)
    parser.add_argument('--export-lp', type=Path)
    parser.add_argument('--cap', type=int, default=30)
    parser.add_argument('--seconds', type=float, default=240)
    run(**vars(parser.parse_args()))
