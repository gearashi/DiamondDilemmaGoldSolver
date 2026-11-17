"""Reoptimize subsets of retained physical bonds around concrete signed incumbents."""
import argparse
import json
from pathlib import Path
from random import Random
from time import monotonic

from bench_r35_loss_boundary344 import problem_scope
from r32_gold_syndrome import OutsideSyndromes
from r35_integer_sat import InventorySAT
from r35_loss_witness import cover_missing, physical_price


def run(tiles, geo, paths, out, cap, seconds, seed, budget):
    problem, reference, physical, required = problem_scope(tiles)
    packed, _, outside = OutsideSyndromes.layout(reference, physical[:13], required)
    def rows(source, key):
        return [frozenset(map(tuple, row)) for row in json.loads(source.read_bytes())[key]]
    groups = [*physical[:13], *(row-required for row in rows(geo, 'geographic_cores')),
              *(row-required for row in rows(paths, 'edge_perfect_cores') if len(row) <= 13)]
    engine = InventorySAT(problem, packed, required, outside, cap, native='cadical195',
                          additional_groups=[row for row in groups if row not in packed])
    def reprice(phase):
        answer = physical_price(problem.tiles, reference, groups, required, phase)
        if answer is None:
            raise ArithmeticError('Native branch violated integer side inventory.')
        cost, forced = answer
        missing = (reference & forced) | cover_missing(groups, required, forced)
        if cost != len(missing) or not all(g & missing for g in groups):
            raise ArithmeticError('The returned loss assignment failed independent replay.')
        return cost, missing
    elite = [0xce77aac6e89509e8867b457e8c580a233a5e5f22, 0xce55aaa6e89109fca073547e8c580ba32e0e5fa2]
    elite = [(phase, *reprice(phase)) for phase in elite]
    rng = Random(seed)
    results = dict(cap=cap, initial_costs=[r[1] for r in elite], trials={}, seen=0)
    used = set()
    started = monotonic()
    try:
        while monotonic() - started < seconds:
            i = results['seen']
            phase, _, missing = elite[0] if i%4 else rng.choice(elite)
            candidates = sorted(reference - missing - required)
            count = (110, 60, 90, 45, 130, 75, 30, 100, 10, 50, 65)[i%11]
            sample = sorted(rng.sample(candidates, min(len(candidates), count)))
            key = tuple(sample)
            if key in used:
                continue
            used.add(key)
            assumptions = [-engine.physical_losses[b] for b in sample]
            engine.solver.set_phases([i+1 if (phase>>i)&1 else -i-1 for i in range(problem.n)] +
                                     [v if bond in missing else -v for bond, v in engine.physical_losses.items()])
            engine.solver.conf_budget(budget)
            answer = engine.solver.solve_limited(assumptions=assumptions)
            label = f'{count}/{answer}'
            results['trials'][label] = 1 + results['trials'].get(label, 0)
            results['seen'] += 1
            if answer:
                ph = engine.phase()
                cost, bonds = reprice(ph)
                if cost > cap:
                    raise ArithmeticError('Invalid signed attainment above the requested cap.')
                results['witness'] = dict(phase=hex(ph), cost=cost, lost=sorted(bonds))
            if answer or i%32==0:
                results.update(seconds=monotonic()-started, **engine.solver.accum_stats())
                out.write_text(json.dumps(results, indent=2)+'\n',encoding='utf8')
            if answer:
                break
    finally:
        results.update(seconds=monotonic()-started, **engine.solver.accum_stats())
        out.write_text(json.dumps(results, indent=2)+'\n',encoding='utf8')
        engine.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('tiles','geo','paths','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--cap',type=int,default=30)
    parser.add_argument('--seed',type=int,default=8135)
    parser.add_argument('--seconds',type=float,default=240)
    parser.add_argument('--budget',type=int,default=2000)
    run(**vars(parser.parse_args()))
