"""Replay native physical costs on all included positive and adverse original phase/cap cases."""
import argparse
from collections import defaultdict
from math import inf
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from bench_r35_loss_proof import original
from io_utils import atomic_json
from r32_gold_loss import read_json
from r35_integer_sat import InventorySAT
from r35_loss_witness import physical_price
from r35_signed_conditions import SignedConditions


def compare(tiles, cases):
    problem, reference, rows, required = problem_scope(tiles)
    _, _, packed, _, outside = original(tiles)
    geo = [frozenset(map(tuple, row)) - required
           for row in read_json('docs/R35_GEOGRAPHIC_LOSS_CORES.json')['geographic_cores']]
    cases_by_budget = defaultdict(set)
    for case in cases:
        cases_by_budget[case['groups'], case['cap']].add(int(case['phase'], 16))
    result = []
    for (group_count, cap), phases in sorted(cases_by_budget.items()):
        groups = rows[:13] if group_count == 13 else [*rows[:13], *geo]
        condition = SignedConditions(problem, mate_cost=True).for_loss(reference, groups, required, cap)
        models = [InventorySAT(problem, packed, required, outside, cap, additional_groups=groups,
                               mate_cost_pairs=pairs) for pairs in (None, condition.native_cost_pairs())]
        try:
            condition.install(models[1])
            for phase in sorted(phases):
                pins = [i + 1 if phase & (1 << i) else -i - 1 for i in range(problem.n)]
                ordinary, _ = physical_price(problem.tiles, reference, groups, required, phase)
                priced = condition.price(phase, ordinary)
                verdicts = []
                decisions = []
                for model in models:
                    before = model.solver.accum_stats()['decisions']
                    verdicts.append(model.decide(inf, pins))
                    decisions.append(model.solver.accum_stats()['decisions'] - before)
                if verdicts != [ordinary <= cap, priced <= cap]:
                    raise AssertionError((phase, cap, ordinary, priced, verdicts))
                result.append(dict(groups=group_count, cap=cap, phase=hex(phase), ordinary=ordinary,
                                   priced=priced, base=verdicts[0], native=verdicts[1], decisions=decisions))
        finally:
            for model in models:
                model.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', required=True)
    parser.add_argument('--cases', type=Path, default=Path('docs/R35_MATE_CLAUSES.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = compare(args.tiles, read_json(args.cases)['checked'])
    atomic_json(args.output, dict(checked=rows))
