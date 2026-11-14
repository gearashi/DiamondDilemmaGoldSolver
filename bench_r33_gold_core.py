"""Shrink the rank-874 phase contradiction and verify every single-equation deletion."""
import argparse
from collections import Counter
from functools import lru_cache
import json
from math import comb
from pathlib import Path

from ortools.sat.python import cp_model
from bench_r33_gold_hit import OBSERVED, ORIGINAL
from r28_geo_color import signed_phase_model
from r29_gold_phase import EndpointCuts, _reciprocal
from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', default='data/tiles.json')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    p = SearchProblem(read_json(args.tiles))
    if p.fingerprint != ORIGINAL:
        raise ValueError('This experiment uses the published 207/52 reference and original tiles.')
    codes = read_json(Path(__file__).parent/'docs/R29_INCUMBENT.json')['codes']
    ref, _, groups = reference_groups(p, codes)
    _, _, outside = OutsideSyndromes.layout(ref, groups, inventory(p).required)
    errors = set(OBSERVED)
    if not errors <= set(outside):
        raise ValueError('Published pattern departs from this outside-bond scope.')
    sides = EndpointCuts(p).sides

    @lru_cache(maxsize=None)
    def decide(conditions):
        # Unrestricted phases: no required mates, reference-loss budget, Euler or group cuts.
        m, phases = signed_phase_model(p, euler=False, bonds=(), pins={})
        for a, b in conditions:
            mask, first = sides[a]
            different = first ^ sides[b][1] ^ (mask.bit_count() & 1) ^ int((a,b) in errors)
            m.add(phases[a//3] != phases[b//3] if different else phases[a//3] == phases[b//3])
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = 3.0
        status = solver.solve(m)
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL):
            return solver.status_name(status), None
        assignment = [solver.value(x) for x in phases]
        labels = [(mask, color ^ assignment[i//3]) for i, (mask, color) in enumerate(sides)]
        counts = Counter(labels)
        for pair, n in counts.items():
            mate = _reciprocal(*pair)
            if (n % 2 if pair == mate else n != counts[mate]):
                raise ArithmeticError('Published phases violate reciprocal inventory.')
        for a, b in conditions:
            if (labels[a] != _reciprocal(*labels[b])) != ((a,b) in errors):
                raise ArithmeticError('Published phases violate a retained phase equation.')
        return solver.status_name(status), hex(sum(x << i for i,x in enumerate(assignment)))

    whole = tuple(reversed(outside))
    status, _ = decide(whole)
    if status != 'INFEASIBLE':
        raise ArithmeticError('Could not reproduce complete-pattern exclusion.')
    core = list(whole)
    for width in (32,16,8,4,2,1):
        i = 0
        while i < len(core):
            trial = core[:i] + core[i+width:]
            if decide(tuple(trial))[0] == 'INFEASIBLE':
                core = trial
            else:
                i += width
    core = sorted(core)
    status, _ = decide(tuple(core))
    if status != 'INFEASIBLE':
        raise ArithmeticError('The reduced core failed independent reordered replay.')
    controls = []
    for i, pair in enumerate(core):
        trial = tuple(core[:i] + core[i+1:])
        decision, assignment = decide(trial)
        if assignment is None:
            raise ArithmeticError('Deletion needs a verified positive control: ' + str(pair))
        controls.append(dict(dropped=pair, status=decision, tile_phases_hex=assignment))
    forced_errors = len(set(core) & errors)
    report = dict(fingerprint=p.fingerprint, reference='docs/R29_INCUMBENT.json', original_rank=874,
                  outside_count=len(outside), outside_errors=OBSERVED, original_status='INFEASIBLE',
                  model='Signed integer reciprocal inventory. Every tile phase free; no Gold bonds, loss cap, Euler equation, geography or group conflicts.',
                  status=status, core_size=len(core), disagree=[x for x in core if x in errors],
                  agree=[x for x in core if x not in errors], exact_four_error_patterns_covered=comb(len(outside)-len(core),len(errors)-forced_errors),
                  minimality='Every one-equation deletion has a checked feasible phase assignment. Bit i of the hexadecimal value is tile i phase.',
                  controls=controls)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:report[k] for k in ('status','core_size','disagree','exact_four_error_patterns_covered')}))


if __name__ == '__main__':
    main()
