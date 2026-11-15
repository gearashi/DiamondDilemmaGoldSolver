"""Measure how many audited geometry conflicts reproduce the fixed-phase loss boundaries."""
import argparse
from collections import Counter
import json
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from r35_loss_witness import cover_missing, physical_price


def undominated(rows):
    kept = []
    for group in sorted(set(rows), key=lambda g: (len(g), sorted(g))):
        if not any(smaller <= group for smaller in kept):
            kept.append(group)
    return kept


def select(reference, physical, required, conflicts, forced_losses, prices, train):
    """Separate explicit counterexample departure sets at exact target costs."""
    chosen = []
    iterations = []
    while True:
        selected = physical + [group - required for group in chosen]
        frequency = Counter()
        for phase in train:
            forced = forced_losses[phase]
            extra = cover_missing(selected, required, forced)
            losses = (reference & forced) | extra
            if len(losses) == prices[phase]:
                continue
            if len(losses) > prices[phase]:
                raise ArithmeticError('Subset of inputs exceeded the full fixed-phase loss price.')
            violated = [i for i, group in enumerate(conflicts) if not (group - required) & losses]
            if not violated:
                raise ArithmeticError('Expected an unsatisfied geographic departure.')
            frequency.update(violated)
        if not frequency:
            return chosen, iterations
        which = min(frequency, key=lambda i: (-frequency[i], len(conflicts[i]), i))
        chosen.append(conflicts[which])
        iterations.append(frequency[which])


def portfolio(tiles, research, destination, selected_file):
    p, ref, physical, req = problem_scope(tiles)
    evidence = json.loads(research.read_bytes())
    if evidence['puzzle'] != p.fingerprint:
        raise ValueError('The donor evidence is for a different puzzle.')
    published = [frozenset(map(tuple, row)) for row in evidence['geographic_cores']]
    potential = undominated(published)
    projected = [group - req for group in potential]
    phases = [int(row['phase'], 16) for row in evidence['held_out'] + evidence['phase_donors']]
    training = {int(row['phase'], 16) for row in evidence['separations'] + evidence['phase_donors']}
    prices = {}
    forced_losses = {}
    for phase in phases:
        cost, failed = physical_price(p.tiles, ref, physical + projected, req, phase)
        prices[phase] = cost
        forced_losses[phase] = failed
    retained = {}
    for name, sample in [('trained', training), ('all', phases)]:
        chosen, iterations = select(ref, physical, req, potential, forced_losses, prices, sample)
        examples = []
        for phase in phases:
            price, _ = physical_price(p.tiles, ref, physical + [g - req for g in chosen], req, phase)
            examples.append(dict(phase=hex(phase), full=prices[phase], selected=price, trained=phase in training))
        retained[name] = dict(count=len(chosen), checks=iterations, prices=examples,
                              indices=[potential.index(g) for g in chosen])
        if name == 'all' and selected_file:
            selected_file.write_text(json.dumps(dict(puzzle=p.fingerprint, geographic_cores=[sorted(g) for g in chosen]),
                                                 separators=(',', ':'))+'\n', encoding='utf8')
    result = dict(puzzle=p.fingerprint, original=len(published), antichain=len(potential), portfolios=retained)
    destination.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles', required=True)
    cli.add_argument('--research', required=True, type=Path)
    cli.add_argument('--output', required=True, type=Path)
    cli.add_argument('--selected-output', type=Path)
    args = cli.parse_args()
    portfolio(args.tiles, args.research, args.output, args.selected_output)
