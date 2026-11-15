"""Replay geographic retention pruning around real unsigned counterexample covers."""
import argparse
import json
from pathlib import Path
from random import Random

from bench_r35_loss_boundary344 import problem_scope
from r27_structure_filters import domain_bits
from r30_gold import search
from r35_geographic_loss import certified_reference, retained_bound


def run(tiles, counterexamples, output):
    problem, reference, groups, required = problem_scope(tiles)
    provenance = json.loads(counterexamples.read_bytes())
    checked, current_limit = certified_reference(problem)
    assert reference == checked
    directory = Path(__file__).parent/'docs'
    for file,key in [('R35_GEOGRAPHIC_RESIDUALS.json','geographic_cores'),
                     ('R35_EDGE_CONTACTS.json','edge_perfect_cores')]:
        groups.extend(frozenset(map(tuple,row))-required for row in json.loads((directory/file).read_bytes())[key])
    clean = domain_bits(problem)
    records = []
    for seed, row in enumerate(provenance['rounds']):
        missing = set(map(tuple,row['lost']))
        assert len(missing)==25 and not missing&required and all(missing&group for group in groups)
        rng = Random(813500 + seed)
        choices = sorted(reference-required-missing)
        rng.shuffle(choices)
        for extra in range(9):
            omitted = missing | set(choices[:extra])
            kept = reference-omitted
            result = retained_bound(problem, list(clean), reference, current_limit, kept|required)
            rec = dict(seed=seed, omitted=len(omitted), forced=result['forced'],
                       previous_bound_closed=len(kept)>180,
                       current_bound_closed=result['over'], forbidden=len(result['forbidden']))
            # Compare one actual mathematical scope against the public embed entry point.
            if seed==0 and len(kept) in (current_limit, current_limit+1, current_limit-1):
                for on in (False,True):
                    answer=search(problem,geography=True,bonds=kept,reference_bound=on,seconds=0.4,max_rounds=1)
                    rec['on' if on else 'off']={k:answer[k] for k in ('status','cp_calls','branches')}
            records.append(rec)
    data=dict(limit=current_limit, previous_limit=180, unsigned_covers=len(provenance['rounds']),
              checked_old_groups=len(groups), records=records)
    output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ('tiles','counterexamples','output'):
        cli.add_argument('--'+name,required=True,type=Path)
    run(**vars(cli.parse_args()))
