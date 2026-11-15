"""Replay genuine signed reference donors through old and new geographic bounds."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

from bench_r35_loss_boundary344 import problem_scope
from r30_gold import search
from r35_geographic_loss import certified_reference


def compare(tiles, output, seconds):
    problem, ref, _, required = problem_scope(tiles)
    directory = Path(__file__).resolve().parent/'docs'
    previous = json.loads((directory/'R35_CONTACT_WIDE_PROOF.json').read_bytes())
    current = json.loads((directory/'R35_SPHERE_CONTACT_PROOF.json').read_bytes())
    verified, limit = certified_reference(problem)
    assert verified == ref and len(ref)-limit == current['certified_lower_bound']
    samples = [('signed31',previous['witness']), *(('sphere33-'+str(i),w) for i,w in enumerate(current['witnesses']))]
    rows = []
    for label, witness in samples:
        lost = sorted(map(tuple,witness['lost']))
        expected = set(ref) - set(lost)
        for delta in (-1, 0, 1):
            if delta < 0:
                kept = expected - {min(expected-required)}
            else:
                kept = expected | set(lost[:delta])
            for maximum in (176,limit):
                with patch('r35_geographic_loss.certified_reference',return_value=(ref,maximum)):
                    answer = search(problem,geography=True,bonds=sorted(kept),reference_bound=True,
                                    seconds=seconds,max_rounds=1)
                rows.append(dict(donor=label, delta=delta, cap=maximum, retained=len(kept),
                                 status=answer['status'], branches=answer['branches'], cp_calls=answer['cp_calls'],
                                 forced=answer['certified_reference']['forced'],
                                 over=answer['certified_reference']['over'],
                                 removed_poses=answer['certified_reference']['unit_poses']))
    output.write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=.4)
    compare(**vars(parser.parse_args()))
