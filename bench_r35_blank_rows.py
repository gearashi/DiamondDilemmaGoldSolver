"""Compare equivalent linear encodings of the residual physical blank matching."""
import argparse
from pathlib import Path
from unittest.mock import patch

from bench_r35_pair_projection import run


def post_rows(engine, flags, sides, blocked, parity, pairs, neighbors):
    # The outer signed inventory already enforces even totals in each color.
    # Every named pair consumes two equal-color sides. This premise permits
    # this comparison to elide either or both explicit residual parity rows.
    for index, active in enumerate(flags):
        count = sum(active)
        if index < parity:
            engine.Add(count == 2 * engine.IntVar(0, len(sides) // 2, ''))
        chosen = dict(zip(sides, active))
        if neighbors:
            for a in sides:
                unavailable = [chosen[b if a == c else c] for b, c in blocked if a in (b, c)]
                engine.Add(count >= 2 * chosen[a] + sum(unavailable))
        if pairs == 'none':
            continue
        for a, b in sorted(blocked):
            scale = 2 if pairs == 'weak' else 3
            engine.Add(count >= scale * (chosen[a] + chosen[b]) - (scale - 1))


def compare(tiles, output, seconds, cases, policies):
    for policy in policies:
        parity, pairs, neighbors = policy.split('-')
        if parity not in {'0', '1', '2'} or pairs not in {'none', 'weak', 'hull'} or neighbors not in {'n', 'y'}:
            raise ValueError('Policies use parity count, pair rows and the neighbor flag, such as 2-weak-n.')
        if pairs == 'none' and neighbors == 'n':
            raise ValueError('At least one condition must rule out forbidden pairs.')
        def install(engine, flags, sides, blocked):
            post_rows(engine, flags, sides, blocked, int(parity), pairs, neighbors == 'y')
        with patch('r35_mip_pairing.blank_rows', install):
            path = output.with_name(f'{output.name}-{policy}.json')
            run(tiles, path, ['blank'], seconds, cases)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True, help='Prefix for one JSON result per policy.')
    ap.add_argument('--seconds', type=float, default=60)
    ap.add_argument('--cases', nargs='+', required=True)
    ap.add_argument('--policies', nargs='+', required=True)
    compare(**vars(ap.parse_args()))
