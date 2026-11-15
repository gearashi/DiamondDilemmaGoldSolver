"""Compare reciprocal seam projections with the complete physical decision on matched residuals."""
import argparse
from pathlib import Path
from time import monotonic

from bench_r31_distribution import compare
from io_utils import atomic_json
from r28_exchange import read_json
from r32_fresh_geometry import projected
from search_problem import SearchProblem


def run(tiles, output, effort):
    p = SearchProblem(read_json(tiles))
    incumbent = read_json('docs/R29_INCUMBENT.json')['codes']
    results = {'fingerprint': p.fingerprint, 'effort': effort, 'scopes': []}
    for size, offset in ((6, 0), (8, 57), (24, 0)):
        pool = list(range(offset, offset + size))
        allowed = sorted(3 * (incumbent[c] // 3) + r for c in pool for r in range(3))
        rows = [allowed if c in pool else [code] for c, code in enumerate(incumbent)]
        ds = tuple(sum(1 << v for v in row) for row in rows)
        for target in (207, 208, 240):
            direct = compare(p, rows, 30, 8, target)
            row = dict(pool=pool, target=target, direct=direct, projections=[])
            # Without seam layers, every physical bijection/rotation meets the
            # projection. Materialize the incumbent and independently count it.
            row['unconstrained'] = dict(status='FEASIBLE', branches=0,
                                        actual_matches=p.validate(incumbent)['matched_edges'])
            results['scopes'].append(row)
            atomic_json(output, results)
            for layers in ((0,), tuple(range(6))):
                for geographic in (False, True):
                    start = monotonic()
                    ans = projected(p, ds, target=target, layers=layers,
                                    regions=(pool,) if geographic else (), effort=effort)
                    if 'codes' in ans:
                        assert all(v in opts for v, opts in zip(ans['codes'], rows))
                        assert len({v // 3 for v in ans['codes']}) == p.n
                        assert p.validate(ans['codes'])['matched_edges'] == ans['actual_matches']
                    if ans['status'] == 'INFEASIBLE':
                        assert direct['global']['status'] != 'witness'
                        assert target > p.validate(incumbent)['matched_edges']
                    row['projections'].append(dict(result=ans, geographic=geographic,
                                                   elapsed=round(monotonic()-start, 3)))
                    print(size, offset, target, len(layers), geographic, ans['status'],
                          ans['branches'], ans.get('actual_matches'), flush=True)
                    atomic_json(output, results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', default='../data/tiles.json')
    parser.add_argument('--effort', type=float, default=8)
    parser.add_argument('--output', type=Path, required=True)
    opts = parser.parse_args()
    run(opts.tiles, opts.output, opts.effort)
