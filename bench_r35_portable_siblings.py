"""Evaluate explicit v1 portable export independently of native checkpoint ownership."""
import argparse
import json
import math
import zlib
from pathlib import Path
from time import perf_counter

from constraint_portable import dfs_roots, export_frontier
from constraint_work import WorkScope, checkpoint_boxes, frontier_snapshot, intersect, restore_frontier
from gpu_domain_runner import inventory_screen, split_cubes
from search_problem import SearchProblem


def volume(row):
    return math.prod(mask.bit_count() for mask in row)


def verify_cover(original, reduced):
    # The decoder checks both covers for disjointness. Verify mutual coverage by
    # intersection volume per output, including nonrectangular holes.
    assert sum(map(volume, original)) == sum(map(volume, reduced))
    for box in reduced:
        covered = sum(volume(part) for source in original if (part := intersect(box, source)) is not None)
        assert covered == volume(box)


def archive_size(payload):
    body = json.dumps(payload, separators=(',', ':')).encode()
    return {'json_bytes': len(body), 'zlib_bytes': len(zlib.compress(body))}


def compare(problem, name, rows, exercise=False):
    scope = WorkScope.for_problem(problem)
    native_rows = frontier_snapshot(scope, rows)
    raw = restore_frontier(native_rows, scope)
    start = perf_counter()
    saved = export_frontier(problem, native_rows, options={'portable_frontier': True})
    elapsed = perf_counter() - start
    merged = restore_frontier(saved, scope)
    verify_cover(raw, merged)
    answer = {'name':name, 'before':{'boxes':len(raw), **archive_size(native_rows)},
              'after':{'boxes':len(merged), **archive_size(saved)}, 'export_seconds': round(elapsed, 5)}
    if exercise:
        from constraint_sat_session import solve
        from constraint_parallel import solve as parallel
        for label, payload in (('before',native_rows),('after',saved)):
            state = {}
            try:
                result = solve(problem, seconds=30, resume=payload, learning_state=state)
                assert result['status'] == 'infeasible' and result['complete'] or result.get('validation',{}).get('valid')
                stats = result['stats']
                cpu = parallel(problem,seconds=120,resume=payload,workers=1)
                assert cpu['status'] == result['status']
                answer[label].update(dfs_import_roots=len(dfs_roots(problem,payload)),
                                     dfs_nodes=cpu['nodes'], dfs_jobs_completed=cpu['stats']['parallel_jobs_completed'],
                                     sat_status=result['status'],
                                     sat_calls=stats['session_native_calls'], sat_decisions=stats['session_total_decisions'],
                                     sat_cubes_closed=stats['session_cubes_closed'])
            finally:
                if state.get('_sat_session') is not None:
                    state['_sat_session'].close()
        assert answer['before']['sat_status'] == answer['after']['sat_status']
    return answer


def benchmark(tiles, runtime=None):
    from test_constraint_search import pair_problem
    from test_gpu_domain_round25 import (joint_hall_fixture, planted160_fixture, precise_reason_fixture,
                                         word_boundary_fixture)
    from test_gpu_exact_round26 import inventory_fixture
    problem = SearchProblem(json.loads(tiles.read_bytes()))
    scope = WorkScope.for_problem(problem)
    results = []
    for lanes in (32,64,128,256,512):
        split = split_cubes([scope.root],lanes)
        live, _ = inventory_screen(split,problem.n)
        results.append(compare(problem, f'original-{lanes}-all', split))
        results.append(compare(problem, f'original-{lanes}-inventory-screened', live))
    for name, small in (('two-piece-gold',pair_problem()[1]), ('two-piece-multiple-loops',pair_problem(True)[1]),
                        ('joint-hall',joint_hall_fixture()), ('word-boundary',word_boundary_fixture()),
                        ('precise-reasons',precise_reason_fixture()), ('inventory',inventory_fixture()),
                        ('planted160',planted160_fixture())):
        split = split_cubes([WorkScope.for_problem(small).root],128)
        live, _ = inventory_screen(split, small.n)
        results.append(compare(small,name,live,True))
    if runtime:
        for algorithm in ('dfs','hybrid'):
            path = runtime / 'searches' / (algorithm+'-gold') / 'checkpoint.json'
            if path.exists():
                checkpoint = json.loads(path.read_bytes())['payload']
                native = checkpoint.get('dfs',checkpoint)
                if native is not None:
                    results.append(compare(problem, algorithm+'-saved-stack', checkpoint_boxes(native)))
    return {'fingerprint':problem.fingerprint,'cases':results}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    cli.add_argument('--runtime',type=Path)
    cli.add_argument('--output',type=Path,required=True)
    args = cli.parse_args()
    args.output.write_text(json.dumps(benchmark(args.tiles,args.runtime),indent=2)+'\n',encoding='utf-8')
