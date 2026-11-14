"""Pair Gold loss queries with per-page suffix rebuilds and cross-page lookup reuse."""
import argparse
from collections import Counter
import gc
import hashlib
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory
from time import perf_counter
import tracemalloc
from unittest.mock import patch

from r32_gold import run_syndrome
from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


CASES = (
    # (budget, errors, first prefix, exclusive last prefix, page size)
    (14, 4, 0, 6903, 256),               # Entire closed four-error layer.
    (15, 5, 0, 266916, 8192),            # Entire closed five-error layer, released default.
    (16, 6, 7044800, 7061184, 2048),     # Four fixed prefix cells on a distant interval.
    (17, 6, 7044800, 7045824, 128),      # First open loss budget, small recovery pages.
)


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def search_comparison(work):
    return {name: work.get(name, 0) for name in ('prefixes', 'indexed_tails', 'joint_queries')}


def pages(query, case):
    cap, errors, lo, hi, page = case
    decisions = []
    work = Counter()
    start_time = perf_counter()
    for start in range(lo, hi, page):
        stop = min(hi, start + page)
        found = query.indexed(cap, errors, float('inf'), start, stop)
        work.update(found['work'])
        decision = {key: value for key, value in found.items() if key != 'work'}
        decisions.append(decision)
        if found['status'] == 'PHASE_RELAXATION_FEASIBLE':
            break
        if found['status'] != 'INFEASIBLE' or found['next_rank'] != stop:
            raise RuntimeError(f'Incomplete interval {start}..{stop}')
    elapsed = perf_counter() - start_time
    return dict(page_decisions=digest(decisions), last=decisions[-1], pages=len(decisions),
                work=dict(work), seconds=round(elapsed, 6))


def memory_sample(make_query, reuse):
    # Query basis construction falls outside the measurement. Collect cyclic temporaries
    # before the retained size reading; tracing stays off during timed experiments.
    query = make_query(reuse)
    gc.collect()
    tracemalloc.start()
    try:
        result = query.indexed(15, 5, float('inf'), 266913, 266914)
        gc.collect()
        retained, peak = tracemalloc.get_traced_memory()
        return dict(status=result['status'], next_rank=result['next_rank'],
                    traced_retained_bytes=retained, traced_peak_bytes=peak)
    finally:
        tracemalloc.stop()


def database_replay(problem, reference, groups, reuse):
    # Each arm starts from a new file. Reopening exercises actual page ownership.
    class ArmQuery(OutsideSyndromes):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, reuse_suffix_tables=reuse, seek_ordered_tails=False,
                             use_bucket_price=False, **kwargs)

    with TemporaryDirectory() as tmp, patch('r32_gold.OutsideSyndromes', ArmQuery):
        db = str(Path(tmp) / 'checkpoint.sqlite')
        before = perf_counter()
        cold = run_syndrome(problem, reference, groups, 14, None, db, 0, page=256)
        elapsed = perf_counter() - before
        warm = run_syndrome(problem, reference, groups, 14, None, db, 0, page=128)
        if not (cold['all_outside_layers'] and cold['status'] == warm['status'] == 'INFEASIBLE'
                and cold['completed_prefixes'] == warm['completed_prefixes']
                and warm['new_pages'] == 0 and not warm['work']):
            raise RuntimeError('Incomplete or repeated persistent coverage')
        return dict(cold=cold, warm=warm, cold_seconds=round(elapsed, 6))


def page_settings(problem, reference, groups):
    """Compare the same full layer under released CLI page size and smaller restart pages."""
    results = {}
    expected = None
    # Reverse the second pass to avoid assigning one size a fixed position.
    for page in (256, 1024, 8192, 8192, 1024, 256):
        with TemporaryDirectory() as tmp:
            db = str(Path(tmp) / 'settings.sqlite')
            before = perf_counter()
            out = run_syndrome(problem, reference, groups, 15, [5], db, 0, page=page)
            seconds = round(perf_counter() - before, 6)
            replay = run_syndrome(problem, reference, groups, 15, [5], db, 0, page=page)
        if not (out['status'] == 'INFEASIBLE' and out['checked_layers'] == [5]
                and out['completed_prefixes'] == 266916 and replay['new_pages'] == 0 and not replay['work']):
            raise RuntimeError('Incomplete settings comparison')
        evidence = (out['status'], out['total_prefixes'], search_comparison(out['work']))
        if expected is not None and expected != evidence:
            raise RuntimeError('Page size affected the completed search')
        expected = evidence
        if page not in results:
            results[page] = dict(page_size=page, pages=out['new_pages'], work=out['work'],
                                 replay_pages=replay['new_pages'], seconds=[])
        if results[page]['work'] != out['work']:
            raise RuntimeError('Inconsistent repeated page-size measurement')
        results[page]['seconds'].append(seconds)
        print('Default search page size', page, 'completed', seconds, flush=True)
    return dict(loss_budget=15, outside_errors=5, start=0, stop=266916,
                status=expected[0], cases=list(results.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', default='data/tiles.json')
    parser.add_argument('--center', default='docs/R29_INCUMBENT.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--released-syndrome', help='Path to the v1.13 r32_gold_syndrome.py for an independent control')
    args = parser.parse_args()
    released = None
    released_sha256 = None
    if args.released_syndrome:
        source = Path(args.released_syndrome).resolve()
        released_sha256 = hashlib.sha256(source.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        spec = spec_from_file_location('released_syndrome', source)
        released = module_from_spec(spec)
        spec.loader.exec_module(released)
    problem = SearchProblem(read_json(args.tiles))
    codes = read_json(args.center)['codes']
    reference, _, groups = reference_groups(problem, codes)
    required = inventory(problem).required

    def make_query(reuse):
        return OutsideSyndromes(problem, reference, groups, required, reuse_suffix_tables=reuse,
                              seek_ordered_tails=False, use_bucket_price=False)

    comparison = []
    for index, case in enumerate(CASES):
        row = dict(zip(('loss_budget', 'outside_errors', 'start', 'stop', 'page'), case))
        baseline = None
        for order in ((False, True), (True, False)):
            for reuse in (order if index % 2 == 0 else reversed(order)):
                run = pages(make_query(reuse), case)
                evidence = (run['page_decisions'], search_comparison(run['work']))
                if baseline is None:
                    baseline = evidence
                if evidence != baseline:
                    raise RuntimeError(f'Page outcome or search work changed for {case}')
                key = 'shared' if reuse else 'rebuild'
                if key not in row:
                    seconds = run.pop('seconds')
                    row[key] = dict(**run, seconds=[seconds])
                else:
                    if run['work'] != row[key]['work']:
                        raise RuntimeError(f'Nondeterministic work in {key} arm')
                    row[key]['seconds'].append(run['seconds'])
        if released is not None:
            previous = pages(released.OutsideSyndromes(problem, reference, groups, required), case)
            if (previous['page_decisions'], search_comparison(previous['work'])) != baseline:
                raise RuntimeError(f'Released v1.13 source disagrees on {case}')
            row['released'] = previous
        comparison.append(row)
        print(case, row['rebuild']['work']['suffix_rows'], row['shared']['work']['suffix_rows'], flush=True)
    checkpoints = {name: database_replay(problem, reference, groups, reuse)
                   for name, reuse in (('rebuild', False), ('shared', True))}
    old = checkpoints['rebuild']['cold']
    new = checkpoints['shared']['cold']
    if (any(old[key] != new[key] for key in old if key != 'work') or
            search_comparison(old['work']) != search_comparison(new['work'])):
        raise RuntimeError('SQLite work ownership or search work changed')
    report = dict(fingerprint=problem.fingerprint, reference_sha256=digest(codes), tiles=problem.n,
                  matched_reference_bonds=len(reference), python=platform.python_version(),
                  cases=comparison, checkpoints=checkpoints, released_source_sha256=released_sha256,
                  shared_controls=dict(seek_ordered_tails=False, use_bucket_price=False),
                  page_settings=page_settings(problem, reference, groups),
                  memory={arm: memory_sample(make_query, enabled)
                          for arm, enabled in (('rebuild', False), ('shared', True))})
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
