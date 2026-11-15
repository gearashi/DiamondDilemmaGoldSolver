"""Compare exact grouped-image and drop searches on identical physical loss candidates."""

import argparse
from collections import Counter
from math import inf
from pathlib import Path
from time import perf_counter
from unittest.mock import patch
import json

from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r33_gold_refine import IntegerPatternFilter
from bench_r35_group_fallback import legacy_drops
from search_problem import SearchProblem


def benchmark(tiles, five_stop):
    problem = SearchProblem(read_json(tiles))
    docs = Path(__file__).parent / 'docs'
    reference, _, groups = reference_groups(problem, read_json(docs / 'R29_INCUMBENT.json')['codes'])
    query = OutsideSyndromes(problem, reference, groups, inventory(problem).required)
    images = query._joint_images
    check = IntegerPatternFilter(problem, reference, query.groups, query.required, query.outside, 17)
    other = OutsideSyndromes(problem, reference, groups, query.required)
    other._joint_images = None
    expected = read_json(docs / 'R34_INTEGER_CORES_FINAL.json')['arrivals']
    positives = []
    original = []
    warm = []
    def baseline_joint(indices, cap, deadline):
        return legacy_drops(other, indices, cap, deadline)
    new_joint = query.joint
    for errors, stop in ((4, 6903), (5, five_stop)):
        trace = []
        def observe(indices, cap, deadline):
            trace.append(tuple(sorted(indices)))
            return new_joint(indices, cap, deadline)

        # Keep the phase-only trace distinct from the integer model, whose
        # verified exclusions may remove candidates before joint evaluation.
        def record(indices, deadline):
            positives.append(sorted(indices))
            return False
        with patch.object(query, 'joint', observe):
            measured = query.indexed(17, errors, inf, 0, stop, accept_errors=record)
        if measured['next_rank'] != stop:
            raise AssertionError('truncated interval')
        def compare(candidates, label, count_inserts):
            images.minima.clear()
            before_images, before_hits = images.images_priced, images.cache_hits
            start = perf_counter()
            new = [new_joint(indices, 17, inf) for indices in candidates]
            new_s = perf_counter() - start
            unique_keys = set()
            for indices in candidates:
                key = images.symbols[0]
                for i in indices:
                    key ^= images.symbols[i+1]
                unique_keys.add(key)
            new_counts = {'images_priced': images.images_priced - before_images,
                          'complete_cost_reuses': images.cache_hits - before_hits,
                          'unique_cosets': len(unique_keys)}
            start = perf_counter()
            old = [baseline_joint(indices, 17, inf) for indices in candidates]
            old_s = perf_counter() - start
            if new != old:
                raise AssertionError('different joint outcome')
            insertions = None
            if count_inserts:
                import bench_r35_group_fallback
                insertions = 0
                raw_insert = bench_r35_group_fallback.insert
                def counted(basis, row):
                    nonlocal insertions
                    insertions += 1
                    return raw_insert(basis, row)
                with patch.object(bench_r35_group_fallback, 'insert', counted):
                    for indices in candidates:
                        baseline_joint(indices, 17, inf)
            out = dict(scope=label, queries=len(candidates), feasible=sum(old), drop_seconds=round(old_s,5),
                       image_seconds=round(new_s,5), baseline_insertions=insertions, **new_counts)
            print(errors, out, flush=True)
            return out
        original.append(dict(errors=errors, stop=stop, indexed_tails=measured['work']['indexed_tails'],
                             full=compare(trace,'all phase arrivals',False),
                             stratified=compare(trace[::max(1,len(trace)//128)],'systematic sample of at least 128',True)))
        surviving = []
        for indices in trace:
            if not check.excludes(indices,inf):
                surviving.append(indices)
                if new_joint(indices,17,inf) and check(indices,inf):
                    raise AssertionError('unexpected integer witness in archived interval')
        warm.append(compare(surviving,'after checked integer cores',False))
    if five_stop == 65536 and positives != expected:
        raise AssertionError((positives, expected))
    return dict(reference=problem.fingerprint, cap=17, image_count=len(images.images), original=original,
                after_integer_cores=warm, integer_work=dict(check.work), phase_hits=positives)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    cli.add_argument('--five-stop',type=int,default=65536)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    result=benchmark(args.tiles,args.five_stop)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
