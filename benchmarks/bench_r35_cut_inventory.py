"""Compare full conditional inventories on saved original DFS frames."""
import argparse
import hashlib
import json
from pathlib import Path

from bench_r35_pose_frontiers import saved_stacks
from constraint_advanced import AdvancedPruner
from r35_cut_inventory import CutInventoryCache
from search_problem import SearchProblem


def measure(problem, domains, wide, dense):
    pruner = AdvancedPruner(problem, 'gold', {'native': 'python', 'wide_cuts': wide})
    if dense:
        pruner._cut_inventory_cache = CutInventoryCache()
    after = list(domains)
    pruner.prune_cuts(after)
    certificates = pruner._cut_inventory_cache.certificates if dense else 0
    names = ('cut_prunes', 'cut_global_matching_steps', 'cut_global_support_steps',
             'cut_global_matching_cache_hits', 'cut_global_support_budget_skips')
    result = {name: pruner.stats[name] for name in names}
    result['cut_global_matching_cache_hits'] -= certificates
    result['cut_dense_inventory_certificates'] = certificates
    return after, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    problem = SearchProblem(json.loads(args.tiles.read_bytes()), physical_gold_support=True)
    records = []
    for name, domains in saved_stacks(args.checkpoints):
        if int(name.rsplit('-', 1)[1]) not in (5, 8, 12, 15, 20, 23):
            continue
        for wide in (False, True):
            original, before = measure(problem, domains, wide, False)
            result, after = measure(problem, domains, wide, True)
            if result != original:
                raise ValueError(f'Final conditional-cut domains differ for {name} with wide={wide}')
            records.append({'frame':name, 'wide':wide, 'baseline':before, 'dense':after,
                            'output_sha256':hashlib.sha256(json.dumps(result).encode()).hexdigest()})
    args.output.write_text(json.dumps(records, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    main()
