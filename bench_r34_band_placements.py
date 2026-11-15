"""Compare exact placement restarts against repeating the same regional SAT call."""
import argparse
from contextlib import closing, nullcontext
import json
from pathlib import Path
import sqlite3
import tempfile
from unittest.mock import patch

from pysat.solvers import Glucose4
from r27_reason_maxsat import SeamFormula, domain_rows
from r31_defect_bands import bands, exact_thresholds, solve
from r34_seam_cnf import encoded_scope, prepare
from search_fixtures import planted_gold
from search_problem import SearchProblem


def compare(opens, seconds):
    problem = SearchProblem(*planted_gold(19, 2, 3)[:2])
    rows = domain_rows(problem)
    result = dict(seed=19, cells=problem.n, edges=len(problem.edges), target=len(problem.edges),
                  calls_per_open=1, seconds_per_open=seconds, conflict_budget=2000)
    with tempfile.TemporaryDirectory() as temp:
        seed = Path(temp) / 'source.sqlite'
        with closing(sqlite3.connect(seed)) as db, Glucose4() as sat:
            prepare(db, seed, problem, rows, True, bands(problem, 3), 0, sat, lambda: None,
                    float('inf'), {}, SeamFormula, exact_thresholds, encoded_scope(problem, rows, True))
        for name in ('repeat', 'partition'):
            path = Path(temp) / (name + '.sqlite')
            with closing(sqlite3.connect(seed)) as src, closing(sqlite3.connect(path)) as db:
                src.backup(db)
            control = patch('r34_band_placements.split_placement', return_value=None)
            stats = []
            with control if name == 'repeat' else nullcontext():
                for _ in range(opens):
                    answer = solve(problem, path, target=len(problem.edges), width=1,
                                   slice_conflicts=2000, max_calls=1, seconds=seconds)
                    stats.append({k: answer[k] for k in ('status', 'calls', 'placement_splits',
                                 'placement_local_unsat', 'pending', 'decisions', 'conflicts')})
                    if answer['status'] != 'open':
                        break
            result[name] = stats
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--opens', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=.35)
    args = parser.parse_args()
    if args.opens < 1 or args.seconds <= 0:
        parser.error('Use a positive number of restarts and seconds.')
    print(json.dumps(compare(args.opens, args.seconds), indent=2))
