"""Exact v1 transfers, including holes, partial scopes and native owner preservation."""
import contextlib
import io
import itertools
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from constraint_portable import export_frontier
from constraint_work import WorkScope, digest, frontier_snapshot, restore_frontier


def assignments(rows):
    found = set()
    for row in rows:
        domains = [[i for i in range(mask.bit_length()) if mask >> i & 1] for mask in row]
        for item in itertools.product(*domains):
            assert item not in found, 'overlapping output'
            found.add(item)
    return found


class SiblingTests(unittest.TestCase):
    def test_all_subsets_of_two_and_three_axis_grids(self):
        for shape in ((3, 3), (2, 2, 2)):
            universe = tuple((1 << size) - 1 for size in shape)
            scope = WorkScope('finite', 'gold', universe, universe)
            space = list(itertools.product(*(range(size) for size in shape)))
            for subset in range(1 << len(space)):
                expected = {x for i, x in enumerate(space) if subset >> i & 1}
                rows = [tuple(1 << code for code in x) for x in sorted(expected)]
                portable = frontier_snapshot(scope, rows, compact=True)
                reduced = restore_frontier(portable, scope)
                self.assertEqual(assignments(reduced), expected)
                self.assertLessEqual(len(reduced), len(rows))
                self.assertEqual(frontier_snapshot(scope, reduced, compact=True), portable)
                if len(expected) == len(space):
                    self.assertEqual(reduced, (universe,))

    def test_nonrectangular_random_subsets_and_high_bits(self):
        from gpu_domain_runner import split_cubes
        rng = random.Random(3567)
        root = (3 << 130, 15, 7 << 63, 5)
        scope = WorkScope('wide', 'edges', root, root)
        for _ in range(60):
            rows = split_cubes([root], rng.randrange(3, 52))
            selected = [row for row in rows if rng.randrange(3)]
            rng.shuffle(selected)
            saved = frontier_snapshot(scope, selected, compact=True)
            self.assertEqual(assignments(restore_frontier(saved, scope)), assignments(selected))

    def test_acceptance_before_compaction_and_narrow_scope_preserved(self):
        from test_constraint_search import pair_problem
        _, problem = pair_problem()
        full = WorkScope.for_problem(problem)
        root = tuple(mask & -mask if i == 0 else mask for i, mask in enumerate(full.root))
        scope = WorkScope.for_problem(problem, root=root)
        rows = [(root[0], bit) for bit in (1 << i for i in range(root[1].bit_length())) if bit & root[1]]
        raw = frontier_snapshot(scope, rows)
        options = {'portable_frontier': True}
        compact = export_frontier(problem, raw, options=options)
        self.assertEqual(compact['scope'], raw['scope'])
        self.assertEqual(restore_frontier(compact, full), (root,))
        broken = dict(raw, boxes=raw['boxes'] * 2)
        broken['digest'] = digest({k:v for k,v in broken.items() if k != 'digest'})
        with self.assertRaisesRegex(ValueError, 'overlap'):
            export_frontier(problem, broken, options=options)
        with self.assertRaisesRegex(ValueError, 'checksum'):
            export_frontier(problem, dict(raw, digest='bad'), options=options)

    def test_gpu_app_writes_compact_transfer_only_for_explicit_export(self):
        import constraint_search as runner
        from gpu_domain_runner import COUNTERS, split_cubes
        from test_constraint_search import pair_problem
        data, problem = pair_problem()
        scope = WorkScope.for_problem(problem)
        raw = frontier_snapshot(scope, split_cubes([scope.root], 12))
        native = {'version': 1, 'scope': scope.payload(), 'queue': raw['boxes'],
                  'native': None, 'totals': dict.fromkeys(COUNTERS, 0), 'completed_lanes': 0,
                  'active_stats': {}, 'frontier': raw}
        native['digest'] = digest(native)
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                path = Path(directory)
                source = path / 'tiles.json'
                source.write_text(json.dumps(data), encoding='utf-8')
                output = path / 'run'
                args = ['--algorithm','gpu-domains','--seconds','5','--data',str(source),
                        '--output',str(output),'--search-options',json.dumps({'portable_frontier':enabled})]
                reply = {'status':'stopped','complete':False,'nodes':0,'stats':{},'checkpoint':native}
                with patch.object(runner,'SearchProblem',return_value=problem), patch.object(runner,'validate_runtime_selection'), patch.object(runner,'run_solver',return_value=reply), contextlib.redirect_stdout(io.StringIO()):
                    runner.main(args)
                folder = output / 'searches/gpu-domains-gold'
                self.assertEqual(json.loads((folder / 'checkpoint.json').read_bytes())['payload'], native)
                transfer = json.loads((folder / 'remaining-work.json').read_bytes())
                if enabled:
                    self.assertEqual(restore_frontier(transfer, scope), (scope.root,))
                else:
                    self.assertEqual(transfer, raw)


if __name__ == '__main__':
    unittest.main()
