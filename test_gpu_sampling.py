"""Sampling correctness and hybrid completeness; native tests are opt-in."""
import json
import os
import time
import unittest
from unittest.mock import patch
import numpy as np

import gpu_sampling as sampling
from constraint_dfs import solve as dfs_solve
from search_problem import SearchProblem
from test_constraint_dfs import board, double_face, simple_tiles, multiple_loops, tetrahedron_tiles


class SamplingTests(unittest.TestCase):
    def test_cpu_score_requires_permutation_fixed_placements_and_raw_reversal(self):
        problem = SearchProblem(simple_tiles(), double_face(), fixed={0: 0})
        self.assertEqual(sampling.score_board(problem, [0, 3]),
                         {'matched_edges': 3, 'required_edges': 3, 'domain_violations': 0, 'objective': 11})
        for codes in ([0, 0], [1, 3], [0, 6], [0], [0.0, 3.0]):
            with self.subTest(codes=codes), self.assertRaises(ValueError):
                sampling.score_board(problem, codes)
        changed = simple_tiles(); changed[1]['segments'] = [[[0, 2], [1, 4]]]
        self.assertEqual(sampling.score_board(SearchProblem(changed, double_face()), [0, 3])['matched_edges'], 1)

    def test_boundary_domain_violations_are_scored_not_hidden(self):
        tiles = [{'id': i, 'segments': [[[0, 2], [0, 10]]]} for i in range(2)]
        problem = SearchProblem(tiles, board(2, [(0, 0, 1, 0)]))
        self.assertEqual(sampling.score_board(problem, [1, 4])['domain_violations'], 2)
        table, count = sampling._inputs(problem)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(table[18*2:21*2].reshape(2,3), np.asarray(problem.neighbors, np.uint32))

    def test_zero_budget_stop_and_conflicting_fixed_tiles_never_start_gpu(self):
        problem = SearchProblem(simple_tiles(), double_face())
        with patch.object(sampling, '_make_sampler', side_effect=AssertionError('GPU should not start')):
            self.assertEqual(sampling.sample(problem, seconds=0)['status'], 'timeout')
            self.assertEqual(sampling.sample(problem, seconds=1, stop=lambda: True)['status'], 'stopped')
            conflict = SearchProblem(simple_tiles(), double_face(), fixed={0:0, 1:0})
            result = sampling.sample(conflict, seconds=1)
            self.assertEqual(result['status'], 'skipped')
            self.assertFalse(result['complete'])

    def test_returned_scores_are_verified_and_repeats_are_not_claimed_unique(self):
        problem = SearchProblem(simple_tiles(), double_face(), fixed={0:0, 1:3})
        clock = [0.0]
        class Fake:
            device = 'test device'
            closed = False
            def top(self): return np.array([[0,3], [0,3]]), np.array([11,11])
            def close(self): clock[0] += .1; self.closed = True
        fake = Fake()
        with patch.object(sampling, 'choose_backend', return_value='cuda'), patch.object(sampling, '_make_sampler', return_value=fake), patch.object(sampling.time, 'monotonic', side_effect=lambda: clock[0]):
            result = sampling.sample(problem, seconds=2, replicas=2)
        self.assertTrue(fake.closed)
        self.assertGreaterEqual(result['elapsed_seconds'], .01)
        self.assertEqual(result['device'], 'test device')
        self.assertEqual(result['samples_considered'], 2)
        self.assertIsNone(result['unique_boards'])
        self.assertFalse(result['duplicates_tracked'])
        self.assertFalse(result['complete'])
        self.assertEqual(result['top_board'], [0,3])
        fake.top = lambda: (np.array([[0,3]]), np.array([10]))
        fake.closed = False
        with patch.object(sampling, 'choose_backend', return_value='cuda'), patch.object(sampling, '_make_sampler', return_value=fake), patch.object(sampling.time, 'monotonic', side_effect=lambda: clock[0]):
            with self.assertRaisesRegex(RuntimeError, 'independent CPU scoring'):
                sampling.sample(problem, seconds=2, replicas=2)
        self.assertTrue(fake.closed)

    def test_invalid_inputs_are_rejected(self):
        problem = SearchProblem(simple_tiles(), double_face())
        for options in ({'seconds':-1}, {'seconds':float('inf')}, {'seconds':True},
                        {'seconds':1,'seed':-1}, {'seconds':1,'replicas':0},
                        {'seconds':1,'replicas':131073}, {'seconds':1,'backend':'cpu'}):
            with self.subTest(options=options), self.assertRaises(ValueError): sampling.sample(problem, **options)


class HybridTests(unittest.TestCase):
    def test_bad_sample_hints_cannot_remove_a_legal_solution(self):
        problem = SearchProblem(simple_tiles(), double_face())
        with patch.object(sampling, 'sample', return_value={'top_board':[1,3]}):
            result = sampling.solve(problem, seconds=2, seed=13)
        self.assertEqual(result['status'], 'solved')
        self.assertTrue(problem.validate(result['codes'])['valid'])
        self.assertEqual(result['method'], 'gpu_sampling_dfs')

    def test_valid_sample_is_returned_even_after_startup_uses_budget(self):
        problem = SearchProblem(simple_tiles(), double_face())
        clock = [0.0]
        def fake_sample(*args, **kwargs):
            clock[0] += 2.0
            return {'top_board':[0,3]}
        with patch.object(sampling, 'sample', side_effect=fake_sample), patch.object(sampling.time, 'monotonic', side_effect=lambda:clock[0]), patch('constraint_dfs.solve', side_effect=AssertionError('Certificate needs no DFS')):
            result = sampling.solve(problem, seconds=1)
        self.assertEqual(result['status'], 'solved')
        self.assertTrue(result['validation']['valid'])
        self.assertEqual(result['elapsed_seconds'], 2.0)
        loops = SearchProblem(multiple_loops(), double_face())
        with patch.object(sampling, 'sample', return_value={'top_board':[0,3]}):
            self.assertEqual(sampling.solve(loops, seconds=2, target='edges')['status'], 'edge_perfect')
            self.assertEqual(sampling.solve(loops, seconds=2, target='gold')['status'], 'infeasible')

    def test_hinted_full_tree_matches_baseline_and_resume_skips_sampling(self):
        problem = SearchProblem(multiple_loops(), double_face())
        baseline = []
        dfs_solve(problem, seconds=3, seed=7, hints=[2,5],
                  progress=lambda event: baseline.append(tuple(event['codes'])) if event['event']=='candidate' else None)
        seen = []
        def progress(event):
            if event['event']=='candidate': seen.append(tuple(event['codes']))
        with patch.object(sampling, 'sample', return_value={'top_board':[2,5]}) as sample:
            first = sampling.solve(problem, seconds=3, seed=7, progress=progress, stop=lambda:len(seen)>=4)
            self.assertEqual(sample.call_count, 1)
        self.assertEqual(first['status'], 'stopped')
        checkpoint = json.loads(json.dumps(first['checkpoint']))
        with patch.object(sampling, 'sample', side_effect=AssertionError('Resume must skip GPU')):
            final = sampling.solve(problem, seconds=3, seed=999, progress=progress, resume=checkpoint)
        self.assertEqual(final['status'], 'infeasible')
        self.assertTrue(final['complete'])
        self.assertEqual(seen, baseline)
        self.assertEqual(len(seen), 18)
        self.assertEqual(len(set(seen)), 18)

    def test_total_budget_counts_sampling_and_zero_budget_is_incomplete(self):
        problem = SearchProblem(simple_tiles(), double_face())
        clock = [0.0]
        def fake_sample(*args, **kwargs):
            clock[0] += .025
            return {'top_board':None}
        def fake_dfs(*args, **kwargs):
            self.assertLess(kwargs['seconds'], .03)
            return {'status':'timeout','codes':None,'elapsed_seconds':0,'nodes':0,'complete':False,'stats':{}}
        with patch.object(sampling, 'sample', side_effect=fake_sample), patch('constraint_dfs.solve', side_effect=fake_dfs), patch.object(sampling.time, 'monotonic', side_effect=lambda: clock[0]):
            result = sampling.solve(problem, seconds=.05)
        self.assertGreaterEqual(result['elapsed_seconds'], .02)
        with patch.object(sampling, 'sample', side_effect=AssertionError('No GPU for zero budget')):
            zero = sampling.solve(problem, seconds=0)
            self.assertEqual(zero['status'], 'timeout')
            self.assertFalse(zero['complete'])
            restored = sampling.solve(problem, seconds=2, resume=zero['checkpoint'])
            self.assertEqual(restored['status'], 'solved')


@unittest.skipUnless(os.environ.get('DIAMOND_TEST_GPU_SAMPLING') == '1', 'Optional native GPU sampling checks')
class NativeTests(unittest.TestCase):
    def test_asymmetric_and_boundary_scores_fixed_cells_and_moves(self):
        backend = os.environ.get('DIAMOND_SAMPLING_BACKEND', 'auto')
        tetra, geometry = tetrahedron_tiles()
        boundary_tiles = [{'id':i, 'segments':[[[0,2],[0,10]]]} for i in range(2)]
        cases = [SearchProblem(tetra, geometry), SearchProblem(tetra, geometry, fixed={0:0}),
                 SearchProblem(boundary_tiles, board(2, [(0,0,1,0)])),
                 SearchProblem(simple_tiles(), double_face(), fixed={0:0,1:3})]
        for problem in cases:
            with self.subTest(n=problem.n, fixed=problem.fixed):
                engine = sampling._make_sampler(problem, *sampling._inputs(problem), 32, 71, sampling.choose_backend(backend))
                try:
                    for _ in range(4):
                        boards, scores = engine.top(32)
                        for codes, score in zip(boards, scores):
                            self.assertEqual(sampling.score_board(problem, codes)['objective'], int(score))
                        engine.launch(64)
                finally: engine.close()


if __name__ == '__main__': unittest.main()
