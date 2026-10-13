"""Sampling correctness and hybrid completeness; native tests are opt-in."""
from functools import partial
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
    def test_software_adapter_opt_in_is_private_and_backend_limits_still_apply(self):
        for info in ({'adapter_type':'CPU','backend_type':'Vulkan'},
                     {'adapter_type':'DiscreteGPU','backend_type':'Vulkan','device':'lavapipe'},
                     {'adapter_type':'Software','backend_type':'D3D12'}):
            with self.subTest(info=info), patch.dict(os.environ, {'DIAMOND_TEST_SOFTWARE_GPU':'1'}):
                with self.assertRaisesRegex(RuntimeError,'software'):
                    sampling._validate_sampling_adapter(info)
                sampling._validate_sampling_adapter(info,allow_software=True)
                with self.assertRaisesRegex(RuntimeError,'software'):
                    sampling._validate_sampling_adapter(info,allow_software='true')
        for allowed in (False,True):
            with self.assertRaisesRegex(RuntimeError,'OpenGL'):
                sampling._validate_sampling_adapter({'adapter_type':'DiscreteGPU','backend_type':'OpenGL'},allow_software=allowed)
        sampling._validate_sampling_adapter({'adapter_type':'DiscreteGPU','backend_type':'Metal'})

    def test_factory_defaults_to_hardware_and_explicit_test_flag_reaches_only_webgpu(self):
        args=(object(),object(),2,128,17)
        with patch.object(sampling,'_WebGPU') as webgpu, patch.object(sampling,'_CUDA') as cuda:
            sampling._make_sampler(*args,'webgpu')
            webgpu.assert_called_once_with(*args,allow_software=False)
            sampling._make_sampler(*args,'webgpu',allow_software=True)
            webgpu.assert_called_with(*args,allow_software=True)
            sampling._make_sampler(*args,'cuda',allow_software=True)
            cuda.assert_called_once_with(*args)

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
        self.assertEqual(result['archive_unique_boards'],1)
        self.assertEqual(result['archive_repeated_observations'],1)
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
            result = sampling.solve(problem, seconds=2, seed=13, repair_fraction=0)
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
            self.assertEqual(sampling.solve(loops, seconds=2, target='gold', repair_fraction=0)['status'], 'infeasible')

    def test_hinted_full_tree_matches_baseline_and_resume_skips_sampling(self):
        problem = SearchProblem(multiple_loops(), double_face())
        # Isolate traversal/resume: new necessary deductions can reject this
        # deliberately impossible toy before it reaches the historical leaves.
        options = {'failure_guided':False, 'lookahead':False, 'strand_connectivity':False,
                   'multi_seam_cuts':False, 'regional_cache':False, 'biconnected':False, 'separator_regions':False,
        'backjumping':False, 'incremental':False, 'adaptive':False, 'persistent_learning':False, 'native':'python'}
        baseline = []
        dfs_solve(problem, seconds=3, seed=7, hints=[2,5], options=options,
                  progress=lambda event: baseline.append(tuple(event['codes'])) if event['event']=='candidate' else None)
        seen = []
        def progress(event):
            if event['event']=='candidate': seen.append(tuple(event['codes']))
        with patch.object(sampling, 'sample', return_value={'top_board':[2,5]}) as sample:
            first = sampling.solve(problem, seconds=3, seed=7, progress=progress, options=options, repair_fraction=0, stop=lambda:len(seen)>=4)
            self.assertEqual(sample.call_count, 1)
        self.assertEqual(first['status'], 'stopped')
        checkpoint = json.loads(json.dumps(first['checkpoint']))
        with patch.object(sampling, 'sample', side_effect=AssertionError('Resume must skip GPU')):
            final = sampling.solve(problem, seconds=3, seed=999, progress=progress, options=options, resume=checkpoint)
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


class RepairIntegrationTests(unittest.TestCase):
    def test_saved_better_board_wins_and_learning_state_is_forwarded(self):
        problem = SearchProblem(simple_tiles(),double_face())
        # Both are imperfect; [1,3] matches one empty seam, [1,5] matches none.
        saved = [1,3]
        other = min(([a,b] for a in range(3) for b in range(3,6)),
                    key=lambda codes:sampling.score_board(problem,codes)['objective'])
        seen, learning = {}, {}
        def repair(problem_arg, codes, **kwargs):
            seen['seed'] = codes
            return {'status':'inconclusive','best_board':codes,'best_score':1,'complete':False,'stats':{}}
        def exact(problem_arg, **kwargs):
            self.assertIs(kwargs['learning_state'],learning)
            self.assertEqual(kwargs['options'], {'regional_cache':False, 'biconnected':False, 'separator_regions':False,
        'backjumping':False, 'incremental':False, 'adaptive':False, 'persistent_learning':False, 'native':'python'})
            self.assertEqual(kwargs['hints'],saved)
            return {'status':'timeout','codes':None,'complete':False,'nodes':0,'elapsed_seconds':0,'stats':{}}
        with patch.object(sampling,'sample',return_value={'top_board':other}), patch('constraint_lns.repair',side_effect=repair), patch('constraint_dfs.solve',side_effect=exact):
            result = sampling.solve(problem,seconds=30,hints=saved,learning_state=learning,
                                    options={'regional_cache':False, 'biconnected':False, 'separator_regions':False,
        'backjumping':False, 'incremental':False, 'adaptive':False, 'persistent_learning':False, 'native':'python'})
        self.assertEqual(seen['seed'],saved)
        self.assertEqual(result['phase'],'exact_dfs')
        self.assertEqual(result['local_repair']['status'],'inconclusive')

    def test_large_duration_still_caps_both_heuristics_and_reports_phases(self):
        problem = SearchProblem(simple_tiles(),double_face())
        events, budgets = [], {}
        def sample(*args,**kwargs):
            budgets['sampling'] = kwargs['seconds']
            return {'top_board':[1,3]}
        def repair(*args,**kwargs):
            budgets['repair'] = kwargs['seconds']
            return {'status':'skipped','reason':'fixture','best_board':[1,3]}
        def exact(*args,**kwargs):
            budgets['exact'] = kwargs['seconds']
            return {'status':'timeout','codes':None,'complete':False,'nodes':0,'elapsed_seconds':0,'stats':{}}
        with patch.object(sampling,'sample',side_effect=sample), patch('constraint_lns.repair',side_effect=repair), patch('constraint_dfs.solve',side_effect=exact):
            sampling.solve(problem,seconds=1e9,progress=events.append)
        self.assertEqual(budgets['sampling'],30)
        self.assertEqual(budgets['repair'],60)
        self.assertGreater(budgets['exact'],1e9-1)
        self.assertEqual([event['phase'] for event in events],['sampling','repair','exact_dfs'])

    def test_zero_budget_with_saved_board_checkpoints_without_heuristic_work(self):
        problem = SearchProblem(simple_tiles(),double_face())
        with patch.object(sampling,'sample',side_effect=AssertionError('No sampling')), patch('constraint_lns.repair',side_effect=AssertionError('No repair')):
            result = sampling.solve(problem,seconds=0,seed_board=[0,3])
        self.assertEqual(result['status'],'timeout')
        self.assertEqual(result['checkpoint']['hybrid_version'],2)
        self.assertTrue(result['checkpoint']['heuristics_complete'])

    def test_old_hybrid_checkpoint_skips_both_heuristics(self):
        problem = SearchProblem(simple_tiles(),double_face())
        checkpoint = dfs_solve(problem,seconds=0)['checkpoint']
        old = {'hybrid_version':1,'dfs':checkpoint,'sampling':{'top_board':[1,3]}}
        with patch.object(sampling,'sample',side_effect=AssertionError('Do not resample')), patch('constraint_lns.repair',side_effect=AssertionError('Do not repeat repair')):
            result = sampling.solve(problem,seconds=2,resume=json.loads(json.dumps(old)))
        self.assertEqual(result['status'],'solved')
        self.assertIsNone(result['local_repair'])

    def test_missing_optional_repair_dependency_still_reaches_exact_dfs(self):
        problem = SearchProblem(simple_tiles(),double_face())
        events = []
        with patch.object(sampling,'sample',return_value={'top_board':[1,3]}), patch('constraint_lns._load_cp_model',side_effect=ImportError('fixture')):
            result = sampling.solve(problem,seconds=2,progress=events.append)
        self.assertEqual(result['status'],'solved')
        self.assertEqual(result['local_repair']['status'],'skipped')
        self.assertTrue(any(event['event']=='repair_skipped' and event['phase']=='repair' for event in events))

    def test_repaired_gold_certificate_skips_exact_search_but_multiloop_does_not(self):
        problem = SearchProblem(simple_tiles(),double_face())
        with patch.object(sampling,'sample',return_value={'top_board':[1,3]}), patch('constraint_lns.repair',return_value={'status':'solved','best_board':[0,3]}), patch('constraint_dfs.solve',side_effect=AssertionError('Already certified')):
            result = sampling.solve(problem,seconds=2)
        self.assertEqual(result['status'],'solved')
        self.assertEqual(result['phase'],'repair')
        loops = SearchProblem(multiple_loops(),double_face())
        with patch.object(sampling,'sample',return_value={'top_board':[0,3]}), patch('constraint_lns.repair',return_value={'status':'inconclusive','best_board':[0,3]}):
            result = sampling.solve(loops,seconds=2)
        self.assertEqual(result['status'],'infeasible')
        self.assertEqual(result['phase'],'exact_dfs')

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_GPU_SAMPLING') == '1', 'Optional native GPU sampling checks')
class NativeTests(unittest.TestCase):
    devices = set()
    def setUp(self):
        if os.environ.get('DIAMOND_TEST_SOFTWARE_GPU') == '1':
            # The environment switch belongs to this opted-in test class only.
            # Patching the private factory also covers sample()/hybrid callers.
            factory = partial(sampling._make_sampler, allow_software=True)
            self.enterContext(patch.object(sampling, '_make_sampler', factory))

    def test_asymmetric_and_boundary_scores_fixed_cells_and_moves(self):
        backend = os.environ.get('DIAMOND_SAMPLING_BACKEND', 'auto')
        tetra, geometry = tetrahedron_tiles()
        boundary_tiles = [{'id':i, 'segments':[[[0,2],[0,10]]]} for i in range(2)]
        cases = [SearchProblem(tetra, geometry), SearchProblem(tetra, geometry, fixed={0:0}),
                 SearchProblem(boundary_tiles, board(2, [(0,0,1,0)])),
                 SearchProblem(simple_tiles(), double_face(), fixed={0:0,1:3})]
        for problem in cases:
            with self.subTest(n=problem.n, fixed=problem.fixed):
                engine = sampling._make_sampler(problem, *sampling._inputs(problem), 128, 71, sampling.choose_backend(backend))
                self.devices.add(engine.device)
                try:
                    for _ in range(4):
                        boards, scores = engine.top(128)
                        for codes, score in zip(boards, scores):
                            self.assertEqual(sampling.score_board(problem, codes)['objective'], int(score))
                        engine.launch(64)
                finally: engine.close()


    def test_long_randomized_score_drift_current_best_and_gather_order(self):
        from search_fixtures import planted_gold
        backend=sampling.choose_backend(os.environ.get('DIAMOND_SAMPLING_BACKEND','auto'))
        tetra,geometry=tetrahedron_tiles()
        rng=np.random.default_rng(891)
        randomized=[]
        for tile in range(4):
            endpoints=rng.choice(33,size=8,replace=False)
            randomized.append({'id':tile,'segments':[[[int(v)//11,int(v)%11+1] for v in endpoints[i:i+2]] for i in range(0,8,2)]})
        data,mesh,known=planted_gold(seed=881,size=2)
        cases=[SearchProblem(randomized,geometry),SearchProblem(data,mesh,fixed={0:known[0],9:known[9]})]
        # Narrowed domains exercise weighted legality deltas as well as seam deltas.
        cases[0].domains=[values[::2] for values in cases[0].domains]
        def read_state(engine):
            if backend=='cuda': return engine.boards.get().T,engine.best.get().T,engine.meta.get()
            arrays=[np.frombuffer(engine.queue.read_buffer(engine.buffers[i]),np.uint32).copy() for i in range(3)]
            return arrays[0].reshape(engine.n,engine.r).T,arrays[1].reshape(engine.n,engine.r).T,arrays[2].reshape(4,engine.r)
        for problem in cases:
            engine=sampling._make_sampler(problem,*sampling._inputs(problem),128,892,backend)
            self.devices.add(engine.device)
            try:
                for epoch in range(5):
                    current,best,meta=read_state(engine)
                    for lane in range(engine.r):
                        self.assertEqual(sampling.score_board(problem,current[lane])['objective'],int(meta[1,lane]))
                        self.assertEqual(sampling.score_board(problem,best[lane])['objective'],int(meta[2,lane]))
                    self.assertTrue(np.all(meta[2]>=meta[1]))
                    # Growing and shrinking readbacks reuse scratch capacity safely.
                    for count in (1,5,128,7,129):
                        boards,scores=engine.top(count)
                        ids=np.argsort(meta[2],kind='stable')[-min(count,engine.r):][::-1]
                        np.testing.assert_array_equal(boards,best[ids])
                        np.testing.assert_array_equal(scores,meta[2,ids])
                    if epoch<4:
                        for _ in range(32): engine.launch(64)
                # Exercise the uint32 iteration wrap without changing placements.
                wrap=np.full(engine.r,2**32-4,np.uint32)
                if backend=='cuda': engine.meta[3]=engine.cp.asarray(wrap)
                else: engine.queue.write_buffer(engine.buffers[2],3*engine.r*4,wrap)
                engine.launch(64)
                current,best,meta=read_state(engine)
                np.testing.assert_array_equal(meta[3],np.full(engine.r,60,np.uint32))
                for lane in range(engine.r):
                    self.assertEqual(sampling.score_board(problem,current[lane])['objective'],int(meta[1,lane]))
                    self.assertEqual(sampling.score_board(problem,best[lane])['objective'],int(meta[2,lane]))
            finally: engine.close()

    def test_native_hybrid_validates_gold_and_keeps_multiloop_incomplete_until_dfs(self):
        backend=os.environ.get('DIAMOND_SAMPLING_BACKEND','auto')
        problem=SearchProblem(simple_tiles(),double_face())
        # Software shader compilation can consume the hardware fixture budget.
        # Bound only this test more generously; certificate assertions stay exact.
        budget=15 if os.environ.get('DIAMOND_TEST_SOFTWARE_GPU') == '1' else 3
        result=sampling.solve(problem,seconds=budget,seed=71,replicas=128,backend=backend)
        self.assertEqual(result['status'],'solved')
        self.assertTrue(problem.validate(result['codes'])['valid'])
        loops=SearchProblem(multiple_loops(),double_face())
        result=sampling.solve(loops,seconds=budget,seed=71,replicas=128,backend=backend)
        self.assertEqual(result['status'],'infeasible')
        self.assertEqual(result['phase'],'exact_dfs')



if __name__ == '__main__': unittest.main()
