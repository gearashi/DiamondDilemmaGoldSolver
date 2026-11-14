import contextlib,io,subprocess,sys,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import benchmark_round26 as benchmark
from search_options import normalize_options,validate_algorithm_options

class Round26BenchmarkTests(unittest.TestCase):
    def test_profiles_are_explicit_compatible_opt_in_controls(self):
        for algorithm,profiles in [('parallel-dfs',benchmark.CPU_PROFILES),('gpu-domains',benchmark.GPU_PROFILES),('hybrid',benchmark.HYBRID_PROFILES)]:
            for name,options in profiles.items():validate_algorithm_options(algorithm,normalize_options(options))
    def test_worker_entrypoint_retains_new_profiles(self):
        with patch.object(benchmark.harness,'main',return_value=7) as run:
            self.assertEqual(benchmark.main(),7)
            self.assertEqual(run.call_args.kwargs['entry_path'],Path(benchmark.__file__).resolve())
        self.assertTrue(benchmark.harness.GPU_PROFILES['round26-cooperative']['gpu_cooperative'])
    def test_worker_keeps_durable_reasoning_inside_each_enabled_cpu_or_hybrid_case(self):
        configs = [('cpu', 'baseline', False), ('cpu', 'round26-reuse', True),
                   ('cpu', 'round26-shared', True), ('hybrid', 'round26-sampling', False),
                   ('hybrid', 'round26-combined', True)]
        with tempfile.TemporaryDirectory() as tmp, patch.dict(benchmark.harness.PROFILES, benchmark.CPU_PROFILES), \
                patch.dict(benchmark.harness.HYBRID_PROFILES, benchmark.HYBRID_PROFILES):
            for engine, profile, durable in configs:
                with self.subTest(engine=engine, profile=profile):
                    path = Path(tmp) / (profile + '.json')
                    args = SimpleNamespace(case='small', seed=817, free_cells=8, seconds=.01,
                                           engine=engine, workers=1, replicas=128, profile=profile, output=path)
                    def fake(problem, **kwargs):
                        state = kwargs['learning_state']
                        self.assertEqual('_reasoning_cache_dir' in state, durable)
                        if durable:
                            cache = Path(state['_reasoning_cache_dir'])
                            self.assertEqual(cache, path.with_suffix('.state') / 'reasoning-cache')
                            self.assertFalse(cache.exists())
                            cache.mkdir()
                        return {'status':'infeasible', 'complete':True, 'phase':'exact_dfs', 'nodes':0, 'stats':{}}
                    target = 'constraint_parallel.solve' if engine == 'cpu' else 'gpu_sampling.solve'
                    with patch(target, side_effect=fake), contextlib.redirect_stdout(io.StringIO()):
                        benchmark.harness.worker(args)
                    state = json.loads(path.read_text())['benchmark_state']
                    self.assertEqual(state['reasoning_directory_supplied'], durable)
                    self.assertEqual(state['reasoning_directory_created'], durable)

    def test_hybrid_releases_retained_repair_after_the_last_slice(self):
        from unittest.mock import Mock
        repair = Mock()
        state = {'_repair_service': repair}
        with patch('gpu_sampling.solve', side_effect=RuntimeError('stop test')):
            with self.assertRaisesRegex(RuntimeError, 'stop test'):
                benchmark.harness.run_hybrid(None, seconds=1, seed=0, replicas=128, backend='auto',
                    options={'durable_hybrid':True}, learning_state=state,
                    publication=SimpleNamespace(progress=None, failed=None))
        repair.close.assert_called_once_with()

    def test_frozen_dirty_guard_precedes_any_search(self):
        with tempfile.TemporaryDirectory() as folder:
            argv=['benchmark_round26.py','--suite','finite','--profiles','round26-reuse','--output',folder,'--evidence-stage','frozen']
            with patch.object(sys,'argv',argv),patch.object(benchmark.harness,'repository_snapshot',return_value={'commit':'abc','dirty':True}),patch.object(benchmark.harness.subprocess,'Popen',side_effect=AssertionError('No child search')):
                with self.assertRaisesRegex(RuntimeError,'clean, readable committed'):benchmark.main()

if __name__=='__main__':unittest.main()
