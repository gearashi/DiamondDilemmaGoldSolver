import subprocess,sys,tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
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
    def test_frozen_dirty_guard_precedes_any_search(self):
        with tempfile.TemporaryDirectory() as folder:
            argv=['benchmark_round26.py','--suite','finite','--profiles','round26-reuse','--output',folder,'--evidence-stage','frozen']
            with patch.object(sys,'argv',argv),patch.object(benchmark.harness,'repository_snapshot',return_value={'commit':'abc','dirty':True}),patch.object(benchmark.harness.subprocess,'Popen',side_effect=AssertionError('No child search')):
                with self.assertRaisesRegex(RuntimeError,'clean, readable committed'):benchmark.main()

if __name__=='__main__':unittest.main()
