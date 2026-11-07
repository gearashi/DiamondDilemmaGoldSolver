import unittest
from pathlib import Path
import benchmark_gpu_gold_scoring_round25 as bench

class GoldScoringEvidenceTests(unittest.TestCase):
    def test_exact_pass_conservation_and_old_work_kept_separate(self):
        counts={'gold_old_base_scans_saved':10,'gold_old_loop_traces_saved':10,'gold_candidate_loop_traces':4,'gold_tabu_traces_skipped':2,'gold_metropolis_traces_skipped':4}
        result=bench.work_counts(counts,10)
        self.assertEqual(result['baseline_old_and_candidate_loop_passes'],20);self.assertEqual(result['lazy_candidate_loop_passes'],4)
        self.assertEqual(result['omitted_loop_passes'],16);self.assertEqual(result['omitted_base_passes'],10)
        for change in ({'gold_candidate_loop_traces':3},{'gold_tabu_traces_skipped':True},{'gold_old_loop_traces_saved':11}):
            with self.assertRaises(ValueError):bench.work_counts({**counts,**change},10)
    def test_guard_and_truthful_fixed_work_scope(self):
        text=Path(bench.__file__).read_text()
        self.assertIn('require_assertions()',text);self.assertIn("repo.get('dirty') is not False",text)
        self.assertIn('source_snapshot(ROOT)',text);self.assertIn("data!=args.data.read_bytes()",text)
        self.assertIn('Fixed work continues even after a synthetic Gold',text);self.assertIn('variable-length loop segment visits',text)
if __name__=='__main__':unittest.main()
