import unittest
from pathlib import Path
import benchmark_gpu_temperatures_round25 as bench

class TemperatureEvidenceTests(unittest.TestCase):
    def test_capped_and_adverse_pairs_never_dropped_or_combined_with_solved_costs(self):
        def row(status,cost,adjust=0):return {'status':status,'lane_proposals':cost,'controller_stats':{'temperature_adjustments':adjust}}
        rows=[{'baseline':row('solved',10),'adaptive':row('solved',8,1)},
              {'baseline':row('solved',10),'adaptive':row('solved',12,1)},
              {'baseline':row('capped',20),'adaptive':row('solved',16,2)},
              {'baseline':row('solved',10),'adaptive':row('capped',20,2)},
              {'baseline':row('solved',5),'adaptive':row('solved',4,0)}]
        result=bench.summarize(rows)
        self.assertEqual(result['both_solved'],3);self.assertEqual(result['baseline_both_solved_proposals'],25);self.assertEqual(result['adaptive_both_solved_proposals'],24)
        self.assertEqual(result['adaptive_both_solved_losses'],1);self.assertEqual(result['post_adjustment_wins'],2)
        self.assertEqual(result['baseline_solved'],4);self.assertEqual(result['adaptive_solved'],4)
    def test_guard_initial_gold_and_burn_in_scope_are_explicit(self):
        text=Path(bench.__file__).read_text();self.assertIn('require_assertions()',text);self.assertIn("repo.get('dirty') is not False",text)
        self.assertIn("'initial_gold':session.launches==0",text);self.assertIn('all adaptation burn-in proposals are included',text)
        self.assertIn('No stationary-distribution',text);self.assertIn('pre_adjustment_native_sha256',text)
if __name__=='__main__':unittest.main()
