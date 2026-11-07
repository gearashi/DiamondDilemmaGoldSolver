"""Host accounting checks for prepared-parent crossover evidence."""
import unittest
from unittest.mock import patch
import benchmark_gpu_crossover_round25 as benchmark
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles

class CrossoverBenchmarkTests(unittest.TestCase):
    def test_fixtures_are_explicit_non_gold_parents_and_known_gold_targets(self):
        rows=benchmark.fixtures();self.assertEqual([row[0] for row in rows],['physical_cycles','rotation_cycles','nearby_160'])
        for name,p,parents,solutions in rows:
            self.assertEqual(len(parents),2);self.assertTrue(all(not p.validate(v)['valid'] for v in parents));self.assertTrue(all(p.validate(v)['valid'] for v in solutions))
        self.assertEqual([len(row[3]) for row in rows],[4,4,1]);self.assertEqual(rows[-1][1].n,160)
    def test_mismatched_prepared_native_states_reject_comparison(self):
        p=SearchProblem(*tetrahedron_tiles())
        with patch.object(benchmark,'fixtures',return_value=[('finite',p,[[1,3,6,9],[0,4,6,9]],{(0,3,6,9)})]),patch.object(benchmark,'baseline',return_value={'initial_native_sha256':'a'}),patch.object(benchmark,'recombine',return_value={'initial_native_sha256':'b'}):
            with self.assertRaisesRegex(ValueError,'native states differ'):benchmark.comparisons('cuda',seeds=1)
    def test_capped_baselines_never_enter_completed_comparison_total(self):
        p=SearchProblem(*tetrahedron_tiles())
        def baseline(*args,**kw):return {'initial_native_sha256':'same','status':'solved' if kw['seed']==1 else 'capped','gpu_lane_proposals':12 if kw['seed']==1 else 256}
        with patch.object(benchmark,'fixtures',return_value=[('finite',p,[[1,3,6,9],[0,4,6,9]],{(0,3,6,9)})]),patch.object(benchmark,'baseline',side_effect=baseline),patch.object(benchmark,'recombine',return_value={'initial_native_sha256':'same','status':'solved'}):
            row=benchmark.comparisons('cuda',seeds=2)[0]
        self.assertEqual(row['solved_baseline_pairs'],1);self.assertEqual(row['solved_baseline_gpu_lane_proposals'],12)
    def test_full160_case_does_not_claim_random_baseline_or_enumeration(self):
        row=benchmark.fixtures()[-1]
        with patch.object(benchmark,'fixtures',return_value=[row]),patch.object(benchmark,'baseline',side_effect=AssertionError('160baseline was not authorized by this contract')),patch.object(benchmark,'recombine',return_value={'initial_native_sha256':'known','status':'solved'}):
            result=benchmark.comparisons('cuda')[0]
        self.assertIsNone(result['pairs'][0]['baseline']);self.assertIn('no enumeration claim',result['target_scope'])
if __name__=='__main__':unittest.main()
