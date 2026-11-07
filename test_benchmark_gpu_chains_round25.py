"""No-device tests of proposal34 finite comparison accounting."""
from copy import deepcopy
import unittest
from unittest.mock import patch
import numpy as np
import benchmark_gpu_chains_round25 as benchmark
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles
import gpu_sampling as sampling

class FiniteGuidanceHarnessTests(unittest.TestCase):
    def test_initial_gold_costs_zero_and_one_round_charges_all_lanes(self):
        p=SearchProblem(*tetrahedron_tiles());goal=[0,3,6,9]
        for starts_solved in (False,True):
            class Engine:
                def __init__(self):self.steps=0;self.codes=goal if starts_solved else [1,3,6,9];self.closed=False
                def snapshot_arrays(self):return [np.array(self.codes,np.uint32)[:,None]]*2+[np.zeros((22,1),np.uint32)]
                def runtime_identity(self):return {'backend':'finite-fake'}
                def top(self,count):return np.tile(self.codes,(count,1)),np.full(count,sampling.gold_score(p,self.codes)['gold_objective'])
                def launch(self,steps):self.steps+=steps;self.codes=goal
                def experiment_stats(self):return {'guided_candidate_checks':self.steps*3}
                def close(self):self.closed=True
            engine=Engine()
            with patch.object(sampling,'_make_sampler',return_value=engine):
                result=benchmark.run_case(p,{tuple(goal)},seed=1,replicas=8,backend='cuda',guided=True,max_proposals=4)
            self.assertEqual(result['lane_proposals'],0 if starts_solved else 8);self.assertEqual(result['initial_gold'],starts_solved);self.assertTrue(engine.closed)
    def test_capped_pairs_are_not_folded_into_solved_work_comparison(self):
        p=SearchProblem(*tetrahedron_tiles())
        def run(*a,**kw):return {'initial_population_sha256':'same','initial_gold':False,'status':'solved' if kw['guided'] else 'capped','lane_proposals':3 if kw['guided'] else 64,'counters':{'guided_candidate_checks':17}}
        with patch.object(benchmark,'cases',return_value=[('finite',p,{(0,3,6,9)})]),patch.object(benchmark,'run_case',side_effect=run):report=benchmark.compare('cuda',seeds=2)
        self.assertEqual(report[0]['both_solved_pairs'],0);self.assertEqual(report[0]['capped_pairs'],2);self.assertEqual(report[0]['both_solved_lane_proposals'],{'baseline':0,'guided':0})
    def test_initial_state_mismatch_rejects_comparison(self):
        p=SearchProblem(*tetrahedron_tiles())
        def run(*a,**kw):return {'initial_population_sha256':str(kw['guided']),'initial_gold':False}
        with patch.object(benchmark,'cases',return_value=[('finite',p,{(0,3,6,9)})]),patch.object(benchmark,'run_case',side_effect=run):
            with self.assertRaisesRegex(ValueError,'initial state'):benchmark.compare('cuda',seeds=1)
    def test_independent_fixture_enumeration_includes_known_gold(self):
        rows=benchmark.cases();self.assertEqual([row[0] for row in rows],['tetrahedron','singleton_domains'])
        for name,p,solutions in rows:self.assertIn((0,3,6,9),solutions)
        self.assertEqual(rows[1][2],{(0,3,6,9)})
if __name__=='__main__':unittest.main()
