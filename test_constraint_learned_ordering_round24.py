"""Frozen fitted ordering must preserve every value and require real weights."""
import unittest
from constraint_branching_round24 import LearnedOrdering,LEARNED_ORDERING_WEIGHTS,LEARNED_ORDERING_PROVENANCE
from constraint_dfs import solve
from benchmark_learned_ordering_round24 import fixture,TRAIN_SEEDS,TEST_SEEDS,LARGE_SEEDS

class LearnedOrderingTests(unittest.TestCase):
    def test_frozen_model_has_completed_training_provenance_and_disjoint_test_seeds(self):
        self.assertEqual(LEARNED_ORDERING_PROVENANCE['training_seeds'],TRAIN_SEEDS)
        self.assertEqual(LEARNED_ORDERING_PROVENANCE['completed_labels'],46)
        self.assertFalse(set(TRAIN_SEEDS)&set(TEST_SEEDS+LARGE_SEEDS))
        self.assertEqual(len(LEARNED_ORDERING_WEIGHTS),5)
        self.assertTrue(any(LEARNED_ORDERING_WEIGHTS))

    def test_predictions_only_permute_existing_values_and_unfitted_call_fails(self):
        problem=fixture(TEST_SEEDS[0]);domains=[sum(1<<c for c in row) for row in problem.domains]
        choices=problem.domains[0];rank={code:i for i,code in enumerate(choices)};stats={}
        with self.assertRaises(ValueError):LearnedOrdering(problem).order(0,domains,choices,{},rank,lambda:None,stats)
        model=LearnedOrdering(problem,LEARNED_ORDERING_WEIGHTS)
        ordered=model.order(0,domains,choices,{choices[-1]:0},rank,lambda:None,stats)
        self.assertEqual(sorted(ordered),sorted(choices));self.assertEqual(ordered[0],choices[-1])
        self.assertEqual(domains,[sum(1<<c for c in row) for row in problem.domains])

    def test_public_flag_uses_fitted_weights_and_finds_validated_gold(self):
        problem=fixture(TEST_SEEDS[0])
        result=solve(problem,seconds=5,seed=TEST_SEEDS[0],options={'native':'python','learned_ordering':True})
        self.assertEqual(result['status'],'solved')
        self.assertGreater(result['stats']['learned_order_calls'],0)
        self.assertTrue(problem.validate(result['codes'])['valid'])

if __name__=='__main__':unittest.main()
