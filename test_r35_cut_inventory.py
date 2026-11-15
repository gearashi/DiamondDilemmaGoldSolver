import copy
import itertools
import unittest
from unittest.mock import patch

from constraint_advanced import AdvancedPruner, _complete_matching_supports, _put
from constraint_dfs import _checkpoint_digest
from constraint_search import run_solver
from r35_cut_inventory import CutInventoryCache, dense_support
from search_fixtures import planted_gold
from search_problem import SearchProblem


class DenseAssignments(unittest.TestCase):
    def test_every_small_rectangular_graph_against_injective_enumeration(self):
        for cells, tiles in ((0, 0), (1, 3), (2, 4), (3, 4), (4, 3)):
            possibilities = list(itertools.permutations(range(tiles), cells))
            for rows in itertools.product(range(1 << tiles), repeat=cells):
                supported = dense_support(rows)
                if supported is None:
                    continue
                exact = [0]*cells
                matches = 0
                for assignment in possibilities:
                    if all(row & (1 << tile) for row,tile in zip(rows,assignment)):
                        matches += 1
                        for c,tile in enumerate(assignment):
                            exact[c] |= 1 << tile
                self.assertGreater(matches, 0, rows)
                self.assertEqual(tuple(exact), supported, rows)

    def test_singletons_empty_chains_collisions_and_sparse_ids(self):
        for rows in ((0,4), (4,4), (1,2,3), (1,3,14), (3,3,7)):
            self.assertIsNone(dense_support(rows), rows)
        tile = 1 << 157
        rows = (tile, tile | 3, tile | 7, tile | 7)
        self.assertEqual(dense_support(rows), (tile,3,7,7))
        self.assertEqual(dense_support(rows), _complete_matching_supports(rows)[0])

    def test_cache_only_supplies_complete_rows_and_retains_existing_eviction(self):
        cache = CutInventoryCache()
        unknown = (3,3,7)
        self.assertNotIn(unknown, cache)
        with patch('r35_cut_inventory.dense_support', side_effect=AssertionError):
            _put(cache,unknown,_complete_matching_supports(unknown)[0],128)
        self.assertIn(unknown,cache)
        self.assertEqual(cache.certificates, 0)
        for index in range(145):
            rows = (1 << index,)
            self.assertIn(rows,cache)
            self.assertEqual(cache[rows],rows)
            self.assertIn(rows,cache)
        self.assertNotIn(unknown,cache)
        self.assertEqual(len(cache),128)
        self.assertEqual(cache.certificates,145)
        self.assertNotIn((3,)*161,cache)

    def test_cancelled_cut_and_budget_fallback_keep_all_native_supports(self):
        from test_constraint_advanced import ConditionalGlobalInventoryTests
        p, ds = ConditionalGlobalInventoryTests.fixture()
        for matching, support in ((0,0), (5,0), (200000,1), (200000,200000)):
            outcomes = []
            for active in (False,True):
                helper = AdvancedPruner(p, options={'native':'python'})
                if active:
                    helper._cut_inventory_cache=CutInventoryCache()
                helper.cut_matching_work_limit = matching
                helper.cut_support_work_limit = support
                def stop():
                    raise InterruptedError
                with self.assertRaises(InterruptedError):
                    helper._cut_support(ds,(0,1),stop)
                outcomes.append(helper._cut_support(ds,(0,1),lambda: None))
            for result in outcomes:
                if result is None:
                    continue
                for codes in ConditionalGlobalInventoryTests.cut_oracle(p,ds):
                    self.assertTrue(all(mask & (1 << codes[c]) for c,mask in result))
            if matching == support == 200000:
                self.assertEqual(*outcomes)


class InventoryCheckpoints(unittest.TestCase):
    def setUp(self):
        tiles,board,self.solution = planted_gold(37,1,5)
        self.problem = SearchProblem(tiles,board,physical_gold_support=True)

    def run_dfs(self, enabled, **kwargs):
        return run_solver(self.problem, 'dfs', reference_pose_cuts=enabled,
                          options={'native':'python'}, **kwargs)

    def test_marker_bound_to_fresh_opt_in_and_validated_before_use(self):
        legacy = self.run_dfs(False,seconds=0)['checkpoint']
        self.assertNotIn('cut_inventory_policy',legacy)
        self.assertNotIn('cut_dense_inventory_certificates',legacy['stats'])
        wrong = copy.deepcopy(legacy)
        wrong.pop('root_symmetry',None)  # Also exercise the older metadata-free schema.
        wrong['stats']['cut_dense_inventory_certificates'] = 1
        wrong['digest'] = _checkpoint_digest(wrong)
        with self.assertRaisesRegex(ValueError,'counts need a marked checkpoint'):
            self.run_dfs(True,seconds=0,resume=wrong)
        self.assertNotIn('cut_inventory_policy',
                         self.run_dfs(True,seconds=0,resume=legacy)['checkpoint'])
        marked = self.run_dfs(True,seconds=0)['checkpoint']
        self.assertEqual(marked['cut_inventory_policy'],'dense-inventory-v1')
        inactive=self.run_dfs(False,seconds=0,resume=marked)['checkpoint']
        self.assertEqual(marked['stack'],inactive['stack'])
        self.assertEqual(inactive['cut_inventory_policy'],marked['cut_inventory_policy'])
        corrupt = copy.deepcopy(marked)
        corrupt['cut_inventory_policy']='unknown'
        corrupt['digest']=_checkpoint_digest(corrupt)
        with self.assertRaisesRegex(ValueError,'inventory checkpoint policy'):
            self.run_dfs(True,seconds=0,resume=corrupt)

    def test_legacy_and_fresh_resume_gold_and_warm_explicit_off(self):
        for fresh, resume in ((False,True), (True,False), (True,True)):
            checkpoint=self.run_dfs(fresh,seconds=0)['checkpoint']
            warm={}
            result=self.run_dfs(resume,seconds=6,resume=checkpoint,learning_state=warm)
            self.assertEqual(result['status'],'solved')
            self.assertTrue(self.problem.validate(result['codes'])['valid'])
            cache=warm['_runtime'].advanced._cut_inventory_cache
            self.assertEqual(isinstance(cache,CutInventoryCache), fresh and resume)
            # The pre-existing result and course of the saved root remain usable
            # after configuration changes even with a shared in-memory runtime.
            again=self.run_dfs(False,seconds=6,resume=checkpoint,learning_state=warm)
            self.assertEqual(again['codes'],result['codes'])
            self.assertIs(type(warm['_runtime'].advanced._cut_inventory_cache),dict)
