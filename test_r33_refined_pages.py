"""A stronger loss filter can inherit plain exclusions but must revisit a hit's entire prefix."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from r32_gold import run_syndrome
from test_r33_gold_lookup import fixture


class RefinedPageTests(unittest.TestCase):
    def test_second_tail_in_same_prefix_and_cached_hit_isolation(self):
        p, ref, groups, q = fixture()
        with TemporaryDirectory() as tmp:
            db = Path(tmp)/'loss.sqlite'
            def search(refined):
                return run_syndrome(p,ref,groups,9,[4],db,0,page=1,refine_integer=refined)
            first = search(False)
            calls = []
            def reject_first(indices, deadline):
                calls.append(tuple(sorted(indices)))
                return len(calls)>1
            with patch('r33_gold_refine.make_integer_filter', return_value=reject_first):
                second = search(True)
            self.assertEqual((first['status'],second['status']),('PHASE_RELAXATION_FEASIBLE','INTEGER_RELAXATION_FEASIBLE'))
            self.assertEqual(first['next_rank'],second['next_rank'])
            self.assertNotEqual(first['outside_errors'],second['outside_errors'])
            self.assertEqual(second['work']['integer_checks'],2)
            self.assertEqual(second['work']['integer_rejections'],1)
            with patch('r33_gold_refine.make_integer_filter', side_effect=AssertionError):
                for flag, hit in ((False,first),(True,second)):
                    saved = search(flag)
                    self.assertEqual(list(map(tuple,saved['outside_errors'])),list(map(tuple,hit['outside_errors'])))
                    self.assertTrue(saved['reused'])
                    self.assertEqual(saved['work'],{})

    def test_inherit_plain_negative_ranges_and_restart_unknown(self):
        p, ref, groups, q = fixture()
        with TemporaryDirectory() as tmp:
            db = Path(tmp)/'loss.sqlite'
            run_syndrome(p,ref,groups,9,[3],db,0,page=1)
            with patch('r33_gold_refine.make_integer_filter',side_effect=AssertionError):
                warm = run_syndrome(p,ref,groups,9,[3],db,0,page=17,refine_integer=True)
            self.assertEqual((warm['work'],warm['new_pages']),({},0))
            with patch('r33_gold_refine.make_integer_filter', return_value=self.raise_timeout):
                interrupted = run_syndrome(p,ref,groups,9,[4],db,0,page=2,refine_integer=True)
            self.assertEqual((interrupted['status'],interrupted['next_rank']),('UNKNOWN',0))
            with patch('r33_gold_refine.make_integer_filter', return_value=lambda *_:False):
                rejected = run_syndrome(p,ref,groups,9,[4],db,0,page=3,refine_integer=True)
            self.assertEqual(rejected['status'],'INFEASIBLE')
            self.assertEqual(rejected['work']['integer_rejections'],15)
            self.assertEqual(run_syndrome(p,ref,groups,9,[4],db,0)['status'],'PHASE_RELAXATION_FEASIBLE')

    @staticmethod
    def raise_timeout(*args):
        raise TimeoutError


if __name__ == '__main__':
    unittest.main()
