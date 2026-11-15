"""Geographic loss input checks agree on shortcuts, native and persistent paths."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r27_structure_fixtures import planted
from r32_gold_loss import geographic, read_json
from search_problem import SearchProblem


class GeographicQueryInputTests(unittest.TestCase):
    def setUp(self):
        self.problem,_=planted(1,3,70,uniform_ports=True)
        self.reference={(0,3),(1,6)}

    def query(self, database=None, *, checked=(), lower=0, upper=1, outside=None, domains=None):
        return geographic(self.problem,self.reference,[],checked,[],lower,upper,outside,
                          5,database,domains=domains)

    def test_integer_loss_counts_checked_before_model_and_database(self):
        with patch('r30_gold.search') as native, patch('r32_gold_loss.save_query') as cache:
            for values in (dict(lower=.5),dict(upper=.5),dict(outside=.5),
                           dict(lower=False),dict(upper=True),dict(outside=False)):
                with self.subTest(values=values), self.assertRaises(ValueError):
                    self.query(**values)
        native.assert_not_called()
        cache.assert_not_called()

    def test_invalid_certificates_rejected_before_exact_and_ancestor_lookup(self):
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/'queries.db'
            # This scope's negative is supplied by the stub native oracle.
            with patch('r30_gold.search',return_value={'status':'INFEASIBLE','cp_calls':1}) as native:
                self.assertFalse(self.query(db).get('reused',False))
                ds=[sum(1<<v for v in row) for row in self.problem.domains]
                ds[0] &= -ds[0]
                for domains in (None,ds):
                    self.assertTrue(self.query(db,checked=[[(0,3)]],domains=domains)['reused'])
                    for pair in ((0,0),(0,1),(0,3*self.problem.n),(True,3)):
                        with self.subTest(pair=pair,domains=domains), self.assertRaises(ValueError):
                            self.query(db,checked=[[pair]],domains=domains)
                native.assert_called_once()

    def test_inconsistent_published_model_cannot_reuse_or_seed_original_database(self):
        root=Path(__file__).resolve().parent
        data=next((p for p in (root/'data/tiles.json',root.parent/'data/tiles.json') if p.is_file()),None)
        if data is None:self.skipTest('Original prepared artwork unavailable')
        self.problem=SearchProblem(read_json(data),physical_gold_support=True)
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/'queries.db'
            with patch('r30_gold.search',return_value={'status':'INFEASIBLE','cp_calls':1}) as native:
                self.query(db,upper=2)
                self.assertTrue(self.query(db,upper=2)['reused'])
                self.problem.masks[0,0] ^= 1
                for upper in (2,1):
                    with self.subTest(upper=upper), self.assertRaisesRegex(ValueError,'authenticated'):
                        self.query(db,upper=upper)
                native.assert_called_once()
