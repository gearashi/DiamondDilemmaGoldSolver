"""Adversarial controls for the independent original-piece proof replay."""
from pathlib import Path
import json
import unittest

from verify_r35_contact_proof import audit


class OriginalProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parent
        candidates=(root/'data/tiles.json',root.parent/'data/tiles.json')
        tiles=next((f for f in candidates if f.is_file()),None)
        if tiles is None:
            raise unittest.SkipTest('requires prepared original pieces')
        cls.tiles=json.loads(tiles.read_bytes())
        cls.codes=json.loads((root/'docs/R29_INCUMBENT.json').read_bytes())['codes']
        cls.proof=json.loads((root/'docs/R35_CONTACT_LOSS_CERTIFICATE.json').read_bytes())

    def test_bound_and_rejected_changes(self):
        proof=self.proof
        self.assertEqual(audit(self.tiles,self.codes,proof)['minimum_lost'],25)
        bad_contact=dict(kind='edge_perfect',weight=1,bonds=[[69,76]])
        for change in ({'fingerprint':'x'}, {'weight_sum':proof['weight_sum']+1},
                       {'maximum':proof['maximum']-1}, {'weighted_cores':[bad_contact]}):
            with self.subTest(fields=tuple(change)), self.assertRaises(ValueError):
                audit(self.tiles,self.codes,{**proof,**change})
        # A single genuine matched reference bond is possible in its original pose.
        bond=next(b for b in proof['reference'] if b not in proof['required'])
        with self.assertRaisesRegex(ValueError,'unproved edge_perfect'):
            audit(self.tiles,self.codes,{**proof,'weighted_cores':[{'kind':'edge_perfect','weight':1,'bonds':[bond]}]})

    def test_regenerated_formula_matches_checked_drat(self):
        from importlib.util import find_spec
        from tempfile import TemporaryDirectory
        if find_spec('pysat') is None:
            self.skipTest('requires python-sat to regenerate DIMACS')
        from verify_r35_contact_parity import write_formula
        root=Path(__file__).resolve().parent
        receipt=json.loads((root/'docs/R35_CONTACT_PARITY_PROOF.json').read_bytes())
        with TemporaryDirectory() as folder:
            path=Path(folder)/'cap25.cnf'
            self.assertEqual(write_formula(path,self.tiles,self.codes,self.proof),receipt['files']['cap25.cnf']['sha256'])
            self.assertEqual(path.stat().st_size,receipt['files']['cap25.cnf']['bytes'])
