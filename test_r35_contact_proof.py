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

    def test_weighted_budget_agrees_with_all_small_loss_sets(self):
        from importlib.util import find_spec
        if find_spec('pysat') is None:
            self.skipTest('requires a SAT solver for exhaustive auxiliary-variable checks')
        from pysat.formula import IDPool
        from pysat.solvers import Solver
        from r35_contact_weight_replay import budget_clauses
        examples=[([({0,1},2),({1,2,3},3),({2,3,4},1)],6),([({0},2),({0,1,2,3},4)],6)]
        for terms,count in examples:
            loss={e:e+1 for e in range(count)}
            maximum=max(sum(w for g,w in terms if e in g) for e in loss)
            for cap in range(count+1):
                ids=IDPool(start_from=count+1)
                clauses=[[loss[e] for e in g] for g,_ in terms]
                clauses.extend(budget_clauses(ids,loss,terms,maximum,cap))
                with Solver(bootstrap_with=clauses) as engine:
                    for mask in range(1<<count):
                        selected={e for e in loss if mask>>e&1}
                        self.assertEqual(engine.solve(assumptions=[v if e in selected else -v for e,v in loss.items()]),
                                         len(selected)<=cap and all(selected&g for g,_ in terms))
