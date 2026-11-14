"""Verify geographic bounds and topology rules against independent small searches."""
import itertools
import unittest
from importlib.util import find_spec
from random import Random

from r27_structure_filters import bits
from r27_structure_fixtures import independent, nested_loops, planted
from r28_geo_color import dual_cuts
from r29_gold_cut_count import cut_witness
from r29_gold_geography import layer_bound, partition_capacity
from r29_gold_phase import EndpointCuts, PhaseBasis, SignedParity, propagate_rules
from test_r28_geometry import phases, seams


class GoldProofTests(unittest.TestCase):
    def test_rollback_explanation_and_projection(self):
        rng = Random(128)
        for trial in range(36):
            basis = PhaseBasis()
            rows = [(rng.randrange(1,64),rng.randrange(2)) for _ in range(5)]
            for i,(row,value) in enumerate(rows):
                basis.add(row,value,[i])
            if basis.conflict is not None:
                mask = rhs = 0
                for i in basis.conflict:
                    mask ^= rows[i][0]
                    rhs ^= rows[i][1]
                self.assertEqual((mask,rhs),(0,1))
                continue
            allowed = {v & 13 for v in range(64) if all(((v&r).bit_count() & 1)==s for r,s in rows)}
            complete = [v for v in range(64) if all(((v&r).bit_count() & 1)==s for r,s in rows)]
            forced = {t:(complete[0]>>t)&1 for t in range(6)
                      if len({v>>t & 1 for v in complete})==1}
            self.assertEqual(basis.forced(),forced)
            projected = basis.project(13)
            self.assertEqual(allowed,{v for v in range(14) if not v & ~13 and
                all(((v&r).bit_count() & 1)==s for r,s,_ in projected)})
            mark = len(basis.trail)
            basis.add(0,1,[51])
            self.assertEqual(basis.project(13)[0][:2],(0,1))
            basis.rollback(mark)
            self.assertEqual(basis.project(13),projected)
            self.assertEqual(basis.projection_hits,1)

    def test_partial_endpoint_cuts_and_phase_flip(self):
        for seed in (11,23,102):
            p,ds = planted(1,2,seed)
            cut = EndpointCuts(p)
            rules,info = cut.clauses(width=3,max_subsets=3000)
            for flip in (0,1):
                values = {t:v^flip for t,v in enumerate(phases(p))}
                self.assertEqual(propagate_rules(rules,values),values)
                self.assertTrue(cut.possible_tour(values,seams(p)))
                parity = SignedParity(p,cut.sides)
                for a,b in seams(p):
                    self.assertTrue(parity.bond(a,b))
                for t,v in values.items():
                    self.assertTrue(parity.pin(t,v))

    def test_partial_capacity_without_a_candidate(self):
        p,_ = planted(1,0,11,True)
        ds = [1,1065048,64,512,4096,32768,266240,2097152,16777216,134217728]
        self.assertIsNone(partition_capacity(p,ds,bags=[[]])["feasible"])
        proof = partition_capacity(p,ds,bags=[[6]])
        self.assertFalse(proof["feasible"])
        self.assertEqual((proof["required"], proof["capacity"]),(2,1))
        self.assertFalse(independent(p,ds))
        if find_spec("ortools"):
            self.assertEqual(dual_cuts(p,ds)["boards"],0)
        p,ds = planted(1,3)
        both = phases(p)
        for flip in (0,1):
            self.assertIsNone(partition_capacity(p,ds,pins={t:i^flip for t,i in enumerate(both)})["feasible"])

    @unittest.skipUnless(find_spec("ortools"),"OR-Tools optional")
    def test_refinements_bound_every_soft_board(self):
        rng = Random(555)
        for seed in range(4):
            p,ds = planted(1,0,seed)
            ds = list(ds)
            cells = rng.sample(range(p.n),3)
            for c in cells:
                choices = [3*t+r for t in cells for r in range(3)]
                ds[c] = sum(1<<v for v in rng.sample(choices,4))
            board = [3*t for t in range(p.n)]
            scores = []
            for row in itertools.product(*(bits(ds[c]) for c in cells)):
                if len({v//3 for v in row}) != 3:
                    continue
                for c,v in zip(cells,row):
                    board[c] = v
                scores.append(p.validate(board)["matched_edges"])
            for detail in ("rings","landmark","sectors","vertices"):
                answer = layer_bound(p,ds,detail=detail)
                if scores:
                    self.assertGreaterEqual(answer["upper"],max(scores))
                fixed = [1<<(3*c) for c in range(p.n)]
                self.assertEqual(layer_bound(p,fixed,gold=True,detail=detail)["status"],"OPTIMAL")
        self.assertIsNone(layer_bound(p,ds,max_atoms=1)["upper"])

    def test_cut_count_independent_validator_and_unknown(self):
        for split in (True,False):
            p,ds = planted(1,3,23,split)
            boards = independent(p,ds)
            r = cut_witness(p,ds,range(5),seed=77,trials=2)
            if boards:
                self.assertIn(tuple(r["witness"]),boards)
            else:
                self.assertIsNone(r["exists"])
                self.assertEqual(r["reason"],"zero-parity")
        p,ds = nested_loops()
        self.assertIsNone(cut_witness(p,ds,range(5),max_work=1)["exists"])
        with self.assertRaises(ValueError):
            cut_witness(p,ds,[-1])


if __name__ == "__main__":
    unittest.main()
