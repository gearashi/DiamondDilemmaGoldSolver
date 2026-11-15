"""Path-loop claims must replay both the geometry and the strict proper loop."""
from pathlib import Path
import unittest
from unittest.mock import patch

from bench_r35_loss_boundary344 import problem_scope
from r28_exchange import read_json
from r30_gold import search
from r32_gold import run_syndrome
from r35_path_mates import circuit_departures, verified_circuits


class PathCircuitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data=next((p for p in (Path("data/tiles.json"),Path("../data/tiles.json")) if p.is_file()),None)
        if data is None:raise unittest.SkipTest("Original source artwork unavailable")
        cls.p,cls.ref,cls.ordinary,cls.req=problem_scope(data)
        cls.hints=read_json(Path(__file__).parent/"docs/R35_PATH_CIRCUITS.json")["gold_path_circuits"]

    def test_untrusted_mate_loop_path_and_reference(self):
        hint=self.hints[0]; group=verified_circuits(self.p,[hint])
        self.assertEqual(len(group),1)
        self.assertEqual(circuit_departures(self.p,self.ref,self.req,[hint]),(group[0]-self.req,))
        self.assertFalse(verified_circuits(self.p,[hint],set()))
        bad=[None,{},dict(hint,path=[]),dict(hint,mate=[-1,2]),dict(hint,loop=[hint["mate"]]),
             dict(hint,path=[[True,5]]),dict(hint,mate=hint["path"][0])]
        self.assertFalse(verified_circuits(self.p,bad))
        self.assertFalse(verified_circuits(self.p,[dict(hint,path=[hint["path"][0]])]))

    def test_repeat_query_reuses_proofs_but_changed_geometry_reopens(self):
        from r35_path_mates import PathMates
        h=self.hints[0]
        self.assertTrue(verified_circuits(self.p,[h]))
        before=self.p._path_circuit_replay
        with patch.object(PathMates,"patterns",side_effect=AssertionError("redundant replay")):
            self.assertTrue(verified_circuits(self.p,[dict(h,path=h["path"][::-1])]))
        saved=self.p.edges
        try:
            self.p.edges=[]
            self.assertFalse(verified_circuits(self.p,[h]))
            self.assertIsNot(before,self.p._path_circuit_replay)
        finally:
            self.p.edges=saved
        self.assertTrue(verified_circuits(self.p,[h]))

    def test_geographic_gold_scope_and_public_zero_search(self):
        h=self.hints[0]
        g=verified_circuits(self.p,[h])[0]
        with self.assertRaisesRegex(ValueError,"require embed"):
            search(self.p,geography=False,bonds=g,gold_path_circuits=[h])
        out=search(self.p,geography=True,bonds=g,gold_path_circuits=[h],seconds=5)
        self.assertEqual((out["status"],out["cp_calls"]),("INFEASIBLE",0))
        self.assertEqual(frozenset(out["gold_path_conflict"]),g)
        with patch("r32_gold._ranked_loss",return_value={}) as run:
            out=run_syndrome(self.p,self.ref,self.ordinary,33,[1],":memory:",1,circuit_hints=[h])
            self.assertEqual(run.call_args.args[-1],(g-self.req,))
            self.assertEqual(out["verified_gold_path_circuits"],1)
