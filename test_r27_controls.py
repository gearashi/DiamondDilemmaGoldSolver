import os
from pathlib import Path
import unittest

from r27_diagrams import edges_of, run_controls
from r27_source_sensitivity import modified_tiles
from search_options import normalize_options
from validator import orientation_masks


class ControlTests(unittest.TestCase):
    def test_pairing_only_vs_changed_endpoint(self):
        data={"tiles":[{"id":"a","segments":[[(0,2),(0,6)],[(1,3),(2,4)]]}]}
        paired=modified_tiles(data,0,[[(0,2),(1,3)],[(0,6),(2,4)]])
        shifted=modified_tiles(data,0,[[(0,1),(1,3)],[(0,6),(2,4)]])
        self.assertEqual(orientation_masks(data),orientation_masks(paired))
        self.assertNotEqual(orientation_masks(data),orientation_masks(shifted))
        self.assertEqual(data["tiles"][0]["segments"][0][0],(0,2))

    def test_directed_triangles_and_open_rim(self):
        cells=[((0,0),(2,0),(1,1)),((2,0),(3,1),(1,1))]
        edges,rim=edges_of(cells,"test")
        self.assertEqual(edges,[(0,1,1,2)])
        self.assertEqual(len(rim),4)

    def test_hybrid_option_preserves_default(self):
        from search_options import ROUND27_OPTIONS
        from test_constraint_ui_round25 import Inputs
        parser=Inputs()
        parser.feed((Path(__file__).resolve().parent/"Dashboard.html").read_text(encoding="utf8"))
        self.assertEqual({name[4:] for name in parser.inputs if name.startswith("r27_")},set(ROUND27_OPTIONS))
        for name in ROUND27_OPTIONS:
            self.assertEqual(len(parser.inputs["r27_"+name]),1)
            self.assertFalse(normalize_options()[name])
        self.assertTrue(normalize_options({"incumbent_polish":True})["durable_hybrid"])

    @unittest.skipUnless(os.environ.get("DIAMOND_TEST_WHITE_DIAGRAMS")=="1","Requires the four public diagrams and prepared tile sheets")
    def test_all_published_white_routes(self):
        rows=run_controls(Path(__file__).resolve().parent/"data")
        full=rows.pop("whole_gold_board")
        self.assertEqual(len(full["solsilver1"]["pairs"]),32)
        self.assertEqual([r["segments"] for r in rows.values()],[48,48,102,179])
        self.assertTrue(all(r["line_components"]==1 for r in rows.values()))

    @unittest.skipUnless(os.environ.get("DIAMOND_TEST_CUDA")=="1","Requires CUDA")
    def test_donated_snapshot_and_grouped_dispatch(self):
        from benchmark_r27_cuda import donation,batching
        cold,shared=donation(False),donation(True)
        self.assertEqual(cold["solutions"],shared["solutions"])
        self.assertLess(shared["strand_joins"],cold["strand_joins"])
        self.assertEqual(cold["bytes"],shared["bytes"])
        for cooperative in (True,False):
            self.assertGreater(batching(8,cooperative,64)["steps"],0)

