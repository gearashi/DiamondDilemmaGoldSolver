"""Check the original conditional pairing costs and physically revalidate cached hints."""
from pathlib import Path
import json
import unittest

from bench_r35_geographic_loss_cores import connected_pose_count
from r31_geographic import ribbon_genus
from r32_gold import reference_groups
from r32_gold_bonds import inventory
from r35_geographic_packing_check import check_packing
from r35_loss_witness import raw_sides, reciprocal
from r35_path_mates import circuit_departures
from r35_phase_sewing import _checked_incumbent, _published_sewings, strand_components
from search_problem import SearchProblem

DOCS = Path(__file__).resolve().parent / "docs"


def document(name):
    return json.loads((DOCS / name).read_bytes())


class MipDonorTests(unittest.TestCase):
    def test_reference_geography_and_checked_donor_choices(self):
        tiles = next((f for f in map(Path,("data/tiles.json","../data/tiles.json")) if f.is_file()),None)
        if tiles is None:
            self.skipTest("Prepare original data to check the geographic donor corpus.")
        p = SearchProblem(json.loads(tiles.read_bytes()),physical_gold_support=True)
        ref, _, original = reference_groups(p, document("R29_INCUMBENT.json")["codes"], [])
        iv = inventory(p)
        req = iv.required
        corpus = document("R35_MIP_DONOR_SEWINGS.json")
        raw_core = frozenset(map(tuple,corpus["geographic_cores"][0]))
        self.assertEqual((len(raw_core),connected_pose_count(p,raw_core)),(29,0))
        self.assertIs(check_packing(p,raw_core,max_steps=20000)["feasible"],False)
        for edge in sorted(raw_core):
            self.assertIs(check_packing(p,raw_core-{edge},max_steps=20000)["feasible"],True)
        def rows(doc,key):
            return [frozenset(map(tuple,g))-req for g in doc[key]]
        geo = rows(document("R35_GEOGRAPHIC_RESIDUALS.json"),"geographic_cores")
        contacts = rows(document("R35_EDGE_CONTACTS.json"),"edge_perfect_cores")
        residual = document("R35_CONTACT_RESIDUALS.json")
        extra = rows(residual,"geographic_cores")+rows(residual,"edge_perfect_cores")
        paths = circuit_departures(p,ref,req,document("R35_PATH_CIRCUITS.json")["gold_path_circuits"])
        self.assertEqual(tuple(map(len,(original,geo,contacts,extra,paths))),(13,194,985,13,19))
        raw = raw_sides(p.tiles)
        for case in corpus["portfolio_checks"]:
            phase = int(case["phase"],16)
            group = [*original,*geo,*contacts]
            if case["add_13"]: group.extend(extra)
            if case["add_29"]: group.append(raw_core-req)
            if case["add_19"]: group.extend(paths)
            for actual in (phase, phase ^ ((1<<p.n)-1)):
                labels = [(slots,color^((actual>>(side//3))&1)) for side,(slots,color) in enumerate(raw)]
                allowed = {e for e in iv.possible if labels[e[0]] == reciprocal(labels[e[1]])}
                valid = [_checked_incumbent(p,hint,allowed,ref,req,group,case["cost"])
                         for hint in _published_sewings(p,actual)]
                mesh = next((m for m in valid if m and len(ref-m)==case["cost"]),None)
                self.assertIsNotNone(mesh)
                self.assertEqual(strand_components(p,mesh),[365])
                self.assertGreater(ribbon_genus(p,sorted(mesh)),0)
                edge = next(iter(mesh-req))
                self.assertIsNone(_checked_incumbent(p,{"pairs":sorted(mesh)}, allowed,ref,req,
                                                     (*group,{edge}),case["cost"]))
        self.assertEqual(list(_published_sewings(p,0)),[])
