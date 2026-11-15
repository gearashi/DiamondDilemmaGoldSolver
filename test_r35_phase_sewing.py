"""Independently validate physical reference costs, whole-loop cuts and actual lifts."""
import json
from importlib.util import find_spec
from pathlib import Path
from unittest import TestCase, skipUnless
from unittest.mock import patch

from r27_structure_fixtures import planted
from r31_geographic import physical_bonds, ribbon_genus
from r32_gold_bonds import inventory
from r35_phase_sewing import search, strand_components
from test_r28_geometry import phases


class PhaseSewingTests(TestCase):
    @skipUnless(find_spec("pysat"), "Optional native SAT is required for pairing decisions.")
    def test_planted_zero_loss_embeds_and_split_board_rejects(self):
        for split in (False,True):
            problem, _ = planted(free=8, seed=61, split=split)
            codes = [3*t for t in range(problem.n)]
            ref = frozenset(physical_bonds(problem,codes))
            phase = sum(value << i for i,value in enumerate(phases(problem)))
            required = inventory(problem).required
            row = search(problem,ref,(),required,phase,0,0,switch_trials=0)
            if split:
                self.assertEqual(row["status"],"INFEASIBLE")
            else:
                self.assertEqual(row["status"],"GOLD")
                self.assertEqual(row["loss"],0)
                self.assertTrue(problem.validate(row["codes"])["valid"])
                self.assertEqual(set(map(tuple,row["pairs"])),set(ref))
            # A group with a compulsory-so-lost bond is impossible at zero.
            if ref - required:
                edge=next(iter(ref-required))
                self.assertEqual(search(problem,ref,({edge},),required,phase,0,1)["status"],"INFEASIBLE")

    def test_no_native_required_for_certified_refutation_or_timeout(self):
        problem,_=planted(seed=19)
        codes=[3*t for t in range(problem.n)]
        ref=frozenset(physical_bonds(problem,codes))
        phase=sum(value << i for i,value in enumerate(phases(problem)))
        req=inventory(problem).required
        with patch("r35_phase_sewing.monotonic",side_effect=[0,2]):
            self.assertEqual(search(problem,ref,(),req,phase,0,1)["status"],"UNKNOWN")
        self.assertEqual(search(problem,ref,[set()],req,phase,0,0)["status"],"INFEASIBLE")

    def test_tracked_reference_portfolios_and_raw_loop_certificates(self):
        from bench_r35_loss_boundary344 import problem_scope
        data=next((p for p in map(Path,("data/tiles.json","../data/tiles.json")) if p.is_file()),None)
        if not data:self.skipTest("Original data unavailable")
        problem,ref,groups,req=problem_scope(data)
        root=Path(__file__).parent/"docs"
        portfolios=json.loads((root/"R35_MATE_PORTFOLIOS.json").read_bytes())
        raw=[frozenset(map(tuple,g))-req for g in json.loads((root/portfolios["geographic_source"]).read_bytes())["geographic_cores"]]
        from r35_loss_witness import raw_sides, reciprocal
        phase=int(portfolios['phase'],16)
        labels=[(slots,color^((phase>>(i//3))&1)) for i,(slots,color) in enumerate(raw_sides(problem.tiles))]
        allowed=inventory(problem).possible
        def full_mesh(mesh):
            self.assertEqual(sorted(side for edge in mesh for side in edge),list(range(3*problem.n)))
            self.assertTrue(mesh<=allowed)
            self.assertTrue(all(labels[b]==reciprocal(labels[a]) for a,b in mesh))
        for case in portfolios["portfolios"]:
            rules=[*groups[:13],*(raw[i] for i in case["rule_indices"])]
            record=case["physical"]
            for tag,cycles in (("first_pairing",record["rounds"][0]["cycles"]),("pairs",[problem.total_segments])):
                mesh=frozenset(map(tuple,record[tag]))
                full_mesh(mesh)
                self.assertTrue(req<=mesh)
                self.assertTrue(all(not group<=mesh for group in rules))
                self.assertEqual(strand_components(problem,mesh),cycles)
                if tag=="pairs":
                    self.assertEqual(len(ref-mesh),record["loss"])
                    self.assertEqual(ribbon_genus(problem,sorted(mesh)),record["genus"])
            for step in record["rounds"]:
                for cut in step["new_loop_cores"]:
                    closed=strand_components(problem,map(tuple,cut),complete=False)
                    self.assertTrue(any(0<size<problem.total_segments for size in closed))

        typed=json.loads((root/portfolios['contact_source']).read_bytes())['edge_perfect_cores']
        for case in portfolios['contact_cases']:
            contact=[frozenset(map(tuple,typed[i]))-req for i in case['contact_indices']]
            rules=[*groups[:13],*raw,*contact]
            public=case['public']
            mesh=frozenset(map(tuple,public['pairs']))
            full_mesh(mesh)
            self.assertTrue(req<=mesh and all(not g<=mesh for g in rules))
            self.assertEqual(len(ref-mesh),public['loss'])
            self.assertEqual(public['loss'],public['certified_loss_lower_bound'])
            self.assertEqual(strand_components(problem,mesh),[problem.total_segments])
            self.assertEqual(ribbon_genus(problem,sorted(mesh)),public['genus'])
            self.assertIsNone(public['codes'])
            self.assertEqual(public['verified_contact_departures'],len(contact))
            for step in public['rounds']:
                for cut in step.get('new_loop_cores',()):
                    self.assertTrue(any(0<size<problem.total_segments for size in strand_components(problem,map(tuple,cut),complete=False)))
