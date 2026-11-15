"""Exercise the checked sphere proof, its working scope and objective boundaries."""
import json
from pathlib import Path
from unittest import TestCase, SkipTest
from unittest.mock import patch

from bench_r35_loss_boundary344 import problem_scope
from r35_contact_certificate import checked_contact_bound
from r35_sphere_contact_bound import RECEIPT, checked_sphere_bound
from verify_r35_contact_wide import read_rows
from verify_r35_sphere_contact import replay, sphere_axioms


class SphereLossProofTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tiles = next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if cls.tiles is None:
            raise SkipTest('Original prepared tiles unavailable')
        cls.problem, cls.ref, initial, cls.required = problem_scope(cls.tiles)
        cls.physical = initial[:13]
        cls.geo = read_rows(RECEIPT.parent/'R35_GEOGRAPHIC_RESIDUALS.json','geographic_cores')
        cls.contacts = read_rows(RECEIPT.parent/'R35_EDGE_CONTACTS.json','edge_perfect_cores')
        cls.groups = [*cls.physical,*cls.geo,*cls.contacts]
        cls.lower = json.loads(RECEIPT.read_bytes())['certified_lower_bound']

    def test_raw_disk_mesh_and_all_pose_premises_and_attained_upper(self):
        checked = replay(self.tiles,RECEIPT)
        self.assertEqual(checked['projected_groups'],1187)
        self.assertEqual([(cost,euler) for _,cost,euler in checked['donors']],[(33,-5),(33,-5)])
        extra, weights, allowed = sphere_axioms(self.problem)
        self.assertEqual((len(extra),len(weights),allowed), (2,160,[-5,-3,-1,1,3,5]))
        self.assertEqual(self.lower,checked['excluded_cap']+1)

    def test_exact_scope_needs_integer_euler_and_full_physical_charges(self):
        def bound(groups, **flags):
            return checked_contact_bound(self.problem,self.ref,self.required,self.physical,
                                         charged=groups,**flags)
        sphere = dict(signed_parity=True,integer=True,sphere=True)
        self.assertEqual(bound(self.groups,**sphere),self.lower)
        self.assertEqual(bound(self.groups,signed_parity=True,integer=True),31)
        self.assertEqual(bound(self.groups,signed_parity=True,sphere=True),27)
        short = [*self.physical,*self.geo,*(c for c in self.contacts if len(c)<=13)]
        self.assertEqual(bound(short,**sphere),31)
        self.assertEqual(checked_sphere_bound(self.problem,self.ref-{min(self.ref)},self.required,self.physical,None),0)
        wrong = [frozenset() for _ in self.physical]
        self.assertEqual(checked_sphere_bound(self.problem,self.ref,self.required,wrong,None),0)
        with patch('r35_sphere_contact_bound.reconstruct',return_value=([],[-5,5],{})):
            self.assertEqual(checked_sphere_bound(self.problem,self.ref,self.required,self.physical,None),0)

    def test_tampered_or_missing_proof_does_not_change_weaker_search(self):
        raw = json.loads(RECEIPT.read_bytes())
        raw['proof_files']['proof.vipr']['sha256'] = '0'*64
        read = Path.read_bytes
        def replacement(path):
            return json.dumps(raw).encode() if path == RECEIPT else read(path)
        with patch.object(Path,'read_bytes',replacement):
            self.assertEqual(checked_sphere_bound(self.problem,self.ref,self.required,self.physical,None),0)
            self.assertEqual(checked_contact_bound(self.problem,self.ref,self.required,self.physical,
                                                  charged=self.groups,signed_parity=True,integer=True,sphere=True),31)
