"""Check source-level VIPR axioms, compact proof binding, and finite signed MIP outcomes."""
import gzip
import json
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from r35_contact_certificate import CERTIFICATE
from r35_loss_witness import physical_price
from verify_r35_contact_vipr import replay_header
from verify_r35_contact_wide import read_rows


class SignedModelTests(unittest.TestCase):
    @unittest.skipUnless(find_spec('ortools'), 'Optional native integer solver')
    def test_all_signed_assignments_bound_every_small_solver_result(self):
        from r35_contact_mip import decide
        from test_r33_gold_refine import fixture
        for split, uniform in ((True,True), (False,True), (False,False)):
            p, ref, groups, req, outside = fixture(split,uniform)
            extra = [frozenset(outside[:2])] if split else []
            groups = [*groups,*extra]
            costs = [ans[0] for phase in range(1<<p.n)
                     if (ans:=physical_price(p.tiles,ref,groups,req,phase)) is not None]
            self.assertTrue(costs)
            low = min(costs)
            for cap in range(max(low-1,0),low+2):
                record=decide(p,ref,groups,req,outside,cap,0)
                feasible=record['status']=='INTEGER_RELAXATION_FEASIBLE'
                self.assertEqual(feasible,low<=cap)
                if feasible:
                    self.assertEqual(record['loss_cost'],low)
                    self.assertEqual(physical_price(p.tiles,ref,groups,req,int(record['phase'],16))[0],low)
                else:
                    self.assertEqual(record['status'],'INFEASIBLE')

    def test_vipr_axioms_match_raw_endpoints_and_reject_adverse_controls(self):
        from bench_r35_loss_boundary344 import problem_scope
        root=CERTIFICATE.parent
        tile_file=next((p for p in map(Path, ('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if tile_file is None:
            self.skipTest('Original physical source unavailable.')
        p,ref,physical,req=problem_scope(tile_file)
        groups=[*physical[:13],*read_rows(root/'R35_GEOGRAPHIC_RESIDUALS.json','geographic_cores'),
                *(g for g in read_rows(root/'R35_EDGE_CONTACTS.json','edge_perfect_cores') if len(g)<=13)]
        cycles=read_rows(root/'R35_CONTACT_SIGNED_CYCLES.json','required_xor')
        original=gzip.decompress((root/'R35_CONTACT_VIPR_AXIOMS.txt.gz').read_bytes())
        with TemporaryDirectory() as tmp:
            header=Path(tmp)/'source.vipr'
            header.write_bytes(original)
            answer=replay_header(header,p,ref,req,groups,cycles,30)
            self.assertEqual((answer['variables'],answer['starting_axioms'],answer['bounds']),(372,1827,744))
            with self.assertRaises(ValueError):
                replay_header(header,p,ref,req,groups,cycles,29)
            with self.assertRaises(ValueError):
                replay_header(header,p,ref,req,groups,cycles[:-1],30)
            for before,after in ((b'RTP infeas',b'RTP range'),(b'OBJ min',b'OBJ max'),
                                 (b'C0 E 0 1 0 1',b'C0 E 1 1 0 1')):
                damaged=original.replace(before,after,1)
                self.assertNotEqual(damaged,original)
                header.write_bytes(damaged)
                with self.assertRaises(ValueError):
                    replay_header(header,p,ref,req,groups,cycles,30)


if __name__=='__main__':
    unittest.main()
