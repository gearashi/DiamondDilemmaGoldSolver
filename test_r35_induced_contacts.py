"""Independently exercise the edge-perfect-only projection and large physical covers."""
from itertools import combinations
import json
from pathlib import Path
from random import Random
import unittest

from bench_r35_loss_boundary344 import problem_scope
from bench_r35_induced_contacts import assess,incidence
from r27_structure_fixtures import planted
from r31_geographic import physical_bonds
from r35_contact_certificate import CERTIFICATE,verify
from r35_cover_search import reduced_cover
from r35_induced_contacts import InducedContacts,contact_departures,verified_contacts
from r35_geographic_replay import geographic_departures
from r35_loss_witness import cover_missing


class InferredContactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=next((f for f in (Path('data/tiles.json'),Path('../data/tiles.json')) if f.is_file()),None)
        if path is None:raise unittest.SkipTest('Original physical source unavailable.')
        cls.p,cls.ref,cls.groups,cls.req=problem_scope(path)
        cls.certificate=json.loads(CERTIFICATE.read_bytes())
        cls.contacts=[frozenset(map(tuple,r['bonds'])) for r in cls.certificate['weighted_cores'] if r['kind']=='edge_perfect']

    def test_positive_geometric_poses_and_unavoidable_raw_mismatch(self):
        g=min(self.contacts,key=len)
        check=assess(self.p,incidence(self.p),g)
        self.assertGreater(check['poses'],0);self.assertEqual(check['compatible'],0)
        self.assertEqual(InducedContacts(self.p).count(g),(check['poses'],0))
        self.assertFalse(geographic_departures(self.p,self.ref,self.req,[g]))
        self.assertEqual(contact_departures(self.p,self.ref,self.req,[g]),(g-self.req,))
        short={min(g)}
        self.assertFalse(verified_contacts(self.p,[short,[(0,0)],[],[(99999,0)]]))
        # Every disconnected or non-physical hint remains untrusted.
        pair=next(p for p in self.ref if not {x//3 for x in p}&{x//3 for e in short for x in e})
        self.assertFalse(verified_contacts(self.p,[short|{pair}]))

    def test_positive_planted_placements_and_raw_pose_agreement(self):
        for seed in (35,803):
            p,_=planted(1,2,seed,False,True);board=[3*t for t in range(p.n)]
            self.assertTrue(p.validate(board)['valid'])
            gold=physical_bonds(p,board)
            engine=InducedContacts(p)
            for row in [gold,*[frozenset([edge]) for edge in sorted(gold)[:6]]]:
                geo,matched=engine.count(row);other=assess(p,incidence(p),row)
                self.assertEqual((geo,matched),(other['poses'],other['compatible']))
                self.assertGreater(matched,0);self.assertFalse(verified_contacts(p,[row]))

    def test_bound_requires_integer_weights_matching_reference_and_charged_groups(self):
        d=self.certificate
        self.assertEqual(verify(self.p,self.ref,self.req,self.groups[:13],d),25)
        self.assertFalse(verify(self.p,self.ref-{min(self.ref)},self.req,self.groups[:13],d))
        self.assertFalse(verify(self.p,self.ref,self.req,self.groups[:13],dict(d,weight_sum=d['weight_sum']+1)))
        self.assertFalse(verify(self.p,self.ref,self.req,self.groups[:13],d,charged=self.groups[:13]))
        bad=dict(d,weighted_cores=[dict(d['weighted_cores'][0],weight=True),*d['weighted_cores'][1:]])
        self.assertFalse(verify(self.p,self.ref,self.req,self.groups[:13],bad))

    def test_only_full_embedded_objective_accepts_contact_cuts(self):
        from r30_gold import search
        core=min(self.contacts,key=len)
        test=search(self.p,bonds=core,geography=True,edge_perfect_cores=[core])
        self.assertEqual((test['status'],test['cp_calls']),('INFEASIBLE',0))
        self.assertEqual(test['edge_perfect_cores'],[sorted(core)])
        self.assertNotIn('geographic_cores',test)
        with self.assertRaises(ValueError):search(self.p,bonds=core,edge_perfect_cores=[core])

    def test_dominated_rows_columns_and_full_bounded_cover(self):
        rng=Random(3581)
        for count in (7,70):
            for _ in range(18):
                labels=range(9)
                groups=[frozenset(rng.sample(labels,rng.randint(1,6))) for i in range(count)]
                exact=next(len(s) for n in range(10) for s in combinations(labels,n) if all(set(s)&g for g in groups))
                for cover in (reduced_cover(groups),cover_missing(groups,set(),set())):
                    self.assertEqual(len(cover),exact);self.assertTrue(all(cover&g for g in groups))


if __name__=='__main__':unittest.main()
