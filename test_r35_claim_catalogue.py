"""Adversarial coverage for unconditional original-mesh claim reuse."""
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import r35_claim_catalogue as catalog
from r27_structure_fixtures import planted
from r31_geographic import physical_bonds
from r32_gold_bonds import inventory
from r35_geographic_replay import replay_for
from search_problem import SearchProblem


class ColdCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tiles = next((p for p in (Path("data/tiles.json"),Path("../data/tiles.json")) if p.is_file()),None)
        if tiles is None:
            raise unittest.SkipTest("Original source unavailable")
        cls.data = json.loads(tiles.read_bytes())
        cls.claims = []
        for name,kind in zip(("R35_GEOGRAPHIC_RESIDUALS","R35_EDGE_CONTACTS","R35_PATH_CIRCUITS"),catalog.KINDS):
            cls.claims.append(json.loads((catalog.DOCS/(name+".json")).read_bytes())[kind][:2])

    def setUp(self):
        self.problem = SearchProblem(self.data)
        codes = json.loads((catalog.DOCS/"R29_INCUMBENT.json").read_bytes())["codes"]
        self.ref = frozenset(physical_bonds(self.problem,codes))
        self.req = inventory(self.problem).required

    def replay(self,p,claims,published=True,ref=None,req=None):
        return catalog.public_departures(p,self.ref if ref is None else ref,self.req if req is None else req,
                                         *claims,published=published)

    def test_canonical_reordering_and_scope_projection(self):
        original = self.replay(self.problem,self.claims)
        self.assertEqual(original[3],dict.fromkeys(catalog.KINDS,2))
        self.assertEqual(self.replay(self.problem,self.claims,False)[:3],original[:3])
        def reverse(row):
            return [[b,a] for a,b in reversed(row)]
        ordinary,typed,loops = self.claims
        reverse_loop = [dict(row,path=reverse(row["path"]),loop=reverse(row["loop"]),mate=row["mate"][::-1])
                        for row in reversed(loops)]
        reversed_claims = ([reverse(g) for g in reversed(ordinary)]*2,
                           [reverse(g) for g in reversed(typed)]*2,reverse_loop*2)
        self.assertEqual(self.replay(self.problem,reversed_claims),original)
        group = frozenset(map(tuple,ordinary[0]))
        for ref,req in ((set(),group),(set(),set()),(group-{min(group)},set()),(group-{min(group)},{min(group)})):
            a=self.replay(self.problem,([ordinary[0]],[],[]),True,ref,req)
            b=self.replay(self.problem,([ordinary[0]],[],[]),False,ref,req)
            self.assertEqual(a[:3],b[:3])
        self.assertEqual(self.replay(self.problem,([ordinary[0]],[],[]),True,set(),group)[0],(frozenset(),))

    def test_mixed_and_wrong_type_claims_are_replayed(self):
        ordinary,typed,loops=self.claims
        unproved=[ordinary[0][0]]
        crossed = [ordinary+[unproved],typed+[ordinary[0]],loops+[dict(loops[0],loop=[loops[0]["mate"]])]]
        fast = self.replay(self.problem,crossed)
        self.assertEqual(fast[3],{})
        live=self.replay(SearchProblem(self.data),crossed,False)
        self.assertEqual(fast[:3],live[:3])
        # Wrong proof type cannot grant a geometric impossibility for a placeable contact.
        self.assertEqual(self.replay(SearchProblem(self.data),([typed[0]],[],[]))[0],())
        for hint in ([[True,1]], [[1,3.0]], [[1]],[[1,3],[1,3]]):
            self.assertIsNone(catalog.checked_groups(catalog.KINDS[0],[hint]))

    def test_missing_tampered_sources_and_incomplete_domains_fail_closed(self):
        original = catalog.DOCS
        with TemporaryDirectory() as tmp:
            path=Path(tmp)
            (path/"R35_CLAIM_CATALOGUE.json").write_bytes((original/"R35_CLAIM_CATALOGUE.json").read_bytes())
            document = json.loads((original/"R35_EDGE_CONTACTS.json").read_bytes())
            document["edge_perfect_cores"][0][0] = [0,1]
            (path/"R35_EDGE_CONTACTS.json").write_text(json.dumps(document))
            with patch.object(catalog,"DOCS",path):
                catalog._catalogue.cache_clear()
                try:
                    self.assertIsNone(catalog.checked_groups(catalog.KINDS[1],[self.claims[1][1]]))
                    (path/"R35_CLAIM_CATALOGUE.json").write_text("{}")
                    catalog._catalogue.cache_clear()
                    self.assertIsNone(catalog.checked_groups(catalog.KINDS[0],[self.claims[0][0]]))
                finally:
                    catalog._catalogue.cache_clear()
        self.assertTrue(catalog.published_scope(self.problem))
        self.problem.domains[0]=self.problem.domains[0][:-1]
        self.assertTrue(catalog.published_scope(self.problem))
        self.assertEqual(self.replay(self.problem,([self.claims[0][0]],[],[]))[3],{"geographic_cores":1})
        self.problem.domains[0].append(500)
        self.assertFalse(catalog.published_scope(self.problem))

    def test_mutated_arc_pairing_masks_mesh_and_warm_proofs(self):
        for field in ("pairing","masks","sides","edges","count"):
            p=SearchProblem(deepcopy(self.data))
            if field=="pairing":
                tile=next(t for t in p.tiles if len(t["segments"])>=2)
                arcs=tile["segments"]
                (a,b),(c,d)=arcs[:2]
                tile["segments"]=((a,c),(b,d),*arcs[2:])
            elif field=="masks":
                p.masks[1,0]^=1
            elif field=="sides":
                p.sides[0,0]=(int(p.sides[0,0])+1)%3
            elif field=="edges":
                p.edges=(*p.edges[1:],p.edges[1])
            else:
                p.total_segments-=1
            self.assertFalse(catalog.published_scope(p),field)

    def test_genuine_stronger_scoped_proof_and_fixed_tile_transfer(self):
        from r27_structure_filters import domain_bits
        group=frozenset(map(tuple,self.claims[0][0]))
        edge=min(group)
        left,right=(e//3 for e in edge)
        # Force the ends of this particular retained bond to non-neighbouring cells.
        other=next(i for i in range(1,self.problem.n) if i not in self.problem.neighbors[0])
        isolated=SearchProblem(self.data,fixed={0:3*left,other:3*right})
        ds=domain_bits(isolated)
        self.assertTrue(catalog.published_scope(isolated))
        single=frozenset([edge])
        p=self.problem
        self.assertEqual(replay_for(p).check(ds,[single]),(single,))
        out=catalog.embedding_claims(p,ds,[group],[],[],published=True)
        self.assertEqual((out[0],out[3]),((single,),{}))
        wide=catalog.embedding_claims(p,domain_bits(p),[group],[],[],published=True)
        self.assertEqual((wide[0],wide[3]),((group,),{"geographic_cores":1}))
        cold=catalog.embedding_claims(isolated,ds,[group],[],[],published=True)
        raw=catalog.embedding_claims(isolated,ds,[group],[],[],published=False)
        self.assertEqual(cold[:3],raw[:3])
        self.assertEqual(cold[3],{"geographic_cores":1})

    def test_weighted_certificate_preserves_live_checks_scope_and_independent_mode(self):
        from r32_gold import reference_groups
        from r35_contact_certificate import verify
        proof=json.loads((catalog.DOCS/"R35_CONTACT_LOSS_CERTIFICATE.json").read_bytes())
        codes=json.loads((catalog.DOCS/"R29_INCUMBENT.json").read_bytes())["codes"]
        _,_,physical=reference_groups(self.problem,codes,())
        charged=[frozenset(map(tuple,row["bonds"]))-self.req for row in proof["weighted_cores"]]
        args=self.problem,self.ref,self.req,set(physical),proof
        with patch("r35_contact_certificate.geographic_departures",side_effect=AssertionError("replayed known geometry")), \
             patch("r35_contact_certificate.contact_departures",side_effect=AssertionError("replayed known contact")):
            self.assertEqual(verify(*args,charged=charged,published_claims=True),25)
            self.assertEqual(verify(*args,charged=[],published_claims=True),0)
            self.assertEqual(verify(self.problem,self.ref,frozenset(),set(physical),proof,
                                    charged=charged,published_claims=True),0)
        with patch.object(catalog,"checked_groups",side_effect=AssertionError("used catalogue in direct verifier")):
            self.assertEqual(verify(*args,charged=charged),25)
        bad=deepcopy(proof)
        next(row for row in bad["weighted_cores"] if row["kind"]=="geographic")["kind"]="edge_perfect"
        self.assertEqual(verify(*args[:4],bad,charged=charged,published_claims=True),0)

    def test_planted_positive_and_independent_mode_bypass_catalogue(self):
        p, _ = planted(1,2,355,False,True)
        real = frozenset(physical_bonds(p,[3*i for i in range(p.n)]))
        self.assertFalse(catalog.published_scope(p))
        self.assertEqual(self.replay(p,([list(real)],[],[]),True,real,set())[0],())
        with patch.object(catalog,"_catalogue",side_effect=AssertionError("independent mode touched catalogue")):
            a = self.replay(self.problem,self.claims,False)
        self.assertEqual([len(s) for s in a[:3]],[2,2,2])
        self.assertEqual(a[3],{})
