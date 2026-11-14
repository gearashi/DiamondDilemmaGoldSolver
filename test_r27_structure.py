"""Independent labelled witnesses, regional multiplicities and conservative fallback checks."""
import importlib.util
import random
import unittest


from r27_structure_cover import cover_page
from r27_structure_filters import Work, _perfect_supports, _regions, filter_domains, separator_ok
from r27_structure_fixtures import independent, nested_loops, planted
from r27_structure_region import _pack, _unpack, regional_count
from r27_structure_sat import sat_boards
from r27_structure_search import enumerate_boards
from test_constraint_sat_domains import fixture


class StructureTests(unittest.TestCase):
    def test_exhaustive_search_and_monotone_filtering(self):
        p,d=planted(1,3)
        p2,d2=fixture()
        for problem,ds in ((p,d),(p2,d2),planted(1,3,19,True)):
            expected=independent(problem,ds)
            for mode in ('cell','seam','seam-table','coarse','cover'):
                r=enumerate_boards(problem,mode,domains=ds,max_work=100000)
                self.assertTrue(r['complete'],mode)
                self.assertEqual(expected,set(r['witnesses']),mode)
                self.assertEqual(len(expected),r['solutions'],mode)
            rng=random.Random(331)
            for i in range(9):
                subset=tuple(x if not i else x & rng.getrandbits(x.bit_length()) or x&-x for x in ds)
                keep={v for v in expected if all(m&(1<<c) for m,c in zip(subset,v))}
                for rule in ('partners','regions','euler','separator','passages'):
                    reduced,_=filter_domains(problem,subset,[rule],max_work=100000)
                    self.assertTrue(all(x&~y==0 for x,y in zip(reduced,subset)),rule)
                    self.assertTrue(all(all(m&(1<<c) for m,c in zip(reduced,v)) for v in keep),rule)

    def test_full_orientation_preserving_physical_assemblies(self):
        for split in (True,False):
            p,d=planted(1,3,split=split)
            expected=independent(p,d)
            for variant in ('glue','embedded','tour','directed-tour'):
                r=enumerate_boards(p,variant,domains=d,max_work=500000)
                self.assertTrue(r['complete'])
                self.assertEqual(set(r['witnesses']),expected)
                self.assertEqual(len(expected),r['solutions'])
        p,d=planted()
        self.assertFalse(enumerate_boards(p,'tour',domains=d,max_work=2)['complete'])

    def test_cover_resume_disjoint_pages_and_binding(self):
        p,d=fixture()
        expected=independent(p,d)
        seen=[]
        cursor=None
        for _ in range(100):
            out=cover_page(p,d,cursor=cursor,node_limit=2,retain=16)
            seen.extend(out['witnesses'])
            if out['complete']:break
            cursor=out['cursor']
        self.assertTrue(out['complete'])
        self.assertEqual(len(seen),len(set(seen)))
        self.assertEqual(set(seen),expected)
        with self.assertRaises(ValueError):cover_page(p,[d[0]&-d[0],*d[1:]],cursor=cursor)

    @unittest.skipUnless(importlib.util.find_spec('pysat'),'optional SAT package missing')
    def test_sat_forms_and_complement_crossing_fallback(self):
        p,d=fixture()
        expected=independent(p,d)
        for mode,m in [('lazy',None),('reach',None),('eliminate',None),('sparse',None),('modular',3),('modular',12)]:
            r=sat_boards(p,d,mode,modulus=m,max_work=2000000,retain=100)
            self.assertTrue(r['complete'],mode)
            self.assertEqual(set(r['witnesses']),expected,mode)
            self.assertEqual(len(expected),r['solutions'],mode)
        with self.assertRaises(ValueError):sat_boards(p,d,'modular',modulus=7)
        self.assertFalse(sat_boards(p,d,'eliminate',max_work=3)['complete'])

    @unittest.skipUnless(importlib.util.find_spec('pysat'),'optional SAT package missing')
    def test_dense_elimination_falls_back_without_losing_boards(self):
        from unittest.mock import patch
        import r27_structure_sat as module
        p,ds=fixture()
        expected=independent(p,ds)
        eliminate=module._eliminate
        with patch.object(module,'_eliminate',lambda f,e,n,cap:eliminate(f,e,n,0)):
            r=sat_boards(p,ds,'sparse',retain=100)
        self.assertEqual(set(r['witnesses']),expected)
        self.assertTrue(r['complete'])
        self.assertEqual(r['work']['dense_elimination_fallback'],1)

    @unittest.skipUnless(importlib.util.find_spec('pysat'),'optional SAT package missing')
    def test_original_moduli_preserve_a_365_segment_witness(self):
        from r27_structure_cycle_fixture import planted_365
        p,d=planted_365()
        for modulus in (5,73,365):
            r=sat_boards(p,d,'modular',modulus=modulus,max_work=1500000)
            self.assertTrue(r['complete'])
            self.assertEqual(r['solutions'],1)

    def test_exact_inventory_regional_join_and_spanning_tree_count(self):
        for p,d in (planted(1,4),planted(1,3,31,True)):
            expected=independent(p,d)
            for cut in ([0],[0,1],range(5)):
                for compact,det in ((False,False),(True,False),(True,True)):
                    r=regional_count(p,cut,d,compact=compact,determinant=det)
                    self.assertTrue(r['complete'])
                    self.assertEqual(r['solutions'],len(expected))
                    if expected:self.assertIn(r['witness'],expected)
        self.assertIsNone(_pack(((0,2),(1,3))))
        a=((1,9),(2,7),(3,4))
        self.assertEqual(set(_unpack(sorted(v for p in a for v in p),_pack(a))),set(a))

    def test_nested_loops_and_gauge(self):
        p,ds=nested_loops()
        validation=p.validate([3*c for c in range(p.n)])
        self.assertEqual(validation['matched_edges'],15)
        self.assertEqual(validation['line_components'],5)
        self.assertFalse(all(filter_domains(p,ds,['regions'])[0]))
        # Alternating annuli plus one disk can have total characteristic one.
        self.assertTrue(all(filter_domains(p,ds,['euler'])[0]))

    def test_lowlink_all_pairs_against_explicit_component_deletions(self):
        from itertools import combinations
        from r27_structure_filters import _components
        rng=random.Random(4949)
        for n in range(2,11):
            for _ in range(25):
                g=[set() for _ in range(n)]
                for a,b in combinations(range(n),2):
                    if rng.randrange(3):
                        g[a].add(b);g[b].add(a)
                expected=all(len(_components(g,removed))<=max(1,k)
                             for k in (0,1,2) for removed in combinations(range(n),k))
                self.assertEqual(separator_ok(g,Work(500000),2),expected)

    def test_adversarial_obstructions_and_limits(self):
        p,d=planted(1,0,split=True)
        for r in ('regions','euler','separator'):
            self.assertFalse(all(filter_domains(p,d,[r],max_work=500000)[0]),r)
        self.assertEqual(d,filter_domains(p,d,['regions'],max_work=1)[0])
        bip=[set(range(2,6)),set(range(2,6)),{0,1},{0,1},{0,1},{0,1}]
        self.assertTrue(separator_ok(bip,Work(10000),1))
        self.assertFalse(separator_ok(bip,Work(10000),2))
        graph=[{1},{0,2},{1,3},{2}]
        self.assertEqual(_perfect_supports(graph,Work(10000)),[{1},{0},{3},{2}])
        self.assertIsNone(_regions((((0,1),(1,2)),((0,3),(1,4)))))


if __name__=='__main__':unittest.main()

