"""Independent controls for geographic and physical single-tour deductions."""
from importlib.util import find_spec
from itertools import combinations, product
import random
import unittest

from r27_structure_filters import domain_bits
from r27_structure_fixtures import independent, nested_loops, planted
from r28_geo_color import signed_phase_model
from r28_geo_sewing import _cycles, PosePacking, side_options
from r30_gold import GoldMatching, search
from r31_geographic import GeographicConflicts, departure_cores, physical_bonds, ribbon_genus, salvage, tighten_bridge
from r31_gold_connectivity import PhaseInterface, connected_colors
from r31_gold_pricing import price_rings
from r31_gold_switch import cover_switches, sewn_switches
from test_r28_geometry import phases, seams
from test_r30_exactgold import bridge_fixture


@unittest.skipUnless(find_spec('ortools'),'Optional OR-Tools')
class GoldExtensionTests(unittest.TestCase):
    def test_partial_poses_and_positive_genus(self):
        p,ds=planted(1,0,23,False,True)
        edges=seams(p)
        cc=GeographicConflicts(p,ds)
        self.assertIsNone(cc.learn(edges))
        self.assertEqual(ribbon_genus(p,edges),0)
        for pair in combinations(edges,2):
            (a,b),(c,d)=pair
            bad=sorted((set(edges)-set(pair))|{(a,c),(b,d)})
            if len({a//3,b//3,c//3,d//3})!=4:continue
            if ribbon_genus(p,bad)>0:break
        else:self.fail('No positive genus trial')
        core=cc.learn(bad)
        self.assertLess(len(core),len(bad))
        self.assertFalse(PosePacking(p,ds).check(core)['feasible'])
        before=cc.queries
        self.assertEqual(cc.learn(bad),core)
        self.assertEqual(before,cc.queries)

    def test_coupled_models_match_independent_oracle(self):
        for p,ds in [planted(1,3,23,False,True),planted(1,3,23,True,True),nested_loops()]:
            valid=bool(independent(p,ds))
            for method in ('parents','flow'):
                out=connected_colors(p,ds,method)
                self.assertIn(out['status'],('INFEASIBLE','OPTIMAL'))
                self.assertEqual(bool(out['boards']),valid)
            q=search(p,domains=ds,channel=True,geography=True,fraction=True,learned_geometry=True,seconds=10)
            self.assertEqual(q['status'],'GOLD' if valid else 'INFEASIBLE')

    def test_fractional_and_two_crossing_on_known_board(self):
        from ortools.sat.python import cp_model
        p,ds=planted(1,0,23,False,True)
        check=GoldMatching(p,ds,None,seams(p))
        for g in ({0},{1},set(range(3))):
            result=tighten_bridge(check,g)
            self.assertIn(result['status'],('CHECKED','NO_TIGHT_BOUND'))
        self.assertEqual(cp_model.CpSolver().solve(check.m),cp_model.OPTIMAL)
        allowed,banned,fixed,pins=bridge_fixture(p)
        self.assertEqual(search(p,pins=pins,bonds=fixed,forbidden=banned,fraction=True,seconds=10)['status'],'INFEASIBLE')

    def test_integer_phase_projection_against_cp(self):
        from ortools.sat.python import cp_model
        p,_=planted()
        interface=PhaseInterface(p)
        for frozen in ({},{0:0},{1:1,3:1,4:0}):
            result=interface.project(frozen,(0,2))
            self.assertEqual(result['status'],'COMPLETE')
            expect=[]
            for mask in range(4):
                pins={0:mask&1,2:mask>>1}
                if any(t in pins and pins[t]!=v for t,v in frozen.items()):continue
                m,_=signed_phase_model(p,pins=pins|frozen)
                if cp_model.CpSolver().solve(m)==cp_model.OPTIMAL:expect.append(mask)
            self.assertEqual(result['patterns'],expect)
            self.assertEqual(interface.project(frozen,(0,2))['transitions'],0)

    def test_pricing_and_salvage(self):
        p,ds=planted(1,2)
        options=side_options(p,ds)
        self.assertEqual(price_rings(p,options)['status'],'FRACTIONAL_COVER')
        self.assertEqual(price_rings(p,options,max_work=1)['status'],'UNKNOWN')
        result=salvage(p,ds,seams(p)[:1],seams(p))
        self.assertEqual(result['retained'],len(seams(p)))
        self.assertTrue(p.validate(result['codes'])['valid'])
        self.assertEqual(departure_cores(p,physical_bonds(p,result['codes']),trials=3)['minimum_breaks'],0)

    def test_geographic_core_recheck_and_disjoint_departure_branches(self):
        from r31_geographic import add_departure_rules
        p,ds=planted(1,0,23,False,True)
        checker=GoldMatching(p,ds,None,seams(p))
        self.assertEqual(add_departure_rules(checker,[seams(p)]),0)
        with self.assertRaises(ValueError):
            search(p,domains=ds,learned_geometry=True,geography=False)
        bad=[(0,1),(3,4)]
        result=departure_cores(p,bad,trials=16)
        self.assertEqual(result['minimum_breaks'],2)
        self.assertEqual(result['lower_bound'],2)
        self.assertEqual(result['branches'][0]['bonds'],[])
        # Guarding the first absent physical bond partitions every potential
        # matching omitting at least one member; each earlier bond is required.
        from itertools import product
        core=[(0,3),(4,8),(12,16)]
        for present in product((0,1),repeat=len(core)):
            homes=[i for i in range(len(core)) if all(present[:i]) and not present[i]]
            self.assertEqual(len(homes),int(not all(present)))
        good=search(p,domains=ds,learned_geometry=True,geography=True,
                    geographic_cores=[seams(p)],channel=True,seconds=3)
        self.assertEqual(good['status'],'GOLD')
        self.assertEqual(good['geographic_cores'],[])

    def test_disconnected_bundled_switches(self):
        rng=random.Random(70)
        for n in (6,9,12):
            for _ in range(10):
                perm=rng.sample(range(n),n)
                starts=rng.sample(range(n),6)
                bundles=[[(starts[0],starts[1]),(starts[2],starts[3])],[(starts[4],starts[5])]]
                hits=[]
                for selection in range(4):
                    v=perm[:]
                    for j,group in enumerate(bundles):
                        if selection>>j&1:
                            for a,b in group:v[a],v[b]=v[b],v[a]
                    if _cycles(v)==1:hits.append(selection)
                self.assertEqual(sorted(cover_switches(perm,bundles)['patterns']),hits)
        p,ds=planted(1,0,23,False,True)
        out=sewn_switches(p,ds,seams(p),phases(p),[])
        self.assertTrue(p.validate(out['witnesses'][0])['valid'])
