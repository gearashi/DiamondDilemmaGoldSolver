"""Independent physical circuit and geographic conjunction regressions."""
from importlib.util import find_spec
from types import SimpleNamespace
import unittest

from r27_structure_fixtures import planted,nested_loops
from r27_structure_filters import domain_bits
from r30_gold import GoldMatching,search
from r35_physical_tour import post_circuit,tour_structure


@unittest.skipUnless(find_spec('ortools'),'requires OR-Tools')
class TourTests(unittest.TestCase):
    @unittest.skipUnless(find_spec('pysat'),'requires PySAT')
    def test_exhaustive_projected_model_sets(self):
        from bench_r35_physical_tour import enumerate_family
        self.assertEqual(enumerate_family(71)['independent_connected'],8)
        self.assertEqual(enumerate_family(157)['independent_connected'],16)

    def test_two_paths_join_twice_and_proper_subtour(self):
        from ortools.sat.python import cp_model
        for n in (2,3):
            m=cp_model.CpModel()
            phases=[m.new_bool_var('phase') for _ in range(2)]
            sel=m.new_bool_var('mate');m.add(sel==1)
            obj=SimpleNamespace(m=m,strands=n,phase=phases,mate={(0,3):sel},
                                inputs=SimpleNamespace(sides=[(3,0)]*6),transitions={(0,3):[(0,1),(0,1)]})
            stats=post_circuit(obj,{(0,3)},'raw')
            self.assertTrue(stats['active'])
            self.assertEqual(cp_model.CpSolver().solve(m)==cp_model.OPTIMAL,n==2)
            count,_,_,closed=tour_structure(obj,{(0,3)},True)
            self.assertEqual(closed,n==3)
            self.assertEqual(count,n-1)

    def test_geographic_objective_and_complement_terminal_controls(self):
        from bench_r35_physical_tour import coupled_region
        p,d=planted(1,3,35)
        positive=search(p,domains=d,geography=True,tour_mode='contract',channel=True)
        self.assertTrue(p.validate(positive['codes'])['valid'])
        for method in ('parents','flow'):
            q,e=nested_loops()
            check=GoldMatching(q,e,None,(),tour_mode='contract')
            self.assertEqual(coupled_region(q,e,method,check)['status'],'INFEASIBLE')
        with self.assertRaises(ValueError):
            search(p,domains=d,matching=False,tour_mode='raw')
        with self.assertRaises(ValueError):
            search(p,domains=d,tour_mode='unknown')
        bad,_=planted(1,3,71)
        bad.tiles[0]['segments']=[[(0,1),(1,1)],[(0,2),(1,2)]]
        with self.assertRaisesRegex(ValueError,'Crossing'):
            GoldMatching(bad,domain_bits(bad),None,(),tour_mode='raw')

    @unittest.skipUnless(find_spec('pysat'),'requires PySAT')
    def test_sat_size_limit_remains_unknown(self):
        from bench_r35_physical_sat import solve_fixed
        from bench_r35_physical_tour import family
        p,_=planted(1,10,71,uniform_ports=True)
        phases,banned,_,_=family(p)
        check=GoldMatching(p,domain_bits(p),dict(enumerate(phases)),(),forbidden=banned,physical_support=False)
        result=solve_fixed(check,phases,(),arc_limit=1)
        self.assertFalse(result['complete'])
        self.assertEqual(result['status'],'SIZE_LIMIT')


if __name__=='__main__':
    unittest.main()
