"""Independent finite controls for the Euler-bound discriminator and physical fingerprint."""
import itertools
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from importlib.util import find_spec

from bench_r35_loss_euler import euler_clause,minimal_named
from r31_gold_connectivity import PhaseInterface
from r35_loss_witness import physical_price
from verify_r35_loss_euler import reconstruct


class EulerControls(unittest.TestCase):
    def test_raw_physical_corner_quotient_and_every_phase_weight(self):
        from bench_r35_loss_boundary344 import problem_scope,verify_21
        path=next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if path is None:self.skipTest('Original pieces unavailable.')
        p,ref,groups,req=problem_scope(path);rows,allowed,degrees=reconstruct(p);phase=PhaseInterface(p)
        self.assertEqual(degrees,{4:5,5:2,6:75})
        self.assertEqual(allowed,[-5,-3,-1,1,3,5])
        self.assertEqual(rows,[(a[-1],b[-1]) for a,b in phase.rows])
        impossible,_,check=verify_21(p,ref,groups,req)
        positive=int('d237aba7e99101e88041403e8c589902025e5326',16)
        e=lambda z:sum(row[z>>i&1] for i,row in enumerate(rows))
        self.assertEqual((check['cost'],e(impossible),e(impossible^((1<<p.n)-1))),(21,-43,43))
        self.assertEqual((physical_price(p.tiles,ref,groups,req,positive)[0],e(positive)),(22,3))

    @unittest.skipUnless(find_spec('pysat'),'finite SAT counter requires optional Python SAT')
    def test_every_assignment_of_weighted_range_including_holes_and_zero(self):
        from pysat.solvers import Solver
        trials=[([-4,10,2,-8,0],-3,{-9,-5,-3,3,7}),([0,0],2,{1}),([0,0],2,{2}),([8],-4,{4})]
        for weights,base,allowed in trials:
            rows=[((base if i==0 else 0,),((base if i==0 else 0)+w,)) for i,w in enumerate(weights)]
            clauses,_=euler_clause(SimpleNamespace(rows=rows,euler=allowed),len(rows))
            with Solver(name='g4',bootstrap_with=clauses) as s:
                for values in itertools.product((0,1),repeat=len(rows)):
                    expected=base+sum(v*w for v,w in zip(values,weights))
                    pins=[i if v else -i for i,v in enumerate(values,1)]
                    self.assertEqual(s.solve(assumptions=pins),expected in allowed)

    @unittest.skipUnless(find_spec('pysat'),'finite SAT counter requires optional Python SAT')
    def test_minimal_representatives_for_every_overlapping_small_phase(self):
        from pysat.solvers import Solver
        pieces=4;edges=[(0,3),(1,6),(4,9),(7,10),(2,11)]
        sets=[{edges[0],edges[1],edges[2]},{edges[1],edges[3]},{edges[2],edges[3]}]
        loss={e:pieces+i+1 for i,e in enumerate(edges)}
        sides=[(0,(i//2)%2) for i in range(3*pieces)]
        with patch('bench_r35_loss_euler.EndpointCuts',return_value=SimpleNamespace(sides=sides)), \
             patch('bench_r35_loss_euler.raw_sides',return_value=[]),patch('bench_r35_loss_euler.parity_conflict',return_value=False):
            clauses,_=minimal_named(SimpleNamespace(tiles=[]),sets,set(),loss,max(loss.values()))
        with Solver(name='g4',bootstrap_with=clauses) as s:
            for phase in range(1<<pieces):
                pins=[i if phase>>(i-1)&1 else -i for i in range(1,pieces+1)]
                errors={e for e in edges if (phase>>(e[0]//3)^phase>>(e[1]//3)^sides[e[0]][1]^sides[e[1]][1])&1}
                feasible=[];minimal=[]
                for selected in range(1<<len(edges)):
                    lost={e for i,e in enumerate(edges) if selected>>i&1}
                    if not errors<=lost or not all(g&lost for g in sets):continue
                    feasible.append(lost);choices=[v if e in lost else -v for e,v in loss.items()]
                    expected=all(any(g&lost=={e} for g in sets) for e in lost-errors)
                    self.assertEqual(s.solve(assumptions=pins+choices),expected)
                    if expected:minimal.append(lost)
                self.assertTrue(all(any(s<=full for s in minimal) for full in feasible))


if __name__=='__main__':unittest.main()
