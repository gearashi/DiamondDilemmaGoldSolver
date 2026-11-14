from importlib.util import find_spec
from itertools import product
from pathlib import Path
import tempfile
import unittest

import numpy as np

from r27_incumbent import BoardModel
from r27_structure_fixtures import planted
from r31_fresh import Assemblies, physical_bonds
from r32_fresh import fresh, speculate
from r32_fresh_geometry import skeletons, projected, slot_layers
from r32_fresh_phases import fit, lift_cube, phase_cube, SignedRepair
from r32_gold_bonds import inventory


class FreshGoldTests(unittest.TestCase):
    def setUp(self):
        self.p,self.ds=planted(1,4,37)
        self.m=BoardModel(self.p)
        self.gold=[3*t for t in range(self.p.n)]
        self.required=inventory(self.p).required

    def test_contour_exhaustion_repeated_frontier_and_hall(self):
        from r32_fresh_geometry import skeleton_domains
        from r32_fresh_skeleton import parity_seed
        p=self.p
        a=skeletons(p,self.ds,limit=3,max_nodes=1000)
        self.assertTrue(a['complete'])
        self.assertEqual(a['patterns'],[(0,)*p.n])
        self.assertEqual(skeleton_domains(p,self.ds,a['patterns'][0]),self.ds)
        self.assertEqual(skeletons(p,self.ds,merge=False,limit=3)['patterns'],a['patterns'])
        bad=list(self.ds)
        bad[0]=bad[1]=1
        self.assertEqual(skeletons(p,bad)['patterns'],[])
        self.assertFalse(skeletons(p,max_nodes=1)['complete'])
        from r27_structure_fixtures import nested_loops
        latitudes,fixed=nested_loops()
        contour=skeletons(latitudes,fixed,limit=3,max_nodes=1000)
        self.assertTrue(contour['complete'])
        self.assertEqual(contour['patterns'],[(5,5,5,5,5,0,0,0,0,0)])
        self.assertIn(parity_seed(p,17,trials=4)['status'],('PARITY_PLACEMENT','UNKNOWN'))

    def test_all_reciprocal_charge_bits(self):
        from validator import reverse_mask
        for m in range(2048):
            self.assertEqual(slot_layers(m),slot_layers(reverse_mask(m)))
            self.assertEqual(sum(slot_layers(m))%2,m.bit_count()%2)

    @unittest.skipUnless(find_spec('ortools'), 'Full CPU profile')
    def test_slot_defects_count_seams_once(self):
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        tiles,mesh,solution=planted_gold(51,1,4)
        p=SearchProblem(tiles,mesh)
        board=list(solution)
        board[0],board[1]=board[1],board[0]
        ds=tuple(1<<v for v in board)
        # Pick endpoints outside and across the region. Independently count projected misses.
        differences=[sum(a!=b for a,b in zip(slot_layers(p.masks[board[i],s]),
                                                slot_layers(p.masks[board[j],t]))) for i,s,j,t in p.edges]
        budget=sum(bool(x) for x in differences)
        target=len(p.edges)-budget
        good=projected(p,ds,layers=tuple(range(6)),regions=((0,1,2),),target=target,effort=.3)
        self.assertEqual(good['projected_defects'],budget)
        if target < len(p.edges):
            self.assertEqual(projected(p,ds,layers=tuple(range(6)),regions=((0,1,2),),
                                       target=target+1,effort=.3)['status'],'INFEASIBLE')
        # Different projections can count the same mismatched seam.
        self.assertGreater(sum(differences),budget)

    @unittest.skipUnless(find_spec('ortools'), 'Full CPU profile')
    def test_cube_partition_and_checked_lifts(self):
        fixed=tuple(1<<c for c in self.gold)
        self.assertEqual([tuple(sorted(phase_cube(self.p,2,i).values())) for i in range(4)],
                         [(0,0,0),(0,0,1),(0,0,1),(0,1,1)])
        with tempfile.TemporaryDirectory() as d:
            db=Path(d)/'lifts.sqlite'
            results=[lift_cube(self.p,self.required,phase_cube(self.p,2,i),fixed,1,db) for i in range(4)]
            self.assertIn('GOLD',[r['status'] for r in results])
            rejected=next(i for i,r in enumerate(results) if r['status']=='INFEASIBLE')
            self.assertEqual(lift_cube(self.p,self.required,phase_cube(self.p,2,rejected),fixed,1,db)['lift_calls'],0)

    @unittest.skipUnless(find_spec('ortools'), 'Full CPU profile')
    def test_signed_caps_and_reversible_part_packing(self):
        phases=fit(self.p,self.required,{0:0},physical_bonds(self.m,self.m.legal_board(self.gold)),
                   effort=.5)
        self.assertEqual(phases['kept'],len(self.p.edges))
        self.assertEqual(phases['ceiling'],len(self.p.edges))
        rng=np.random.default_rng(29)
        assembly=Assemblies(self.m)
        board,info=speculate(self.m,assembly,np.asarray(self.gold),phases['phases'],rng,merges=3)
        self.assertTrue(assembly.bonds<=physical_bonds(self.m,board))
        self.assertFalse(assembly.forbidden & physical_bonds(self.m,board))
        self.assertLessEqual(info['retained'],3)
        row=fresh(self.p,3,rounds=6)
        self.assertTrue(self.required<=physical_bonds(self.m,self.m.legal_board(row['codes'])))
        self.assertEqual(row['gold'],self.p.validate(row['codes'])['valid'])

    @unittest.skipUnless(find_spec('ortools'), 'Full CPU profile')
    def test_cache_keys_and_direct_block_optimum(self):
        from itertools import permutations
        phase=fit(self.p,self.required,{0:0})['phases']
        board=np.array(self.gold)
        cells=[0,next(c for c in range(1,self.p.n) if c not in self.p.neighbors[0])]
        tiles=board[cells]//3
        board[cells]=board[list(reversed(cells))]
        with tempfile.TemporaryDirectory() as d:
            db=Path(d)/'repairs.sqlite'
            a=SignedRepair(self.m,phase,database=db)
            chosen=a.block(board,cells)
            max_score=-1
            for donors in permutations(tiles):
                for rotations in product(range(3),repeat=2):
                    b=board.copy()
                    b[cells]=3*np.array(donors)+rotations
                    max_score=max(max_score,a.score(b)[0])
            self.assertEqual(a.score(chosen)[0],max_score)
            a.stats.clear()
            self.assertEqual(a.block(board,cells).tolist(),chosen.tolist())
            self.assertEqual(a.stats['rows_computed'],0)
            self.assertEqual(a.stats['row_hits'],len(cells))
            b=SignedRepair(self.m,[1-v for v in phase],database=db)
            b.block(board,cells)
            self.assertEqual(b.stats['row_hits'],0)


if __name__=='__main__':
    unittest.main()
