"""Geographic signed-anchor deductions must preserve every planted Gold and its rotations."""
from importlib.util import find_spec
from itertools import permutations, product
import unittest

from bench_r35_mixed_sewings import ring_donors
from r27_structure_filters import domain_bits
from r30_gold import search
from r35_phase_anchors import allowed_mates,narrow,_rotate
from r35_mixed_sewings import geographic_roots
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face


class PhaseAnchorTests(unittest.TestCase):
    def test_all_two_tile_boards_domains_and_signs(self):
        p=SearchProblem(simple_tiles(),double_face())
        full=domain_bits(p)
        for signs in product((0,1),repeat=p.n):
            pins=dict(enumerate(signs));allowed=allowed_mates(p,pins)
            for ds in (full,(full[0]&19,full[1]),(1<<2,full[1])):
                expected=[]
                for order in permutations(range(p.n)):
                    for rots in product(range(3),repeat=p.n):
                        codes=[3*t+r for t,r in zip(order,rots)]
                        if not all(d&(1<<c) for d,c in zip(ds,codes)):continue
                        for a,s,b,t in p.edges:
                            left=3*(codes[a]//3)+(s-codes[a]%3)%3
                            right=3*(codes[b]//3)+(t-codes[b]%3)%3
                            if tuple(sorted((left,right))) not in allowed:break
                        else:expected.append(codes)
                for unique,project in product((False,True),repeat=2):
                    out=narrow(p,ds,pins,unique=unique,project=project)
                    if expected:self.assertEqual(out['status'],'SURVIVING')
                    for board in expected:
                        self.assertTrue(all(d&(1<<c) for d,c in zip(out['domains'],board)))

    def test_planted_boards_partial_complement_and_symmetric_splits(self):
        for seed in (103,817):
            p,parents,boards=ring_donors(seed)
            for donor,board in zip(parents,boards):
                self.assertTrue(p.validate(board)['valid'])
                for pins in (dict(enumerate(donor['phases'])),{i:1^v for i,v in enumerate(donor['phases'])},{1:donor['phases'][1]}):
                    full=domain_bits(p)
                    for ds in (full,tuple(1<<c for c in board)):
                        bare=narrow(p,ds,pins,symmetry=False,anchors=[0,2],repeat=False)
                        quot=narrow(p,ds,pins,symmetry=True,anchors=[0,2],repeat=False)
                        self.assertEqual(bare['domains'],quot['domains'])
                        self.assertEqual(quot['status'],'SURVIVING')
                        self.assertTrue(all(d&(1<<c) for d,c in zip(quot['domains'],board)))
                    expired=narrow(p,full,pins,deadline=0)
                    self.assertEqual(expired['status'],'LIMIT')
                    self.assertEqual(expired['domains'],tuple(full))

    def test_reject_physical_only_mode_and_conflicting_mates(self):
        p,parents,boards=ring_donors(103)
        with self.assertRaisesRegex(ValueError,'embed'):
            search(p,geography=False,phase_anchors='roots')
        for option in ('tour_mode','corner_mode','neighbor_mode','turn_length','project_blanks','mate_correlations'):
            with self.subTest(option=option), self.assertRaises(ValueError):
                search(p,geography=True,phase_anchors='shared',bonds=[(0,3)],forbidden=[(0,3)],**{option:'bad-mode'})
        ds=domain_bits(p)
        self.assertEqual(narrow(p,ds,{},required=[(0,3),(0,6)])['status'],'INFEASIBLE')
        with self.assertRaises(ValueError):narrow(p,ds,{},anchors=[p.n])
        with self.assertRaises(ValueError):geographic_roots(p,ds,set(),anchor=-1)

    def test_crossing_artwork_is_rejected_before_a_terminal_preflight(self):
        p,_,_=ring_donors(103)
        tiles=[{**t,'segments':[list(seg) for seg in t['segments']]} for t in p.tiles]
        tiles[0]['segments']=[[[0,1],[1,1]],[[0,2],[1,2]]]
        bad=SearchProblem(tiles,p.board)
        with self.assertRaisesRegex(ValueError,'Crossing'):
            search(bad,geography=True,phase_anchors='shared',bonds=[(0,3)],forbidden=[(0,3)])

    def test_rotation_transports_every_pose_bit(self):
        for n in range(1,6):
            full=(1<<(3*n))-1
            for turn in range(3):
                for code in range(3*n):
                    self.assertEqual(_rotate(1<<code,turn,full//7,full),1<<(3*(code//3)+(code%3+turn)%3))

    @unittest.skipUnless(find_spec('pysat'),'requires independent SAT control')
    def test_explicit_signed_cnf_preserves_planted_board(self):
        from bench_r35_phase_board import solve_board
        p,donors,boards=ring_donors(103)
        phase=sum(v<<i for i,v in enumerate(donors[0]['phases']))
        answer=solve_board(p,phase,[1<<code for code in boards[0]],3)
        self.assertEqual(answer['status'],'GOLD')
        self.assertEqual(answer['codes'],boards[0])
