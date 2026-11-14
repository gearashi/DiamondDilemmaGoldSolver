import copy
from importlib.util import find_spec
from itertools import permutations, product
import unittest

import numpy as np

from r27_incumbent import BoardModel
from r30_fresh import fresh_start
from r31_fresh_ties import optimize_ties
from r31_fresh import Assemblies, Engine, joint_repair, physical_bonds, relink, tie_repair, pack_cycles, simultaneous_swaps
from r28_exchange_patches import AssignmentPrices
from r31_rotation_messages import CyclicRotationOracle
from r28_exchange_rotations import RotationOracle
from search_fixtures import planted_gold
from search_problem import SearchProblem


class FreshPortfolioTests(unittest.TestCase):
    def setUp(self):
        tiles,mesh,gold=planted_gold(51,1,4)
        self.problem=SearchProblem(tiles,mesh)
        self.gold=gold
        self.model=BoardModel(self.problem)
        self.rng=np.random.default_rng(43)
        self.board=fresh_start(self.model,self.rng)

    def test_bond_constructor_and_moves(self):
        assembly=Assemblies(self.model)
        self.assertTrue(assembly.bonds)
        board=assembly.seed(self.board,self.rng)
        self.assertTrue(assembly.bonds<=physical_bonds(self.model,board))
        for _ in range(8):
            board=assembly.relocate(board,self.rng,trials=25)
            self.model.legal_board(board)
            self.assertTrue(assembly.bonds<=physical_bonds(self.model,board))
        fixed=SearchProblem(self.problem.tiles,self.problem.board,dict(enumerate(self.gold)))
        row=Engine(fixed,bonds=True,joint=True).run(0,rounds=2)
        self.assertTrue(row['gold'])
        self.assertEqual(row['work']['repair_steps'],0)

    @unittest.skipUnless(find_spec('ortools'), 'CP-SAT requires the full Python profile')
    def test_joint_optimum_one_adjacent_pair_and_free_support(self):
        cells=[0,1,2,3]
        pairs=[(a,b) for a,_,b,_ in self.problem.edges if a in cells and b in cells]
        for adjacent in (False,True):
            scores=[]
            for order in permutations((self.board[cells]//3).tolist()):
                for rotations in product(range(3),repeat=len(cells)):
                    row=self.board.copy()
                    row[cells]=3*np.array(order)+rotations
                    if not np.all(self.model.legal[cells,row[cells]]):
                        continue
                    changed=set(c for c in cells if self.board[c]!=row[c])
                    if sum(a in changed and b in changed for a,b in pairs)<=int(adjacent):
                        scores.append(int(self.model.matches(row).sum()))
            board,status=joint_repair(self.model,self.board,cells,adjacent=adjacent,effort=10.)
            self.assertEqual(status['status'],'OPTIMAL')
            self.assertEqual(int(self.model.matches(board).sum()),max(scores))
            if not adjacent:
                answer,work=pack_cycles(self.model,self.board,cells,AssignmentPrices(self.model),relaxations=1000)
                self.assertTrue(work['complete'])
                self.assertEqual(int(self.model.matches(answer).sum()),max(scores))
        restricted=SearchProblem(self.problem.tiles,self.problem.board)
        for c,code in enumerate(self.board):
            choices=[int(code),int(self.board[(c+1)%len(self.board)])]
            restricted.domains[c]=choices
        constrained=BoardModel(restricted)
        row,_=simultaneous_swaps(constrained,self.board,constrained.free,AssignmentPrices(constrained))
        constrained.legal_board(row)

    def test_five_donor_cycle_without_any_compatible_pair_swap(self):
        tiles,mesh,solution=planted_gold(55,2,4)
        p=SearchProblem(tiles,mesh)
        m=BoardModel(p)
        target=np.array(solution)
        cells=m.independent_set(target,self.rng)[:5]
        self.assertEqual(len(cells),5)
        perturbed=target.copy()
        perturbed[cells]=np.roll(target[cells],1)
        # Each allowed donor graph consists of identity loops and one directed 5-cycle.
        for c in range(p.n):
            p.domains[c]=sorted({int(perturbed[c]),int(target[c])})
        model=BoardModel(p)
        swaps,_=simultaneous_swaps(model,perturbed,cells,AssignmentPrices(model))
        np.testing.assert_array_equal(swaps,perturbed)
        fixed,work=pack_cycles(model,perturbed,cells,AssignmentPrices(model),relaxations=2)
        self.assertTrue(work['complete'])
        self.assertTrue(p.validate(fixed.tolist())['valid'])

    def test_message_labels_reconstruct_each_tied_optimum(self):
        oracle=CyclicRotationOracle(self.model)
        plain=RotationOracle(self.model)
        for _ in range(6):
            board=fresh_start(self.model,self.rng)
            a,aw=oracle.optimum(board,ties=True,limit=10000)
            b,bw=plain.optimum(board,ties=True,limit=10000)
            self.assertEqual(a,b)
            self.assertEqual(set(map(tuple,aw)),set(map(tuple,bw)))

    def test_ranked_repair_layers_and_relink(self):
        independent=self.model.independent_set(self.board,self.rng)[:2]
        opt,optimal_count=tie_repair(self.model,self.board,independent,cap=500)
        slack,slack_count=tie_repair(self.model,self.board,independent,cap=500,slack=2)
        self.assertGreaterEqual(slack_count,optimal_count)
        self.assertGreaterEqual(int(self.model.matches(slack).sum()),int(self.model.matches(opt).sum())-2)
        rows=relink(self.model,self.board,self.gold,None)
        self.assertGreater(len(rows),0)
        self.assertEqual(len({r.tobytes() for r in rows}),len(rows))
        for row in rows:
            self.model.legal_board(row)

    def test_resume_replays_no_finished_rounds(self):
        engine=Engine(self.problem,slate=2,fork=True,relink=True)
        cold=engine.run(9,rounds=12)
        states=[]
        def interrupt(state):
            states.append(copy.deepcopy(state))
            if state['cursor']==3:
                raise InterruptedError
        with self.assertRaises(InterruptedError):
            engine.run(9,rounds=12,save=interrupt,batch=3)
        resumed=engine.run(9,rounds=12,saved=states[-1])
        self.assertEqual(cold,resumed)

    def test_single_path_extension_reuses_prior_round_cap(self):
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        engine=Engine(self.problem)
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'prefix.sqlite'
            short=engine.run(19,rounds=3,trajectory_db=db,batch=2)
            self.assertFalse(short['gold'])
            with patch.object(engine,'_archive',side_effect=AssertionError('Repeated sampling')):
                extended=engine.run(19,rounds=8,trajectory_db=db,batch=2)
            cold=engine.run(19,rounds=8)
            self.assertEqual(extended['codes'],cold['codes'])
            self.assertEqual(extended['work'],cold['work'])
            self.assertEqual((extended['reused_repair_steps'],extended['new_repair_steps']),(3,5))

    @unittest.skipUnless(find_spec('ortools'), 'Exact path tie queries require the full profile')
    def test_exact_path_tie_optima_match_exhaustive_assignments(self):
        cells=self.model.independent_set(self.board,self.rng)[:2]
        all_rows=[]
        for donor in permutations(self.board[cells]//3):
            for rotations in product(range(3),repeat=len(cells)):
                row=self.board.copy()
                row[cells]=3*np.array(donor)+rotations
                if np.all(self.model.legal[cells,row[cells]]):
                    checked=self.problem.validate(row.tolist())
                    all_rows.append((checked['matched_edges'],checked['line_components']))
        layers=optimize_ties(self.model,self.board,cells,seconds=5)
        for item in layers:
            self.assertTrue(item['closed'])
            expected=min(comps for score,comps in all_rows if score>=item['score_floor'])
            self.assertEqual(item['best_components'],expected)
            self.assertEqual(self.problem.validate(item['codes'])['line_components'],expected)
