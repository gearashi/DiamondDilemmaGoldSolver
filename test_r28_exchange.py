"""Differential checks for shell coverage, exact orientations and neighborhood bounds."""
import importlib.util
import itertools
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, combination_batch, derangement_unrank, derangements, domain_binding, run_shell
from r28_exchange_bounds import partial_bound, patch_dominance
from r28_exchange_cuda import CudaShell, unrank
from r28_exchange_patches import AssignmentPrices, canonical_paths, joint_patch, trade_patches
from r28_exchange_rotations import RotationOracle
from search_fixtures import planted_gold
from search_problem import SearchProblem


def brute(model,codes,movable):
    choices = [np.flatnonzero(model.legal[c,3*(codes[c]//3):3*(codes[c]//3)+3]) for c in movable]
    states = []
    for rotations in itertools.product(*choices):
        row = codes.copy()
        for c,r in zip(movable,rotations):
            row[c] = 3*(codes[c]//3)+r
        states.append((int(model.matches(row).sum()),tuple(row)))
    score = max(s for s,_ in states)
    return score,{r for s,r in states if s==score}


class Exchanges(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data,board,codes = planted_gold(19,1,3)
        cls.problem = SearchProblem(data,board)
        cls.original = np.array(codes)
        cls.model = BoardModel(cls.problem)
        cls.small = BoardModel(SearchProblem(data,board,{i:codes[i] for i in range(6,len(codes))}))

    def test_scope_and_rotate(self):
        board = self.original.copy()
        mask = [0,1,2,3,4,5]
        oracle = RotationOracle(self.small)
        for indices in [(2,1),(0,1,2),(3,4,0)]:
            board[list(indices)] = board[list(indices[1:]+indices[:1])]
            score,winners = brute(self.small,board,mask)
            result,generated = oracle.optimum(board,ties=True,limit=None)
            self.assertEqual((result,{tuple(c) for c in generated}),(score,winners))
        self.assertGreater(oracle.work["avoided"],0)
        self.assertGreater(oracle.work["dominated"],0)

    def test_bounds_and_cuda_unrank(self):
        n = self.problem.n
        priced = ExchangeBound(self.model,self.original)
        for size in (2,3,4,5):
            count = math.comb(n,size)
            all_cells = np.array([unrank(rank,n,size) for rank in range(count)])
            self.assertEqual(set(map(tuple,all_cells)),set(itertools.combinations(range(n),size)))
            for order in derangements(size)[:2]:
                batch = priced.bounds(all_cells,order)
                for cells,value in zip(all_cells,batch):
                    child = priced.witness_seed(cells,order)
                    full = ExchangeBound(self.model,child)
                    self.assertEqual(value,full.total//2)
                    self.assertGreaterEqual(value,self.model.matches(child).sum())

    def test_optional_cuda_against_every_cpu_bound(self):
        try:
            import cupy as cp
            usable = cp.cuda.runtime.getDeviceCount()>0
        except Exception:
            usable = False
        if not usable:
            self.skipTest("CUDA device unavailable")
        n = self.problem.n
        bound = ExchangeBound(self.model,self.original)
        for size in (2,3,4,5,6):
            allowed = CudaShell(bound,size)
            count = math.comb(n,size)
            cells = np.array([unrank(rank,n,size) for rank in range(count)])
            for order in derangements(size)[::max(1,len(derangements(size))//7)]:
                scores = bound.bounds(cells,order)
                for target in range(int(scores.min()),int(scores.max())+2):
                    self.assertEqual(allowed.survivors(0,count,order,target),
                                     cells[scores>=target].tolist())

    def test_assignment_upper_and_patch_dominance(self):
        moving = [0,1,2]
        bound = ExchangeBound(self.small,self.original)
        best = -1
        for order in itertools.permutations(moving):
            seed = bound.witness_seed(moving,order)
            score,_ = brute(self.small,seed,[0,1,2,3,4,5])
            best = max(best,score)
        for weight in [(1,1),(1,2),(2,1),(0,1)]:
            for use_all in (False,True):
                self.assertGreaterEqual(partial_bound(bound,moving,weights=weight,assignment=use_all),best)
        p = (0,1)
        before = self.original[list(p)].tolist()
        after = [(c//3)*3+(c%3+1)%3 for c in before]
        outside = {i:[self.original[i]] for i in range(len(self.original))}
        self.assertTrue(patch_dominance(self.model,outside,p,after,before))

    def test_all_ties_prices_and_trade(self):
        prices = AssignmentPrices(self.model)
        prices.update(self.original)
        cells = self.model.independent_set(self.original,np.random.default_rng(1))[:3]
        out = list(prices.independent(cells))
        self.assertIn(tuple(self.original),set(map(tuple,out)))
        self.assertTrue(all(self.model.matches(b).all() for b in out))
        child = self.original.copy()
        child[0] = (child[0]//3)*3+(child[0]%3+1)%3
        prices.update(child)
        cold = AssignmentPrices(self.model)
        cold.update(child)
        np.testing.assert_array_equal(prices.prices,cold.prices)
        self.assertGreater(prices.rows_reused,0)
        a,_,b,_ = self.problem.edges[0]
        c,_,d,_ = next(row for row in self.problem.edges if set(row[::2]).isdisjoint((a,b)))
        trades = list(trade_patches(self.model,self.original,[{a,b},{c,d}]))
        self.assertGreater(len(trades),0)
        for row in trades:
            self.model.legal_board(row)
            self.assertEqual(set(row[[a,b]]//3),set(self.original[[c,d]]//3))
        self.assertEqual(len(list(canonical_paths([[(0,1),(2,3)],[(3,2),(0,1)]]))),1)

    @unittest.skipUnless(importlib.util.find_spec("ortools"),"optional OR-Tools")
    def test_joint_threshold_against_complete_assignments(self):
        a,b = 0,1
        choices = (3*(self.original[[a,b]]//3)[:,None]+np.arange(3)).ravel().tolist()
        rows = {a:choices,b:choices}
        references = []
        for x,y in itertools.product(rows[a],rows[b]):
            if x//3==y//3:
                continue
            copy = self.original.copy()
            copy[a],copy[b] = x,y
            report = self.problem.validate(copy.tolist())
            references.append((report["matched_edges"],report["line_components"]))
        for score in (13,14,15,16):
            for comp in (1,2):
                status,candidate = joint_patch(self.model,self.original,rows,min_score=score,max_components=comp)
                self.assertEqual(candidate is not None,any(s>=score and c<=comp for s,c in references),status)

    def test_scope_changes_seek_and_dependencies(self):
        original = list(range(10))
        for k in (2,3,4,5,6,7):
            orders = derangements(k)
            for i in (0,len(orders)//2,len(orders)-1):
                self.assertEqual(derangement_unrank(k,i),orders[i])
        for k in (2,3,4):
            for offset in (0,4,8,11):
                got = combination_batch(original,k,offset,10)
                self.assertEqual(got.tolist(),[unrank(i,10,k) for i in range(offset,offset+10)])
        model = BoardModel(self.problem)
        old = domain_binding(model)
        model.legal[1,3] = not model.legal[1,3]
        self.assertNotEqual(old,domain_binding(model))
        paths = [((0,1),(2,3)),((2,3),(0,1))]
        self.assertEqual(len(list(canonical_paths(paths))),1)
        self.assertEqual(len(list(canonical_paths(paths,reads=lambda m:{3}))),2)
        self.assertEqual(len(list(canonical_paths(paths,capped=True))),2)

    def test_two_workers_share_one_shell(self):
        import concurrent.futures
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp)/"shared.db"
            start = self.original.copy()
            start[[0,1]] = start[[1,0]]
            def work():
                return run_shell(self.problem,start,2,15,db,batch=2)
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                answers = list(pool.map(lambda _:work(),range(2)))
            self.assertTrue(all(a["complete"] for a in answers))
            self.assertEqual(answers[0]["tested"],45)
            self.assertEqual(answers[0]["hits"],answers[1]["hits"])

    def test_checkpoint_positive_negative_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp)/"checkpoint.db"
            # Corrupt two physical placements, then recover the planted witness.
            base = self.original.copy()
            base[[0,1]] = base[[1,0]]
            hit = run_shell(self.problem,base,2,15,db,batch=1,stop_on_hit=True)
            self.assertEqual(hit["best"]["score"],15)
            full = run_shell(self.problem,base,2,15,db,batch=5)
            self.assertTrue(full["complete"])
            self.assertEqual(full["tested"],math.comb(self.problem.n,2))
            restored = run_shell(self.problem,base,2,15,db)
            self.assertEqual(restored["tested"],full["tested"])
            self.assertEqual(full["hits"],restored["hits"])


if __name__=="__main__":
    unittest.main()
