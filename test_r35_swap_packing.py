"""Check physical fixed-code scores and the exact MWIS fallback independently."""
from collections import Counter
from itertools import combinations
from importlib.util import find_spec
from pathlib import Path
import tempfile
from random import Random
from types import SimpleNamespace
import unittest

import numpy as np

from r27_incumbent import BoardModel
from r30_fresh import fresh_start
from r34_swap_packing import pack_swaps
from r35_touch_packing import augment
from search_fixtures import planted_gold
from search_problem import SearchProblem


class PackingTests(unittest.TestCase):
    def model_graph(self, n, edges):
        adj=[[] for _ in range(2*n)]
        for a,b in edges:
            adj[2*a].append(2*b)
            adj[2*b].append(2*a)
        return SimpleNamespace(n=2*n,problem=SimpleNamespace(neighbors=adj)), [(2*i,2*i+1) for i in range(n)]

    def test_odd_cycle_and_articulation_beyond_small_limit(self):
        for kind in ('articulation','odd'):
            n=25
            if kind=='articulation':
                edges=[(0,j) for j in range(1,n)]+[(1,2)]
                weights=[10]+[2]*(n-1)
                expected=2*(n-2)
            else:
                edges=[(i,(i+1)%n) for i in range(n)]
                weights=[10,9]+[1]*(n-3)+[9]
                def path(values):
                    skip,take=0,0
                    for v in values:
                        skip,take=max(skip,take),skip+v
                    return max(skip,take)
                expected=max(path(weights[1:]),weights[0]+path(weights[2:-1]))
            model,pairs=self.model_graph(n,edges)
            raw=pack_swaps(model,pairs,weights)
            stats=Counter()
            actual=pack_swaps(model,pairs,weights,separator_depth=3,stats=stats)
            self.assertLess(sum(weights[i] for i in raw),expected)
            self.assertEqual(sum(weights[i] for i in actual),expected)
            self.assertEqual(stats['certified_components'],1)
            self.assertGreater(stats['articulation_splits' if kind=='articulation' else 'odd_cycle_splits'],0)

    def test_three_mutually_touching_moves_reconstruct_one_gold_loop(self):
        tiles,geometry,reference = planted_gold(3534,1,5)
        model = BoardModel(SearchProblem(tiles,geometry))
        board = np.array(reference)
        pairs = [(0,2),(1,5),(3,6)]
        for a,b in pairs:
            board[a],board[b]=board[b],board[a]
        codes = [(reference[a],reference[b]) for a,b in pairs]
        for a,b in combinations(pairs,2):
            self.assertTrue(any(y in model.problem.neighbors[x] for x in a for y in b))
        weights=[]
        initial = int(model.matches(board).sum())
        for (a,b),(new_a,new_b) in zip(pairs,codes):
            partial=board.copy()
            partial[a],partial[b]=new_a,new_b
            weights.append(int(model.matches(partial).sum())-initial)
        self.assertEqual(weights,[2,1,3])
        result,chosen,_=augment(model,board,pairs,codes,weights,[2],'complete')
        self.assertEqual(set(chosen),{0,1,2})
        self.assertTrue(model.problem.validate(result.tolist())['valid'])
        self.assertEqual(initial+sum(weights),9)
        self.assertEqual(int(model.matches(result).sum()),15)

    def test_tied_nonbipartite_certificate_preserves_released_choices(self):
        n = 25
        model,pairs = self.model_graph(n,[(i,(i+1)%n) for i in range(n)])
        weights = [1]*n
        raw = pack_swaps(model,pairs,weights)
        stats = Counter()
        new = pack_swaps(model,pairs,weights,separator_depth=3,stats=stats)
        self.assertEqual(new,raw)
        self.assertEqual(stats['preserved_ties'],1)

    @unittest.skipUnless(find_spec('ortools'), 'requires OR-Tools')
    def test_repair_settings_bind_resume_and_preserve_normal_entrypoint(self):
        from r31_fresh import Engine
        tiles,geo,_ = planted_gold(854,1,5)
        problem = SearchProblem(tiles,geo)
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp)/'prefix.sqlite'
            for policy in ({'exact_packing':True},{'touch_packing':'complete'},
                           {'separator_packing':True},{'touch_packing':'cp'}):
                engine = Engine(problem,pool=4,**policy)
                self.assertTrue(engine.joint)
                first = engine.run(3540,rounds=2,trajectory_db=db)
                self.assertEqual(first['reused_repair_steps'],0)
                second = engine.run(3540,rounds=3,trajectory_db=db)
                self.assertGreater(second['reused_repair_steps'],0)
                cold = engine.run(3540,rounds=3)
                self.assertEqual(second['codes'],cold['codes'])
                self.assertEqual(second['work'],cold['work'])
            with self.assertRaises(ValueError):
                Engine(problem,touch_packing='unknown')

    def test_complete_small_contact_groups_match_physical_enumeration(self):
        tiles,geo,_=planted_gold(853,1,5)
        model=BoardModel(SearchProblem(tiles,geo))
        rng=np.random.default_rng(853)
        for trial in range(24):
            board=fresh_start(model,rng)
            initial=int(model.matches(board).sum())
            pairs=[]; codes=[]; weights=[]
            all_pairs=list(combinations(range(model.n),2))
            for i in rng.permutation(len(all_pairs)):
                p,q=all_pairs[i]
                candidate=board.copy()
                candidate[[p,q]]=3*(board[[q,p]]//3)+rng.integers(0,3,2)
                weight=int(model.matches(candidate).sum())-initial
                if weight>0:
                    pairs.append((p,q))
                    codes.append(tuple(candidate[[p,q]]))
                    weights.append(weight)
                if len(pairs)>=9:
                    break
            base=[]
            greedy=-1
            for mode in ('neutral','priced','complete'):
                result,chosen,_=augment(model,board,pairs,codes,weights,base,mode)
                score=int(model.problem.validate(model.legal_board(result).tolist())['matched_edges'])
                self.assertEqual(len(set(c for i in chosen for c in pairs[i])),2*len(chosen))
                self.assertGreaterEqual(score,greedy)
                greedy=score
            optimum=initial
            for subset in range(1<<len(pairs)):
                row=board.copy()
                used=set()
                for i,(p,q) in enumerate(pairs):
                    if subset>>i&1:
                        if p in used or q in used:
                            break
                        row[p],row[q]=codes[i]
                        used.update((p,q))
                else:
                    optimum=max(optimum,int(model.matches(row).sum()))
            self.assertEqual(greedy,optimum)
            if trial < 8 and find_spec('ortools'):
                from r35_fixed_swap_cp import exact
                selected,report = exact(model,board,pairs,codes,base,2)
                self.assertTrue(report['optimal'])
                self.assertEqual(report['score'],optimum)
                row=board.copy()
                for selected_id in selected:
                    a,b=pairs[selected_id]
                    row[a],row[b]=codes[selected_id]
                self.assertEqual(int(model.matches(row).sum()),optimum)

    def test_budgeted_separator_never_loses_greedy_on_arbitrary_graph(self):
        rng=Random(3534)
        for n in (12,24,28):
            for _ in range(6):
                weights=sorted([rng.randint(1,12) for _ in range(n)],reverse=True)
                edges=[p for p in combinations(range(n),2) if rng.random()<.22]
                model,pairs=self.model_graph(n,edges)
                old=pack_swaps(model,pairs,weights,exact_limit=5)
                new=pack_swaps(model,pairs,weights,exact_limit=5,separator_depth=3)
                self.assertTrue(all(not(a in new and b in new) for a,b in edges))
                self.assertGreaterEqual(sum(weights[i] for i in new),sum(weights[i] for i in old))


if __name__=='__main__':
    unittest.main()
