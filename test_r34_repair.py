"""Compare typed exchange enumeration and predecessor exclusions with independent finite references."""
import itertools
from importlib.util import find_spec
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r29_exchange_route import swap_orders
from r30_multicenter import cycles, run, SharedSupports
from r34_cycle_orders import CycleOrders, FIVE_FAMILIES
from r34_exchange_undo import CoveredUndo
from search_fixtures import planted_gold
from search_problem import SearchProblem


class FiveSwapTests(unittest.TestCase):
    def test_complete_family_iteration_and_random_access(self):
        for family in FIVE_FAMILIES:
            options = CycleOrders(family)
            expected = [p for p in swap_orders(sum(family),5) if cycles(p)==family]
            self.assertEqual(list(options),expected)
            self.assertEqual(len(options),len(set(expected)))
            for rank,order in enumerate(expected):
                self.assertEqual(options[rank],order)
                self.assertEqual(options.rank(order),rank)
            with self.assertRaises(IndexError):
                options[len(options)]
            with self.assertRaises(ValueError):
                options.rank(range(sum(family)))

    def test_nine_and_ten_mover_ownership_and_reopen(self):
        tiles, geo, codes = planted_gold(29,1,4)
        problem = SearchProblem(tiles,geo)
        model = BoardModel(problem)
        for family in FIVE_FAMILIES[-2:]:
            options = CycleOrders(family)
            cells = list(range(sum(family)))
            left,right = list(codes),list(codes)
            for i,c in enumerate(cells):
                left[c] = codes[options[0][i]]
                right[c] = codes[options[1][i]]
            graph = SharedSupports(model,[left,right],(family,))
            undo = tuple(options[1].index(i) for i in cells)
            self.assertTrue(graph.covered_by_earlier(1,cells,undo))
            with tempfile.TemporaryDirectory() as tmp:
                db = Path(tmp)/'routes.sqlite'
                args = dict(families=(family,),target=len(problem.edges),stop=1,page=1)
                with patch('r30_bounds.budget_upper',wraps=__import__('r30_bounds').budget_upper) as cap:
                    cold = run(problem,[left,right],db,**args)
                    self.assertEqual({call.args[2] for call in cap.call_args_list},{5})
                self.assertEqual(cold['work']['raw_center_endpoints'],2*len(options))
                self.assertEqual(cold['best']['score'],len(problem.edges))
                warm = run(problem,[left,right],db,**args)
                self.assertEqual(warm['new_pages'],0)
                self.assertEqual(warm['best'],cold['best'])

    def test_undo_only_with_a_covered_complement_and_consistent_bound(self):
        tiles,geo,codes = planted_gold(47,2,4)
        model = BoardModel(SearchProblem(tiles,geo))
        bound = ExchangeBound(model,codes)
        available = list(map(int,model.free))
        isolated = []
        for c in available:
            if not any(c in bound.neighbors[d] for d in isolated):
                isolated.append(c)
            if len(isolated)==10:
                break
        self.assertEqual(len(isolated),10)
        group = tuple(zip(isolated[::2],isolated[1::2]))
        missing = CoveredUndo(bound,lambda group: False)
        self.assertFalse(missing.rejects(group))
        seen = []
        # This solved board has strict local preferences in some exterior settings.
        # Use an identical physical-response instance to guarantee neutral exchanges.
        identical = dict(tiles)
        identical['tiles'] = [dict(tiles['tiles'][0],id=str(i)) for i in range(bound.n)]
        equal = BoardModel(SearchProblem(identical,geo))
        covered = CoveredUndo(ExchangeBound(equal,[3*c for c in available]),lambda g: seen.append(g) or True)
        self.assertTrue(covered.rejects(group))
        self.assertEqual(len(seen[0]),4)
        self.assertTrue(all(covered.gain(a,b)==0 for a,b in group))
        covered.covered = lambda g: False
        self.assertFalse(covered.rejects(group))


class QuotaTests(unittest.TestCase):
    def test_exact_scalar_envelope_and_forced_bounds(self):
        from fractions import Fraction
        import numpy as np
        from r34_quota import quota_price
        rng = np.random.default_rng(114)
        for size in range(2,7):
            rows = np.arange(size)
            for _ in range(9):
                weights = rng.integers(-8,11,(size,size))
                allowed = rng.random((size,size)) < .68
                dirty = rng.random((size,size)) < .4
                matchings = [(p,int(weights[rows,p].sum()),int(dirty[rows,p].sum()))
                             for p in itertools.permutations(range(size)) if allowed[rows,p].all()]
                for quota in range(size+1):
                    result = quota_price(weights,allowed,dirty,quota,(2,3))
                    legal = [(p,v) for p,v,d in matchings if d<=quota]
                    if not legal:
                        self.assertIsNone(result)
                        continue
                    lines = {(v,d) for _,v,d in matchings}
                    # Brute intersections for all competing assignments.
                    points = {Fraction(0)}
                    best_by_count = {d:max(v for v,m in lines if m==d) for _,d in lines}
                    for (d,v),(e,w) in itertools.combinations(best_by_count.items(),2):
                        point = Fraction(v-w,d-e)
                        if point >= 0:
                            points.add(point)
                    opt = min(max(v+x*(quota-d) for d,v in best_by_count.items()) for x in points)
                    self.assertEqual(result['upper'],opt.numerator//opt.denominator)
                    for perm,value in legal:
                        self.assertTrue((result['forced'][rows,perm]>=value).all())


class CoverTests(unittest.TestCase):
    def test_subsumption_requires_every_clause_and_both_budgets(self):
        from r32_route_pool import CoverCores
        cache = CoverCores()
        cache.add([((1,),12),((4,5),15)],3,2)
        self.assertTrue(cache.covered([((1,),4),((4,5),0),((5,),6)],2,2))
        self.assertFalse(cache.covered([((1,),4)],2,2))
        self.assertFalse(cache.covered([((1,),13),((4,5),1)],2,2))
        self.assertFalse(cache.covered([((1,),4),((4,5),0)],4,2))
        self.assertFalse(cache.covered([((1,),4),((4,5),0)],2,3))

    @unittest.skipUnless(find_spec('ortools'), 'CP-SAT requires the full Python profile')
    def test_rechecked_cp_with_changed_pool_center_and_threshold(self):
        from r32_route_pool import CoverCores,defect_cover
        tiles,geo,codes = planted_gold(72,1,5)
        model = BoardModel(SearchProblem(tiles,geo))
        altered = list(codes)
        altered[0],altered[1] = altered[1],altered[0]
        boards = [ExchangeBound(model,b) for b in (altered,codes)]
        cache = CoverCores()
        for bound in boards:
            for pool in (range(8),range(6),()):
                for group in (((2,3),),((4,5),),()):
                    for target in (15,14):
                        expected = defect_cover(bound,group,pool,0,target,seconds=3)
                        got = defect_cover(bound,group,pool,0,target,cores=cache,seconds=3)
                        self.assertEqual(got['closed'],expected['closed'])
        self.assertGreater(cache.work['avoided_solves'],0)
        self.assertTrue(cache.cores)


@unittest.skipUnless(find_spec('ortools'), 'CP-SAT requires the full Python profile')
class ContactTests(unittest.TestCase):
    def test_path_and_disjoint_contact_optima(self):
        import numpy as np
        from r31_fresh import joint_repair
        from r30_fresh import fresh_start
        tiles,geo,_ = planted_gold(55,1,4)
        model = BoardModel(SearchProblem(tiles,geo))
        board = fresh_start(model,np.random.default_rng(22))
        cells = (0,1,2,3)
        local = list(cells)
        pairs = [(a,b) for a,_,b,_ in model.problem.edges if a in cells and b in cells]
        variants = ((2,False),(2,True),(-1,False),(-1,True))
        best = dict.fromkeys(variants,0)
        for donors in itertools.permutations(board[local]//3):
            for rotations in itertools.product(range(3),repeat=len(cells)):
                row = board.copy()
                row[local] = 3*np.array(donors)+rotations
                changed = set(c for c in cells if row[c]!=board[c])
                shared = [(a,b) for a,b in pairs if a in changed and b in changed]
                disjoint = len({c for e in shared for c in e})==2*len(shared)
                score = int(model.matches(row).sum())
                for (limit,restricted) in variants:
                    if (limit<0 or len(shared)<=limit) and (not restricted or disjoint):
                        best[limit,restricted] = max(best[limit,restricted],score)
        for limit,restricted in variants:
            row,work = joint_repair(model,board,cells,contacts=limit,disjoint=restricted,effort=10.)
            self.assertEqual(work['status'],'OPTIMAL')
            self.assertEqual(int(model.matches(row).sum()),best[limit,restricted])


class PackingTests(unittest.TestCase):
    @staticmethod
    def instance(n, edges):
        from types import SimpleNamespace
        neighbors = [[] for _ in range(2*n)]
        for a,b in edges:
            neighbors[2*a].append(2*b)
            neighbors[2*b].append(2*a)
        model = SimpleNamespace(n=2*n,problem=SimpleNamespace(neighbors=neighbors))
        return model, [(2*i,2*i+1) for i in range(n)]

    def test_small_components_match_all_subsets(self):
        from r34_swap_packing import pack_swaps
        import random
        rng = random.Random(834)
        for _ in range(12):
            weights = sorted((rng.randrange(1,10) for i in range(12)),reverse=True)
            edges = [e for e in itertools.combinations(range(12),2) if rng.random()<.23]
            model,pairs = self.instance(12,edges)
            chosen = pack_swaps(model,pairs,weights)
            self.assertFalse(any(a in chosen and b in chosen for a,b in edges))
            reference = max(sum(weights[i] for i in range(12) if mask>>i&1) for mask in range(1<<12)
                            if all(not(mask>>a&1 and mask>>b&1) for a,b in edges))
            self.assertEqual(sum(weights[i] for i in chosen),reference)

    @unittest.skipUnless(find_spec('ortools'), 'Native min-cut requires the full Python profile')
    def test_bipartite_cut_replaces_a_heavy_greedy_center(self):
        from r34_swap_packing import pack_swaps
        model,pairs = self.instance(8,[(0,i) for i in range(1,8)])
        self.assertEqual(pack_swaps(model,pairs,[5,4,3,3,2,2,1,1],exact_limit=2),list(range(1,8)))


class SharedSeamTests(unittest.TestCase):
    def test_two_swaps_match_directly_rotated_boards(self):
        from r30_fresh import fresh_start
        from r34_pair_cycles import pair_repair
        import numpy as np
        tiles,geo,_ = planted_gold(53,1,4)
        m = BoardModel(SearchProblem(tiles,geo))
        rng = np.random.default_rng(934)
        for _ in range(5):
            board = fresh_start(m,rng)
            pool = [0,1,2,3]
            base = int(m.matches(board).sum())
            values = {}
            for swap in itertools.combinations(pool,2):
                a,b = swap
                scores=[]
                for r in itertools.product(range(3),repeat=2):
                    row=board.copy()
                    row[[a,b]]=3*(board[[b,a]]//3)+r
                    scores.append(int(m.matches(row).sum())-base)
                values[swap]=max(scores)
            expected=max([0,*values.values()])
            for c in pool[1:]:
                first=(0,c)
                second=tuple(k for k in pool if k not in first)
                if min(values[first],values[second])>0:
                    if all(j not in m.problem.neighbors[i] for i in first for j in second):
                        expected=max(expected,values[first]+values[second])
                    continue
                a,b=second
                for r in itertools.product(range(3),repeat=4):
                    row=board.copy()
                    row[[0,c,a,b]]=3*(board[[c,0,b,a]]//3)+r
                    expected=max(expected,int(m.matches(row).sum())-base)
            found,_=pair_repair(m,board,pool)
            m.legal_board(found)
            self.assertEqual(int(m.matches(found).sum())-base,expected)


class ExteriorReuseTests(unittest.TestCase):
    def test_neighboring_and_evicted_responses_match_witness(self):
        from r32_exterior import CompactExterior,ExteriorCache
        from r28_exchange_rotations import RotationOracle
        tiles,geo,codes=planted_gold(82,1,4)
        m=BoardModel(SearchProblem(tiles,geo))
        b=ExchangeBound(m,codes)
        oracle=RotationOracle(m)
        for cache in (ExteriorCache(b),ExteriorCache(b,max_bytes=800)):
            for cells in ((0,1,2),(0,1,3),(3,6,8),(0,1,2)):
                cold=CompactExterior(b,cells)
                warm=CompactExterior(b,cells,cache=cache)
                self.assertTrue(warm.fixed)
                for order in itertools.permutations(range(3)):
                    score=warm.score(order)
                    self.assertEqual(score,cold.score(order))
                    value,boards=oracle.optimum(b.witness_seed(cells,order))
                    witness=m.legal_board(next(boards))
                    self.assertEqual(score,value)
                    self.assertEqual(score,int(m.matches(witness).sum()))
                self.assertLessEqual(cache.bytes,cache.max_bytes)
        with self.assertRaises(ValueError):
            CompactExterior(ExchangeBound(m,codes),cells,cache=cache)
