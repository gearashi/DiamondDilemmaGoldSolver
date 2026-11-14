"""Exact small oracles for the move budget, remote-singleton and rotation bounds."""
from itertools import combinations, permutations, product
import unittest

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r29_exchange_clusters import StarClusters
from r30_bounds import (RotationForest, RotationRegions, ResponseGroups, budget_upper,
    can_hit_cores, cell_mask, defect_cores, defect_packings, family_upper,
    grouped_rotation_bound, pattern_distance_upper, report_family, family_case_upper)
from search_fixtures import planted_gold
from search_problem import SearchProblem


class Bounds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tiles,geo,codes = planted_gold(31,1,1)
        cls.codes = codes
        p = SearchProblem(tiles,geo,{i:codes[i] for i in range(5,len(codes))})
        cls.model = BoardModel(p)
        cls.board = np.array(codes)
        cls.board[:4]=cls.board[[2,3,0,1]]
        cls.bound = ExchangeBound(cls.model,cls.board)
        cls.clusters = StarClusters(cls.bound)
        cls.real = {}
        cls.target = len(p.edges)
        for perm in permutations(map(int,cls.model.free)):
            candidates = []
            for r in product(range(3),repeat=len(perm)):
                row = cls.board.copy()
                row[:5] = [3*cls.bound.tiles[j]+v for j,v in zip(perm,r)]
                candidates.append(row)
            rows = np.array(candidates)
            m = cls.model
            scores = (m.masks[rows[:,m.a],m.sa] == m.reverse[m.masks[rows[:,m.b],m.sb]]).sum(1)
            cls.real[perm] = int(scores.max())

    def test_core_coverage_and_budget(self):
        b = self.bound
        packed = defect_packings(b)
        for target in range(self.target-5,self.target+2):
            cores = defect_cores(b,target)
            for perm,score in self.real.items():
                if score>=target:
                    self.assertTrue(all(core & cell_mask(i for i,j in enumerate(perm) if i!=j) for core in cores))
        for d in (0,1,2,3):
            actual = max(s for p,s in self.real.items() if distance(p)<=d)
            upper = budget_upper(b,packed,d,self.model.free)
            self.assertGreaterEqual(upper,actual)
            cap = pattern_distance_upper(b,self.model.free,d,max_nodes=5)
            self.assertGreaterEqual(cap['upper'],actual)
            exact = pattern_distance_upper(b,self.model.free,d)
            self.assertTrue(exact['complete'])
            self.assertLessEqual(exact['upper'],upper)
            self.assertGreaterEqual(exact['upper'],actual)
        for budget in range(4):
            small = (3,6,12)
            right = any(all(cell_mask(cs)&v for v in small) for cs in combinations(range(4),budget))
            self.assertEqual(can_hit_cores(small,0,15,budget),right)

    def test_physical_family_with_extra_donors_and_cluster_refinement(self):
        c = self.clusters
        for active in [(),(0,), (1,2), (0,1,2)]:
            for extra in (0,1,3):
                for pin in ({},{active[0]:4} if active else {}):
                    upper = family_upper(c,active,extra=extra,fixed_donors=pin)
                    for perm,score in self.real.items():
                        changed = {i for i,j in enumerate(perm) if i!=j}
                        others = changed-set(active)
                        if (not set(active)<=changed or len(others)>extra
                            or any(perm[i]!=j for i,j in pin.items())
                            or any(c.close[i,j] for i in others for j in changed-{i})):
                            continue
                        self.assertGreaterEqual(upper,2*score)
        report = report_family(c,(0,1),self.target+1,2,nodes=100)
        self.assertTrue(report['closed'])
        complete = family_case_upper(c,(0,),[1,2,3],merge_slots=5,max_parts=300)
        values = [score for p,score in self.real.items() if p[0]!=0 and p[4]==4]
        self.assertGreaterEqual(complete['upper_twice'],2*max(values))
        capped = family_case_upper(c,(0,),[1,2,3],merge_slots=0,max_parts=2)
        self.assertGreaterEqual(capped['upper_twice'],2*max(values))

    def test_budget_scopes_share_scores_and_keep_all_witnesses(self):
        from r29_plateau import walk
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from search_fixtures import planted_gold
        tiles,geo,base = planted_gold(31,1,1)
        p = SearchProblem(tiles,geo,{i:base[i] for i in range(6,len(base))})
        start = list(base)
        for i in range(6):
            start[i] = base[(i+1)%6]
        with TemporaryDirectory() as tmp:
            d = Path(tmp)/'dip.sqlite'
            args = dict(floor=0,goal=len(p.edges),max_centers=100,tie_limit=1)
            cold = walk(p,start,':memory:',max_swaps=1,potentials=False,**args)
            warm = walk(p,start,d,max_swaps=1,**args)
            self.assertFalse(cold['reached'])
            self.assertFalse(warm['reached'])
            self.assertLess(warm['solved'],cold['solved'])
            self.assertEqual(warm['potential_closed'],1)
            wider = walk(p,start,d,max_swaps=2,**args)
            check = walk(p,start,':memory:',max_swaps=2,potentials=False,**args)
            self.assertEqual(wider['reached'],check['reached'])
        near = base.copy()
        near[0],near[1] = near[1],near[0]
        self.assertTrue(walk(p,near,':memory:',max_swaps=1,**args)['reached'])

    def test_rotation_regions_can_keep_discounts_with_partial_moves(self):
        r = RotationRegions(self.bound,max_group=5,rings=True)
        t = self.bound.tiles
        forest = RotationForest(self.model)
        for perm,score in list(self.real.items())[::5]:
            moved = [i for i,j in enumerate(perm) if i!=j]
            ids = t.copy()
            ids[:5] = t[list(perm)]
            cold = ExchangeBound(self.model,[3*int(tile)+int(np.flatnonzero(
                self.model.legal[c,3*tile:3*tile+3])[0]) for c,tile in enumerate(ids)])
            self.assertGreaterEqual(cold.total-r.discount(cell_mask(moved)),2*score)
            self.assertGreaterEqual(forest.upper(ids),score)
        self.assertGreater(forest.hits,0)

    def test_whole_four_swap_budget_closes_pending_cursor_once(self):
        from unittest.mock import patch
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from r29_exchange_route import run
        tiles,geo,codes = planted_gold(31,2,1)
        problem = SearchProblem(tiles,geo)
        start = np.random.default_rng(311).permutation(codes)
        target = len(problem.edges)
        args = dict(distance=4,size=7,batch=2,max_pages=1,backend='cpu')
        with TemporaryDirectory() as folder:
            db = Path(folder)/'route.sqlite'
            with patch('r30_bounds.budget_upper',return_value=target):
                pilot = run(problem,start,target,db,**args)
            self.assertFalse(pilot['complete'])
            self.assertEqual(pilot['offset'],2)
            done = run(problem,start,target,db,**args)
            self.assertTrue(done['complete'])
            self.assertLess(done['budget_upper'],target)
            self.assertEqual(done['budget_closed_cursor'],done['pages_end']-2)
            self.assertEqual(done['core_closed'],pilot['core_closed'])
            self.assertEqual(done['considered'],pilot['considered'])
            with patch('r30_bounds.budget_upper',side_effect=AssertionError('replayed closed budget')):
                warm = run(problem,start,target,db,**args)
            self.assertEqual(warm['budget_closed_cursor'],done['budget_closed_cursor'])
            control = run(problem,start,target,':memory:',cluster_screen=False,**args)
            self.assertEqual(control['offset'],2)
            self.assertNotIn('budget_closed_cursor',control)

    def test_route_hook_keeps_synthetic_witness_at_four_swaps(self):
        from r29_exchange_route import run
        tiles,geo,codes=planted_gold(31,1,1)
        p=SearchProblem(tiles,geo,{i:codes[i] for i in range(6,len(codes))})
        start=list(codes)
        for group in ((0,1,2),(3,4,5)):
            for i,c in enumerate(group):
                start[c]=codes[group[(i+1)%len(group)]]
        args=dict(size=6,distance=4,backend='cpu',target=len(p.edges),db=':memory:')
        screened = run(p,start,**args)
        control = run(p,start,cluster_screen=False,**args)
        self.assertGreater(screened['hits'],0)
        self.assertEqual(screened['hits'],control['hits'])
        self.assertTrue(p.validate(screened['best']['codes'])['valid']
                        or p.validate(screened['best']['codes'])['matched_edges']==len(p.edges))

    def test_rotated_family_anchors_respect_center_and_swap_scope(self):
        from unittest.mock import patch
        from r28_exchange import domain_binding
        from r29_exchange_route import physical_hash, legal_cell_rotations
        from r30_bounds import issued_pair_anchors
        tiles,geo,codes = planted_gold(31,1,1)
        model = BoardModel(SearchProblem(tiles,geo))
        b = ExchangeBound(model,codes)
        binding = domain_binding(model)
        key = physical_hash(b.tiles)
        manifest = dict(domain=binding,target=15,centers=[dict(sha256=key,pieces=b.tiles.tolist())])
        limited = {key[:16]:dict(domain=binding,target=15,extra=4,closed_anchors='0-1 2-3')}
        unbounded = {key[:16]:dict(domain=binding,target=15,closed_anchors='0-1')}
        def fake(path):
            return {'R29_EXCHANGE_CENTERS.json':manifest,'R30_BOUNDS_RESULTS.json':limited,
                    'R30_BOUNDS_ALL_SIZES.json':unbounded}[path.name]
        with patch('r28_exchange.read_json',side_effect=fake):
            for perm in legal_cell_rotations(model):
                codes_rotated = np.empty(len(codes),int)
                codes_rotated[list(perm)] = codes
                bound = ExchangeBound(model,codes_rotated)
                candidates = issued_pair_anchors(bound,15,4)
                self.assertEqual(candidates,{tuple(sorted((perm[0],perm[1]))),
                                             tuple(sorted((perm[2],perm[3])))})
                self.assertEqual(issued_pair_anchors(bound,15,7),{tuple(sorted((perm[0],perm[1])))})
                self.assertFalse(issued_pair_anchors(bound,14,4))
            other = list(codes)
            other[0],other[1] = other[1],other[0]
            self.assertFalse(issued_pair_anchors(ExchangeBound(model,other),15,4))

    def test_grouped_response_preserves_every_identity_and_rotation(self):
        b = self.bound
        groups = ResponseGroups(b)
        domains={int(c):[1,2,4] for c in b.model.problem.neighbors[0] if c>=0}
        for choice in (None,domains):
            grouped = groups.groups(0,range(b.n),choice)
            scores,count = grouped_rotation_bound(b,0,range(b.n),neighbor_domains=choice)
            self.assertCountEqual([v for row in grouped.values() for v in row],range(b.n))
            self.assertEqual(count,len(grouped))
            for t in range(b.n):
                rotations = np.flatnonzero(self.model.legal[0,3*t:3*t+3])
                by_rot = []
                for r in rotations:
                    value = 0
                    for side,c in enumerate(b.neighbors[0]):
                        if c==b.n:
                            continue
                        pool = (choice or {}).get(int(c),(b.tiles[c],))
                        other = self.model.problem.sides[0,side]
                        value += int(any(self.model.reverse[b.masks[j,rr,other]]==b.masks[t,r,side]
                                     for j in pool for rr in range(3)))
                    by_rot.append(value)
                self.assertEqual(scores[t],max(by_rot,default=-1))


def distance(perm):
    seen = set()
    d = 0
    for start in range(len(perm)):
        if start in seen:
            continue
        i = start
        while i not in seen:
            seen.add(i)
            i = perm[i]
            if i!=start:
                d += 1
    return d


if __name__=='__main__':
    unittest.main()
