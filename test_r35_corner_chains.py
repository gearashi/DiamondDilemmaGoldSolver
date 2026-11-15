"""Exact small map and independent physical placement checks for vertex-ring propagation."""
from collections import Counter
from importlib.util import find_spec
import random
from types import SimpleNamespace
import unittest

from r27_structure_fixtures import planted
from r28_geo_sewing import ring_census
from r30_gold import search
from r31_geographic import physical_bonds
from r35_corner_chains import CornerChains, degree_inventory, filter_options, pack_lengths, post_positions
from test_r28_geometry import matchings


def direct_rings(n,pairs,degrees):
    nxt = {}
    for a,b in pairs:
        for i,j in ((a,b),(b,a)):
            nxt[i] = (j//3)*3 + (j%3+1)%3
    rings = Counter()
    unseen = set(range(3*n))
    while unseen:
        root=min(unseen)
        members=[]
        v=root
        while v in unseen:
            unseen.remove(v)
            members.append(v)
            v=nxt[v]
        if v!=root or len(set(i//3 for i in members))!=len(members):
            return False
        rings[len(members)]+=1
    return rings==degrees


class CornerTests(unittest.TestCase):
    def test_all_tetrahedral_sewings_and_rollback(self):
        degrees = {3:4}
        state=CornerChains(4,degrees)
        accepted=0
        for pairs in matchings(tuple(range(12))):
            good = direct_rings(4,pairs,degrees)
            result = all(state.push(pair) for pair in pairs)
            self.assertEqual(result, good, pairs)
            accepted+=result
            state.rollback(0)
            self.assertEqual(state.open,[0,12,0,0])
        self.assertGreater(accepted,0)

    def test_size_packing_against_exhaustive_bin_assignment(self):
        rng=random.Random(35)
        for _ in range(300):
            bins = [rng.randint(2,6) for _ in range(rng.randint(1,3))]
            weights=[]
            remaining=sum(bins)
            while remaining:
                size=rng.randint(1,min(remaining,6))
                weights.append(size)
                remaining-=size
            def allocate(i):
                if i==len(weights):
                    return True
                for j,capacity in enumerate(bins):
                    if weights[i]<=capacity:
                        bins[j]-=weights[i]
                        found=allocate(i+1)
                        bins[j]+=weights[i]
                        if found:
                            return True
                return False
            packed=pack_lengths(tuple(weights.count(i) for i in range(7)),
                                tuple(bins.count(i) for i in range(7)))
            self.assertEqual(packed,allocate(0),(weights,bins))
        self.assertIsNone(pack_lengths((0,0,0,2,0),(0,0,0,2,0),0))

    def test_required_map_controls_and_other_mesh_degree(self):
        p,_=planted(1,3,351)
        degrees=degree_inventory(p)
        self.assertEqual(degrees,{4:5,5:2})
        pairs=physical_bonds(p,[3*t for t in range(10)])
        for seed in range(10):
            rng=random.Random(seed)
            order=rng.sample(pairs,len(pairs))
            state=CornerChains(p.n,degrees)
            for i,pair in enumerate(order):
                mark=state.mark()
                self.assertTrue(state.push(pair))
                state.rollback(mark)
                self.assertTrue(state.push(pair))
                if i==len(order)//2:
                    self.assertIsNone(state.conflict(order[i+1:]))
            self.assertEqual(sum(state.slots),0)
        allowed,_,_=filter_options(p,set(pairs),set(pairs))
        self.assertEqual(allowed,set(pairs))

    def test_length_packing_rejects_before_evaluating_thirty_complete_sewings(self):
        from r32_gold_bonds import inventory
        p,_=planted(1,10,35,uniform_ports=True)
        ds=degree_inventory(p)
        selected=[(15,27),(11,12),(6,23),(19,29),(3,26),(18,21),(0,10),(1,5),(7,28),(17,22)]
        simple=CornerChains(p.n,ds,False)
        strong=CornerChains(p.n,ds,True)
        self.assertTrue(all(simple.push(pair) for pair in selected))
        self.assertTrue(all(strong.push(pair) for pair in selected[:-1]))
        self.assertFalse(strong.push(selected[-1]))
        self.assertEqual(strong.reason,'length_packing')
        live=sorted(set(range(3*p.n))-{x for pair in selected for x in pair})
        possible=inventory(p).possible
        completions=[list(pairs) for pairs in matchings(live) if all(pair in possible for pair in pairs)]
        self.assertEqual(len(completions),30)
        self.assertFalse(any(direct_rings(p.n,selected+row,ds) for row in completions))

    @unittest.skipUnless(find_spec('ortools'),'optional CP backend')
    def test_degree_position_model_agrees_with_independent_full_ring_census(self):
        from ortools.sat.python import cp_model
        p,_=planted(1,3,72)
        base=physical_bonds(p,[3*t for t in range(p.n)])
        for trial in range(12):
            rng=random.Random(trial)
            q=set(base)
            for _ in range(3):
                (a,b),(c,d)=rng.sample(base,2)
                if a//3 != d//3 and b//3 != c//3:
                    q.update((tuple(sorted((a,d))),tuple(sorted((b,c)))))
            if trial%2:
                q.discard(rng.choice(base))
            rows=[set() for _ in range(3*p.n)]
            m=cp_model.CpModel()
            edges={pair:m.new_bool_var('mate') for pair in q}
            for a,b in q:
                rows[a].add(b)
                rows[b].add(a)
            for i,row in enumerate(rows):
                m.add_exactly_one([edges[tuple(sorted((i,j)))] for j in row])
            stats=post_positions(SimpleNamespace(problem=p,m=m,mate=edges))
            self.assertTrue(stats['applicable'])
            solver=cp_model.CpSolver()
            solver.parameters.num_search_workers=1
            result=solver.solve(m)
            oracle=ring_census(p,rows,max_work=100000,seconds=5)
            self.assertIn(oracle['status'],('OPTIMAL','INFEASIBLE'))
            self.assertEqual(solver.status_name(result),oracle['status'])

    @unittest.skipUnless(find_spec('ortools'),'optional CP backend')
    def test_complete_geographic_endpoints_and_target_scope(self):
        for split in (False,True):
            p,ds=planted(1,3,35,split)
            for mode in ('chains','packing','positions'):
                out=search(p,domains=ds,geography=True,corner_mode=mode,seconds=5)
                self.assertEqual(out['status'],'INFEASIBLE' if split else 'GOLD')
                if not split:
                    self.assertTrue(p.validate(out['codes'])['valid'])
        with self.assertRaisesRegex(ValueError,'require an embed'):
            search(p,corner_mode='chains')
