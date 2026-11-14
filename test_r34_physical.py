"""Physical support and compact geographic channel checks on finite controls."""
from itertools import combinations, product
from importlib.util import find_spec
from random import Random
import unittest
from unittest.mock import patch

from r27_structure_filters import Work, _perfect_supports
from r27_structure_fixtures import independent, planted
from r28_geo_sewing import _pack_footprints
from r31_geographic import physical_bonds
from r32_gold_bonds import _closes_path, _open_paths, endpoints, gold_matching_supports, loop_groups


class GoldPhysicalTests(unittest.TestCase):
    def test_contracted_paths_match_direct_tracing(self):
        for seed in (8, 17, 23, 41):
            p, ds = planted(1, 3, seed, False, True)
            solution = next(iter(independent(p, ds)))
            actual = set(physical_bonds(p, solution))
            arcs = endpoints(p)
            sides = [[k for k in range(1,12) if (d,k) in arcs] for d in range(3*p.n)]
            for count in (0, 3, 9, len(actual)):
                mandatory = Random(seed+count).sample(sorted(actual), count)
                used = {u for pair in mandatory for u in pair}
                paths = _open_paths(arcs, mandatory, sides)
                rows = [set() for _ in sides]
                for a,b in combinations(range(3*p.n),2):
                    if (a,b) in mandatory:
                        rows[a].add(b)
                        rows[b].add(a)
                    elif (a not in used and b not in used and a//3 != b//3
                          and sides[a] == sorted(12-k for k in sides[b])):
                        direct = bool(loop_groups(arcs, sides, [*mandatory, (a,b)], p.total_segments))
                        self.assertEqual(_closes_path(paths,sides,a,b,p.total_segments), direct)
                        rows[a].add(b)
                        rows[b].add(a)
                result = gold_matching_supports(p, rows, Work(200000))
                self.assertIsNotNone(result)
                self.assertTrue(all(b in result[a] for a,b in actual))

    def test_sparse_large_class_matches_unique_perfect_matching(self):
        n = 46
        rows = [set() for _ in range(n)]
        for first in (0, 23):
            for i in range(23):
                a,b = first+i,first+(i+1)%23
                rows[a].add(b)
                rows[b].add(a)
        rows[0].add(23)
        rows[23].add(0)
        required = {(0,23)} | {(offset+i,offset+i+1) for offset in (0,23) for i in range(1,23,2)}
        w=Work(50000)
        out=_perfect_supports(rows,w)
        self.assertEqual({(a,b) for a,r in enumerate(out) for b in r if a<b},required)
        self.assertEqual(w.counters['large_partner_classes_checked'],1)

    def test_conditioned_contours_keep_required_count_and_apices(self):
        import numpy as np
        from geometry import build_board
        from r32_fresh_skeleton import _condition_count
        vertices=np.array([c.vertices for c in build_board(1).cells])
        width=int(vertices.max())+1
        movable=list(range(1,width-1))
        incident={}
        neighbors={}
        for v in movable:
            cells,j=np.where(vertices==v)
            left,right=vertices[cells,(j+1)%3],vertices[cells,(j+2)%3]
            incident[v]=cells,left,right
            neighbors[v]=set(left)|set(right)
        rng=np.random.default_rng(41)
        for _ in range(15):
            initial=rng.integers(0,2,width,dtype=np.int8)
            for target in range(11):
                colors=initial.copy()
                seed=int(rng.integers(100000))
                order=np.random.default_rng(seed).permutation(movable)
                free=[];blocked=set()
                for v in order:
                    if v not in blocked:
                        free.append(v)
                        blocked.update(neighbors[v])
                def active(row):
                    f=row[vertices]
                    return int(np.any(f!=f[:,:1],axis=1).sum())
                counts=set()
                for values in product((0,1),repeat=len(free)):
                    row=initial.copy()
                    row[free]=values
                    counts.add(active(row))
                found=_condition_count(vertices,incident,neighbors,movable,colors,target,np.random.default_rng(seed))
                self.assertEqual(found,target in counts)
                if found:
                    self.assertEqual(active(colors),target)
                    self.assertEqual(colors[[0,-1]].tolist(),initial[[0,-1]].tolist())

    def test_early_hall_skeletons_preserve_every_completion(self):
        from geometry import build_board
        from r32_fresh_geometry import skeletons
        from search_problem import SearchProblem
        p=SearchProblem([{'id':t,'segments':[((0,3),(1,9))] if t<6 else [((0,3),(0,9))]}
                         for t in range(10)],build_board(1))
        old_states = trial_states = survivors = 0
        for seed in range(34,84):
            rng=Random(seed)
            ds=tuple(sum(1<<v for v in rng.sample(range(30),rng.randrange(3,11))) for _ in range(p.n))
            results=[skeletons(p,ds,max_nodes=20000,limit=3000,early_hall=on) for on in (False,True)]
            old,trial=results
            self.assertTrue(all(r['complete'] for r in results))
            self.assertEqual(old['patterns'],trial['patterns'])
            old_states += old['work']['states']
            trial_states += trial['work'].get('states',0)
            survivors += len(trial['patterns'])
        self.assertLess(trial_states,old_states)
        self.assertGreater(survivors,0)

    def test_packing_matches_all_small_assignments(self):
        rng = Random(34)
        for _ in range(80):
            groups = [sorted({sum(1<<c for c in rng.sample(range(7),rng.randrange(1,3)))
                              for _ in range(rng.randrange(1,5))}) for _ in range(rng.randrange(1,6))]
            expected = any(all(not (x & y) for x,y in combinations(chosen,2)) for chosen in product(*groups))
            self.assertEqual(_pack_footprints(groups,5000)[0],expected)

    @unittest.skipUnless(find_spec('ortools'),'Optional OR-Tools')
    def test_large_odd_set_has_valid_matching_crossing(self):
        from r31_gold_pricing import _odd_mate_sets
        sides = list(range(18))
        values = {tuple(sorted((offset+i,offset+(i+1)%9))):.5 for offset in (0,9) for i in range(9)}
        values[0,9] = 0.
        crossing = lambda p,s: (p[0] in s) != (p[1] in s)
        found = [part for part in _odd_mate_sets(sides,values,100000)
                 if sum(v for p,v in values.items() if crossing(p,part))<1]
        self.assertTrue(found)
        exact = {(0,9)} | {(offset+i,offset+i+1) for offset in (0,9) for i in range(1,9,2)}
        for part in found:
            self.assertEqual(len(part)%2,1)
            self.assertGreaterEqual(sum(crossing(p,part) for p in exact),1)

    @unittest.skipUnless(find_spec('ortools'),'Optional OR-Tools')
    def test_element_channel_matches_checked_gold_placements(self):
        from r30_gold import search
        from r31_geographic import _partner_channel
        for seed in (11, 23):
            for split in (False,True):
                p,ds = planted(1,3,seed,split,True)
                expected=independent(p,ds)
                with patch('r31_geographic.channel_geographic',_partner_channel):
                    out=search(p,domains=ds,channel=True,geography=True,seconds=4)
                self.assertEqual(out['channel']['encoding'],'partner-elements')
                self.assertEqual(out['status'],'GOLD' if expected else 'INFEASIBLE')
                if expected:
                    self.assertIn(tuple(out['codes']),expected)


if __name__=='__main__':
    unittest.main()
