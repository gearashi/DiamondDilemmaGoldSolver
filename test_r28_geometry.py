"""Independent orbit, exact counting, sewing and topology controls."""
from collections import Counter
from importlib.util import find_spec
from itertools import combinations
from random import Random
import unittest

from r27_structure_fixtures import independent, nested_loops, planted
from r27_structure_filters import _regions, _vertex_links, domain_bits, oriented_segments
from r28_geo_color import dual_cuts, phase_probe
from r28_geo_regions import _one_tour, count_partition, representatives
from r28_geo_sewing import PosePacking, _cycles, boundary_lower, ring_census, sign_filter, supply, supply_supports, switch_family


def seams(problem):
    return [(3*a+s,3*b+t) for a,s,b,t in problem.edges]


def phases(problem):
    phase = {0:0}
    regions = [_regions(oriented_segments(problem,3*t)) for t in range(problem.n)]
    todo = [0]
    while todo:
        a = todo.pop()
        for s,(b,t) in enumerate(problem.board.neighbors[a]):
            color = lambda row,k:row[k].bit_count() & 1
            v = phase[a] ^ color(regions[a],12*s) ^ color(regions[b],12*t+11)
            if b not in phase:
                phase[b] = v
                todo.append(b)
            elif phase[b] != v:raise ValueError("Board cannot be colored.")
    return [phase[i] for i in range(problem.n)]


def matchings(xs):
    if not xs:
        yield ()
        return
    a,*tail = xs
    for b in tail:
        for pairs in matchings([x for x in tail if x != b]):yield ((a,b),*pairs)


class GeometryTests(unittest.TestCase):
    @unittest.skipUnless(find_spec("ortools"),"optional CP backend")
    def test_phases_complement_and_guarded_dual(self):
        for split in (False,True):
            p,ds = planted(1,3,23,split)
            pins = dict(enumerate(phases(p)))
            from r28_geo_color import prune_signed_core
            kept,info = prune_signed_core(p,ds,seams(p))
            if not split:
                self.assertEqual(kept,ds)
                self.assertEqual(info["removed"],0)
            for flip in (0,1):
                full = {t:x^flip for t,x in pins.items()}
                record = phase_probe(p,euler=True,pins=full,bonds=seams(p))
                if not split:self.assertEqual(record["status"],"OPTIMAL")
            out = dual_cuts(p,ds)
            self.assertTrue(out["complete"])
            self.assertEqual(out["exists"], bool(independent(p,ds)))
        p,ds = nested_loops()
        result = dual_cuts(p,ds)
        self.assertFalse(result["exists"])
        self.assertGreater(result["cuts"],0)

    @unittest.skipUnless(find_spec("ortools"),"optional CP backend")
    def test_ring_coverage_and_cutoff(self):
        p,ds = planted()
        options = [set() for _ in range(3*p.n)]
        for a,b in seams(p):
            options[a].add(b)
            options[b].add(a)
        exact = ring_census(p,options)
        self.assertEqual(exact["columns"],len(set(_vertex_links(p))))
        self.assertEqual(exact["status"],"OPTIMAL")
        self.assertEqual(ring_census(p,options,max_work=1)["status"],"UNKNOWN")
        a,b = seams(p)[0]
        options[a].remove(b)
        options[b].remove(a)
        self.assertEqual(ring_census(p,options)["status"],"INFEASIBLE")

    def test_disjoint_poses_and_inventory(self):
        p,ds = planted(1,0)
        self.assertTrue(PosePacking(p,ds).check(seams(p))["feasible"])
        more = list(ds)
        more[2] = (1<<(3*0)) | (1<<(3*2))
        more[1] = (1<<(3*0)) | (1<<(3*1))
        more[0] = (1<<(3*1)) | (1<<(3*2))
        # Every singleton tile individually fits; three tiles forced to two cells.
        more[2] = (1<<(3*3))
        actual = PosePacking(p,more).check([])
        self.assertTrue(actual["independent"])
        self.assertFalse(actual["feasible"])
        self.assertIsNone(PosePacking(p,ds).check([],max_steps=0)["feasible"])
        lbs,_ = boundary_lower(p,5)
        for k in range(1,6):
            for cells in combinations(range(p.n),k):
                S = set(cells)
                boundary = sum((a in S) != (b in S) for a,s,b,t in p.edges)
                self.assertLessEqual(lbs[k],boundary)
                self.assertEqual(supply(p,S,seams(p)),boundary)
                self.assertGreaterEqual(supply(p,S),boundary)

    def test_parity_and_switches(self):
        for p,ds in (planted(1,0),planted(1,0,split=True),nested_loops()):
            gold = p.validate([3*t for t in range(p.n)])
            outcome = sign_filter(p,phases(p),seams(p))
            self.assertEqual(outcome["reference_cycles"],gold["line_components"])
            self.assertEqual(outcome["feasible"],bool(gold["line_components"]%2))
            self.assertTrue(sign_filter(p,phases(p))["feasible"] or not gold["valid"])
        from r28_geo_sewing import switch_boards
        p,_ = planted(1,0)
        codes = [3*t for t in range(p.n)]
        self.assertEqual(switch_boards(p,codes,[])["single_tours"],1)
        with self.assertRaises(ValueError):switch_boards(p,codes,[{0:codes[0]}])
        with self.assertRaises(ValueError):switch_boards(p,codes,[{0:codes[1]}])
        rng = Random(91)
        for size in (3,5,7):
            tour = list(range(1,size*2))+[0]
            order = rng.sample(range(2*size),2*size)
            switches = list(zip(order[::2],order[1::2]))
            for forced in ((),(0,),(0,1)):
                total = 0
                for mask in range(1<<size):
                    if any(not mask >> x & 1 for x in forced):continue
                    v = tour.copy()
                    for j,(a,b) in enumerate(switches):
                        if mask>>j & 1:v[a],v[b] = v[b],v[a]
                    total += _cycles(v)==1
                r = switch_family(tour,switches,forced)
                self.assertTrue(r["complete"])
                self.assertEqual(r["single_tours"],total)

    def test_matching_basis_all_exteriors(self):
        for n in (2,4,6,8):
            rows = list(matchings(list(range(n))))
            pick,_ = representatives(rows)
            self.assertEqual(len(pick),1<<(n//2-1))
            rng = Random(n)
            for size in (1,3,9,30):
                sub = rng.sample(rows,min(size,len(rows)))
                keep,_ = representatives(sub)
                for exterior in rows:
                    self.assertEqual(any(_one_tour(r,exterior) for r in sub),
                                     any(_one_tour(sub[i],exterior) for i in keep))
                _,groups = representatives(sub,rows[:4])
                for profile,indices in groups.items():
                    for j in indices:
                        self.assertEqual(profile,sum(1<<i for i,z in enumerate(rows[:4]) if _one_tour(sub[j],z)))

    def test_multiboundary_count_and_join(self):
        p,ds = planted(2,4)
        # Remove both apex stars; their complement is a connected annulus.
        rings = [[c.index for c in p.board.cells if any(p.board.vertices[v] == ((pole,2),)
                  for v in c.vertices)] for pole in (0,1)]
        belt = sorted(set(range(p.n))-set(rings[0])-set(rings[1]))
        expected = len(independent(p,ds))
        self.assertGreater(expected,0)
        for mode in ("plain","trie","response","basis","orbit"):
            r = count_partition(p,[belt,rings[0]+rings[1]],ds,mode=mode)
            self.assertTrue(r["complete"])
            self.assertEqual(r.get("solutions",r.get("exists")),expected if mode != "basis" else bool(expected))
        for mode in ("trie","plain"):
            self.assertEqual(count_partition(p,[belt,*rings],ds,mode=mode)["solutions"],expected)

    def test_trie_cursor_matches_uninterrupted(self):
        from r28_geo_regions import paged_trie
        p,ds = planted(1,5,97,uniform_ports=True)
        parts = ((0,1,2),(3,4,5),(6,7,8,9))
        control = paged_trie(p,parts,ds)
        self.assertTrue(control["complete"])
        cursor = None
        pages = 0
        while True:
            r = paged_trie(p,parts,ds,max_steps=5,cursor=cursor)
            cursor = r["cursor"]
            pages += 1
            if r["complete"]:break
        self.assertGreater(pages,1)
        self.assertEqual((r["checks"],r["solutions"]),(control["checks"],control["solutions"]))
        self.assertEqual(r["solutions"],count_partition(p,parts,ds)["solutions"])
        self.assertEqual(paged_trie(p,parts,ds,cursor=cursor)["steps"],0)
        with self.assertRaises(ValueError):paged_trie(p,parts[::-1],ds,cursor=cursor)

    def test_c5_orbits_stabilizers(self):
        p,_ = planted(1,0,23,uniform_ports=True)
        # All rotations of the planted physical assignment are admitted.
        ds = tuple(sum(1<<(3*t) for t in (range(5) if c<5 else range(5,10))) for c in range(p.n))
        parts = (range(5),range(5,10))
        expected = count_partition(p,parts,ds,mode="trie")
        r = count_partition(p,parts,ds,mode="orbit")
        self.assertGreater(r["work"]["orbit_representatives"],0)
        self.assertEqual(r["solutions"],expected["solutions"])
        self.assertGreater(r["solutions"],0)


if __name__ == "__main__":unittest.main()
