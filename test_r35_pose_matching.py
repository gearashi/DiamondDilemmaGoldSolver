"""Check weighted pose departures against exhaustive signed-orientation assignments."""
from itertools import permutations, product
from random import Random
from unittest import TestCase

from r35_pose_matching import owner_bounds, clear_owner_cache, _price, owner_candidate
from bench_r35_pose_matching import exact_margins


def oracle(domains, references, cap):
    n = len(domains)
    support = [0]*n
    for tiles in permutations(range(n)):
        rotations = [[3*t+r for r in range(3) if ds & (1<<(3*t+r))]
                     for t, ds in zip(tiles,domains)]
        for board in product(*rotations):
            if any(sum(a==b for a,b in zip(board,reference))>cap for reference in references):
                continue
            for c, code in enumerate(board):
                support[c] |= 1<<code
    return support


class PoseMatchingTests(TestCase):
    def test_independent_board_permutations_and_orientations(self):
        rng = Random(8903)
        count = negative = removed = 0
        for n in (2,3,4,5):
            for _ in range(75):
                ds = tuple(sum(1<<code for code in range(3*n) if rng.random()<.33)
                           or 1<<rng.randrange(3*n) for c in range(n))
                refs=[]
                for _ in range(2):
                    order=list(range(n));rng.shuffle(order)
                    refs.append([3*t+rng.randrange(3) for t in order])
                images=[tuple(reversed(tuple((c,1<<code) for c,code in enumerate(ref)))) for ref in refs]
                cap=rng.randrange(n)
                expected=oracle(ds,refs,cap)
                actual,stats=owner_bounds(ds,images,cap)
                if stats['over']:
                    self.assertFalse(any(expected))
                    negative+=1
                else:
                    self.assertTrue(all(bits & ~retained == 0 for bits,retained in zip(expected,actual)))
                removed+=stats['removed']
                count+=1
        self.assertEqual(count,300)
        self.assertGreater(negative,0)
        self.assertGreater(removed,0)

    def test_rotation_alternative_and_individually_optional_owners(self):
        n=5
        ref=[3*t for t in range(n)]
        image=tuple((c,1<<v) for c,v in enumerate(ref))
        ds=[1 | sum(7<<(3*t) for t in range(1,n))]
        ds.extend((1<<(3*t)) | 7 for t in range(1,n))
        self.assertTrue(all(mask.bit_count()>1 for mask in ds))
        for cap,negative in ((2,True),(3,False)):
            out,stats=owner_bounds(ds,[image],cap)
            self.assertEqual(stats['over'],negative)
            self.assertEqual(stats['lower'],3)
            if not negative:
                expected=oracle(ds,[ref],cap)
                self.assertTrue(all(b & ~a == 0 for b,a in zip(expected,out)))
                # No particular leaf has to retain its reference pose.
                for i in range(1,n):
                    self.assertTrue(expected[i] & 7)
        ds[2]|=1<<(3*2+1)
        out,stats=owner_bounds(ds,[image],2)
        self.assertFalse(stats['over'])
        self.assertTrue(any(oracle(ds,[ref],2)))

    def test_empty_domain_duplicate_images_and_distinct_turns(self):
        n=5
        base=[(1<<(3*n))-1]*n
        image=tuple((i,1<<(3*i)) for i in range(n))
        base[3]=0
        out,stats=owner_bounds(base,[image],3)
        self.assertTrue(stats['over'])
        self.assertEqual(stats['reason'],'empty_domain')
        self.assertEqual(stats['weighted_queries'],0)
        ds=[1 | sum(7<<(3*t) for t in range(1,n)),*(1<<(3*t) | 7 for t in range(1,n))]
        one,stats=owner_bounds(ds,[image],3)
        twice,other=owner_bounds(ds,[image,tuple(reversed(image))],3)
        self.assertFalse(stats['over'] or other['over'])
        self.assertEqual(one,twice)
        for bad in (image[:-1], image[:-1]+(image[0],),
                    image[:-1]+((n-1,2),), image[:-1]+((n-1,3),)):
            with self.subTest(image=bad), self.assertRaises(ValueError):
                owner_bounds(ds,[bad],3)
        rotated=tuple((i,1<<(3*i+1)) for i in range(n))
        extra=oracle(ds,[[3*i for i in range(n)],[3*i+1 for i in range(n)]],3)
        both,other=owner_bounds(ds,[image,rotated],3)
        self.assertFalse(other['over'])
        self.assertTrue(all(d & ~v == 0 for d,v in zip(extra,both)))

    def test_real_consumer_rejects_changed_mesh_and_missing_theorem(self):
        from pathlib import Path
        from unittest.mock import patch
        import json
        from bench_r35_pose_matching import cases
        from r35_pose_departures import original_pose_departures
        from r35_repair_seeds import active_catalog
        from search_problem import SearchProblem
        tiles=next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if tiles is None:
            self.skipTest('Original prepared pieces unavailable')
        problem=SearchProblem(json.loads(tiles.read_bytes()),physical_gold_support=True)
        ds=cases(problem,active_catalog(problem)['codes'])['chain-bank-0-5']
        _,stat=original_pose_departures(problem,ds)
        self.assertTrue(stat['over'])
        old=int(problem.masks[0,0])
        try:
            problem.masks[0,0]=old^1
            result,wrong=original_pose_departures(problem,ds)
            self.assertIsNone(wrong)
            self.assertEqual(result,ds)
        finally:
            problem.masks[0,0]=old
        with patch('r35_pose_departures.proof_version',return_value=('different version',)), \
             patch('r35_repair_seeds.active_catalog',return_value=None):
            result,absent=original_pose_departures(problem,ds)
        self.assertIsNone(absent)
        self.assertEqual(result,ds)

    def test_absolute_edge_and_spare_costs_against_every_rectangular_injection(self):
        from constraint_connectivity_round26 import assignment_upper_bound
        rng=Random(8908)
        seen=0
        for k,m in ((2,3),(3,3),(3,5),(4,4),(4,6),(5,7)):
            for _ in range(20):
                matrix=[[rng.choice((None,0,1,1)) for j in range(m)] for i in range(k)]
                scored=[(sum(matrix[i][j] for i,j in enumerate(perm)),perm) for perm in permutations(range(m),k)
                        if all(matrix[i][j] is not None for i,j in enumerate(perm))]
                if not scored:
                    continue
                optimum=max(v for v,_ in scored)
                proof=assignment_upper_bound(matrix)
                self.assertEqual(optimum,proof['bound'])
                for slack in (0,1,2):
                    inside,outside=exact_margins(list(range(k)),list(range(m)),matrix,proof,slack)
                    for (i,j),bound in inside.items():
                        expected=min([optimum-value for value,perm in scored if perm[i]==j]+[slack+1])
                        self.assertEqual(min(slack+1,bound),expected)
                    for j,bound in outside.items():
                        expected=min([optimum-value for value,perm in scored if j not in perm]+[slack+1])
                        self.assertEqual(min(slack+1,bound),expected)
                    seen+=1
        self.assertGreater(seen,300)

    def test_identically_priced_components_only_solve_once(self):
        n=10
        image=tuple((i,1<<(3*i)) for i in range(n))
        ds=[]
        for i in range(n):
            start=i//5*5
            alternatives=range(start+1,start+5) if i==start else [start]
            ds.append(1<<(3*i)|sum(7<<(3*t) for t in alternatives))
        clear_owner_cache()
        _,stats=owner_bounds(ds,[image],6)
        self.assertFalse(stats['over'])
        self.assertEqual(stats['weighted_queries'],1)
        self.assertEqual(stats['repeated_matrices'],1)
        self.assertEqual(stats['lower'],6)
        self.assertGreater(stats['removed'],0)

    def test_broad_guard_and_identical_cache_return(self):
        n=5
        ref=tuple((i,1<<(3*i)) for i in range(n))
        clear_owner_cache()
        ds=[(1<<(3*n))-1]*n
        self.assertEqual(owner_bounds(ds,[ref],3)[1]['weighted_queries'],0)
        self.assertEqual(_price.cache_info().currsize,0)
        # Tile-only and even two-rotation domains cannot force the oriented
        # reference when using a physical tile in a cell.
        for turns in (3,5,6,7):
            broad=[sum(turns<<(3*t) for t in range(n) if (c+t)%3) for c in range(n)]
            self.assertFalse(owner_candidate(broad,3))
        thin=[(1<<(3*c)) | (7<<(3*((c+1)%n))) for c in range(n)]
        self.assertTrue(owner_candidate(thin,3))
        ds=[1 | sum(7<<(3*t) for t in range(1,n)),*(1<<(3*t) | 7 for t in range(1,n))]
        expected,first=owner_bounds(ds,[ref],3)
        actual,repeat=owner_bounds(ds,[ref],3)
        self.assertEqual(actual,expected)
        self.assertGreater(first['weighted_queries'],0)
        self.assertEqual(repeat['weighted_queries'],0)
        self.assertEqual(repeat['cache_hits'],1)
