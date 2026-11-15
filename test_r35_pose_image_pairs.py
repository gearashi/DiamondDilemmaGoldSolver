"""Independent exact cell/color/tile controls for the joint pose quotas."""
from itertools import permutations, product
from random import Random
import unittest

from bench_r35_pose_pair_weights import away_matching_size, exact_pair_price, pair_owner_bounds
from r35_pose_image_pairs import pair_bounds, pair_candidate


def reference(n, offsets):
    return tuple(tuple((c,1<<(3*((c+d)%n))) for c in range(n)) for d in offsets)


def solutions(ds, images, cap, *, owners):
    tables=[dict(image) for image in images]
    support=[0]*len(ds)
    for codes in product(*[[v for v in range(3*len(ds)) if m>>v & 1] for m in ds]):
        if owners and len({v//3 for v in codes})!=len(codes): continue
        if any(sum(1<<v==table[c] for c,v in enumerate(codes))>cap for table in tables):continue
        for i,v in enumerate(codes):support[i]|=1<<v
    return support


class PairQuotaTests(unittest.TestCase):
    def test_pure_hall_matches_every_supported_pose_with_transfer_fixed_point(self):
        from r35_pose_departures import propagate
        rng=Random(89359);n=5;images=reference(n,(0,1,2));cap=2
        # Three colors have 6 slots across 5 sites. The only tight Hall subsets are singles and pairs.
        samples=zero=positive=0
        for _ in range(160):
            domains=[]
            for c in range(n):
                options=[dict(im)[c] for im in images]+[1<<(3*c+1)]
                domains.append(sum(rng.sample(options,rng.randint(1,3))))
            support=solutions(domains,images,cap,owners=False)
            current=domains
            while True:
                narrowed,one=propagate(current,images,{},cap)
                if one['over']:break
                after,two=pair_bounds(narrowed,images,cap)
                if two['over']:break
                if list(after)==list(current):break
                current=after
            if one['over'] or two['over']:
                self.assertFalse(any(support));zero+=1
            else:
                self.assertEqual(list(after),support);positive+=1
            samples+=1
        self.assertEqual(samples,160);self.assertGreater(zero,0);self.assertGreater(positive,0)

    def test_conditional_weighted_price_agrees_with_complete_placement_enumeration(self):
        n=4;rng=Random(89363);images=reference(n,(0,1,2))
        tables=[dict(im) for im in images];feasible=refuted=pruned=0
        for _ in range(140):
            domains=[sum(1<<v for v in rng.sample(range(3*n),rng.randint(1,4))) for _ in range(n)]
            cap=rng.randrange(1,3)
            paid=tuple(tables[0][c]|tables[1][c] for c in range(n))
            exact=[0]*n
            for codes in product(*[[v for v in range(3*n) if d>>v & 1] for d in domains]):
                if len({v//3 for v in codes})!=n or sum(bool(mask & (1<<v)) for mask,v in zip(paid,codes))>cap:continue
                for c,v in enumerate(codes):exact[c]|=1<<v
            filtered,over,_=exact_pair_price(tuple(domains),paid,cap)
            if over:
                self.assertFalse(any(exact));refuted+=1
            else:
                self.assertEqual(list(filtered),exact);feasible+=1
                pruned+=sum((a^b).bit_count() for a,b in zip(domains,exact))
            supported=solutions(domains,images,cap,owners=True)
            narrowed,stats=pair_owner_bounds(domains,images,cap)
            if stats['over']:self.assertFalse(any(supported))
            else:self.assertTrue(all(not (s&~d) for s,d in zip(supported,narrowed)))
        self.assertGreater(feasible,0);self.assertGreater(refuted,0);self.assertGreater(pruned,0)

    def test_off_image_matching_handles_augmentations_and_capped_early_exit(self):
        rng=Random(89365);n=5
        for _ in range(100):
            rows=[rng.randrange(1<<n) for _ in range(n)]
            optimum=max(sum(bool(m&(1<<t)) for t,m in zip(p,rows)) for p in permutations(range(n)))
            for limit in (1,3,6):
                self.assertEqual(away_matching_size(rows,limit),min(limit,optimum))

    def test_threshold_and_broad_no_op(self):
        n=160;images=reference(n,(0,1));full=(1<<(3*n))-1
        tables=[dict(i) for i in images]
        for size,expected in ((129,0),(130,60),(131,0)):
            domains=[(tables[0][c]|tables[1][c]) if c<size else full for c in range(n)]
            after,stats=pair_bounds(domains,images,65)
            self.assertEqual(stats['over'],size==131)
            self.assertEqual(stats['removed'],expected)
            if size==130:
                self.assertTrue(all(after[c]==full^(tables[0][c]|tables[1][c]) for c in range(size,n)))
        self.assertFalse(pair_candidate([full]*n,65))
        self.assertEqual(pair_owner_bounds([full]*n,images,65)[1]['matching_queries'],0)

    def test_competing_away_tile_and_recurrent_exact_query(self):
        from bench_r35_pose_pair_weights import _owner_pair
        n=10; cap=4
        images=reference(n,(0,1));tables=[dict(i) for i in images]
        ds=[tables[0][c]|tables[1][c]|(1<<2) for c in range(n)]
        self.assertFalse(pair_bounds(ds,images,cap)[1]['over'])
        _owner_pair.cache_clear()
        _,initial=pair_owner_bounds(ds,images,cap)
        _,again=pair_owner_bounds(ds,images,cap)
        self.assertTrue(initial['over'] and again['over'])
        self.assertEqual(initial['lower'],9)
        self.assertEqual(initial['matching_queries'],1)
        self.assertEqual((again['cache_hits'],again['matching_queries']),(1,0))
