import itertools
import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

from r35_pose_constructor import PoseConstructor,weighted_donors,path_cycle_assignment


def setup(n,domains,groups):
    legal=np.zeros((n,3*n),dtype=bool)
    for tile,slots in enumerate(domains):
        for cell in slots:
            legal[cell,3*tile:3*tile+3]=True
    return SimpleNamespace(model=SimpleNamespace(n=n,legal=legal),groups=groups,
                           tiles=set().union(*(tiles for tiles,_ in groups)))


class PoseConstructionTests(unittest.TestCase):
    def test_all_ground_embeddings_with_restricted_physical_resources(self):
        rng=np.random.default_rng(35039)
        for _ in range(48):
            n=7
            domains=[{int(x) for x in rng.choice(n,int(rng.integers(2,n+1)),replace=False)} for i in range(n)]
            groups=[]
            for a,b in ((0,1),(2,3)):
                poses=[((i,3*a),(j,3*b)) for i in sorted(domains[a]) for j in sorted(domains[b]) if i!=j and rng.random()<.5]
                groups.append(({a,b},poses[:6]))
            assembly=setup(n,domains,groups)
            def admits(fixed):
                free=set(range(n))-fixed.keys()
                return any(all(slot in domains[t] for t,slot in zip((4,5,6),order)) for order in itertools.permutations(free))
            expected=set()
            for p,q in itertools.product(groups[0][1],groups[1][1]):
                fixed=dict((*p,*q))
                if len(fixed)==4 and admits(fixed):expected.add(tuple(sorted(fixed.items())))
            for mode in ('forward','hall','matching'):
                actual=set()
                def leaf(fixed):
                    if admits(fixed):actual.add(tuple(sorted(fixed.items())))
                stats=Counter()
                PoseConstructor(assembly,[r for _,r in groups]).find(leaf,None,mode,stats)
                self.assertEqual(actual,expected)

    def test_full_matching_rejects_large_hidden_hall_family(self):
        domains=[set(range(6))-{i} for i in range(6)]+[set(range(6)),{6,7}]
        assembly=setup(8,domains,[])
        for mode,expected in (('forward',1),('hall',1),('matching',0)):
            stats=Counter()
            found=PoseConstructor(assembly,[]).find(lambda fixed:True,None,mode,stats)
            self.assertEqual(bool(found),bool(expected))
            if mode=='matching':self.assertEqual(stats['residual_matching_rejections'],1)

    def test_interrupted_pose_budget_cannot_remove_a_later_legal_lift(self):
        groups=[({0,1},[((0,0),(1,3)),((2,0),(3,3))]),({2,3},[((0,6),(1,9))])]
        a=setup(4,[set(range(4)) for _ in range(4)],groups)
        find=PoseConstructor(a,[r for _,r in groups]).find
        stats=Counter()
        self.assertIsNone(find(lambda f:dict(f),1,'matching',stats))
        self.assertGreater(stats['budget_stops'],0)
        self.assertEqual(find(lambda f:dict(f),None,'matching',Counter()),{0:6,1:9,2:0,3:3})

    def test_geographic_pose_cache_tracks_hard_domains_and_exact_group_rows(self):
        groups=[({0,1},[((0,0),(1,3)),((2,0),(3,3))])]
        a=setup(4,[set(range(4)) for _ in range(4)],groups)
        stats=Counter()
        x=PoseConstructor.for_assemblies(a,stats)
        self.assertIs(x,PoseConstructor.for_assemblies(a,stats))
        self.assertEqual(stats['geometry_pose_rows_reused'],2)
        a.model.legal[0,6]=False
        y=PoseConstructor.for_assemblies(a,stats)
        self.assertIsNot(x,y)
        groups[0][1].pop()
        z=PoseConstructor.for_assemblies(a,stats)
        self.assertIsNot(y,z)
        self.assertEqual(len(z.poses[0]),1)

    def test_all_three_by_three_sparse_weighted_graphs_include_paths_cycles_and_hall_failures(self):
        rng=np.random.default_rng(353909)
        seen=Counter()
        for mask in range(1<<9):
            legal=np.array([bool(mask&(1<<i)) for i in range(9)]).reshape(3,3)
            if max(legal.sum(axis=0).max(),legal.sum(axis=1).max())>2:continue
            weights=rng.integers(-2,9,size=(3,3))
            feasible=[sum(weights[i,j] for i,j in enumerate(p)) for p in itertools.permutations(range(3))
                      if all(legal[i,j] for i,j in enumerate(p))]
            if not feasible:
                with self.assertRaises(ValueError):path_cycle_assignment(weights,legal,Counter())
                continue
            stats=Counter()
            donors=path_cycle_assignment(weights,legal,stats)
            seen.update(stats)
            self.assertEqual(sum(weights[i,j] for i,j in enumerate(donors)),max(feasible))
        self.assertGreater(seen['weighted_cycle_components'],0)
        self.assertGreater(seen['forced_weighted_rows'],0)

    def test_compact_matches_every_weighted_optimum_and_falls_back_when_hints_interfere(self):
        rng=np.random.default_rng(35391)
        fallbacks=Counter()
        for n in (2,3,5,7):
            for _ in range(100):
                allowed=rng.random((n,n,3))<.26
                allowed[np.arange(n),np.arange(n),0]=True
                weights=rng.integers(0,8,size=allowed.shape)
                for i,j in enumerate(rng.permutation(n)):
                    if rng.random()<.8:
                        weights[i,j,2]+=8*n+1
                weights[~allowed]=-1
                for compact in (False,True):
                    stats=Counter()
                    donors=weighted_donors(weights,allowed,stats,compact=compact)
                    value=int(weights.max(axis=2)[np.arange(n),donors].sum())
                    self.assertTrue(allowed.any(axis=2)[np.arange(n),donors].all())
                    if compact:
                        self.assertEqual(value,reference)
                        fallbacks.update(stats)
                    else:reference=value
        self.assertGreater(fallbacks['compact_fallbacks'],0)
        self.assertGreater(fallbacks['fixed_hint_rows'],0)


if __name__=='__main__':
    unittest.main()
