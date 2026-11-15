"""Exercise source-independent geometric rechecks, dominance and objective separation."""
from itertools import combinations,permutations,product
import unittest
from unittest.mock import patch

from r27_structure_filters import domain_bits
from r27_structure_fixtures import planted,independent
from r31_geographic import physical_bonds
from r32_gold_watch import BondWatcher
from r35_geographic_replay import replay_for
from validator import reverse_mask


def triangle(p):
    for x,y,z in combinations(range(p.n),3):
        for trip in product(permutations(range(3),2),repeat=3):
            path=tuple(zip((x,y,z),trip))
            edges=[tuple(sorted((3*a+out,3*b+inside))) for (a,(_,out)),(b,(inside,_)) in zip(path,path[1:]+path[:1])]
            if all(int(p.masks[3*(a//3),a%3])==reverse_mask(int(p.masks[3*(b//3),b%3])) for a,b in edges):
                return frozenset(edges)
    raise AssertionError('Expected at least one triangle of compatible physical side pairs.')


class GeographicReplayTests(unittest.TestCase):
    def test_cached_domain_dominance_open_variants_and_limits(self):
        p,_=planted(1,2,35,False,True)
        verified=replay_for(p);root=verified.root
        core=triangle(p);board=[3*i for i in range(p.n)];good=set(physical_bonds(p,board))
        self.assertTrue(verified.check(root,[core]))
        count=verified.work['replays']
        self.assertTrue(verified.check(root,[core]))
        self.assertEqual(verified.work['replays'],count)
        self.assertFalse(verified.check(root,[good,[(0,0)],[(0,-1)],[(9999,5)]]))
        actual=next(iter(good));a,b=(i//3 for i in actual)
        remote=next(i for i in range(p.n) if i!=a and i!=b and i not in p.neighbors[a])
        swapped=board.copy();swapped[b],swapped[remote]=swapped[remote],swapped[b]
        narrow=tuple(1<<v for v in swapped);sibling=tuple(1<<v for v in board)
        self.assertTrue(verified.check(narrow,[[actual]]))
        self.assertFalse(verified.check(root,[[actual]]))
        self.assertFalse(verified.check(sibling,[[actual]]))
        self.assertFalse(verified.check(root,[core],max_steps=0))
        watch=BondWatcher(p,(),geographic=[[actual]],domains=narrow)
        allowed=list(sibling);watch.prune(allowed)
        self.assertEqual(tuple(allowed),sibling)
        foreign,_=planted(1,2,40,False,False)
        self.assertIsNot(replay_for(foreign),verified)
        self.assertFalse(replay_for(foreign).check(domain_bits(foreign),[physical_bonds(foreign,board)]))

    def test_scoped_geographic_units_through_native_model_callbacks(self):
        from pathlib import Path
        from r28_exchange import read_json
        from r34_geographic_mates import _post
        controls=read_json(Path(__file__).parent/'docs/R35_GEOGRAPHIC_UNITS.json')
        for case in controls:
            p,_=planted(1,4,case['seed'],False,True)
            ds=[int(v,16) for v in case['domains']]
            p.geographic_conflicts=[case['core']]
            live=ds.copy()
            def remove(c,v,b,j,positive):
                self.assertTrue(positive)
                self.assertIsNone(j)
                live[c]&=~(1<<v)
            emitted=_post(p,[[v for v in range(3*p.n) if d>>v&1] for d in ds],remove,lambda:None,physical=False)
            self.assertEqual(emitted['geographic_core_units'],case['removals'])
            self.assertEqual(independent(p,ds),independent(p,live))

    def test_complete_gold_and_edge_objectives_on_all_cpu_engines(self):
        from constraint_models import solve
        from constraint_dfs import solve as dfs
        for split in (False,True):
            p,ds=planted(1,3,35,split,True);hint=triangle(p)
            p.domains=[[v for v in range(3*p.n) if d>>v&1] for d in ds]
            p.physical_gold_support=True;p.geographic_conflicts=[hint]
            expected=independent(p,ds)
            for engine in ('sat','cp-sat','cp-sat-circuit','cp','dfs'):
                for target in ('gold','edges'):
                    out=dfs(p,target=target,seconds=5) if engine=='dfs' else solve(p,engine=engine,target=target,seconds=5)
                    if target=='gold':
                        self.assertEqual(out['status'],'solved' if expected else 'infeasible',(engine,target,out))
                    else:
                        self.assertIn(out['status'],('solved','edge_perfect'))
                        self.assertNotIn('physical_forbidden_clauses',out['stats'])

if __name__=='__main__':unittest.main()
