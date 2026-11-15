"""Round 34 follow-ups on physical Gold, phase cubes and count-preserving contours."""
from importlib.util import find_spec
from random import Random
import unittest

import numpy as np

from constraint_symmetry import proper_rotations
from r27_incumbent import BoardModel
from r27_structure_fixtures import planted, independent
from r29_gold_phase import SignedParity
from r32_fresh_phases import SignedRepair, independent_phases, phase_cube
from r32_gold_bonds import inventory
from r32_gold_watch import BondWatcher

from geometry import build_board
from search_problem import SearchProblem
def ring_puzzle():
    mesh=build_board(1);n=len(mesh.cells)
    def path(sequence):
        candidates=[v for v,_ in mesh.neighbors[sequence[-1]]]
        if len(sequence)==n:
            return sequence if sequence[0] in candidates else None
        for v in candidates:
            if v not in sequence and (found:=path([*sequence,v])):
                return found
    route=path([0])
    ends=[[] for _ in range(n)]
    for i,a in enumerate(route):
        b=route[(i+1)%n];s=next(j for j,(v,_) in enumerate(mesh.neighbors[a]) if v==b)
        t=mesh.neighbors[a][s][1];k=1+(i%11)
        ends[a].append((s,k));ends[b].append((t,12-k))
    return SearchProblem([{'id':t,'segments':[ends[t]]} for t in range(n)],mesh)


class PhysicalFollowups(unittest.TestCase):
    def test_two_owner_poses_conflict_and_sibling_replay(self):
        p,_ = planted(1,0,23,False,True)
        x,y = 3,14
        self.assertIn((x,y),inventory(p).loops)
        gold=[3*t for t in range(p.n)]
        a,s,b,t=p.edges[0]
        remaining=iter(i for i in range(p.n) if i not in (x//3,y//3))
        bad=[]
        for c in range(p.n):
            if c==a:bad.append(3*(x//3)+(s-x%3)%3)
            elif c==b:bad.append(3*(y//3)+(t-y%3)%3)
            else:bad.append(3*next(remaining))
        for rotation in (1,2):
            symmetry=proper_rotations(p.board)[rotation]
            def two_poses(layout):
                return [1<<v | 1<<w for v,w in zip(layout,symmetry.apply(layout))]
            narrow=two_poses(bad)
            sibling=two_poses(gold)
            self.assertEqual({row.bit_count() for row in narrow},{2})
            watcher=BondWatcher(p,[[(x,y)]])
            for domains in ([u|v for u,v in zip(narrow,sibling)],narrow,sibling,narrow):
                old=list(domains); expected=independent(p,old)
                trial=list(old);watcher.prune(trial)
                if domains==narrow:
                    self.assertNotEqual(trial,old)
                if domains==sibling:
                    self.assertEqual(len(expected),2)
                self.assertEqual(independent(p,trial),expected)
                again=list(old);n=watcher.pair_checks
                watcher.prune(again)
                self.assertEqual(trial,again)
                self.assertEqual(watcher.pair_checks,n)

    def test_phase_cubes_partition_all_affine_assignments(self):
        p=ring_puzzle()
        q=SignedParity(p)
        for a,b in sorted(inventory(p).required):q.bond(a,b)
        rows=[(mask,rhs) for mask,rhs,_ in q.base.pivots.values()]
        solutions=[x for x in range(0,1<<p.n,2) if all((x&m).bit_count()%2==rhs for m,rhs in rows)]
        order=independent_phases(p)
        self.assertEqual(len(solutions),1<<len(order))
        homes={}
        for i in range(1<<len(order)):
            pins=phase_cube(p,len(order),i,independent=True)
            selected=[x for x in solutions if all(((x>>t)&1)==v for t,v in pins.items())]
            self.assertEqual(len(selected),1)
            homes[selected[0]]=i
        self.assertEqual(set(homes),set(solutions))
        with self.assertRaises(ValueError):
            phase_cube(p,len(order)+1,0,independent=True)

    def test_mandatory_pose_survivors_include_every_gold_skeleton(self):
        from r32_fresh_geometry import skeletons
        p=ring_puzzle();rng=Random(34)
        ds=[(1<<(3*t))|sum(1<<v for v in rng.sample([c for c in range(3*p.n) if c!=3*t],3)) for t in range(p.n)]
        results=[skeletons(p,ds,max_nodes=5000,limit=5000,mandatory_poses=active) for active in (False,True)]
        old,trial=results
        self.assertTrue(old['complete'] and trial['complete'])
        self.assertTrue(set(trial['patterns'])<set(old['patterns']))
        actual=set()
        for board in independent(p,ds):
            actual.add(tuple(sum((int(m).bit_count()%2)<<s for s,m in enumerate(p.masks[v])) for v in board))
        self.assertTrue(actual)
        self.assertLessEqual(actual,set(trial['patterns']))

    def test_dart_factorization_matches_direct_prices_for_changed_phases(self):
        p=ring_puzzle();model=BoardModel(p);forbidden=inventory(p).forbidden
        rng=Random(34)
        for _ in range(3):
            phases=[rng.randrange(2) for t in range(p.n)]
            signed=SignedRepair(model,phases,forbidden)
            for dart in range(3*p.n):
                vector=signed.geometry.price(dart,signed.dart_phases,signed.stats)
                t,u=divmod(dart,3)
                for side in range(3):
                    baseline=signed.labels[:,side]==signed.reverse[signed.labels[3*t,u]]
                    if dart in signed.blocked:
                        baseline &= ~np.isin(signed.sides[:,side],signed.blocked[dart])
                    np.testing.assert_array_equal(vector[signed.sides[:,side]],baseline)
        self.assertEqual(signed.stats['dart_templates_reused'],3*p.n)

    @unittest.skipUnless(find_spec('ortools'),'optional geographic lift')
    def test_flip_lifts_preserve_counts_and_validated_gold(self):
        from r27_structure_filters import domain_bits
        from r32_fresh_geometry import skeleton_domains
        from r32_fresh_phases import lift_cube
        from r34_contour_walk import contour_walk
        p=ring_puzzle()
        result=contour_walk(p,0,limit=6)
        self.assertEqual(len({tuple(row) for row in result['patterns']}),6)
        gold=0
        for pattern in result['patterns']:
            self.assertEqual(sum(map(bool,pattern)),p.n)
            row=lift_cube(p,inventory(p).required,{0:0},skeleton_domains(p,domain_bits(p),pattern),4)
            self.assertIn(row['status'],('GOLD','INFEASIBLE'))
            if row['status']=='GOLD':
                self.assertTrue(p.validate(row['codes'])['valid'])
                gold+=1
        self.assertGreater(gold,0)
        self.assertLess(result['assignment_cells'],6*p.n)

    @unittest.skipUnless(find_spec('ortools') and find_spec('pysat'),'optional search backends')
    def test_mate_clauses_preserve_gold_and_edges_objectives(self):
        from constraint_models import solve
        for split in (False,True):
            p,ds=planted(1,3,34,split,True)
            p.domains=[[v for v in range(3*p.n) if d>>v&1] for d in ds]
            p.physical_gold_support=True
            expected=independent(p,ds)
            for engine in ('sat','cp-sat','cp-sat-circuit','cp'):
                gold=solve(p,engine=engine,target='gold',seconds=4)
                self.assertEqual(gold['status'],'solved' if expected else 'infeasible')
                edges=solve(p,engine=engine,target='edges',seconds=4)
                self.assertIn(edges['status'],('solved','edge_perfect'))
                self.assertNotIn('physical_forbidden_clauses',edges['stats'])


if __name__=='__main__':
    unittest.main()
