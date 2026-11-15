"""Check labelled contact constraints independently of footprint packing and speculative scoring."""
import itertools
import random
import unittest
from types import SimpleNamespace

from r27_structure_filters import domain_bits
from r32_gold_bonds import inventory
from r35_joint_poses import narrowed_poses
from r35_pose_contacts import conditioned_components
from r35_speculative_poses import PoseJoins
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,double_face,simple_tiles
from test_r35_joint_poses import cluster
from validator import reverse_mask


def feasible(p,codes,physical):
    allowed=inventory(p).possible if physical else None
    for a,s,b,t in p.edges:
        v,w=codes[a],codes[b]
        if int(p.masks[v,s])!=reverse_mask(int(p.masks[w,t])):return False
        pair=tuple(sorted((3*(v//3)+(s-v%3)%3,3*(w//3)+(t-w%3)%3)))
        if physical and pair not in allowed:return False
    return True


class ContactTests(unittest.TestCase):
    def test_exact_all_singleton_component_codes_under_restricted_domains(self):
        p=SearchProblem(*tetrahedron_tiles());rng=random.Random(85)
        full=[tuple(3*t+r for t,r in zip(tiles,rot)) for tiles in itertools.permutations(range(4))
              for rot in itertools.product(range(3),repeat=4)]
        for physical in (False,True):
            valid=[x for x in full if feasible(p,x,physical)]
            self.assertTrue(valid)
            for k in range(32):
                seed=rng.choice(valid)
                ds=tuple(sum(1<<v for v in range(12) if rng.random()<.36 or (k%2==0 and v==seed[c])) for c in range(4))
                groups=[cluster([t],[((c,3*t+r),) for c in range(4) for r in range(3)]) for t in range(4)]
                expected=[x for x in valid if all(ds[c]>>v&1 for c,v in enumerate(x))]
                oracle=tuple(sum(1<<v for v in {x[c] for x in expected}) for c in range(4))
                result=narrowed_poses(ds,groups,problem=p,contact_options={'physical':physical})
                self.assertEqual(result['domains'],oracle)
                self.assertTrue(result['complete'])
                capped=narrowed_poses(ds,groups,problem=p,work_limit=1,contact_options={'physical':physical})
                self.assertTrue(all(a&b==a for a,b in zip(oracle,capped['domains'])))

    def test_parallel_extra_seams_do_not_remove_scored_repair_possibilities(self):
        tiles=simple_tiles()
        tiles[1]['segments']=[[[0,10],[1,6]]]
        p=SearchProblem(tiles,double_face());ds=domain_bits(p)
        class Legal:
            def __getitem__(self,key):
                c,v=key
                return bool(ds[c]>>v&1)
        a=SimpleNamespace(model=SimpleNamespace(problem=p,legal=Legal(),n=p.n),groups=[])
        chosen=(0,3)
        initial=PoseJoins(a)
        scored=initial.replay([chosen])[0][1]
        self.assertTrue(scored)
        self.assertTrue(all(not feasible(p,dict(pose),False) for pose in scored))
        self.assertIsNone(PoseJoins(a,gold=True).append([],chosen))
        conditioned,stats=conditioned_components(p,ds,[],[chosen])
        state=narrowed_poses(ds,conditioned,problem=p)
        self.assertFalse(any(state['domains']))
        self.assertEqual(stats['incompatible_incidental_rows'],len(scored))

    def test_named_cycle_replay_remains_consistent_on_sparse_mesh_without_neighbor_property(self):
        p=SearchProblem(simple_tiles(),double_face());ds=domain_bits(p)
        # All three seams are forced by the actual labelled planted poses.
        for bonds in ([(0,3),(1,4),(2,5)],[(2,5),(0,3),(1,4)]):
            families,work=conditioned_components(p,ds,[],bonds)
            kept=narrowed_poses(ds,families,problem=p)
            self.assertTrue(all(kept['domains']))
            for cell,v in enumerate([0,3]):self.assertTrue(kept['domains'][cell]>>v&1)
            self.assertGreater(work['cycle_pose_tests'],0)
        wrong=[(0,3),(1,5)]
        families,_=conditioned_components(p,ds,[],wrong)
        self.assertFalse(any(narrowed_poses(ds,families,problem=p)['domains']))

    def test_public_early_failure_and_held_out_single_loop_positive(self):
        from r30_gold import search
        from r27_structure_fixtures import planted,independent
        p,ds=planted(1,4,85)
        solution=min(independent(p,ds))
        a,s,b,t=p.edges[0];v,w=solution[a],solution[b]
        left=3*(v//3)+(s-v%3)%3;right=3*(w//3)+(t-w%3)%3
        third=next(c for c in range(3*p.n) if c//3 not in (left//3,right//3))
        failed=search(p,domains=ds,bonds=[(left,right),(left,third)],geography=True,joint_poses=True)
        self.assertEqual(failed['status'],'INFEASIBLE')
        self.assertEqual(failed['cp_calls'],0)
        from test_dependencies import HAS_ORTOOLS
        if HAS_ORTOOLS:
            answer=search(p,domains=ds,bonds=[(left,right)],geography=True,channel=True,joint_poses=True)
            self.assertEqual(answer['status'],'GOLD')
            self.assertTrue(p.validate(answer['codes'])['valid'])

if __name__=='__main__':unittest.main()
