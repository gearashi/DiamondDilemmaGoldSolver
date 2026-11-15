"""Independent labelled-pose/owner projection and disjoint-branch controls."""
from importlib.util import find_spec
from itertools import permutations,product
from random import Random
import unittest

from r35_joint_poses import narrowed_poses,branch_domains


def cluster(tiles,poses):
    return tuple(tiles),sum(7<<(3*t) for t in tiles),tuple(poses)


def arrangements(n):
    for perm in permutations(range(n)):
        for rots in product(range(3),repeat=n):
            yield tuple(3*t+r for t,r in zip(perm,rots))


def allowed(board,groups):
    return all(any(all(board[c]==v for c,v in pose) for pose in poses) for _,_,poses in groups)


class JointPoseTests(unittest.TestCase):
    def test_all_complete_assignments_and_projection_at_random_restrictions(self):
        n=4;rng=Random(35)
        boards=list(arrangements(n));origin=boards[0]
        for _ in range(40):
            groups=[]
            for tile in (0,1):
                poses=[((c,3*tile+r),) for c,r in product(range(n),range(3)) if rng.random()<.35]
                poses.append(((origin.index(3*tile),3*tile),))
                groups.append(cluster([tile],sorted(set(poses))))
            ds=[sum(1<<v for v in range(3*n) if rng.random()<.55 or v==origin[c]) for c in range(n)]
            true={b for b in boards if all(ds[c]>>v&1 for c,v in enumerate(b)) and allowed(b,groups)}
            state=narrowed_poses(ds,groups,work_limit=100000)
            self.assertTrue(state['complete'])
            expected=tuple(sum(1<<v for v in {b[c] for b in true}) for c in range(n))
            self.assertEqual(expected,state['domains'])
            children=branch_domains(state)
            if children:
                self.assertTrue(all(sum(all(row>>v&1 for row,v in zip(ds,b)) for ds in children)==1 for b in true))
            capped=narrowed_poses(ds,groups,work_limit=1)
            self.assertFalse(capped['complete'])
            self.assertTrue(all(a&b==a for a,b in zip(expected,capped['domains'])))

    @unittest.skipUnless(find_spec("ortools"),"requires OR-Tools")
    def test_public_geographic_search(self):
        from r27_structure_fixtures import planted
        from r30_gold import search
        p,d=planted(1,3,35)
        result=search(p,domains=d,geography=True,channel=True,joint_poses=True)
        self.assertTrue(p.validate(result['codes'])['valid'])
        self.assertTrue(result['joint_pose_support']['complete'])
        with self.assertRaises(ValueError):
            search(p,domains=d,joint_poses=True)

    def test_same_footprint_different_labels_and_incompatible_pair_of_poses(self):
        ds=[(1<<15)-1]*5
        ga=cluster([0,1],[((0,0),(1,3)),((0,1),(1,4)),((2,0),(3,3))])
        gb=cluster([2,3],[((2,6),(4,9)),((2,8),(4,11))])
        s=narrowed_poses(ds,[ga,gb])
        self.assertTrue(s['complete'])
        self.assertEqual(len(s['poses'][0]),2)
        self.assertEqual(s['domains'][0],3)
        self.assertEqual(s['domains'][3],7<<(3*4))
        self.assertEqual(len(branch_domains(s)),2)
        bad=cluster([2,3],[((0,6),(3,9))])
        self.assertTrue(narrowed_poses(ds,[ga,bad])['stats']['impossible'])


if __name__=='__main__':
    unittest.main()
