"""Host mirror and independent complete-pose controls for classic GPU future mates."""
from itertools import combinations
from random import Random
import unittest

import numpy as np
from r27_structure_fixtures import planted
from r31_geographic import physical_bonds
from r35_gpu_mates import host_conflict,pose_table
from validator import reverse_mask


class FuturePoseTests(unittest.TestCase):
    def test_every_component_anchor_and_known_legal_partial_board(self):
        p,_=planted(1,3,71)
        joins=physical_bonds(p,[3*t for t in range(p.n)])
        faces=p.masks
        rev=np.array([[reverse_mask(int(m)) for m in row] for row in faces])
        rng=Random(35)
        for selected in (joins,joins[:6],joins[::3]):
            for cap in (1,2000000):
                table=pose_table(p.neighbors,p.sides,selected,max_words=cap)
                for _ in range(40):
                    known=set(rng.sample(range(p.n),rng.randrange(p.n)))
                    board=[3*c if c in known else -1 for c in range(p.n)]
                    for c in set(range(p.n))-known:
                        self.assertEqual(host_conflict(table,board,c,3*c,p.neighbors,p.sides,faces,rev),-1)
        self.assertFalse(pose_table(p.neighbors,p.sides,()).size)

    def test_two_future_owners_competing_for_same_empty_cell(self):
        p,_=planted(1,3,103)
        center=0
        a,b=map(int,p.neighbors[center,:2])
        sa=list(p.neighbors[a]).index(center);sb=list(p.neighbors[b]).index(center)
        required={(sa,3),(6+sb,9)}
        table=pose_table(p.neighbors,p.sides,required)
        board=[-1]*p.n;board[a]=0
        blank=np.zeros_like(p.masks)
        self.assertEqual(host_conflict(table,board,b,6,p.neighbors,p.sides,blank,blank),a)



if __name__=='__main__':
    unittest.main()
