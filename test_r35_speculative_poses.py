import unittest
from collections import Counter
from types import SimpleNamespace

import numpy as np

from r27_incumbent import BoardModel
from r32_fresh import _poses
from r35_speculative_poses import PoseJoins,conflicting_groups,rollback_index
from search_fixtures import planted_gold
from search_problem import SearchProblem


class SpeculativeTests(unittest.TestCase):
    def test_joint_tables_equal_full_root_traversal_with_restrictions_and_different_orders(self):
        tiles,geo,reference=planted_gold(3540,1,5)
        model=BoardModel(SearchProblem(tiles,geo))
        board=np.array(reference)
        edges=[]
        for a,s,b,t in model.problem.edges:
            left=3*(board[a]//3)+(s-board[a]%3)%3
            right=3*(board[b]//3)+(t-board[b]%3)%3
            edges.append(tuple(sorted((int(left),int(right)))))
        for restrict in (False,True):
            if restrict:
                rng=np.random.default_rng(90)
                model.legal &= rng.random(model.legal.shape)<.36
                model.legal[np.arange(model.n),board]=True
            a=SimpleNamespace(model=model,groups=[])
            joins=PoseJoins(a)
            tree=[edges[0]]
            group={x//3 for x in tree[0]}
            for size in range(2,5):
                edge=next(e for e in edges if len(group&{x//3 for x in e})==1)
                tree.append(edge)
                group|={x//3 for x in edge}
                expected=_poses(model,group,set(tree))
                self.assertTrue(expected)
                observed=joins.replay(tree)
                self.assertEqual(observed[0][1],expected)
                backwards=joins.replay(tree[::-1])
                self.assertEqual(backwards[0][1],expected)
            self.assertGreater(joins.work['joined_tables_reused'],0)

    def test_conflict_components_preserve_irrelevant_later_choice_and_incomplete_is_unknown(self):
        first,last=(0,3),(12,15)
        groups=[({0,1},[((0,0),(1,3))],(first,)),
                ({2,3},[((1,6),(2,9))],()),
                ({4,5},[((4,12),(5,15))],(last,))]
        stats=Counter()
        self.assertEqual(set(conflicting_groups(groups,100,stats)),{0,1})
        self.assertEqual(rollback_index([first,last],groups,100,stats),0)
        self.assertEqual(stats['later_choices_preserved'],1)
        unproved=Counter()
        self.assertEqual(rollback_index([first,last],groups,0,unproved),1)
        self.assertEqual(unproved['packing_cores'],0)
        self.assertEqual(unproved['unproved_packing_budgets'],1)
        # An incomplete attempt must never become a negative cache entry.
        groups[1]=({2,3},[((1,6),(2,9)),((2,6),(3,9))],())
        self.assertFalse(conflicting_groups(groups,0,Counter()))
        self.assertFalse(conflicting_groups(groups,100,Counter()))
        self.assertTrue(set(dict(groups[0][1][0])).isdisjoint(dict(groups[1][1][1])))


if __name__=='__main__':
    unittest.main()
