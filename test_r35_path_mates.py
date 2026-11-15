"""Reference placements and independent completed boards for geographic path mates."""
from importlib.util import find_spec
from itertools import permutations, product
import unittest
from time import monotonic

from bench_r35_mixed_sewings import ring_donors
from r27_structure_filters import domain_bits
from r30_gold import search
from r31_geographic import physical_bonds
from r35_path_mates import PathMates, alternatives, propagate
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face


class PathMateTests(unittest.TestCase):
    def test_complete_two_tile_poses_and_shared_edges(self):
        p=SearchProblem(simple_tiles(),double_face())
        checker=PathMates(p)
        ds=domain_bits(p)
        boards=[]
        for order in permutations(range(p.n)):
            for rot in product(range(3),repeat=p.n):
                codes=tuple(3*t+r for t,r in zip(order,rot))
                if p.validate(codes)["matched_edges"]==len(p.edges):
                    boards.append((codes,frozenset(physical_bonds(p,codes))))
        self.assertTrue(boards)
        for coding,pairs in boards:
            for pair in pairs:
                required={pair}
                for restricted in (ds,tuple(1<<code for code in coding)):
                    expected={all_pairs-required for board,all_pairs in boards if pair in all_pairs and
                              all(mask>>code&1 for code,mask in zip(board,restricted))}
                    patterns,_,_=checker.patterns(required,restricted)
                    self.assertEqual(patterns,expected)
                    for blocked in (set(),next(iter(expected))):
                        status,forced,absent,tables,stats=propagate(p,restricted,required,blocked)
                        live=[row for row in expected if not row&blocked]
                        self.assertEqual(status=="INFEASIBLE",not live)
                        for row in live:
                            self.assertTrue(forced<=row)
                            self.assertFalse(absent&row)
                            for columns,options in tables:
                                self.assertIn(tuple(x in row for x in columns),
                                              {tuple(x in q for x in columns) for q in options})

    def test_minimal_positive_alternatives_and_no_vacuous_claims(self):
        a,b,c=(3,6),(4,8),(5,10)
        rows={frozenset((a,b)),frozenset((b,c))}
        self.assertEqual(set(alternatives(rows)),{frozenset((b,)),frozenset((a,c))})
        self.assertEqual(alternatives(set()),())
        self.assertEqual(alternatives({*rows,frozenset()}),())

    def test_expired_scan_and_conflicting_side_ownership(self):
        p,donors,boards=ring_donors(103)
        pairs=set(map(tuple,donors[0]['pairs']))
        outcome=propagate(p,domain_bits(p),pairs,(),monotonic()-1)
        self.assertEqual(outcome[:3],("LIMIT",set(),set()))
        self.assertEqual(propagate(p,domain_bits(p),[(0,3),(0,6)],())[0],"SKIPPED")
        with self.assertRaisesRegex(ValueError,"embed"):
            search(p,geography=False,path_mates="units")

    @unittest.skipUnless(find_spec("ortools"),"requires CP-SAT")
    def test_public_modes_preserve_planted_gold_under_partial_bonds(self):
        for seed in (103,817):
            p,donors,boards=ring_donors(seed)
            for d,board in zip(donors,boards):
                fixed=sorted(d["pairs"])[::2]
                for mode in ("units","table"):
                    out=search(p,geography=True,bonds=fixed,domains=tuple(1<<c for c in board),
                               path_mates=mode,seconds=10)
                    self.assertEqual(out["status"],"GOLD")
                    self.assertTrue(p.validate(out["codes"])["valid"])
