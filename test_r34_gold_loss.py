"""Cross-check the compiled joint oracle and ordering of filtered suffix buckets."""
from collections import Counter
from contextlib import closing
from importlib.util import find_spec
from itertools import combinations, product
from math import inf
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
from unittest import skipUnless, TestCase
from unittest.mock import patch

from r32_gold import run_syndrome
from r32_gold_syndrome import OutsideSyndromes
from r33_gold_refine import IntegerCoreStore, IntegerPatternFilter
from test_r33_gold_lookup import fixture, phase_costs
from test_r33_gold_refine import fixture as integer_fixture, exact_costs


class LossIndexTests(TestCase):
    def test_every_joint_decision_price_against_enumerated_tile_phases(self):
        for seed in (1, 11):
            problem, reference, groups, query = fixture(seed)
            costs = phase_costs(problem, set(reference), query)
            for assignment in product((0,1), repeat=len(query.outside)):
                indices = tuple(i for i, v in enumerate(assignment) if v)
                for budget in range(len(reference)+1):
                    self.assertEqual(query.joint(set(indices),budget,inf),
                                     costs.get(indices,inf) <= budget, (seed,indices,budget))
            with self.assertRaises(TimeoutError):
                query.joint(set(),12,0)

    def test_second_anchor_keeps_order_prices_and_skips_only_unowned_tails(self):
        tails = list(combinations(range(10),2))
        bins = {s:sum(1<<i for i,(a,b) in enumerate(tails) if (a^b)==s) for s in range(16)}
        prices = {0:3, 1:2, 3:1, 4:2, 10:5}
        for prefix in (0,1,4,9,12):
            for available in (0,1,2,7):
                for last in (-1,0,5,8):
                    expected = [t for t in tails if prices.get(t[0]^t[1]^prefix,inf)<=available+1]
                    counts = Counter()
                    emitted = list(OutsideSyndromes._secondary_tails(tails,bins,prefix,prices,
                                                                      available,last,counts))
                    self.assertEqual(emitted, [t for t in expected if t[0]>last])
                    self.assertEqual(counts["unowned_tail_visits_skipped"],
                                     sum(t[0]<=last for t in expected))


@skipUnless(find_spec("ortools"), "Integer refinement needs optional OR-Tools")
class IntegerCoreTests(TestCase):
    def test_stored_false_claim_is_checked_and_scopes_do_not_mix(self):
        p,ref,groups,req,out = integer_fixture(True,True)
        args = (p,ref,groups,req,out,8)
        costs = exact_costs(p,ref,groups,req,out)
        real = next(iter(indices for indices,cost in costs.items() if cost<=8))
        with sqlite3.connect(":memory:") as db:
            saved = IntegerCoreStore(db,"scope-A")
            saved.remember(0,0)  # invalid suggestion: would exclude every assignment
            self.assertEqual(list(IntegerCoreStore(db,"scope-B").read()),[])
            checker = IntegerPatternFilter(*args,store=saved)
            self.assertTrue(checker(real,inf))
            self.assertEqual(checker.work["integer_core_replays"],1)
            self.assertEqual(checker.verified,set())
            self.assertNotIn((0,0),checker.pending)
            self.assertEqual(list(checker.model.proto.assumptions),[])

    def test_replayed_negative_subsumes_other_patterns_and_keeps_unknown_open(self):
        p,ref,groups,req,out = integer_fixture(True,True)
        args = (p,ref,groups,req,out,5)
        choices = [frozenset(i for i in range(len(out)) if bits & (1<<i)) for bits in range(1<<len(out))]
        with sqlite3.connect(":memory:") as db:
            store = IntegerCoreStore(db,"scope")
            first = IntegerPatternFilter(*args,store=store)
            truth = [first(c,inf) for c in choices]
            self.assertIn(True,truth)
            self.assertIn(False,truth)
            self.assertGreater(first.work["integer_native_calls_avoided"],0)
            warm = IntegerPatternFilter(*args,store=store)
            self.assertTrue(warm.pending)
            failing = choices[truth.index(False)]
            with patch.object(warm.solver,"solve",return_value=warm.cp.UNKNOWN):
                with self.assertRaises(TimeoutError):
                    warm(failing,inf)
            self.assertFalse(warm.verified)
            self.assertEqual([warm(c,inf) for c in choices],truth)
            self.assertGreater(warm.work["integer_core_replays"],0)


class CrossBudgetPageTests(TestCase):
    def test_only_negative_weaker_pages_can_cover_a_stronger_cap(self):
        p,ref,g,q = fixture()
        with TemporaryDirectory() as folder:
            path = Path(folder)/"loss.sqlite"
            def check(cap,layer,db=path,refine=False):
                return run_syndrome(p,ref,g,cap,[layer],db,0,page=1,refine_integer=refine)
            self.assertEqual(check(6,4)["status"],"INFEASIBLE")
            self.assertEqual(check(7,4)["status"],"PHASE_RELAXATION_FEASIBLE")
            self.assertEqual(check(8,3)["status"],"INFEASIBLE")
            readonly = path.as_uri()+"?mode=ro"
            with closing(sqlite3.connect(path)) as writer:
                writer.execute("BEGIN IMMEDIATE")
                with patch.object(OutsideSyndromes,"__init__",side_effect=AssertionError("compiled inherited proof")):
                    for refined in (False,True):
                        warmed = check(7,3,readonly,refined)
                        self.assertEqual((warmed["status"],warmed["work"],warmed["new_pages"]),("INFEASIBLE",{},0))
            # A higher-budget hit cannot be reported at the lower budget either.
            self.assertEqual(check(5,4)["status"],"INFEASIBLE")

    def test_integer_negative_is_invisible_to_plain_search_and_high_caps_are_discovered(self):
        p,ref,g,q = fixture()
        with TemporaryDirectory() as folder:
            path = Path(folder)/"loss.sqlite"
            with patch("r33_gold_refine.make_integer_filter",return_value=lambda *_:False):
                refined = run_syndrome(p,ref,g,100,[4],path,0,1,refine_integer=True)
                self.assertEqual(refined["status"],"INFEASIBLE")
            with patch.object(OutsideSyndromes,"__init__",side_effect=AssertionError):
                read_only = path.as_uri()+"?mode=ro"
                warmed = run_syndrome(p,ref,g,8,[4],read_only,0,1,refine_integer=True)
                self.assertEqual(warmed["work"],{})
            plain = run_syndrome(p,ref,g,7,[4],path,0,1)
            self.assertEqual(plain["status"],"PHASE_RELAXATION_FEASIBLE")
