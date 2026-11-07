"""Independent edge-solution checks for support-cover and witness reuse."""
import random
import unittest
from constraint_sat_assist import SatAssist
from test_constraint_sat_assist import HAS_SAT
from test_constraint_sat_domains import fixture, admitted
from test_constraint_dfs import brute


@unittest.skipUnless(HAS_SAT, 'optional python-sat missing')
class SatRound24Tests(unittest.TestCase):
    def helper(self, problem, **options):
        initial=[sum(1<<v for v in row) for row in problem.domains]
        h=SatAssist(problem,initial,**options)
        self.addCleanup(h.close)
        self.assertTrue(h.prepare(seconds=5))
        return h

    def test_support_cover_exact_supported_values_across_random_contexts(self):
        problem, domains=fixture(); solutions=brute(problem,'edges')
        h=self.helper(problem,support_cover=True,witness_pool=True)
        rng=random.Random(240012)
        for case in range(35):
            current=domains if case==0 else [sum(1<<c for c in row if rng.random()<.8) for row in problem.domains]
            expected=admitted(solutions,current)
            result=h.filter_domains(current,seconds=2,max_candidates=128)
            self.assertEqual(admitted(solutions,result),expected)
            if expected:
                supported=tuple(sum(1<<v for v in {row[c] for row in expected}) for c in range(problem.n))
                self.assertEqual(result,supported)
            else:self.assertIn(0,result)
        self.assertGreater(h.stats['sat_assist_support_cover_prunes'],0)

    def test_support_cover_uses_fewer_queries_than_single_placement_scans(self):
        problem, domains=fixture()
        baseline=self.helper(problem)
        cover=self.helper(problem,support_cover=True)
        a=baseline.filter_domains(domains,seconds=3,max_candidates=128)
        b=cover.filter_domains(domains,seconds=3,max_candidates=128)
        self.assertEqual(a,b)
        self.assertLess(cover.stats['sat_assist_domain_calls'],baseline.stats['sat_assist_domain_calls'])

    def test_pool_revalidates_vectors_in_other_contexts_and_new_gold_cuts(self):
        problem, domains=fixture(); solutions=brute(problem,'edges')
        h=self.helper(problem,support_cover=True,witness_pool=True)
        h.filter_domains(domains,seconds=3,max_candidates=128)
        self.assertTrue(h._witnesses)
        witness=next(iter(h._witnesses))
        widened=list(h.initial)
        result=h.filter_domains(widened,seconds=3,max_candidates=128)
        self.assertGreater(h.stats['sat_assist_witness_pool_hits'],0)
        self.assertEqual(admitted(solutions,result),admitted(solutions,widened))
        supported=[0]*problem.n
        narrow=list(h.initial);narrow[0]&=~(1<<witness[0])
        h._pool_supports(narrow,supported,lambda:None)
        self.assertFalse(supported[0]&(1<<witness[0]))
        h._gold_known.add(tuple(enumerate(witness)))
        h._witnesses.clear();h._witnesses[witness]=None
        supported=[0]*problem.n;h._pool_supports(h.initial,supported,lambda:None)
        self.assertEqual(supported,[0]*problem.n)

    def test_cover_selectors_are_disabled_on_cancellation(self):
        problem,domains=fixture()
        h=self.helper(problem,support_cover=True)
        old=h._model_supports
        def interrupted(*args):raise InterruptedError('stop after SAT')
        h._model_supports=interrupted
        with self.assertRaises(InterruptedError):h.filter_domains(domains,seconds=3,max_candidates=128)
        selector=h.ids
        self.assertFalse(h.solver.solve(assumptions=[selector]))
        h._model_supports=old
        result=h.filter_domains(h.initial,seconds=3,max_candidates=128)
        self.assertEqual(admitted(brute(problem,'edges'),result),brute(problem,'edges'))

    def test_captured_base_excludes_temporary_support_selectors(self):
        problem,domains=fixture();h=self.helper(problem,capture_clauses=True,support_cover=True)
        before=h.proof_base()
        h.filter_domains(domains,seconds=2,max_candidates=128)
        self.assertEqual(h.proof_base(),before)
        self.assertGreater(len(before['clauses']),0)
        with self.assertRaises(ValueError):self.helper(problem).proof_base()

if __name__=='__main__':unittest.main()
