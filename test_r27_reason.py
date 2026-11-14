"""Exhaustive and adverse checks for Round 27 scoped reasoning."""
from itertools import product
import random
import unittest

from r27_reason_algebra import algebra
from r27_reason_cost import DefectCores, cost_bound
from r27_reason_maxsat import maximize_seams
from r27_reason_orbits import orbit_choices
from r27_reason_parallel import concurrent_sat
from r27_reason_search import scoped_search
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles
from test_constraint_models import pair_problem
from test_constraint_separator import ring
from test_dependencies import requires_sat,requires_ortools


def answers(problem,rows):
    output = []
    for codes in product(*rows):
        if len({code//3 for code in codes}) != problem.n:
            continue
        checked = problem.validate(list(codes))
        output.append((codes,checked['matched_edges'],checked['valid']))
    return output


def corpus(problem,rng,count,density):
    for _ in range(count):
        yield [[x for x in row if rng.random()<density] for row in problem.domains]


class ScopedReasonTests(unittest.TestCase):
    @requires_sat
    def test_maxsat_full_objective_every_threshold_and_gold_cut(self):
        rng = random.Random(270016)
        for multi in (False,True):
            p = pair_problem(multiple_loops=multi)
            for rows in [p.domains,*corpus(p,rng,7,.45)]:
                boards = answers(p,rows)
                optimum = max((s for _,s,_ in boards),default=-1)
                for threshold in (None,0,1,2,3):
                    r = maximize_seams(p,domains=rows,target='edges',threshold=threshold,seconds=3)
                    if threshold is None:
                        self.assertEqual(r['edge_upper_bound'],optimum)
                    elif optimum < threshold:
                        self.assertLess(r['edge_upper_bound'],threshold)
                        self.assertGreaterEqual(r['edge_upper_bound'],optimum)
                    if r['codes'] is not None:
                        self.assertIn((tuple(r['codes']),r['matched_edges'],r['gold_valid']),boards)
                    if optimum >= (threshold or 0):
                        self.assertGreaterEqual(r['matched_edges'],optimum if threshold is None else threshold)
                gold = maximize_seams(p,domains=rows,target='gold',threshold=3,seconds=3)
                self.assertEqual(gold['gold_valid'],any(v for _,_,v in boards))
            if multi:
                self.assertGreater(maximize_seams(p,target='gold',threshold=3)['gold_cuts'],0)
        p = pair_problem()
        with self.assertRaises(ValueError):
            maximize_seams(p,seed_board=[0,0])
        self.assertEqual(maximize_seams(p,seconds=1,stop=lambda:True)['status'],'unknown')

    def check_multiloop_target(self,method):
        p = pair_problem(multiple_loops=True)
        edges = method(p,'edges')
        self.assertEqual(edges['status'],'edge_perfect')
        self.assertEqual(edges['target'],'edges')
        self.assertIs(edges['gold_valid'],False)
        self.assertFalse(p.validate(edges['codes'])['valid'])
        gold = method(p,'gold')
        self.assertNotEqual(gold['status'],'solved')
        self.assertIs(gold['gold_valid'],False)

    def test_edge_perfect_status_is_distinct_in_scoped_modes(self):
        for mode in ('mrv','backdoor','strong','dynamic','backjump','orbit'):
            self.check_multiloop_target(lambda p,t:scoped_search(p,target=t,mode=mode))

    @requires_sat
    def test_edge_perfect_status_is_distinct_on_native_and_seed_returns(self):
        self.check_multiloop_target(lambda p,t:concurrent_sat(p,target=t))
        self.check_multiloop_target(lambda p,t:maximize_seams(p,target=t))
        p = pair_problem(multiple_loops=True)
        codes = maximize_seams(p,target='edges')['codes']
        self.check_multiloop_target(lambda p,t:maximize_seams(p,target=t,seed_board=codes))

    def test_additive_defect_cores_shifts_inheritance_and_corruption(self):
        p = pair_problem()
        rng = random.Random(270017)
        positive = 0
        for rows in corpus(p,rng,70,.35):
            boards = answers(p,rows)
            if not boards:
                continue
            r = cost_bound(p,domains=rows)
            best = max(score for _,score,_ in boards)
            self.assertLessEqual(r['lower_bound'],len(p.edges)-best)
            positive += r['lower_bound'] > 0
            child = [row[:1] for row in rows]
            child_result = cost_bound(p,domains=child,inherited=r['certificate'])
            self.assertGreaterEqual(child_result['vac_bound'],r['vac_bound'])
        self.assertGreater(positive,0)
        forged = DefectCores(p.fingerprint,tuple(tuple(row) for row in p.domains),((0,),))
        with self.assertRaises(ValueError):
            cost_bound(p,inherited=forged)

    def test_cutset_backjump_dynamic_strong_all_match_full_enumeration(self):
        rng = random.Random(270019)
        for p in (pair_problem(),pair_problem(multiple_loops=True),SearchProblem(*tetrahedron_tiles()),ring(4)):
            for rows in [p.domains,*corpus(p,rng,8,.5)]:
                boards = answers(p,rows)
                for target in ('edges','gold'):
                    valid = {b for b,s,g in boards if (g if target=='gold' else s==len(p.edges))}
                    for mode in ('mrv','strong','backdoor','backjump','dynamic'):
                        with self.subTest(mode=mode,target=target,rows=rows):
                            r = scoped_search(p,domains=rows,target=target,mode=mode)
                            if valid:
                                self.assertIn(tuple(r['codes']),valid)
                            else:
                                self.assertEqual(r['status'],'infeasible')
        self.assertEqual(scoped_search(ring(4),max_checks=1)['status'],'unknown')

    def test_verified_same_residual_orbits_and_caps(self):
        rng = random.Random(270020)
        for p in (ring(4),pair_problem(),pair_problem(multiple_loops=True)):
            for rows in [p.domains,*corpus(p,rng,6,.7)]:
                result = orbit_choices(p,domains=rows)
                c = result['cell']
                valid = {b[c] for b,_,gold in answers(p,rows) if gold}
                for v,rep in result['suppressed'].items():
                    self.assertEqual(v in valid,rep in valid)
                self.assertEqual(bool(valid),bool(valid.intersection(result['choices'])))
        self.assertEqual(len(orbit_choices(ring(4))['choices']),1)
        result = scoped_search(ring(4),mode='orbit')
        self.assertTrue(ring(4).validate(result['codes'])['valid'])
        self.assertGreater(result['stats']['orbit_mappings'],0)
        self.assertEqual(orbit_choices(ring(4),work_limit=1)['suppressed'],{})

    def compare_algebra(self,method):
        rng = random.Random(270021)
        p = pair_problem()
        certified = 0
        for rows in [p.domains,*corpus(p,rng,12,.3)]:
            feasible = any(s==len(p.edges) for _,s,_ in answers(p,rows))
            for degree in (1,2):
                report = algebra(p,domains=rows,degree=degree,method=method)
                if report['status']=='certified_infeasible':
                    self.assertFalse(feasible)
                    self.assertTrue(report['certificate'])
                    certified += 1
        self.assertGreater(certified,0)
        self.assertEqual(algebra(p,max_vars=1)['status'],'skipped')

    def test_lifting_separates_odd_cycles_and_edges_only_remainder(self):
        from bench_r27_reason import odd_cycles
        p,rows = odd_cycles()
        self.assertFalse(any(s==len(p.edges) for _,s,_ in answers(p,rows)))
        self.assertEqual(algebra(p,domains=rows,method='gf2',degree=1)['status'],'unknown')
        self.assertEqual(algebra(p,domains=rows,method='gf2',degree=2)['status'],'certified_infeasible')
        r = scoped_search(ring(5),mode='backdoor')
        self.assertEqual(r['status'],'solved')
        self.assertEqual(r['stats'].get('matching_remainders'),1)

    def test_polynomial_certificates(self):
        self.compare_algebra('gf2')

    @requires_ortools
    def test_lp_certificates(self):
        self.compare_algebra('lp')
        from bench_r27_reason import odd_cycles
        p,rows = odd_cycles()
        self.assertEqual(algebra(p,domains=rows,method='lp',degree=1)['status'],'unknown')
        self.assertEqual(algebra(p,domains=rows,method='lp',degree=2)['status'],'certified_infeasible')

    @requires_sat
    def test_concurrent_helper_and_single_owner_agree_with_oracle(self):
        rng = random.Random(270024)
        for p in (pair_problem(),pair_problem(multiple_loops=True),ring(4)):
            for rows in [p.domains,*corpus(p,rng,4,.55)]:
                valid = {b for b,_,g in answers(p,rows) if g}
                for pilots in (0,2):
                    r = concurrent_sat(p,domains=rows,pilots=pilots)
                    if valid:
                        self.assertIn(tuple(r['codes']),valid)
                    else:
                        self.assertEqual(r['status'],'infeasible')
        self.assertEqual(concurrent_sat(pair_problem(),seconds=0)['status'],'unknown')

