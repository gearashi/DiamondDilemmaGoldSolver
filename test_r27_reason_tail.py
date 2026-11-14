"""Production DFS tail integration, scoped failure, Stop/resume and 160-cell fixtures."""
from functools import partial
import random
import unittest
from unittest.mock import patch

from constraint_dfs import solve
from r27_reason_tail import bounded_tail
from test_constraint_dfs import tetrahedron_tiles
from test_constraint_models import pair_problem
from test_constraint_separator import ring,permutation_ring
from test_gpu_domain_round25 import planted160_fixture
from test_r27_reason import answers,corpus
from search_problem import SearchProblem


class DFSTailIntegration(unittest.TestCase):
    def test_normal_entry_and_accounting(self):
        p = ring(6)
        baseline = solve(p,seconds=5,options={'easy_remainder':False})
        accelerated = solve(p,seconds=5,options={'easy_remainder':True})
        self.assertTrue(p.validate(accelerated['codes'])['valid'])
        self.assertLess(accelerated['nodes'],baseline['nodes'])
        self.assertEqual(accelerated['stats']['easy_remainder_witnesses'],1)
        self.assertGreater(accelerated['stats']['easy_remainder_work'],0)
        p = pair_problem(multiple_loops=True)
        r = solve(p,seconds=5,target='edges',options={'easy_remainder':True})
        self.assertEqual(r['status'],'edge_perfect')
        self.assertFalse(r['validation']['valid'])

    def test_failure_uses_full_scope_and_unknown_resumes_baseline(self):
        p = permutation_ring([(0,1,2)]*4)
        options = dict(easy_remainder=True,native='python',strand_connectivity=False,
                       biconnected=False,regional_cache=False,separator_regions=False)
        gold = solve(p,target='gold',seconds=5,options=options)
        self.assertEqual(gold['status'],'infeasible')
        self.assertEqual(gold['stats']['easy_remainder_infeasible'],1)
        self.assertFalse(any(ok for _,_,ok in answers(p,p.domains)))
        edges = solve(p,target='edges',seconds=5,options=options)
        self.assertEqual(edges['status'],'edge_perfect')
        p = ring(4)
        with patch('r27_reason_tail.bounded_tail',partial(bounded_tail,max_checks=0)):
            r = solve(p,seconds=5,options=options)
        self.assertEqual(r['status'],'solved')
        self.assertGreater(r['stats']['easy_remainder_unknown'],0)
        self.assertTrue(p.validate(r['codes'])['valid'])
        with patch('r27_reason_tail.bounded_tail',return_value=('witness',[999]*p.n,1)):
            with self.assertRaisesRegex(RuntimeError,'escapes'):
                solve(p,seconds=5,options=options)

    def test_stop_inside_query_keeps_owned_frame(self):
        p = ring(4)
        stopped = []
        def stop_inside(problem,masks,target,check):
            stopped.append(True)
            return bounded_tail(problem,masks,target,check)
        with patch('r27_reason_tail.bounded_tail',stop_inside):
            r = solve(p,seconds=5,stop=lambda:bool(stopped),options={'native':'python','easy_remainder':True})
        self.assertEqual(r['status'],'stopped')
        self.assertTrue(r['checkpoint']['stack'])
        result = solve(p,seconds=5,resume=r['checkpoint'])
        self.assertTrue(p.validate(result['codes'])['valid'])

    def test_random_scopes_and_original_size_tail(self):
        p = SearchProblem(*tetrahedron_tiles())
        for rows in corpus(p,random.Random(271822),24,.38):
            masks = [sum(1<<x for x in row) for row in rows]
            for target in ('gold','edges'):
                result = bounded_tail(p,masks,target,lambda:None,seconds=2)
                if result is None or result[0]=='unknown':
                    continue
                feasible = {b for b,s,g in answers(p,rows) if (g if target=='gold' else s==len(p.edges))}
                if feasible:
                    self.assertIn(tuple(result[1]),feasible)
                else:
                    self.assertEqual(result[0],'infeasible')
        p = planted160_fixture()
        masks = [sum(1<<x for x in row) for row in p.domains]
        status,codes,work = bounded_tail(p,masks,'gold',lambda:None,seconds=2)
        self.assertEqual(status,'witness')
        self.assertTrue(p.validate(codes)['valid'])
        self.assertLess(work,8000)

