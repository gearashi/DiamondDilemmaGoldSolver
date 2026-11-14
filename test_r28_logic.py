"""Round 28 exactness cross-checks against physical enumeration and native SAT."""
from itertools import product
import random
import unittest

from test_constraint_models import pair_problem
from test_dependencies import requires_sat,requires_ortools
from search_fixtures import planted_gold
from search_problem import SearchProblem
from r28_logic_tail import binary_search,binary_tail,cut_bound
from r28_logic_sat import seam_hitting_set,cegar,checked_interpolant,verify_interpolant,determining_support,project_models
from r28_logic_reuse import direct_cnf,ClauseCache,TrailReplay,dominance,live_session
from r28_logic_region import regional_interpolant


def oracle(p,rows):
    return [(b,report['matched_edges'],report['valid']) for b in product(*rows)
            if len({v//3 for v in b})==p.n for report in (p.validate(list(b)),)]


class LogicTests(unittest.TestCase):
    def test_scc_against_enumeration_both_targets_and_unknown(self):
        rng = random.Random(19566)
        for multiple in (True,False):
            p = pair_problem(multiple_loops=multiple)
            for _ in range(40):
                rows = [rng.sample(r,rng.choice([1,2,2,3])) for r in p.domains]
                ans = oracle(p,rows)
                for target in ('gold','edges'):
                    ok = {b for b,score,gold in ans if (gold if target=='gold' else score==len(p.edges))}
                    result,board,_ = binary_search(p,rows,target,wide_limit=2)
                    self.assertEqual(result,'witness' if ok else 'infeasible')
                    if ok:
                        self.assertIn(tuple(board),ok)
            self.assertEqual(binary_search(p,p.domains,'edges',max_work=0)[0],'unknown')

    def test_binary_160(self):
        data,surface,board = planted_gold()
        p = SearchProblem(data,surface)
        rows = [[v,3*(v//3)+(v+1)%3] for v in board]
        masks = [sum(1<<v for v in row) for row in rows]
        status,codes,work = binary_tail(p,masks,'gold',lambda:None,seconds=2)
        self.assertEqual(status,'witness')
        self.assertTrue(p.validate(codes)['valid'])
        self.assertLess(work,50000)
        from r27_reason_tail import bounded_tail
        via_dfs = bounded_tail(p,masks,'gold',lambda:None,seconds=2)
        self.assertEqual(via_dfs[0],'witness')
        self.assertEqual(bounded_tail(p,masks,'gold',lambda:None,max_checks=0)[0],'unknown')
        illegal = list(masks)
        illegal[0] = 1 << (p.n*3+100) | 1
        with self.assertRaises(ValueError):
            bounded_tail(p,illegal,'gold',lambda:None,seconds=2)

    @requires_ortools
    def test_cut_rephasing_bound_and_unconstrained_inventory(self):
        rng = random.Random(11)
        for multiple in (True,False):
            p = pair_problem(multiple_loops=multiple)
            for _ in range(45):
                rows = [rng.sample(row,rng.choice([1,2])) for row in p.domains]
                results = oracle(p,rows)
                cut = cut_bound(p,rows)
                if results:
                    self.assertGreaterEqual(cut['upper'],max(s for _,s,_ in results))
                if cut['codes']:
                    self.assertEqual(cut['score'],p.validate(cut['codes'])['matched_edges'])
                    if cut['frustrated']==0:
                        self.assertEqual(cut['upper'],cut['score'])
        from test_constraint_dfs import tetrahedron_tiles
        tetra = SearchProblem(*tetrahedron_tiles())
        frustrated = 0
        for _ in range(300):
            rows = [rng.sample(r,2) for r in tetra.domains]
            cut = cut_bound(tetra,rows)
            relaxed = max(tetra.validate(list(b))['matched_edges'] for b in product(*rows))
            self.assertGreaterEqual(cut['upper'],relaxed)
            frustrated += int(cut['frustrated']>0)
        self.assertGreater(frustrated,0)
        data,surface,codes = planted_gold(size=1)
        p = SearchProblem(data,surface)
        # Covers frustrated cycles and parallel/full-board coupling.
        for _ in range(20):
            rows = [rng.sample(row,2) for row in p.domains]
            bound = cut_bound(p,rows)
            for _ in range(10):
                choices = [rng.choice(r) for r in rows]
                self.assertLessEqual(p.validate(choices)['matched_edges'],bound['upper'])

    @requires_sat
    @requires_ortools
    def test_core_reuse_check_and_optimum(self):
        p = pair_problem()
        for rows in ([[0,1],[3,4]],[[0,1],[5]],[[0],[0]]):
            result = seam_hitting_set(p,rows,previous=[[0],[0,1],[2]])
            expected = oracle(p,rows)
            if not expected:
                self.assertEqual(result['status'],'infeasible')
                continue
            self.assertEqual(result['upper'],max(s for _,s,_ in expected))
            self.assertEqual(result['status'],'optimal')
            replay = seam_hitting_set(p,rows,previous=result['cores'])
            self.assertEqual(replay['upper'],result['upper'])
            self.assertEqual(p.validate(result['codes'])['matched_edges'],result['upper'])

    @requires_sat
    def test_lazy_refinement_edge_gold_inventory(self):
        for multiple in (True,False):
            p = pair_problem(multiple_loops=multiple)
            for target in ('edges','gold'):
                result = cegar(p,p.domains,target)
                ok = [b for b,s,g in oracle(p,p.domains) if (g if target=='gold' else s==len(p.edges))]
                self.assertEqual(result['status'],'witness' if ok else 'infeasible')
                if ok:
                    self.assertIn(tuple(result['codes']),ok)
                if multiple and target=='gold':
                    self.assertGreater(result['gold_cuts'],0)
        self.assertEqual(cegar(p,[[0],[0]])['status'],'infeasible')

    @requires_sat
    def test_interpolant_checks_and_region(self):
        a = [[1,2],[-2,3],[-1,3]]
        b = [[-3,4],[-3,-4]]
        result = checked_interpolant(a,b,[3])
        self.assertEqual(result['status'],'separated')
        self.assertTrue(verify_interpolant(a,b,[3],result['clauses']))
        self.assertFalse(verify_interpolant(a,b,[3],[[-3]]))
        self.assertFalse(verify_interpolant(a,b,[3],[[4]]))
        self.assertEqual(checked_interpolant(a,b,[3],max_cubes=0)['status'],'unknown')
        self.assertEqual(checked_interpolant(a,[[3]],[3])['status'],'joined')
        for multiple in (True,False):
            p = pair_problem(multiple_loops=multiple)
            result = regional_interpolant(p,p.domains,[0],target='gold')
            self.assertEqual(result['status'],'separated' if multiple else 'joined')

    @requires_sat
    def test_support_reconstruction_and_model_scope(self):
        clauses = [[1,2],[-1,-2],[2,-3],[-2,3]]
        variables = [1,2,3,4]
        reduced = determining_support(clauses,variables)
        self.assertEqual(len(reduced),2)
        self.assertEqual(set(project_models(clauses,variables,variables)),set(project_models(clauses,reduced,variables)))

    def test_dominance_keeps_all_extensions_and_distinguishes_gold(self):
        for multiple in (True,False):
            p = pair_problem(multiple_loops=multiple)
            for target in ('edges','gold'):
                r = dominance(p,p.domains,[0,1],target=target)
                self.assertEqual(r['status'],'complete')
                ok = {b for b,s,g in oracle(p,p.domains) if (g if target=='gold' else s==len(p.edges))}
                self.assertEqual(len(r['kept']),int(bool(ok)))
                self.assertTrue(set(r['kept'])<=ok)
        self.assertEqual(dominance(p,p.domains,[0],max_assignments=1)['status'],'unknown')

    @requires_sat
    def test_biclique_encoding_and_component_cache(self):
        p = pair_problem()
        expected = {b for b,s,g in oracle(p,p.domains) if s==len(p.edges)}
        for use in ('pair','biclique','support','bitcode'):
            ids,clauses,_ = direct_cnf(p,p.domains,encoding=use)
            variables = [x for row in ids for x in row.values()]
            models = project_models(clauses,variables,variables)
            actual = {tuple(next(code for code,v in row.items() if v in model) for row in ids) for model in models}
            self.assertEqual(actual,expected)
        cache = ClauseCache()
        rows = [[1,2],[-1,-2],[3,4],[-3,-4]]
        self.assertIsNotNone(cache.solve(rows,[1]))
        self.assertIsNotNone(cache.solve(rows,[-1]))
        self.assertGreater(cache.hits,0)
        self.assertIsNone(cache.solve(rows,[-1,-2]))
        self.assertIsNone(cache.solve([[1],[-1]],[]))

    def test_replay_entailment_survives_only_applicable_reasons(self):
        clauses = [[-4,5],[-3,4],[-1,3],[1,2]]
        replay = TrailReplay(clauses)
        for assumptions in ([1,-8],[1,8],[-1],[],[-1,-2],[1,8]):
            self.assertEqual(replay.propagate(assumptions),TrailReplay(clauses).propagate(assumptions))
        self.assertGreater(replay.reused,0)

    @requires_sat
    def test_retained_worker_reconnects(self):
        from r28_logic_worker import session_probe
        result = session_probe([[1],[-1]])
        self.assertTrue(result['reconnected'])
        self.assertFalse(result['answer'])

    @requires_sat
    def test_pause_and_cold_replay_agree(self):
        for clauses in ([[1],[-1]],[[1,2],[-1,-2]]):
            report = live_session(clauses)
            self.assertEqual(report['runs']['warm']['answer'],report['runs']['cold']['answer'])
            self.assertFalse(report['full_native_state_serializable'])


if __name__=='__main__':
    unittest.main()
