"""Exact differential and cancellation checks for incremental affine snapshots."""
import copy
import random
import unittest
from constraint_algebra import AffinePruner
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,brute
from test_constraint_sat_domains import admitted


def affine_fixture():
    problem=SearchProblem(*tetrahedron_tiles())
    solutions=brute(problem,'edges');witness=min(solutions)
    root=[sum(1<<v for v in row) for row in problem.domains];states=[root.copy()]
    for _ in range(3):
        for cell in range(problem.n):
            choices=[v for v in problem.domains[cell] if states[-1][cell]>>v&1 and v!=witness[cell]]
            if choices:
                current=states[-1].copy();current[cell]&=~(1<<choices[0]);states.append(current)
    return problem,solutions,states


def compare_affine_updates():
    problem,solutions,states=affine_fixture();report={}
    for enabled in (False,True):
        helper=AffinePruner(problem,incremental=enabled);answers=[]
        for original in states:
            current=original.copy();helper.prune(current,max_candidates=0)
            assert current==original and admitted(solutions,current)
            answers.append(current)
        report['incremental' if enabled else 'rebuild']={'stats':dict(helper.stats),'answers':answers}
    assert report['rebuild']['answers']==report['incremental']['answers']
    for row in report.values():row.pop('answers')
    return {'contexts':len(states),'independent_edge_solutions':len(solutions),**report}


class IncrementalAffineTests(unittest.TestCase):
    def test_reuses_unchanged_equations_with_identical_consistent_bases(self):
        report=compare_affine_updates();old=report['rebuild']['stats'];new=report['incremental']['stats']
        self.assertLess(new['affine_equations'],old['affine_equations'])
        self.assertLess(new['affine_xors'],old['affine_xors'])
        self.assertEqual(new['affine_full_builds'],1)
        self.assertEqual(new['affine_incremental_builds'],report['contexts']-1)

    def test_random_narrowing_widening_and_rollback_matches_fresh_bases(self):
        problem,solutions,states=affine_fixture();root=states[0];rng=random.Random(250009)
        helper=AffinePruner(problem,incremental=True);previous=root.copy()
        for index in range(180):
            if index%4==0:context=root.copy()
            elif index%4==3:context=states[index%len(states)].copy()
            else:context=[sum(1<<v for v in problem.domains[c] if previous[c]>>v&1 and rng.random()<.85)
                          for c in range(problem.n)]
            expected=context.copy();actual=context.copy()
            AffinePruner(problem).prune(expected,max_candidates=128)
            helper.prune(actual,max_candidates=128)
            self.assertEqual(actual,expected,(index,context))
            self.assertEqual(admitted(solutions,actual),admitted(solutions,context))
            previous=context

    def test_ancestor_snapshots_survive_backtracking_and_uncached_widening(self):
        problem,_,states=affine_fixture();helper=AffinePruner(problem,incremental=True)
        for row in states[:4]:helper.prune(row.copy(),max_candidates=0)
        basis=copy.deepcopy(helper.cache[tuple(states[1])]['basis'])
        helper.prune(states[1].copy(),max_candidates=0)
        self.assertEqual(helper.cache[tuple(states[1])]['basis'],basis)
        self.assertEqual(helper.stats['affine_cache_hits'],1)
        for row in states[4:]:helper.prune(row.copy(),max_candidates=0)
        self.assertNotIn(tuple(states[0]),helper.cache)
        old=helper.stats['affine_full_builds'];helper.prune(states[0].copy(),max_candidates=0)
        self.assertEqual(helper.stats['affine_full_builds'],old+1)
        self.assertLessEqual(len(helper.cache),4)

    def test_interrupted_extension_never_mutates_ancestor_or_callers_domains(self):
        problem,solutions,states=affine_fixture()
        for limit in (1,2,3,10,30,80,150,300,600):
            helper=AffinePruner(problem,incremental=True)
            helper.prune(states[0].copy(),max_candidates=0)
            parent=copy.deepcopy(helper.cache[tuple(states[0])]);current=states[1].copy();ticks=[0]
            def stop():
                ticks[0]+=1
                if ticks[0]==limit:raise InterruptedError('stop')
            try:helper.prune(current,stop,max_candidates=128)
            except InterruptedError:self.assertEqual(current,states[1])
            self.assertEqual(helper.cache[tuple(states[0])],parent)
            resumed=states[1].copy();helper.prune(resumed,max_candidates=128)
            expected=states[1].copy();AffinePruner(problem).prune(expected,max_candidates=128)
            self.assertEqual(resumed,expected)
            self.assertEqual(admitted(solutions,resumed),admitted(solutions,states[1]))

    def test_binding_and_domain_validation(self):
        problem,_,states=affine_fixture();helper=AffinePruner(problem,incremental=True)
        for current in ([1],[-1]*4,[True]*4,[1<<20]*4):
            with self.assertRaises(ValueError):helper.prune(current)
        problem.masks[0,0]^=1
        with self.assertRaisesRegex(ValueError,'binding'):helper.prune(states[0].copy())
        with self.assertRaises(ValueError):AffinePruner(problem,incremental=1)

    def test_actual_dfs_uses_incremental_affine_and_preserves_the_solution(self):
        from constraint_dfs import solve
        from search_options import normalize_options
        self.assertTrue(normalize_options({'incremental_affine':True})['affine_reasoning'])
        problem,solutions,_=affine_fixture()
        result=solve(problem,seconds=10,target='edges',options={'incremental_affine':True,'native':'python',
                     'lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertTrue(result['complete']);self.assertIn(tuple(result['codes']),solutions)
        self.assertGreater(result['stats']['affine_checks'],0)


if __name__=='__main__':unittest.main()
