from test_constraint_sat_assist import HAS_SAT
import itertools,random,unittest
from constraint_sat_simplify_round26 import SimplifiedCNF,replay,canonical
from constraint_sat_session import PersistentSatSession
from constraint_work import WorkScope
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,multiple_loops,double_face,brute


def models(clauses,variables):
    for bits in itertools.product((False,True),repeat=len(variables)):
        values=dict(zip(variables,bits))
        if all(any(values[abs(lit)]==(lit>0) for lit in row) for row in clauses):yield values


class SimplificationTests(unittest.TestCase):
    def test_exhaustive_random_protected_projection_and_every_lift(self):
        rng=random.Random(260502)
        for case in range(160):
            n=rng.randrange(2,7);variables=list(range(1,n+1));protected=frozenset(rng.sample(variables,rng.randrange(n+1)))
            cnf=[tuple(rng.choice((-1,1))*v for v in rng.sample(variables,rng.randrange(1,min(4,n)+1))) for _ in range(rng.randrange(1,13))]
            original=canonical(cnf);result=SimplifiedCNF(cnf,protected,max_variables=8,max_resolvents=128,max_subsumption=64)
            before={tuple(m[v] for v in sorted(protected)) for m in models(original,variables)}
            after=set()
            for assignment in models(result.clauses,variables):
                after.add(tuple(assignment[v] for v in sorted(protected)))
                lifted=result.reconstruct([v if assignment[v] else -v for v in variables])
                values={abs(v):v>0 for v in lifted}
                self.assertTrue(all(any(values[abs(lit)]==(lit>0) for lit in row) for row in original))
                self.assertEqual(tuple(values[v] for v in sorted(protected)),tuple(assignment[v] for v in sorted(protected)))
            self.assertEqual(before,after,case)
            self.assertEqual(replay(cnf,protected,result.steps)[0],result.projection_clauses)

    def test_checked_rules_reject_forged_equivalence_protected_elimination_and_model(self):
        cnf=[(-3,1),(3,-1),(3,2),(-3,-2,4),(-4,2)]
        result=SimplifiedCNF(cnf,{1,2})
        self.assertGreater(result.stats['simplify_equivalences'],0)
        with self.assertRaises(ValueError):replay(cnf,{1,2},[('equiv',3,2)])
        with self.assertRaises(ValueError):replay(cnf,{1,2},[('eliminate',1)])
        with self.assertRaises(ValueError):replay(cnf,{1,2},[('subsume',(1,),(3,2))])
        with self.assertRaises(ValueError):result.reconstruct([1,-1])
        with self.assertRaises(ValueError):SimplifiedCNF([(True,)],set())
        cold=SimplifiedCNF(cnf,{1,2},max_variables=0,max_subsumption=0)
        self.assertEqual(cold.clauses,canonical(cnf))
        calls=0
        def stop():
            nonlocal calls
            calls+=1
            if calls==3:raise InterruptedError
        with self.assertRaises(InterruptedError):SimplifiedCNF(cnf,{1,2},stop)

    @unittest.skipUnless(HAS_SAT,'optional python-sat unavailable')
    def test_actual_puzzle_formula_preserves_all_placement_assumptions(self):
        problem=SearchProblem(simple_tiles(),double_face());scope=WorkScope.for_problem(problem,'edges')
        answers=set(brute(problem,'edges'));session=PersistentSatSession(problem,scope,simplification=True,guard_reuse=True)
        try:
            self.assertTrue(session.prepare(lambda:None,2));original=session._base_clauses
            self.assertGreater(session.stats['simplify_clauses_removed'],0)
            for a,b in itertools.product(range(6),repeat=2):
                literals=[session.variables[0][a],session.variables[1][b]]
                actual=session.solver.solve(assumptions=literals)
                self.assertEqual(actual,(a,b) in answers)
                if actual:
                    lifted=session._simplified.reconstruct(session.solver.get_model())
                    values={abs(v):v>0 for v in lifted}
                    self.assertTrue(all(any(values[abs(v)]==(v>0) for v in row) for row in original))
            self.assertEqual(session._base_clauses,original)
        finally:session.close()

    @unittest.skipUnless(HAS_SAT,'optional python-sat unavailable')
    def test_actual_session_restore_policy_toggle_and_gold_exhaustion(self):
        problem=SearchProblem(multiple_loops(),double_face());scope=WorkScope.for_problem(problem,'gold')
        session=PersistentSatSession(problem,scope,simplification=True,guard_reuse=True,regional_compiler=True)
        try:
            session.prepare(lambda:None,2);saved=session.snapshot()
            restored=PersistentSatSession.restore(problem,saved)
            try:
                self.assertTrue(restored.simplification);restored.prepare(lambda:None,2)
                pending=tuple(restored.boxes);restored.configure_simplification(False)
                self.assertEqual(tuple(restored.boxes),pending);restored.prepare(lambda:None,2)
                restored.configure_simplification(True);restored.prepare(lambda:None,2)
                self.assertGreater(restored.stats['simplify_cache_hits'],0)
                result=restored.advance(seconds=2)
                self.assertEqual(result['status'],'infeasible');self.assertFalse(restored.boxes)
            finally:restored.close()
        finally:session.close()

if __name__=='__main__':unittest.main()
