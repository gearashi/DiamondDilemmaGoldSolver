"""Independent support, scope and worker-lifetime checks for proposal 2."""
import copy
import random
import unittest
from constraint_job_sat import SharedSatAssistant, JobSatAssist
from constraint_sat_assist import SatAssist
from constraint_dfs import solve as dfs_solve, _checkpoint_digest
from constraint_parallel import _WorkerCaches
from search_options import normalize_options
from test_constraint_sat_assist import HAS_SAT
from test_constraint_sat_domains import fixture, admitted
from test_constraint_dfs import brute
from test_constraint_parallel import OPTIONS


def compare_job_models(jobs=20):
    problem, narrow = fixture()
    root = tuple(sum(1 << v for v in row) for row in problem.domains)
    contexts = [tuple(narrow), root]
    solutions = brute(problem, 'edges')
    output = {}
    for shared in (False, True):
        owner = SharedSatAssistant(problem, 'edges') if shared else None
        builds = calls = steps = 0
        answers = []
        try:
            for index in range(jobs):
                current = contexts[index % len(contexts)]
                helper = owner.view(current) if owner else SatAssist(problem, current, 'edges')
                if not owner:
                    builds += 1
                try:
                    assert helper.prepare(seconds=5)
                    answer = helper.filter_domains(current, seconds=2, max_candidates=256)
                    assert admitted(solutions, answer) == admitted(solutions, current)
                    answers.append(answer if all(answer) else None)
                    if not owner:
                        calls += helper.stats['sat_assist_domain_calls']
                        steps += helper.stats['sat_assist_build_steps']
                finally:
                    helper.close()
            if owner:
                builds = 1
                calls = owner.helper.stats['sat_assist_domain_calls']
                steps = owner.helper.stats['sat_assist_build_steps']
            output['shared' if shared else 'baseline'] = {
                'model_builds': builds, 'native_support_calls': calls,
                'model_build_steps': steps, 'answers': answers}
        finally:
            if owner: owner.close()
    assert output['shared']['answers'] == output['baseline']['answers']
    return output


@unittest.skipUnless(HAS_SAT, 'optional python-sat missing')
class SharedJobSatTests(unittest.TestCase):
    def test_alternating_jobs_reuse_one_model_and_completed_queries(self):
        result = compare_job_models()
        self.assertEqual(result['shared']['model_builds'], 1)
        self.assertLess(result['shared']['native_support_calls'], result['baseline']['native_support_calls'])
        self.assertLess(result['shared']['model_build_steps'], result['baseline']['model_build_steps'])

    def test_scoped_decision_cores_never_escape_their_job(self):
        problem, narrow = fixture()
        owner = SharedSatAssistant(problem, 'edges'); self.addCleanup(owner.close)
        self.assertTrue(owner.helper.prepare(seconds=5))
        solutions = brute(problem, 'edges')
        rng = random.Random(250002)
        for _ in range(80):
            root = tuple(sum(1 << v for v in row if rng.random() < .7) for row in problem.domains)
            if not all(root): continue
            view = owner.view(root)
            decisions = tuple((cell, rng.choice([v for v in range(3 * problem.n) if root[cell] >> v & 1]))
                              for cell in range(problem.n) if rng.random() < .4)
            core = view.query(decisions, seconds=1, conflicts=10000)
            if core is not None:
                self.assertFalse(any(all(board[c] == value for c, value in core)
                                     for board in admitted(solutions, root)))
            after = view.filter_domains(root, seconds=1, max_candidates=256)
            self.assertEqual(admitted(solutions, after), admitted(solutions, root))
        wide = owner.view(owner.root)
        self.assertEqual(admitted(solutions, wide.filter_domains(owner.root, seconds=2, max_candidates=256)), solutions)

    def test_worker_eviction_keeps_shared_native_solver_alive(self):
        problem, _ = fixture(); options = normalize_options({**OPTIONS, 'shared_sat_assistant':True})
        cache = _WorkerCaches(problem, limit=1); self.addCleanup(cache.close)
        root = dfs_solve(problem, seconds=0, target='edges', options=options)['checkpoint']
        root.pop('root_symmetry', None); root['digest'] = _checkpoint_digest(root)
        owners = []
        for job in range(4):
            request = {'id':job, 'region_key':str(job), 'target':'edges', 'options':options, 'checkpoint':root}
            state = cache.get(request); owners.append(state['_shared_sat'])
            result = dfs_solve(problem, seconds=2, target='edges', resume=root, options=options, learning_state=state)
            self.assertIn(result['status'], ('solved','edge_perfect'))
            self.assertIsInstance(state['_runtime'].sat, JobSatAssist)
            self.assertIsNotNone(state['_shared_sat'].helper.solver)
            cache.remember(request,state)
        self.assertEqual(len({id(owner) for owner in owners}), 1)
        self.assertEqual(cache.stats['shared_sat_models'], 1)
        self.assertEqual(cache.stats['job_cache_evictions'], 3)
        self.assertEqual(cache.stats['shared_sat_job_views'],4)
        self.assertEqual(cache.stats['shared_sat_clauses'],owners[0].helper.stats['sat_assist_clauses'])
        cache.get(request);cache.remember(request,state)
        self.assertEqual(cache.stats['shared_sat_job_views'],4)

    def test_scope_validation_and_view_close_do_not_close_owner(self):
        problem, narrow = fixture(); owner = SharedSatAssistant(problem, 'gold'); self.addCleanup(owner.close)
        view = owner.view(narrow); view.close()
        self.assertIsNotNone(owner.helper.solver)
        with self.assertRaises(ValueError): owner.view([1 << 999] * problem.n)
        with self.assertRaises(ValueError): view.filter_domains(owner.root)
        self.assertFalse(owner.matches(problem, 'edges'))
        with self.assertRaises(ValueError):
            dfs_solve(problem, seconds=0, target='edges', options={'shared_sat_assistant':True},
                      learning_state={'_shared_sat':owner})


if __name__ == '__main__':
    unittest.main()
