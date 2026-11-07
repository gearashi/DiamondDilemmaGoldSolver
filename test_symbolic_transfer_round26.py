import copy
import itertools
import json
import unittest
from constraint_work import WorkScope, frontier_snapshot, digest
from constraint_symbolic_frontier import SymbolicFrontier
from constraint_symbolic_transfer import portable_snapshot, restore_portable, from_checkpoint, engine_remaining
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles, multiple_loops, double_face
from test_constraint_sat_assist import HAS_SAT
from test_symbolic_frontier_round26 import represented, points

class SymbolicTransferTests(unittest.TestCase):
    def test_legacy_and_symbolic_roundtrips_with_leases(self):
        scope = WorkScope('test', 'gold', (7, 7, 7), (7, 7, 7))
        old = frontier_snapshot(scope, [(1, 7, 7), (6, 3, 7)])
        source = restore_portable(old, scope)
        lease = source.claim('cpu')
        expected = points((1, 7, 7)) | points((6, 3, 7))
        saved = portable_snapshot(source)
        self.assertNotIn('boxes', saved)
        target = restore_portable(json.loads(json.dumps(saved)), scope)
        self.assertEqual(represented(target.sets, target.pending), expected)
        self.assertFalse(target.leases)

    def test_native_cpu_pending_work_roundtrip(self):
        from constraint_dfs import _solve_core
        problem = SearchProblem(simple_tiles(), double_face())
        native = _solve_core(problem, seconds=0)['checkpoint']
        source = from_checkpoint(problem, native)
        self.assertEqual(source.remaining_volume(), 36)
        scope, rows = engine_remaining(problem, 'cpu', native)
        self.assertEqual(represented(source.sets, source.pending), set().union(*(points(row) for row in rows)))

    @unittest.skipUnless(HAS_SAT, 'optional SAT dependency')
    def test_native_sat_pending_work_roundtrip(self):
        from constraint_sat_session import PersistentSatSession
        problem = SearchProblem(multiple_loops(), double_face())
        scope = WorkScope.for_problem(problem)
        session = PersistentSatSession(problem, scope, [(1, 63), (2, 63)])
        try: native = session.snapshot()
        finally: session.close()
        source = from_checkpoint(problem, native)
        self.assertEqual(source.remaining_volume(), 12)
        self.assertEqual(restore_portable(portable_snapshot(source), scope).remaining_volume(), 12)

    def test_wrong_scope_and_extra_roots_are_rejected(self):
        scope = WorkScope('test', 'gold', (7, 7), (7, 7))
        saved = portable_snapshot(SymbolicFrontier(scope))
        other = WorkScope('different', 'gold', scope.original, scope.root)
        with self.assertRaises(ValueError): restore_portable(saved, other)
        bad = copy.deepcopy(saved); bad['graph']['roots'].append(0)
        bad['digest'] = digest({k:v for k,v in bad.items() if k != 'digest'})
        with self.assertRaises(ValueError): restore_portable(bad, scope)

if __name__ == '__main__': unittest.main()