"""Guard/accounting tests only: no actual native solver, GPU, or timed search."""
from contextlib import contextmanager, ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_repair_round25 as benchmark

ROOT = Path(__file__).resolve().parent
SOURCES = {'constraint_lns.py': {'raw_sha256': 'a'*64, 'lf_sha256': 'a'*64},
           'constraint_repair_resume_round25.py': {'raw_sha256': 'b'*64, 'lf_sha256': 'b'*64}}
RUNTIME = {'python': 'test', 'optimization_level': 0, 'assertions_enabled': True,
           'repair_native_files': {'ortools/sat/python/cp_model_helper.pyd': {'sha256': 'd'*64, 'bytes': 17}}}
REPOSITORY = {'available': True, 'commit': 'c'*40, 'dirty': False, 'status': []}


def native(count):
    row = {key: 0 for key in benchmark.NATIVE_KEYS}
    row.update(model_constructions=count, solver_constructions=count,
               native_calls=count, native_completed_calls=count,
               native_statuses={'OPTIMAL': count} if count else {})
    return row


def comparisons():
    rows = []
    for case in ('ordinary_completed_query', 'ordinary_pending_query', 'legacy_core', 'legacy_fragment', 'legacy_scoped', 'legacy_trade'):
        full, prefix, tail, fresh = native(3), native(1), native(2), native(3)
        if case == 'legacy_trade': full = prefix = tail = fresh = native(0)
        outcome = {'continuation_complete': True, 'status': 'completed'}
        rows.append({'case': case, 'same_independent_outcomes': True, 'completed_discriminator': True,
            'uninterrupted': {'outcome': dict(outcome), 'native': full, 'controller': {'trade_states': 10}},
            'saved_prefix': {'outcome': {'continuation_complete': False}, 'native': prefix, 'controller': {'trade_states': 2}},
            'resumed_total': {'outcome': dict(outcome), 'native': benchmark._sum_native(prefix, tail), 'controller': {'trade_states': 10}},
            'fresh_restart_total': {'outcome': dict(outcome), 'native': benchmark._sum_native(prefix, fresh), 'controller': {'trade_states': 10}},
            'resumed_tail_native': tail, 'fresh_tail_native': fresh,
            'actual_experiment_native': benchmark._sum_native(full, prefix, tail, fresh)})
    return rows


@contextmanager
def mocked(output, *, stage='frozen', sources=None, runtimes=None, repositories=None, answer=None, error=None):
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys, 'argv', [benchmark.__file__, '--output', str(output), '--evidence-stage', stage]))
        source = stack.enter_context(patch.object(benchmark, 'source_snapshot', side_effect=sources or [SOURCES, SOURCES]))
        runtime = stack.enter_context(patch.object(benchmark, '_runtime_snapshot', side_effect=runtimes or [RUNTIME, RUNTIME]))
        repository = stack.enter_context(patch.object(benchmark, 'repository_snapshot', side_effect=repositories or [REPOSITORY, REPOSITORY]))
        run = stack.enter_context(patch.object(benchmark, 'run_comparisons', return_value=comparisons() if answer is None else answer, side_effect=error))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield SimpleNamespace(run=run, source=source, runtime=runtime, repository=repository)


class RepairBenchmarkTests(unittest.TestCase):
    def folder(self): return tempfile.TemporaryDirectory(prefix='repair-benchmark-', dir=ROOT)

    def test_frozen_dirty_unreadable_and_existing_output_stop_before_native_work(self):
        for repository in ({}, {**REPOSITORY, 'dirty': True}):
            with self.folder() as folder:
                output = Path(folder)/'result.json'
                with mocked(output, repositories=[repository]) as calls:
                    with self.assertRaisesRegex(RuntimeError, 'clean, readable committed'): benchmark.main()
                calls.run.assert_not_called(); self.assertFalse(output.exists())
        for name in ('occupied.json', 'source.py'):
            with self.folder() as folder:
                output = Path(folder)/name; output.write_text('preserve', encoding='utf-8')
                with mocked(output) as calls:
                    with self.assertRaisesRegex(ValueError, 'fresh .json'): benchmark.main()
                calls.run.assert_not_called(); calls.source.assert_not_called()
                self.assertEqual(output.read_text(encoding='utf-8'), 'preserve')

    def test_frozen_success_binds_sources_runtime_git_and_all_six_cases(self):
        with self.folder() as folder:
            output = Path(folder)/'result.json'
            with mocked(output) as calls: benchmark.main()
            calls.run.assert_called_once()
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(report['evidence_stage'], 'frozen'); self.assertEqual(report['evidence_status'], 'passed')
            self.assertEqual(report['source_before'], SOURCES); self.assertEqual(report['source_after'], SOURCES)
            self.assertEqual(report['runtime_before'], RUNTIME); self.assertEqual(report['runtime_after'], RUNTIME)
            self.assertEqual(report['repository_before'], REPOSITORY); self.assertEqual(report['repository_after'], REPOSITORY)
            self.assertEqual(len(report['results']), 6)
            self.assertTrue(any('No timing winner' in limit for limit in report['limits']))

    def test_changed_sources_runtime_or_repository_never_publish_success(self):
        cases = ({'sources': [SOURCES, {}]}, {'runtimes': [RUNTIME, {}]},
                 {'repositories': [REPOSITORY, {**REPOSITORY, 'commit': 'f'*40}]},
                 {'repositories': [REPOSITORY, {**REPOSITORY, 'dirty': True}]})
        for changes in cases:
            with self.folder() as folder:
                output = Path(folder)/'result.json'
                with mocked(output, **changes):
                    with self.assertRaisesRegex(RuntimeError, 'changed'): benchmark.main()
                self.assertFalse(output.exists())

    def test_development_allows_dirty_checkout_but_not_head_change(self):
        dirty = {**REPOSITORY, 'dirty': True, 'status': [' M controller.py']}
        for changed in (False, True):
            with self.folder() as folder:
                output = Path(folder)/'result.json'; after = {**dirty, 'commit': 'e'*40} if changed else dirty
                with mocked(output, stage='development', repositories=[dirty, after]):
                    if changed:
                        with self.assertRaisesRegex(RuntimeError, 'HEAD changed'): benchmark.main()
                    else: benchmark.main()
                self.assertEqual(output.exists(), not changed)

    def test_failure_incomplete_matrix_or_omitted_cost_never_publish(self):
        mutations = [lambda rows: rows.pop(),
                     lambda rows: rows[0].__setitem__('same_independent_outcomes', False),
                     lambda rows: rows[0]['resumed_total']['outcome'].__setitem__('continuation_complete', False),
                     lambda rows: rows[0]['resumed_total'].__setitem__('native', native(2)),
                     lambda rows: rows[0].__setitem__('actual_experiment_native', native(99)),
                     lambda rows: rows[-1]['resumed_total']['controller'].__setitem__('trade_states', 12)]
        for mutate in mutations:
            rows = comparisons(); mutate(rows)
            with self.folder() as folder:
                output = Path(folder)/'result.json'
                with mocked(output, answer=rows):
                    with self.assertRaises(RuntimeError): benchmark.main()
                self.assertFalse(output.exists())
        with self.folder() as folder:
            output = Path(folder)/'result.json'
            with mocked(output, error=RuntimeError('Independent solution set mismatch')):
                with self.assertRaisesRegex(RuntimeError, 'solution set'): benchmark.main()
            self.assertFalse(output.exists())

    def test_late_collision_is_not_overwritten(self):
        with self.folder() as folder:
            output = Path(folder)/'result.json'
            def collision():
                output.write_text('another owner', encoding='utf-8'); return comparisons()
            with mocked(output) as calls:
                calls.run.side_effect = collision
                with self.assertRaises(FileExistsError): benchmark.main()
            self.assertEqual(output.read_text(encoding='utf-8'), 'another owner')

    def test_native_meter_distinguishes_completed_unknown_and_exception_calls(self):
        class Model:
            def __init__(self): pass
        class Solver:
            def __init__(self): self.num_branches = 2; self.num_conflicts = 1
            def solve(self, status):
                if isinstance(status, BaseException): raise status
                return status
            def status_name(self, status): return {1: 'OPTIMAL', 2: 'FEASIBLE', 3: 'INFEASIBLE', 4: 'UNKNOWN'}[status]
        cp = SimpleNamespace(CpModel=Model, CpSolver=Solver, OPTIMAL=1, FEASIBLE=2, INFEASIBLE=3, UNKNOWN=4)
        original = Solver.solve
        with patch.dict(sys.modules, {'ortools.sat.python': SimpleNamespace(cp_model=cp)}):
            with benchmark.native_work() as work:
                model = Model(); solver = Solver()
                for status in (1, 3, 4): self.assertEqual(solver.solve(status), status)
                with self.assertRaisesRegex(RuntimeError, 'interrupted'): solver.solve(RuntimeError('interrupted'))
        self.assertIs(Solver.solve, original)
        self.assertEqual(work, {'model_constructions': 1, 'solver_constructions': 1, 'native_calls': 4,
            'native_completed_calls': 2, 'native_unknown_calls': 1, 'native_exception_calls': 1,
            'native_branches': 6, 'native_conflicts': 3, 'native_statuses': {'OPTIMAL': 1, 'INFEASIBLE': 1, 'UNKNOWN': 1}})

    def test_optimized_python_refuses_before_native_import_or_output(self):
        with self.folder() as folder:
            output = Path(folder)/'result.json'
            result = subprocess.run([sys.executable, '-B', '-O', str(ROOT/'benchmark_repair_round25.py'), '--output', str(output)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0); self.assertIn('require assertions', result.stderr)
            self.assertFalse(output.exists())


if __name__ == '__main__': unittest.main()
