"""Finite CPU-only repair-continuation discriminator with frozen provenance.

Counts completed work, discarded-prefix work, native calls and reconstruction.
It does not rank wall-clock performance or predict the original puzzle runtime.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

from benchmark_provenance import require_assertions, source_snapshot, runtime_snapshot, repository_snapshot

ROOT = Path(__file__).resolve().parent
NATIVE_KEYS = ('model_constructions', 'solver_constructions', 'native_calls',
               'native_completed_calls', 'native_unknown_calls', 'native_exception_calls',
               'native_branches', 'native_conflicts')


def _runtime_snapshot():
    """Also bind the CP-SAT wrapper and packaged native binaries, without loading them."""
    result = runtime_snapshot()
    bindings = {}
    try:
        distribution = importlib.metadata.distribution('ortools')
    except importlib.metadata.PackageNotFoundError:
        result['repair_native_files'] = bindings
        return result
    for item in distribution.files or ():
        name = str(item).replace('\\', '/')
        leaf = name.rsplit('/', 1)[-1]
        if leaf == 'cp_model.py' or '.so' in leaf or leaf.endswith(('.pyd', '.dll', '.dylib')):
            path = Path(distribution.locate_file(item))
            if path.is_file():
                data = path.read_bytes()
                bindings[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
    result['repair_native_files'] = bindings
    try: result['protobuf_version'] = importlib.metadata.version('protobuf')
    except importlib.metadata.PackageNotFoundError: result['protobuf_version'] = None
    return result


@contextmanager
def native_work():
    """Count every CP-SAT solve, including repeated solves inside a patch query."""
    require_assertions()
    from ortools.sat.python import cp_model as cp
    stats = {key: 0 for key in NATIVE_KEYS}; stats['native_statuses'] = {}
    model_init, solver_init, solve = cp.CpModel.__init__, cp.CpSolver.__init__, cp.CpSolver.solve
    def model(self, *args, **kwargs):
        stats['model_constructions'] += 1
        return model_init(self, *args, **kwargs)
    def solver(self, *args, **kwargs):
        stats['solver_constructions'] += 1
        return solver_init(self, *args, **kwargs)
    def run(self, *args, **kwargs):
        stats['native_calls'] += 1
        try:
            status = solve(self, *args, **kwargs)
        except BaseException:
            stats['native_exception_calls'] += 1
            raise
        name = self.status_name(status)
        stats['native_statuses'][name] = stats['native_statuses'].get(name, 0)+1
        stats['native_branches'] += int(self.num_branches)
        stats['native_conflicts'] += int(self.num_conflicts)
        if status in (cp.OPTIMAL, cp.FEASIBLE, cp.INFEASIBLE): stats['native_completed_calls'] += 1
        elif status == cp.UNKNOWN: stats['native_unknown_calls'] += 1
        else: raise RuntimeError('Invalid native repair model: '+name)
        return status
    with patch.object(cp.CpModel, '__init__', model), patch.object(cp.CpSolver, '__init__', solver), patch.object(cp.CpSolver, 'solve', run):
        yield stats
    if stats['native_calls'] != stats['native_completed_calls']+stats['native_unknown_calls']+stats['native_exception_calls']:
        raise RuntimeError('Native call accounting mismatch.')


def _sum_native(*records):
    total = {key: sum(row[key] for row in records) for key in NATIVE_KEYS}
    total['native_statuses'] = {}
    for row in records:
        for name, count in row['native_statuses'].items():
            total['native_statuses'][name] = total['native_statuses'].get(name, 0)+count
    return total


def _checkpoint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    return json.loads(encoded), {'json_bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest()}


def _finite(problem):
    from test_constraint_dfs import brute
    gold = brute(problem, 'gold'); edges = brute(problem, 'edges')
    return gold, {'cells': problem.n, 'independently_enumerated_gold': len(gold),
                  'independently_enumerated_edge_perfect': len(edges),
                  'gold_codes_sha256': hashlib.sha256(json.dumps(sorted(gold)).encode()).hexdigest()}


def _checked(problem, report, gold, *, finished=True):
    from constraint_lns import score
    board = report.get('best_board', report.get('codes'))
    if board is None: raise RuntimeError('Finite repair omitted a candidate board.')
    metrics = score(problem, board)
    validation = problem.validate(board)
    if metrics['domain_violations']: raise RuntimeError('Finite repair candidate escaped domains.')
    if finished and not report.get('continuation_complete'): raise RuntimeError('Finite repair did not complete its configured work.')
    if report['status'] == 'solved' and (not validation['valid'] or tuple(board) not in set(map(tuple, gold))):
        raise RuntimeError('Finite repair Gold witness failed independent enumeration.')
    for core in report.get('exclusions', {}).get('cores', []):
        if any(all(row[cell] == code for cell, code in core) for row in gold):
            raise RuntimeError('A scoped repair exclusion removed an independent Gold solution.')
    return {'status': report['status'], 'matched_edges': metrics['matched_edges'],
            'gold_valid': validation['valid'], 'line_components': validation['line_components'],
            'minimum_changes': report.get('minimum_changes'),
            'complete_local_query': report.get('complete_local_query'),
            'continuation_complete': bool(report.get('continuation_complete'))}


def _result(problem, report, gold, native, *, finished=True):
    return {'outcome': _checked(problem, report, gold, finished=finished), 'controller': report['stats'],
            'native': dict(native), 'best_board': list(report['best_board'])}


def ordinary_comparison(*, inside_query=False):
    require_assertions()
    from constraint_lns_resume_round25 import RepairSession
    from test_constraint_dfs import multiple_loops, double_face
    from search_problem import SearchProblem
    problem = SearchProblem(multiple_loops(), double_face()); gold, fixture = _finite(problem)
    if gold: raise RuntimeError('Loop fixture unexpectedly has a Gold completion.')
    policy = {'phase_seconds': 20, 'patch_sizes': (2,), 'max_attempts': 4, 'target': 'gold'}
    def session(resume=None): return RepairSession(problem, [0, 3], resume=resume, **policy)
    with native_work() as full_native:
        full = session(); full.advance(20); full_report = full.report()
    with native_work() as prefix_native:
        first = session()
        if inside_query:
            first.advance(20, stop=lambda: first.pending is not None and len(first.pending['cuts']['gold_cuts']) >= 2)
        else:
            first.advance(20, max_attempts_per_slice=1)
        prefix_report = first.report(); saved, saved_info = _checkpoint(first.snapshot())
    if inside_query:
        if saved['pending'] is None or len(saved['pending']['cuts']['gold_cuts']) < 2 or saved['index'] != 0:
            raise RuntimeError('Native patch interruption failed to preserve pending cuts.')
    elif saved['index'] != 1 or not saved['cache']:
        raise RuntimeError('Completed-query prefix did not reach the discriminator boundary.')
    with native_work() as tail_native:
        resumed = session(saved); resumed.advance(20); resumed_report = resumed.report()
    with native_work() as fresh_native:
        fresh = session(); fresh.advance(20); fresh_report = fresh.report()
    baseline = _result(problem, full_report, gold, full_native)
    warm = _result(problem, resumed_report, gold, _sum_native(prefix_native, tail_native))
    cold = _result(problem, fresh_report, gold, _sum_native(prefix_native, fresh_native))
    prefix = _result(problem, prefix_report, gold, prefix_native, finished=False)
    if not baseline['outcome'] == warm['outcome'] == cold['outcome']:
        raise RuntimeError('Ordinary finite repair outcomes changed after restart.')
    for key in ('patch_selections', 'attempt_cache_hits', 'conclusive_attempts'):
        if full_report['stats'][key] != resumed_report['stats'][key]:
            raise RuntimeError('Resumed ordinary completed-query work differs: '+key)
    if full_report['stats']['patch_selections'] != 4 or full_report['stats']['attempt_cache_hits'] < 1:
        raise RuntimeError('Ordinary finite cache discriminator was not completed.')
    if inside_query and resumed_report['stats']['repair_native_retries'] < 1:
        raise RuntimeError('Interrupted identical-query retry was not counted.')
    if cold['native']['native_calls'] <= warm['native']['native_calls']:
        raise RuntimeError('No completed native-work reuse discriminator.')
    cold['controller_total_work'] = {key: prefix_report['stats'][key]+fresh_report['stats'][key]
                                     for key in ('attempts', 'patch_selections', 'attempt_cache_hits', 'conclusive_attempts', 'loop_cuts', 'repair_model_rebuilds', 'repair_native_retries')}
    return {'case': 'ordinary_pending_query' if inside_query else 'ordinary_completed_query',
            'fixture': fixture, 'policy': policy, 'checkpoint': saved_info,
            'uninterrupted': baseline, 'saved_prefix': prefix, 'resumed_total': warm,
            'fresh_restart_total': cold, 'resumed_tail_native': tail_native, 'fresh_tail_native': fresh_native,
            'actual_experiment_native': _sum_native(full_native, prefix_native, tail_native, fresh_native),
            'same_independent_outcomes': True, 'completed_discriminator': True,
            'cost_scope': 'Both restart totals include the discarded/saved prefix. Prefix was executed once and is shared in the two policy totals; actual_experiment_native counts that execution once.'}


def legacy_comparison(kind):
    require_assertions()
    import constraint_repair_round24 as legacy
    import constraint_repair_resume_round25 as continued
    from search_problem import SearchProblem
    from test_constraint_dfs import tetrahedron_tiles, multiple_loops, double_face
    if kind == 'trade':
        problem = SearchProblem(multiple_loops(), double_face()); seed = [0, 3]
        fn, args, kwargs = legacy.loop_merge_trade, (seed, [0, 1]), {}
    else:
        problem = SearchProblem(*tetrahedron_tiles()); seed = [0, 4, 6, 9]
        if kind == 'core': fn, args, kwargs = legacy.core_guided_repair, (seed,), {}
        elif kind == 'fragment': fn, args, kwargs = legacy.fragment_crossover, ([[1, 3, 6, 9], seed],), {'max_fragment': 1}
        elif kind == 'scoped': fn, args, kwargs = legacy.scoped_repair_failure, ([1, 3, 6, 9], [1, 2, 3]), {}
        else: raise ValueError('Unknown finite repair mode.')
    gold, fixture = _finite(problem)
    def run(**extra): return fn(problem, *args, durable=True, seconds=10, **kwargs, **extra)
    with native_work() as full_native: full = run()
    with native_work() as prefix_native:
        if kind in ('core', 'fragment'):
            owner, attribute = (legacy, '_minimum_hitting_set') if kind == 'core' else (continued, '_fragment_master')
            original = getattr(owner, attribute); ready = [False]
            def pause_after_master(*a, **k):
                answer = original(*a, **k)
                ready[0] = answer[0] is not None
                return answer
            with patch.object(owner, attribute, pause_after_master): first = run(stop=lambda: ready[0])
            if first['checkpoint']['state']['phase'] != 'query' or first['stats']['repair_completed_queries']:
                raise RuntimeError('Repair master interruption missed its exact boundary.')
        elif kind == 'trade':
            calls = [0]
            def stop():
                calls[0] += 1
                return calls[0] > 5
            first = run(stop=stop)
            if first['continuation_complete'] or first['stats']['trade_states'] == 0:
                raise RuntimeError('Trade interruption did not save work.')
        else:
            first = run()
            if not first['continuation_complete'] or first['status'] != 'infeasible':
                raise RuntimeError('Scoped finite failure did not complete.')
        saved, saved_info = _checkpoint(first['checkpoint'])
    with native_work() as tail_native: resumed = run(resume=saved)
    with native_work() as fresh_native: fresh = run()
    baseline = _result(problem, full, gold, full_native)
    warm = _result(problem, resumed, gold, _sum_native(prefix_native, tail_native))
    cold = _result(problem, fresh, gold, _sum_native(prefix_native, fresh_native))
    prefix = _result(problem, first, gold, prefix_native, finished=kind == 'scoped')
    if not baseline['outcome'] == warm['outcome'] == cold['outcome']:
        raise RuntimeError('Legacy finite repair outcomes changed: '+kind)
    if kind == 'core':
        minimum = min(sum(a != b for a, b in zip(seed, row)) for row in gold)
        if full['minimum_changes'] != minimum or resumed['minimum_changes'] != minimum:
            raise RuntimeError('Minimum-change repair failed independent enumeration.')
    if kind == 'trade':
        if not resumed['complete_local_query']: raise RuntimeError('Trade finite query was not exhausted.')
        for key in ('trade_states', 'trade_histories'):
            if resumed['stats'][key] != full['stats'][key]: raise RuntimeError('Trade repeated or skipped completed work.')
        discriminator = first['stats']['trade_states']+fresh['stats']['trade_states']-resumed['stats']['trade_states']
        unit = 'processed_patch_transitions'
    else:
        discriminator = cold['native']['native_calls']-warm['native']['native_calls']; unit = 'actual_cp_sat_solve_calls'
    if discriminator <= 0: raise RuntimeError('No finite completed-work reuse discriminator: '+kind)
    cold['controller_total_work'] = {key: first['stats'][key]+fresh['stats'][key] for key in first['stats']}
    return {'case': 'legacy_'+kind, 'fixture': fixture, 'checkpoint': saved_info,
            'uninterrupted': baseline, 'saved_prefix': prefix, 'resumed_total': warm,
            'fresh_restart_total': cold, 'resumed_tail_native': tail_native, 'fresh_tail_native': fresh_native,
            'actual_experiment_native': _sum_native(full_native, prefix_native, tail_native, fresh_native),
            'same_independent_outcomes': True, 'completed_discriminator': True,
            'fresh_restart_extra_work': discriminator, 'extra_work_unit': unit,
            'cost_scope': 'Restart totals include the prefix; actual_experiment_native counts the shared prefix only once. Master-boundary Stop occurs before the next native feasibility query.'}


def run_comparisons():
    require_assertions()
    return [ordinary_comparison(inside_query=False), ordinary_comparison(inside_query=True),
            *[legacy_comparison(kind) for kind in ('core', 'fragment', 'scoped', 'trade')]]


def validate_comparisons(results):
    expected = {'ordinary_completed_query', 'ordinary_pending_query', 'legacy_core', 'legacy_fragment', 'legacy_scoped', 'legacy_trade'}
    if type(results) is not list or len(results) != len(expected) or {row.get('case') for row in results} != expected:
        raise RuntimeError('Incomplete finite repair comparison matrix.')
    for row in results:
        if row.get('same_independent_outcomes') is not True or row.get('completed_discriminator') is not True:
            raise RuntimeError('Finite repair preservation/discriminator did not pass.')
        outcomes = [row[name]['outcome'] for name in ('uninterrupted', 'resumed_total', 'fresh_restart_total')]
        if not all(outcome.get('continuation_complete') is True for outcome in outcomes) or any(outcome != outcomes[0] for outcome in outcomes):
            raise RuntimeError('Finite repair comparison has incomplete or unequal outcomes.')
        native_records = [row[name]['native'] for name in ('uninterrupted', 'saved_prefix', 'resumed_total', 'fresh_restart_total')]
        native_records += [row['resumed_tail_native'], row['fresh_tail_native'], row['actual_experiment_native']]
        for native in native_records:
            if any(type(native.get(key)) is not int or native[key] < 0 for key in NATIVE_KEYS):
                raise RuntimeError('Invalid native work accounting.')
            if native['native_calls'] != native['native_completed_calls']+native['native_unknown_calls']+native['native_exception_calls']:
                raise RuntimeError('Native work accounting is not exhaustive.')
            if type(native.get('native_statuses')) is not dict or any(type(value) is not int or value < 0 for value in native['native_statuses'].values()) or sum(native['native_statuses'].values()) != native['native_calls']-native['native_exception_calls']:
                raise RuntimeError('Native status accounting is not exhaustive.')
        prefix = row['saved_prefix']['native']
        if row['resumed_total']['native'] != _sum_native(prefix, row['resumed_tail_native']) or row['fresh_restart_total']['native'] != _sum_native(prefix, row['fresh_tail_native']):
            raise RuntimeError('Discarded/saved prefix work was omitted from policy totals.')
        if row['actual_experiment_native'] != _sum_native(row['uninterrupted']['native'], prefix, row['resumed_tail_native'], row['fresh_tail_native']):
            raise RuntimeError('Actual experiment native work was omitted or double-counted.')
        if row['case'] == 'legacy_trade':
            extra = row['saved_prefix']['controller']['trade_states']+row['fresh_restart_total']['controller']['trade_states']-row['resumed_total']['controller']['trade_states']
        else:
            extra = row['fresh_restart_total']['native']['native_calls']-row['resumed_total']['native']['native_calls']
        if extra <= 0: raise RuntimeError('Completed finite-work discriminator was not positive.')


def main():
    require_assertions()
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence-stage', choices=('development', 'frozen'), default='development')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.json' or args.output.exists() or args.output.is_symlink():
        raise ValueError('Repair benchmark output must be a fresh .json path.')
    before, runtime_before, repository_before = source_snapshot(ROOT), _runtime_snapshot(), repository_snapshot(ROOT)
    if args.evidence_stage == 'frozen' and (not repository_before.get('commit') or repository_before.get('dirty') is not False):
        raise RuntimeError('Frozen repair evidence requires a clean, readable committed checkout.')
    results = run_comparisons()
    validate_comparisons(results)
    after, runtime_after, repository_after = source_snapshot(ROOT), _runtime_snapshot(), repository_snapshot(ROOT)
    if before != after or runtime_before != runtime_after:
        raise RuntimeError('Sources or runtime changed during finite repair evaluation.')
    if repository_before.get('commit') != repository_after.get('commit'):
        raise RuntimeError('Git HEAD changed during finite repair evaluation.')
    if args.evidence_stage == 'frozen' and repository_before != repository_after:
        raise RuntimeError('Git identity changed during frozen finite repair evaluation.')
    report = {'proposal': 33, 'evidence_stage': args.evidence_stage, 'evidence_status': 'passed',
              'python': platform.python_version(), 'platform': platform.platform(), 'results': results,
              'source_before': before, 'source_after': after, 'runtime_before': runtime_before, 'runtime_after': runtime_after,
              'repository_before': repository_before, 'repository_after': repository_after,
              'source_unchanged': True, 'runtime_unchanged': True,
              'limits': [
                  'Finite CPU repair controllers only; no GPU sampling or whole-puzzle performance claim.',
                  'No timing winner. Native branches/conflicts may vary with runtime; completed query/transition reuse is the discriminator.',
                  'Native CP-SAT state is rebuilt. A logical patch query can contain several completed native solve calls; those units are reported separately.',
                  'Native UNKNOWN and raised calls are separate from completed native calls. A controller may stop between completed native calls, leaving its enclosing logical query incomplete.',
                  'Saved checkpoints are trusted-local continuation, not portable proof certificates; exported cores still require independent verification.',
                  'Configured local repair completion is not global puzzle exhaustion or a Gold impossibility result.',
                  'Fixture enumeration, checkpoint serialization and restoration checks are required overhead; no wall-time or avoided-CPU-time claim is made.',
                  'Model/solver construction and every native solve call are instrumented. Shared prefix work appears once per restart policy and once in actual experiment totals.']}
    encoded = json.dumps(report, indent=2, allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream: stream.write(encoded)
    for row in results:
        print(json.dumps({'case': row['case'], 'resumed_native_calls': row['resumed_total']['native']['native_calls'],
                          'fresh_restart_native_calls': row['fresh_restart_total']['native']['native_calls'],
                          'extra_work_unit': row.get('extra_work_unit'), 'fresh_restart_extra_work': row.get('fresh_restart_extra_work')}), flush=True)


if __name__ == '__main__': main()
