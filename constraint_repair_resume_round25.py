"""Trusted-local resumable controllers for the four exact Gold repair modes.

A digest detects corruption; it is NOT a proof certificate. These controllers
must never be accepted through an external proof/frontier import route. Native
CP-SAT state is not serialized: an unfinished native call is explicitly retried.
Completed master choices, assumption cores and patch DFS transitions survive.
The completed_work_reused counter counts saved work records loaded on each
restore (including repeated loads), not uniquely avoided native calls.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import time

from constraint_lns import score, _Control, _Cancelled, _watch
from validator import reverse_mask

VERSION = 1
MAX_BYTES = 8 * 1024 * 1024
MAX_CELLS = 512
MAX_QUERIES = 1024
MAX_BOARDS = 32
MAX_STATES = 100000000
COMMON_STATS = (
    'repair_resume_loads', 'repair_model_rebuilds', 'repair_model_build_attempts', 'repair_master_model_rebuilds',
    'repair_native_attempts', 'repair_native_retries', 'repair_native_unknown', 'repair_native_interrupted',
    'repair_completed_queries', 'repair_master_attempts', 'repair_master_retries',
    'repair_completed_masters', 'repair_completed_work_reused',
    'oracle_queries', 'oracle_branches', 'oracle_conflicts', 'oracle_unknown',
    'oracle_core_cells')
MODE_STATS = {
    'core': ('hitting_set_solves', 'hitting_set_branches', 'queries'),
    'fragment': ('fragment_queries', 'fragment_master_branches', 'fragments'),
    'scoped': (),
    'trade': ('trade_states', 'trade_histories', 'trade_improvements'),
}
SCOPE = {'core': 'minimum_change_repair', 'fragment': 'archive_fragment_recombination',
         'scoped': 'outside_patch_assignments', 'trade': 'fixed_patch_boundary'}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


class _RepairControl(_Control):
    def cancelled(self):
        try:
            return super().cancelled()
        except BaseException as exc:
            # An application callback error, even ImportError/CircuitLimit,
            # must not masquerade as an unavailable optional dependency.
            self.error = exc
            raise


def _seconds(seconds):
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Repair slice seconds must be finite and nonnegative.')


def _binding(problem, kind, inputs, policy):
    if not _integer(problem.n, 1, MAX_CELLS):
        raise ValueError('Repair continuation cell limit exceeded.')
    domains = [list(row) for row in problem.domains]
    if len(domains) != problem.n or any(
            len(set(row)) != len(row) or any(not _integer(c, 0, 3*problem.n-1) for c in row)
            for row in domains):
        raise ValueError('Invalid repair continuation domains.')
    # Bind actual mutable semantic inputs as well as the advertised fingerprint.
    source_names = ('constraint_repair_resume_round25.py', 'constraint_repair_round24.py',
                    'constraint_lns.py', 'constraint_circuit.py', 'search_problem.py',
                    'validator.py', 'geometry.py')
    sources = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
               for name in source_names}
    return _digest({'version': VERSION, 'target': 'gold', 'kind': kind,
                    'fingerprint': problem.fingerprint, 'tiles': problem.tiles,
                    'edges': list(problem.edges), 'board_edges': list(problem.board.edges),
                    'masks': problem.masks.tolist(), 'neighbors': problem.neighbors.tolist(),
                    'sides': problem.sides.tolist(), 'fixed': sorted(problem.fixed.items()),
                    'domains': domains, 'inputs': inputs, 'policy': policy, 'sources': sources})


def _patch(problem, patch):
    cells = list(patch)
    if any(not _integer(cell, 0, problem.n-1) for cell in cells):
        raise ValueError('Invalid repair patch cell.')
    return sorted(set(cells))


def _fresh(kind, board):
    return {'phase': 'walk' if kind == 'trade' else 'query' if kind == 'scoped' else 'master',
            'status': 'unknown', 'reason': None, 'best_board': list(board),
            'pending': None, 'retry': False, 'master_retry': False,
            'failures': [], 'cores': [], 'conflicts': [], 'minimum_changes': None,
            'sources': [], 'walk': {'prefix': [], 'next': [0]} if kind == 'trade' else None,
            'complete_local_query': False,
            'stats': {key: 0 for key in (*COMMON_STATS, *MODE_STATS[kind])}}


def _ints(row, low, high, *, distinct=True, ordered=False):
    return (type(row) is list and all(_integer(v, low, high) for v in row)
            and (not distinct or len(row) == len(set(row)))
            and (not ordered or row == sorted(row)))


def _validate_state(problem, kind, state, fresh, inputs, policy, trade=None, fragments=None):
    """Strict shape/cursor checks for local checkpoints, not negative proof replay."""
    bad = lambda: ValueError('Invalid trusted-local repair continuation state.')
    if type(state) is not dict or set(state) != set(fresh): raise bad()
    if state['phase'] not in ('master', 'query', 'walk', 'done'): raise bad()
    if state['status'] not in ('unknown', 'stopped', 'timeout', 'solved', 'infeasible', 'completed', 'unavailable'): raise bad()
    if state['reason'] is not None and (type(state['reason']) is not str or len(state['reason']) > 4096): raise bad()
    for key in ('retry', 'master_retry', 'complete_local_query'):
        if type(state[key]) is not bool: raise bad()
    stats = state['stats']
    if type(stats) is not dict or set(stats) != set(fresh['stats']) or any(
            not _integer(value, 0, 2**53-1) for value in stats.values()): raise bad()
    if stats['repair_completed_queries'] > policy.get('max_queries', 1): raise bad()
    if stats['repair_native_retries'] > stats['repair_native_attempts']: raise bad()
    if stats['repair_native_unknown'] > stats['repair_native_attempts']: raise bad()
    if stats['repair_completed_masters'] > stats['repair_master_attempts']: raise bad()
    score(problem, state['best_board'])
    if state['status'] == 'solved':
        if state['phase'] != 'done' or not problem.validate(state['best_board'])['valid'] or any(
                code not in problem.domains[cell] for cell, code in enumerate(state['best_board'])): raise bad()
    elif state['status'] in ('infeasible', 'completed', 'unavailable') and state['phase'] != 'done': raise bad()
    if state['phase'] == 'done' and (state['retry'] or state['master_retry']): raise bad()
    if state['retry'] and state['phase'] != 'query': raise bad()
    if state['master_retry'] and state['phase'] != 'master': raise bad()
    if state['phase'] == 'master' and state['pending'] is not None: raise bad()
    if state['phase'] != 'done' and state['status'] not in ('unknown', 'stopped', 'timeout'): raise bad()
    failures = state['failures']
    if type(failures) is not list or len(failures) > policy.get('max_queries', 1): raise bad()
    for core in failures:
        if type(core) is not list or len(core) > problem.n: raise bad()
        if any(type(pair) is not list or len(pair) != 2 or
               not _integer(pair[0], 0, problem.n-1) or not _integer(pair[1], 0, 3*problem.n-1)
               for pair in core): raise bad()
        if core != sorted(core) or len({cell for cell, code in core}) != len(core): raise bad()
    if len({_digest(core) for core in failures}) != len(failures): raise bad()
    if len(failures) > stats['repair_completed_queries']: raise bad()
    if state['minimum_changes'] is not None and not _integer(state['minimum_changes'], 0, problem.n): raise bad()
    if kind != 'core' and (state['cores'] or state['minimum_changes'] is not None): raise bad()
    if kind != 'fragment' and (state['conflicts'] or state['sources']): raise bad()
    if kind != 'trade' and (state['walk'] is not None or state['complete_local_query']): raise bad()
    pending = state['pending']
    if kind == 'core':
        movable = set(range(problem.n))-set(problem.fixed)
        cores = state['cores']
        if type(cores) is not list or len(cores) > policy['max_queries']: raise bad()
        if any(not _ints(core, 0, problem.n-1, ordered=True) or not core or not set(core)<=movable for core in cores): raise bad()
        if len({tuple(core) for core in cores}) != len(cores): raise bad()
        if cores != [[cell for cell, code in core] for core in failures if core]: raise bad()
        if any(code != inputs['codes'][cell] or cell not in movable for core in failures for cell, code in core): raise bad()
        if state['status'] == 'infeasible' and [] not in failures: raise bad()
        if state['status'] != 'solved' and state['best_board'] != inputs['codes']: raise bad()
        if pending is not None and (not _ints(pending, 0, problem.n-1, ordered=True)
                or not set(pending)<=movable or any(not set(pending).intersection(core) for core in cores)): raise bad()
        if state['phase'] == 'query' and pending is None: raise bad()
        if state['phase'] not in ('master', 'query', 'done'): raise bad()
        if state['status'] == 'solved' and state['minimum_changes'] != sum(a != b for a, b in zip(inputs['codes'], state['best_board'])): raise bad()
    elif kind == 'fragment':
        count = len(fragments)
        if stats['fragments'] != count: raise bad()
        if type(state['conflicts']) is not list or len(state['conflicts']) > policy['max_queries']: raise bad()
        if any(not _ints(row, 0, count-1, ordered=True) or not row for row in state['conflicts']): raise bad()
        if not _ints(state['sources'], 0, len(inputs['boards'])-1, ordered=True): raise bad()
        allowed = {tuple(pair) for fragment in fragments for pair in fragment['placements']}
        if any(not set(map(tuple, core)) <= allowed for core in failures): raise bad()
        if state['status'] == 'infeasible' and [] not in failures: raise bad()
        if state['status'] != 'solved' and state['best_board'] != inputs['boards'][0]: raise bad()
        if pending is not None:
            if not _ints(pending, 0, count-1, ordered=True): raise bad()
            places = [pair for i in pending for pair in fragments[i]['placements']]
            if len({cell for cell, code in places}) != len(places) or len({code//3 for cell, code in places}) != len(places): raise bad()
            if any(set(row)<=set(pending) for row in state['conflicts']): raise bad()
        if state['phase'] == 'query' and pending is None: raise bad()
        if state['phase'] not in ('master', 'query', 'done'): raise bad()
    elif kind == 'scoped':
        if pending is not None or state['phase'] not in ('query', 'done'): raise bad()
        allowed = {(cell, code) for cell, code in enumerate(inputs['codes']) if cell not in inputs['patch']}
        if any(not set(map(tuple, core))<=allowed for core in failures): raise bad()
        if state['status'] == 'infeasible' and len(failures) != 1: raise bad()
        if state['phase'] == 'query' and stats['repair_completed_queries'] != 0: raise bad()
        if state['status'] != 'solved' and state['best_board'] != inputs['codes']: raise bad()
        if state['status'] == 'solved' and any(state['best_board'][cell] != code for cell, code in allowed): raise bad()
    else:
        if pending is not None or state['phase'] not in ('walk', 'done') or failures: raise bad()
        walk = state['walk']
        if type(walk) is not dict or set(walk) != {'prefix', 'next'}: raise bad()
        prefix, indexes = walk['prefix'], walk['next']
        order, choices = trade
        if not _ints(prefix, 0, 3*problem.n-1) or len(prefix)>len(order): raise bad()
        if type(indexes) is not list or len(indexes) != len(prefix)+1: raise bad()
        if len({code//3 for code in prefix}) != len(prefix): raise bad()
        assignment = {}
        for depth, code in enumerate(prefix):
            cell = order[depth]
            if code not in choices[cell] or indexes[depth] != choices[cell].index(code)+1: raise bad()
            if not _compatible(problem, assignment, cell, code): raise bad()
            assignment[cell] = code
        limit = len(choices[order[len(prefix)]]) if len(prefix)<len(order) else 0
        if not _integer(indexes[-1], 0, limit): raise bad()
        if stats['trade_states'] > policy['max_states']: raise bad()
        metric = score(problem, state['best_board'])
        if metric['domain_violations'] or metric['matched_edges'] != len(problem.edges): raise bad()
        patch = set(inputs['patch'])-set(problem.fixed)
        if any(state['best_board'][cell] != inputs['codes'][cell] for cell in range(problem.n) if cell not in patch): raise bad()
        if state['complete_local_query'] and state['status'] != 'completed': raise bad()


def _load(problem, kind, inputs, policy, resume, *, trade=None, fragments=None):
    seed = inputs['boards'][0] if kind == 'fragment' else inputs['codes']
    fresh = _fresh(kind, seed)
    if kind == 'fragment': fresh['stats']['fragments'] = len(fragments)
    binding = _binding(problem, kind, inputs, policy)
    if resume is None: return fresh, binding
    try:
        raw = _encoded(resume)
        if len(raw)>MAX_BYTES: raise ValueError('Repair continuation exceeds its byte cap.')
        record = json.loads(raw)
        if type(record) is not dict or set(record) != {'version', 'kind', 'binding', 'state', 'digest'}:
            raise ValueError('Invalid repair continuation envelope.')
        if type(record['version']) is not int or record['version'] != VERSION or record['kind'] != kind or record['binding'] != binding:
            raise ValueError('Repair continuation scope, policy or source mismatch.')
        unsigned = {key: value for key, value in record.items() if key != 'digest'}
        if type(record['digest']) is not str or record['digest'] != _digest(unsigned):
            raise ValueError('Repair continuation digest mismatch.')
        state = record['state']
        _validate_state(problem, kind, state, fresh, inputs, policy, trade=trade, fragments=fragments)
    except (TypeError, OverflowError, RecursionError) as exc:
        raise ValueError('Invalid repair continuation JSON.') from exc
    state['stats']['repair_resume_loads'] += 1
    state['stats']['repair_completed_work_reused'] += (
        state['stats']['repair_completed_queries'] + state['stats']['repair_completed_masters']
        + state['stats'].get('trade_states', 0))
    if state['phase'] != 'done': state['status'], state['reason'] = 'unknown', None
    return state, binding


def _done(state, status='unknown', reason=None):
    state.update(phase='done', status=status, reason=reason, retry=False, master_retry=False)


def _output(problem, kind, state, binding, began, inputs, policy):
    if _binding(problem, kind, inputs, policy) != binding:
        raise ValueError('Repair model or source changed during the slice; no continuation published.')
    checkpoint = {'version': VERSION, 'kind': kind, 'binding': binding, 'state': state}
    checkpoint['digest'] = _digest(checkpoint)
    raw = _encoded(checkpoint)
    if len(raw)>MAX_BYTES:
        raise ValueError('Repair continuation exceeds its byte cap; no partial checkpoint published.')
    checkpoint = json.loads(raw)  # Caller ownership is independent of every result container.
    result = {'status': state['status'], 'scope': SCOPE[kind], 'best_board': list(state['best_board']),
              'stats': dict(state['stats']), 'checkpoint': checkpoint,
              'continuation_complete': state['phase'] == 'done',
              'elapsed_seconds': time.monotonic()-began,
              'continuation_authority': 'trusted_local_only',
              'native_state_restored': False}
    if state['reason'] is not None: result['reason'] = state['reason']
    if kind == 'core':
        result.update(cores=[list(core) for core in state['cores']], minimum_changes=state['minimum_changes'])
    if kind == 'fragment' and state['status'] == 'solved': result['sources'] = list(state['sources'])
    if kind == 'trade':
        result.update(validation=problem.validate(state['best_board']), complete_local_query=state['complete_local_query'])
    else:
        result['binding'] = (problem.fingerprint, tuple(tuple(row) for row in problem.domains))
        result['exclusions'] = {'kind': 'gold_repair_assumption_cores', 'fingerprint': problem.fingerprint,
                                'domains': [list(row) for row in problem.domains],
                                'cores': [[list(pair) for pair in core] for core in state['failures']]}
        if state['status'] == 'solved':
            result.update(validation=problem.validate(state['best_board']), codes=list(state['best_board']))
        if kind == 'scoped' and state['status'] == 'infeasible':
            result['core'] = tuple(map(tuple, state['failures'][-1]))
    return result


def _oracle(problem, control, state):
    from constraint_repair_round24 import GoldRepairOracle
    control.check()
    state['stats']['repair_model_build_attempts'] += 1
    oracle = GoldRepairOracle(problem, control)
    state['stats']['repair_model_rebuilds'] += 1
    oracle.failures = [tuple(map(tuple, row)) for row in state['failures']]
    return oracle


def _query(oracle, assumptions, state):
    """Commit only a finished query; retry telemetry counts actual native starts."""
    oracle.control.check()
    before = dict(oracle.stats)
    was_retry = state['retry']
    answer = None
    try:
        answer = oracle.query(assumptions)
    finally:
        for key, value in oracle.stats.items(): state['stats'][key] += value-before.get(key, 0)
        starts = oracle.stats['oracle_queries']-before['oracle_queries']
        state['stats']['repair_native_attempts'] += starts
        if starts and was_retry: state['stats']['repair_native_retries'] += 1
        if starts: state['retry'] = True
        if starts and answer is None: state['stats']['repair_native_interrupted'] += starts
    if answer['status'] not in ('solved', 'infeasible'):
        state['stats']['repair_native_unknown'] += starts
        return answer
    state['retry'] = False
    state['stats']['repair_completed_queries'] += 1
    if answer['status'] == 'infeasible':
        core = [list(pair) for pair in sorted(answer['core'])]
        if any(len(pair) != 2 or type(pair[0]) is not int or type(pair[1]) is not int for pair in core) or len({cell for cell, code in core}) != len(core) or not set(map(tuple, core)) <= set(assumptions):
            raise RuntimeError('Repair oracle returned an invalid assumption core.')
        if core not in state['failures']: state['failures'].append(core)
    else:
        codes = answer['codes']
        score(oracle.problem, codes)
        if not oracle.problem.validate(codes)['valid'] or any(
                code not in oracle.problem.domains[cell] for cell, code in enumerate(codes)) or any(
                codes[cell] != code for cell, code in assumptions):
            raise RuntimeError('Resumable repair candidate failed independent scope/Gold validation.')
    return answer


def _caught(exc, state, control):
    if control.error is exc: raise exc
    if isinstance(exc, _Cancelled): state['status'] = control.reason or 'unknown'; return
    if isinstance(exc, ImportError): _done(state, 'unavailable'); return
    from constraint_circuit import CircuitLimit
    if isinstance(exc, CircuitLimit): _done(state, reason=str(exc)); return
    raise exc


def core_guided_repair(problem, codes, *, seconds=10, stop=None, max_queries=128, resume=None):
    from constraint_repair_round24 import _minimum_hitting_set
    _seconds(seconds); score(problem, codes)
    if not _integer(max_queries, 1, MAX_QUERIES): raise ValueError('Invalid repair query limit.')
    inputs, policy = {'codes': list(codes)}, {'max_queries': max_queries}
    state, binding = _load(problem, 'core', inputs, policy, resume)
    control = _RepairControl(seconds, stop); stats = state['stats']
    if state['phase'] != 'done' and stats['repair_completed_queries'] >= max_queries:
        _done(state, reason='query_limit')
    try:
        if state['phase'] != 'done':
            oracle = _oracle(problem, control, state)
            cells = [cell for cell in range(problem.n) if cell not in problem.fixed]
            while state['phase'] != 'done':
                control.check()
                if stats['repair_completed_queries'] >= max_queries:
                    _done(state, reason='query_limit'); break
                if state['phase'] == 'master':
                    if state['master_retry']: stats['repair_master_retries'] += 1
                    stats['repair_master_attempts'] += 1; stats['repair_master_model_rebuilds'] += 1
                    stats['hitting_set_solves'] += 1; state['master_retry'] = True
                    released, work = _minimum_hitting_set([frozenset(row) for row in state['cores']], cells, control, oracle.cp_model)
                    if released is None:
                        if work == 'infeasible': _done(state, 'infeasible')
                        break
                    stats['hitting_set_branches'] += work; stats['repair_completed_masters'] += 1
                    state.update(pending=sorted(released), phase='query', master_retry=False)
                assumptions = [(cell, codes[cell]) for cell in cells if cell not in state['pending']]
                answer = _query(oracle, assumptions, state); stats['queries'] += 1
                if answer['status'] == 'solved':
                    actual = sum(a != b for a, b in zip(codes, answer['codes']))
                    if actual != len(state['pending']): raise RuntimeError('Minimum-change repair bound mismatch.')
                    state.update(best_board=list(answer['codes']), minimum_changes=actual)
                    _done(state, 'solved'); break
                if answer['status'] != 'infeasible': break
                core = sorted(cell for cell, code in answer['core'])
                if not core: _done(state, 'infeasible'); break
                if core in state['cores']: raise RuntimeError('Resumable hitting set failed to advance.')
                state['cores'].append(core); state.update(phase='master', pending=None)
    except Exception as exc: _caught(exc, state, control)
    return _output(problem, 'core', state, binding, control.began, inputs, policy)


def _fragment_master(fragments, conflicts, control, cp):
    model = cp.CpModel(); select = [model.new_bool_var(f'fragment_{i}') for i in range(len(fragments))]
    by_cell, by_tile = {}, {}
    for index, fragment in enumerate(fragments):
        for cell, code in fragment['placements']:
            by_cell.setdefault(cell, []).append(select[index]); by_tile.setdefault(code//3, []).append(select[index])
    for row in [*by_cell.values(), *by_tile.values()]: model.add_at_most_one(row)
    for conflict in conflicts: model.add_bool_or([select[i].Not() for i in conflict])
    model.maximize(sum(len(fragment['placements'])*select[i] for i, fragment in enumerate(fragments)))
    solver = cp.CpSolver(); solver.parameters.num_search_workers = 1
    solver.parameters.max_time_in_seconds = max(.000001, control.deadline-time.monotonic())
    with _watch(control, control.deadline, solver): status = solver.solve(model)
    if status == cp.MODEL_INVALID: raise RuntimeError('Invalid fragment master: '+model.validate())
    chosen = [i for i, var in enumerate(select) if solver.value(var)] if status in (cp.FEASIBLE, cp.OPTIMAL) else None
    return chosen, int(solver.num_branches)


def fragment_crossover(problem, boards, *, seconds=10, stop=None, max_fragment=8, max_queries=32, resume=None):
    from constraint_repair_round24 import matched_fragments
    _seconds(seconds)
    if type(boards) not in (list, tuple) or not 1<=len(boards)<=MAX_BOARDS: raise ValueError('Invalid repair archive size.')
    for board in boards: score(problem, board)
    if not _integer(max_queries, 1, MAX_QUERIES) or not _integer(max_fragment, 1, MAX_CELLS): raise ValueError('Invalid fragment policy.')
    fragments = matched_fragments(problem, boards, max_fragment=max_fragment)
    inputs = {'boards': [list(board) for board in boards]}
    policy = {'max_queries': max_queries, 'max_fragment': max_fragment}
    state, binding = _load(problem, 'fragment', inputs, policy, resume, fragments=fragments)
    control = _RepairControl(seconds, stop); stats = state['stats']
    if state['phase'] != 'done' and stats['repair_completed_queries'] >= max_queries:
        _done(state, reason='query_limit')
    try:
        if state['phase'] != 'done':
            oracle = _oracle(problem, control, state)
            while state['phase'] != 'done':
                control.check()
                if stats['repair_completed_queries'] >= max_queries: _done(state, reason='query_limit'); break
                if state['phase'] == 'master':
                    if state['master_retry']: stats['repair_master_retries'] += 1
                    stats['repair_master_attempts'] += 1; stats['repair_master_model_rebuilds'] += 1
                    state['master_retry'] = True
                    chosen, branches = _fragment_master(fragments, state['conflicts'], control, oracle.cp_model)
                    stats['fragment_master_branches'] += branches
                    if chosen is None: break
                    stats['repair_completed_masters'] += 1
                    state.update(pending=chosen, phase='query', master_retry=False)
                chosen = state['pending']
                assumptions = [pair for i in chosen for pair in fragments[i]['placements']]
                answer = _query(oracle, assumptions, state); stats['fragment_queries'] += 1
                if answer['status'] == 'solved':
                    state.update(best_board=list(answer['codes']), sources=sorted({fragments[i]['source'] for i in chosen}))
                    _done(state, 'solved'); break
                if answer['status'] != 'infeasible': break
                core = set(map(tuple, answer['core']))
                conflict = [i for i in chosen if core.intersection(fragments[i]['placements'])]
                if not conflict: _done(state, 'infeasible'); break
                state['conflicts'].append(conflict); state.update(phase='master', pending=None)
    except Exception as exc: _caught(exc, state, control)
    return _output(problem, 'fragment', state, binding, control.began, inputs, policy)


def scoped_repair_failure(problem, codes, patch, *, seconds=5, stop=None, resume=None):
    _seconds(seconds); score(problem, codes); patch = _patch(problem, patch)
    inputs, policy = {'codes': list(codes), 'patch': patch}, {'max_queries': 1}
    state, binding = _load(problem, 'scoped', inputs, policy, resume)
    control = _RepairControl(seconds, stop)
    try:
        if state['phase'] != 'done':
            oracle = _oracle(problem, control, state)
            answer = _query(oracle, [(cell, code) for cell, code in enumerate(codes) if cell not in patch], state)
            if answer['status'] == 'solved': state['best_board'] = list(answer['codes'])
            if answer['status'] in ('solved', 'infeasible'): _done(state, answer['status'])
    except Exception as exc: _caught(exc, state, control)
    return _output(problem, 'scoped', state, binding, control.began, inputs, policy)


def _compatible(problem, assignment, cell, code):
    return all(int(other) not in assignment or int(problem.masks[code, side]) ==
               reverse_mask(problem.masks[assignment[int(other)], int(problem.sides[cell, side])])
               for side, other in enumerate(problem.neighbors[cell]) if other >= 0)


def _trade_plan(problem, codes, patch):
    cells = [cell for cell in patch if cell not in problem.fixed]
    inside = set(cells); inventory = {codes[cell]//3 for cell in cells}
    choices = {cell: [code for code in problem.domains[cell] if code//3 in inventory and
                      all(int(problem.masks[code, side]) == reverse_mask(problem.masks[codes[int(other)], int(problem.sides[cell, side])])
                          for side, other in enumerate(problem.neighbors[cell]) if other>=0 and int(other) not in inside)]
               for cell in cells}
    return sorted(cells, key=lambda cell: (len(choices[cell]), cell)), choices


def loop_merge_trade(problem, codes, patch, *, seconds=2, stop=None, max_states=200000, resume=None):
    _seconds(seconds); metrics = score(problem, codes); patch = _patch(problem, patch)
    if metrics['domain_violations'] or metrics['matched_edges'] != len(problem.edges):
        raise ValueError('Loop trades require an edge-perfect legal board.')
    if not _integer(max_states, 1, MAX_STATES): raise ValueError('Invalid patch state limit.')
    inputs, policy = {'codes': list(codes), 'patch': patch}, {'max_states': max_states}
    order, choices = _trade_plan(problem, codes, patch)
    state, binding = _load(problem, 'trade', inputs, policy, resume, trade=(order, choices))
    control = _RepairControl(seconds, stop); stats = state['stats']; walk = state['walk']
    try:
        while state['phase'] != 'done':
            control.check()
            prefix, indexes = walk['prefix'], walk['next']; depth = len(prefix)
            if depth == len(order):
                candidate = list(codes)
                for cell, code in zip(order, prefix): candidate[cell] = code
                validation = problem.validate(candidate); checked = score(problem, candidate)
                if checked['domain_violations'] or checked['matched_edges'] != len(problem.edges) or not validation['unique_tiles']:
                    raise RuntimeError('Resumed trade failed independent seam/inventory validation.')
                stats['trade_histories'] += 1
                if validation['line_components'] < problem.validate(state['best_board'])['line_components']:
                    state['best_board'] = candidate; stats['trade_improvements'] += 1
                if validation['valid']:
                    state['best_board'] = candidate; _done(state, 'solved'); break
                if prefix: prefix.pop(); indexes.pop()
                else: state['complete_local_query'] = True; _done(state, 'completed')
                continue
            cell = order[depth]
            if indexes[-1] == len(choices[cell]):
                if prefix: prefix.pop(); indexes.pop()
                else: state['complete_local_query'] = True; _done(state, 'completed')
                continue
            if stats['trade_states'] >= max_states: _done(state, reason='state_limit'); break
            code = choices[cell][indexes[-1]]; indexes[-1] += 1; stats['trade_states'] += 1
            if code//3 in {old//3 for old in prefix}: continue
            if not _compatible(problem, dict(zip(order, prefix)), cell, code): continue
            prefix.append(code); indexes.append(0)
    except _Cancelled: state['status'] = control.reason or 'unknown'
    return _output(problem, 'trade', state, binding, control.began, inputs, policy)
