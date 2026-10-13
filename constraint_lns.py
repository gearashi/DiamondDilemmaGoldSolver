"""Bounded CPU large-neighborhood repair; local failure is never global UNSAT.

Each CP-SAT model moves only one selected set of cells and exactly the physical
pieces already in those cells. Outside placements are constants. Returned boards
are independently scored and checked; only a Gold certificate means solved.
"""
from __future__ import annotations
from contextlib import contextmanager
import math
import random
import threading
import time
from validator import reverse_mask


def diverse_archive(candidates, *, capacity=8, score_slack=4):
    """Bounded exact-board deduplication, then greedy Hamming diversity.

    Inputs must already have independently checked scores. This archive makes
    no claim about uniqueness of discarded or unobserved search proposals.
    """
    if type(capacity) is not int or capacity < 1 or score_slack < 0:
        raise ValueError('Invalid candidate archive limits.')
    unique = {}
    for item in candidates:
        key = tuple(item['codes'])
        if key not in unique or item['objective'] > unique[key]['objective']:
            unique[key] = {**item, 'codes':list(key)}
    pool = sorted(unique.values(), key=lambda item:item['objective'], reverse=True)
    if not pool: return []
    floor = pool[0]['objective']-score_slack
    pool = [item for item in pool if item['objective'] >= floor]
    selected = [pool.pop(0)]
    while pool and len(selected) < capacity:
        winner = max(range(len(pool)), key=lambda i:(
            min(sum(a!=b for a,b in zip(pool[i]['codes'],old['codes'])) for old in selected),
            pool[i]['objective']))
        selected.append(pool.pop(winner))
    return selected


def _load_cp_model():
    from ortools.sat.python import cp_model
    return cp_model


def score(problem, codes):
    """Verify a complete physical-tile permutation, then score its raw seams."""
    if hasattr(codes, 'tolist'):
        codes = codes.tolist()
    if not isinstance(codes, (list, tuple)) or len(codes) != problem.n:
        raise ValueError('Repair needs one orientation code per cell.')
    codes = list(codes)
    if any(type(code) is not int or not 0 <= code < 3*problem.n for code in codes):
        raise ValueError('Repair orientation codes are invalid.')
    if len({code//3 for code in codes}) != problem.n:
        raise ValueError('Repair must use each physical tile once.')
    if any(codes[cell] != code for cell, code in problem.fixed.items()):
        raise ValueError('Repair seed changes a fixed placement.')
    bad = [cell for cell, code in enumerate(codes) if code not in problem.domains[cell]]
    mismatch = [(a, sa, b, sb) for a, sa, b, sb in problem.edges
                if int(problem.masks[codes[a], sa]) != reverse_mask(problem.masks[codes[b], sb])]
    return {'matched_edges': len(problem.edges)-len(mismatch), 'domain_violations': len(bad),
            'objective': (problem.n-len(bad))*(len(problem.edges)+1)+len(problem.edges)-len(mismatch),
            'bad_cells': bad, 'unmatched_edges': mismatch}


class _Cancelled(Exception):
    pass


class _Control:
    def __init__(self, seconds, stop):
        self.began = time.monotonic()
        self.deadline = self.began + seconds
        self.stop, self.reason, self.error = stop, None, None
    def cancelled(self):
        if self.reason is not None: return True
        requested = self.stop() if callable(self.stop) else self.stop.is_set() if self.stop is not None else False
        if requested: self.reason = 'stopped'
        elif time.monotonic() >= self.deadline: self.reason = 'timeout'
        return self.reason is not None
    def check(self):
        if self.cancelled(): raise _Cancelled


@contextmanager
def _watch(control, deadline, solver):
    done = threading.Event()
    def monitor():
        while not done.wait(.01):
            try:
                expired = control.cancelled() or time.monotonic() >= deadline
            except BaseException as exc:
                control.error, expired = exc, True
            if expired:
                solver.stop_search()
                return
    thread = threading.Thread(target=monitor, name='diamond-repair-stop', daemon=True)
    thread.start()
    try: yield
    finally:
        done.set(); thread.join()
    if control.error is not None: raise control.error


def _select_patch(problem, current, metrics, size, rng, target):
    movable = [cell for cell in range(problem.n) if cell not in problem.fixed]
    bad = [cell for cell in metrics['bad_cells'] if cell in movable]
    roots = list({cell for a, _, b, _ in metrics['unmatched_edges'] for cell in (a,b) if cell in movable})
    if not roots and target == 'gold':
        roots = [cell for cell, _ in problem.gold_nogood(current) if cell in movable]
    if not roots: roots = movable.copy()
    rng.shuffle(roots)
    selected = set(bad)
    if roots: selected.add(roots[0])
    desired = min(len(movable), max(size, len(selected)))
    while len(selected) < desired:
        # Include matched neighbors too: restricting to defective cells traps
        # useful pieces outside the movable neighborhood.
        neighbors = list({int(other) for cell in selected for other in problem.neighbors[cell]
                          if other >= 0 and int(other) in movable and int(other) not in selected})
        options = neighbors or [cell for cell in movable if cell not in selected]
        selected.add(rng.choice(sorted(options)))
    return sorted(selected)


def _solve_patch(problem, current, patch, *, cp_model, control, seconds, seed, target, loop_aware=True, diagnostics=None):
    control.check()
    deadline = min(control.deadline, time.monotonic()+seconds)
    cells = sorted(patch)
    inventory = {current[cell]//3 for cell in cells}
    model = cp_model.CpModel()
    variables, domains, tiles = {}, {}, []
    for cell in cells:
        control.check()
        allowed = [code for code in problem.domains[cell] if code//3 in inventory
                   and (cell not in problem.fixed or code == problem.fixed[cell])]
        if not allowed: return None, 'NO_LOCAL_DOMAIN'
        domains[cell] = allowed
        variables[cell] = model.new_int_var_from_domain(cp_model.Domain.from_values(allowed), f'code_{cell}')
        tile = model.new_int_var_from_domain(cp_model.Domain.from_values(sorted(inventory)), f'tile_{cell}')
        model.add_allowed_assignments([variables[cell], tile], [(code,code//3) for code in allowed])
        tiles.append(tile)
        if current[cell] in allowed: model.add_hint(variables[cell], current[cell])
    model.add_all_different(tiles)
    # An equal-score alternative is useful; solving the unchanged board is not.
    model.add_forbidden_assignments([variables[cell] for cell in cells], [tuple(current[cell] for cell in cells)])
    matches, constant = [], 0
    masks = {}
    def side_value(cell, side, reverse=False):
        if cell not in variables:
            value = int(problem.masks[current[cell], side])
            return reverse_mask(value) if reverse else value
        key = cell, side, reverse
        if key not in masks:
            rows = [(code, reverse_mask(problem.masks[code,side]) if reverse else int(problem.masks[code,side])) for code in domains[cell]]
            values = sorted({value for _, value in rows})
            masks[key] = model.new_int_var_from_domain(cp_model.Domain.from_values(values), f'mask_{cell}_{side}_{int(reverse)}')
            model.add_allowed_assignments([variables[cell], masks[key]], rows)
        return masks[key]
    for index, (a,sa,b,sb) in enumerate(problem.edges):
        control.check()
        if a not in variables and b not in variables:
            constant += int(problem.masks[current[a],sa]) == reverse_mask(problem.masks[current[b],sb])
            continue
        match = model.new_bool_var(f'matched_{index}')
        left, right = side_value(a,sa), side_value(b,sb,True)
        model.add(left == right).only_enforce_if(match)
        model.add(left != right).only_enforce_if(match.Not())
        matches.append(match)
    model.maximize(sum(matches))
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault('loop_cuts',0)
    cut_keys = set()
    def add_loop_cut(board):
        # A full edge-matched rejected witness certifies this Gold nogood.
        # Outside literals already hold as constants, so only patch literals
        # need to appear in the local clause. An untouched external loop is
        # left to a later, larger/different patch; it is never global UNSAT.
        cut = [(cell,code) for cell,code in problem.gold_nogood(board) if cell in variables]
        key = tuple(cut)
        if not cut or key in cut_keys: return False
        model.add_forbidden_assignments([variables[cell] for cell,_ in cut], [tuple(code for _,code in cut)])
        cut_keys.add(key); diagnostics['loop_cuts'] += 1
        return True
    if loop_aware and target == 'gold':
        initial = score(problem,current)
        if not initial['domain_violations'] and initial['matched_edges'] == len(problem.edges) and not problem.validate(current)['valid']:
            add_loop_cut(current)
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed % 2147483647
    fallback = None
    while True:
        control.check()
        if time.monotonic() >= deadline: return fallback, 'MODEL_BUDGET' if fallback is None else 'UNKNOWN'
        solver.parameters.max_time_in_seconds = max(.000001, deadline-time.monotonic())
        with _watch(control, deadline, solver):
            status = solver.solve(model)
        name = solver.status_name(status)
        if status == cp_model.MODEL_INVALID:
            raise RuntimeError('Repair model is invalid: '+model.validate())
        if status not in (cp_model.FEASIBLE, cp_model.OPTIMAL): return fallback, name
        candidate = current.copy()
        for cell in cells: candidate[cell] = int(solver.value(variables[cell]))
        checked = score(problem, candidate)
        if checked['domain_violations'] or {candidate[cell]//3 for cell in cells} != inventory:
            raise RuntimeError('Repair violated domains or its physical-tile inventory.')
        if any(candidate[cell] != current[cell] for cell in range(problem.n) if cell not in variables):
            raise RuntimeError('Repair changed a cell outside its patch.')
        if checked['matched_edges'] != constant+int(round(solver.objective_value)):
            raise RuntimeError('Repair failed independent CPU seam scoring.')
        if fallback is None or checked['objective'] >= score(problem,fallback)['objective']:
            fallback = candidate
        if not loop_aware or target != 'gold' or checked['matched_edges'] != len(problem.edges):
            return fallback,name
        if problem.validate(candidate)['valid']:
            return candidate,name
        if not add_loop_cut(candidate): return fallback,name
        if control.cancelled() or time.monotonic() >= deadline: return fallback,'UNKNOWN'


def repair(problem, codes, *, seconds=10, seed=0, target='gold', stop=None, progress=None,
           patch_sizes=(8,16,24,40), per_attempt_seconds=2, max_attempts=64,
           candidate_boards=None, archive_size=8, adaptive=True, cache_attempts=True, loop_aware=True):
    """Return the best checked board; an inconclusive patch never proves UNSAT.

    ``codes`` may come from GPU sampling or a saved full-board hint. It must be
    a permutation honoring fixed placements. A feasible repair also obeys every
    domain; all initially illegal movable cells are included in each patch.
    """
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative.')
    if target not in ('gold','edges') or type(seed) is not int or seed < 0:
        raise ValueError('Invalid repair target or seed.')
    if not patch_sizes or any(type(size) is not int or size < 1 for size in patch_sizes):
        raise ValueError('patch_sizes must contain positive integers.')
    if isinstance(per_attempt_seconds,bool) or not isinstance(per_attempt_seconds,(int,float)) or not math.isfinite(per_attempt_seconds) or per_attempt_seconds <= 0 or type(max_attempts) is not int or max_attempts < 1:
        raise ValueError('Repair attempt limits must be positive and finite.')
    if stop is not None and not callable(stop) and not callable(getattr(stop, 'is_set', None)):
        raise ValueError('stop must be a callback or Event.')
    control = _Control(seconds, stop)
    current = list(codes.tolist() if hasattr(codes,'tolist') else codes)
    metrics = score(problem, current)
    validation = problem.validate(current)
    best, best_metrics, best_validation = current.copy(), metrics, validation
    stats = {'attempts':0, 'accepted':0, 'equal_alternatives':0, 'improvements':0,
             'local_no_candidate':0, 'candidates_checked':1, 'revisited_boards':0,
             'workers':1, 'native_statuses':{}, 'patch_sizes':list(patch_sizes)}
    archive = diverse_archive([{'codes':current.copy(), **metrics}] +
        [{'codes':board.tolist() if hasattr(board,'tolist') else list(board), **score(problem,board)}
         for board in ([] if candidate_boards is None else candidate_boards)], capacity=archive_size)
    if archive:
        current = archive[0]['codes'].copy(); metrics = score(problem,current)
        validation = problem.validate(current)
        best,best_metrics,best_validation = current.copy(),metrics,validation
    for item in archive:
        report = problem.validate(item['codes'])
        if not item['domain_violations'] and (report['valid'] or target=='edges' and report['matched_edges']==len(problem.edges)):
            current = item['codes'].copy(); metrics=score(problem,current); validation=report
            best,best_metrics,best_validation=current.copy(),metrics,report
            break
    stats.update(loop_cuts=0, attempt_cache_hits=0, conclusive_attempts=0, patch_selections=0,
                 patch_size_counts={}, size_changes=0, seed_switches=0)
    seen = {tuple(current)}
    conclusive = set()
    size_index, stalls, seed_cursor, previous_size = 0,0,0,None
    def snapshot_stats():
        return {**stats, 'native_statuses':dict(stats['native_statuses']), 'patch_sizes':list(stats['patch_sizes']), 'patch_size_counts':dict(stats['patch_size_counts'])}
    def finish(status, reason=None):
        result = {'status':status, 'complete':status in ('solved','edge_perfect'),
                  'scope':'local_repair_only', 'best_board':best.copy(), 'best_score':best_metrics['matched_edges'],
                  'best_objective':best_metrics['objective'], 'domain_violations':best_metrics['domain_violations'],
                  'validation':best_validation, 'elapsed_seconds':time.monotonic()-control.began, 'stats':snapshot_stats(),
                  'candidate_archive':[{**item,'codes':item['codes'].copy()} for item in archive]}
        if reason is not None: result['reason'] = reason
        return result
    def solved(report, values):
        return not values['domain_violations'] and (report['valid'] or target == 'edges' and report['unique_tiles'] and report['matched_edges'] == len(problem.edges))
    def emit(event, **extra):
        if progress is not None:
            progress({'event':event, 'phase':'repair', 'elapsed_seconds':time.monotonic()-control.began,
                      'codes':best.copy(), 'validation':best_validation, 'best_score':best_metrics['matched_edges'],
                      'stats':snapshot_stats(), **extra})
    try:
        control.check()
        if solved(validation, metrics): return finish('solved' if validation['valid'] else 'edge_perfect')
        if any(cell in problem.fixed for cell in metrics['bad_cells']):
            return finish('skipped', 'A fixed cell is outside its legal domain; exact DFS must decide feasibility.')
        if len(problem.fixed) == problem.n: return finish('inconclusive', 'No movable cells.')
        try: cp_model = _load_cp_model()
        except ImportError:
            result = finish('skipped', 'OR-Tools is unavailable; local repair was skipped and exact DFS can continue.')
            emit('repair_skipped', reason=result['reason'])
            return result
        rng = random.Random(seed)
        for attempt in range(max_attempts):
            control.check()
            chosen_size = patch_sizes[size_index] if adaptive else patch_sizes[attempt % len(patch_sizes)]
            patch = _select_patch(problem,current,metrics,chosen_size,rng,target)
            if not patch: break
            stats['patch_selections'] += 1
            size = len(patch)
            stats['patch_size_counts'][str(size)] = stats['patch_size_counts'].get(str(size),0)+1
            if previous_size is not None and size != previous_size: stats['size_changes'] += 1
            previous_size = size
            # Exact query identity includes all outside placements, the current
            # excluded board, and the movable region. This invocation has fixed
            # problem domains and objective. Timeout/FEASIBLE is never cached.
            query = (tuple(current),tuple(patch),target,bool(loop_aware))
            if cache_attempts and query in conclusive:
                stats['attempt_cache_hits'] += 1
                if adaptive: size_index = (size_index+1)%len(patch_sizes)
                continue
            stats['attempts'] += 1
            emit('repair_attempt',patch_cells=patch)
            details = {}
            candidate,native_status = _solve_patch(problem,current,patch,cp_model=cp_model,control=control,
                seconds=per_attempt_seconds,seed=seed+attempt,target=target,loop_aware=loop_aware,diagnostics=details)
            stats['loop_cuts'] += details.get('loop_cuts',0)
            stats['native_statuses'][native_status] = stats['native_statuses'].get(native_status,0)+1
            if cache_attempts and native_status in ('OPTIMAL','INFEASIBLE','NO_LOCAL_DOMAIN'):
                conclusive.add(query); stats['conclusive_attempts'] += 1
            if candidate is None:
                stats['local_no_candidate'] += 1
                stalls += 1
            else:
                candidate_metrics = score(problem,candidate)
                candidate_validation = problem.validate(candidate)
                stats['candidates_checked'] += 1
                key = tuple(candidate)
                if key in seen: stats['revisited_boards'] += 1
                seen.add(key)
                archive = diverse_archive([*archive,{'codes':candidate.copy(),**candidate_metrics}],capacity=archive_size)
                improved = candidate_metrics['objective'] > metrics['objective']
                if candidate_metrics['objective'] >= metrics['objective']:
                    stats['accepted'] += 1
                    if not improved: stats['equal_alternatives'] += 1
                    current,metrics = candidate,candidate_metrics
                better = candidate_metrics['objective'] > best_metrics['objective']
                if candidate_metrics['objective'] >= best_metrics['objective'] or solved(candidate_validation,candidate_metrics):
                    if better: stats['improvements'] += 1
                    best,best_metrics,best_validation = candidate.copy(),candidate_metrics,candidate_validation
                emit('repair_progress',candidate_score=candidate_metrics['matched_edges'])
                if solved(candidate_validation,candidate_metrics):
                    emit('candidate')
                    return finish('solved' if candidate_validation['valid'] else 'edge_perfect')
                stalls = 0 if improved else stalls+1
            if adaptive:
                if stalls == 0: size_index = 0
                elif native_status in ('UNKNOWN','MODEL_BUDGET','FEASIBLE'):
                    size_index = max(0,size_index-1)
                else: size_index = (size_index+1)%len(patch_sizes)
            if stalls >= 2*len(patch_sizes) and len(archive)>1:
                seed_cursor = (seed_cursor+1)%len(archive)
                alternate = archive[seed_cursor]['codes']
                if alternate != current:
                    current=alternate.copy();metrics=score(problem,current);stats['seed_switches']+=1
                stalls=0
        return finish('inconclusive', 'The bounded local attempt limit was reached; global feasibility is unknown.')
    except _Cancelled:
        return finish(control.reason or 'timeout')
