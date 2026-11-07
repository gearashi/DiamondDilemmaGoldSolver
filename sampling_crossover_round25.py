"""Bounded host cycle crossover for heuristic sampler archives.

Whole relative-permutation cycles preserve inventory. Every parent and proposed
child is independently scored. No outcome excludes exact search work. History
is a bounded FIFO of exact child boards, not global deduplication or coverage.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

VERSION = 1
MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
STAT_KEYS = ('cycle_crossover_rounds', 'cycle_parent_pairs', 'cycle_components',
             'cycle_candidates', 'cycle_duplicate_skips', 'cycle_domain_rejections',
             'cycle_improvements', 'cycle_gold_solutions', 'cycle_history_evictions',
             'cycle_injections', 'cycle_parent_scores', 'cycle_parent_duplicates',
             'cycle_pair_cursor_evictions', 'cycle_offers')


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _board(problem, values):
    if type(values) not in (list, tuple) or len(values) != problem.n or any(
            not _integer(code, 0, 3*problem.n-1) for code in values):
        raise ValueError('Crossover needs integer orientation codes for every cell.')
    if len({code//3 for code in values}) != problem.n:
        raise ValueError('Crossover parent/child must preserve every physical piece.')
    if any(values[cell] != code for cell, code in problem.fixed.items()):
        raise ValueError('Crossover cannot change fixed placements.')
    return tuple(values)


def relative_cycles(left, right):
    """Cell cycles of owner_left[piece_right[cell]], ordered by minimum cell.

    Callers validate both complete piece permutations first. Rotation-only
    differences are singleton physical cycles and can be mixed independently.
    """
    n = len(left)
    if len(right) != n or len({code//3 for code in left}) != n or len({code//3 for code in right}) != n or {code//3 for code in left} != {code//3 for code in right}:
        raise ValueError('Relative cycles require matching piece permutations.')
    owner = {code//3: cell for cell, code in enumerate(left)}
    unseen = set(range(n)); result = []
    while unseen:
        first = min(unseen); cell = first; cycle = []
        while cell in unseen:
            unseen.remove(cell); cycle.append(cell); cell = owner[right[cell]//3]
        if cell != first: raise ValueError('Relative permutation cycle did not close.')
        result.append(tuple(cycle))
    return tuple(result)


class CycleCrossover:
    def __init__(self, problem, *, gold_aware=False, max_pairs=4, max_candidates=16,
                 history_limit=256, resume=None):
        if not _integer(problem.n, 1, 160) or type(gold_aware) is not bool:
            raise ValueError('Invalid cycle crossover puzzle/objective.')
        if not _integer(max_pairs, 1, 28) or not _integer(max_candidates, 1, 256) or not _integer(history_limit, 1, 4096):
            raise ValueError('Invalid bounded cycle crossover policy.')
        self.problem = problem
        self.policy = {'gold_aware': gold_aware, 'max_pairs': max_pairs,
                       'max_candidates': max_candidates, 'history_limit': history_limit}
        self.binding = self._identity()
        self._state = {'pair_cursor': 0, 'offsets': [], 'history': [], 'injection_pending': False,
                       'stats': {key: 0 for key in STAT_KEYS}}
        if resume is not None: self._restore(resume)

    def _identity(self):
        p = self.problem
        sources = ('sampling_crossover_round25.py', 'gpu_sampling.py', 'constraint_lns.py',
                   'search_problem.py', 'validator.py', 'geometry.py')
        identity = {'version': VERSION, 'policy': self.policy, 'fingerprint': p.fingerprint,
                    'n': p.n, 'tiles': p.tiles, 'edges': p.edges, 'board_edges': p.board.edges,
                    'fixed': sorted(p.fixed.items()), 'domains': [list(row) for row in p.domains],
                    'masks': p.masks.tolist(), 'neighbors': p.neighbors.tolist(), 'sides': p.sides.tolist(),
                    'segments': p.total_segments, 'segment_ports': p._segment_ports,
                    'sources': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in sources}}
        return _digest(identity)

    def _check_scope(self):
        if self._identity() != self.binding:
            raise ValueError('Cycle crossover input/source/policy changed.')

    @property
    def stats(self):
        self._check_scope()
        return dict(self._state['stats'])

    def report(self):
        return self.stats

    def _restore(self, value):
        try:
            raw = _encoded(value)
            if len(raw) > MAX_SNAPSHOT_BYTES: raise ValueError('Crossover snapshot exceeds byte limit.')
            value = json.loads(raw)
        except (TypeError, OverflowError, RecursionError) as exc:
            raise ValueError('Invalid crossover checkpoint JSON.') from exc
        if type(value) is not dict or set(value) != {'cycle_crossover_version', 'binding', 'policy', 'state', 'digest'}:
            raise ValueError('Invalid crossover checkpoint envelope.')
        policy = value['policy']
        if type(value['cycle_crossover_version']) is not int or value['cycle_crossover_version'] != VERSION or value['binding'] != self.binding or type(policy) is not dict or set(policy) != set(self.policy) or any(type(policy[key]) is not type(expected) or policy[key] != expected for key, expected in self.policy.items()):
            raise ValueError('Crossover checkpoint scope/source/policy differs.')
        if type(value['digest']) is not str or value['digest'] != _digest({key: data for key, data in value.items() if key != 'digest'}):
            raise ValueError('Crossover checkpoint digest mismatch.')
        state = value['state']
        if type(state) is not dict or set(state) != set(self._state) or not _integer(state['pair_cursor'], 0, 27) or type(state['injection_pending']) is not bool:
            raise ValueError('Invalid crossover controller cursor.')
        stats = state['stats']
        if type(stats) is not dict or set(stats) != set(STAT_KEYS) or any(not _integer(count, 0, 2**53-1) for count in stats.values()):
            raise ValueError('Invalid crossover counters.')
        if not stats['cycle_injections'] <= stats['cycle_offers'] <= stats['cycle_crossover_rounds'] or stats['cycle_offers'] > stats['cycle_candidates']:
            raise ValueError('Inconsistent crossover event counters.')
        if state['injection_pending'] and stats['cycle_injections'] >= stats['cycle_offers']:
            raise ValueError('Crossover pending injection has no offer.')
        if type(state['history']) is not list or len(state['history']) > self.policy['history_limit']:
            raise ValueError('Crossover history exceeds policy.')
        history = [_board(self.problem, row) for row in state['history']]
        if len(set(history)) != len(history) or len(history) != min(self.policy['history_limit'], stats['cycle_candidates']) or stats['cycle_history_evictions'] != stats['cycle_candidates']-len(history):
            raise ValueError('Invalid exact crossover history.')
        if any(stats[key] > stats['cycle_candidates'] for key in ('cycle_domain_rejections', 'cycle_improvements', 'cycle_gold_solutions')) or stats['cycle_parent_pairs'] > stats['cycle_crossover_rounds']*self.policy['max_pairs'] or stats['cycle_parent_scores']+stats['cycle_parent_duplicates'] > 8*stats['cycle_crossover_rounds']:
            raise ValueError('Inconsistent bounded crossover work counters.')
        if type(state['offsets']) is not list or len(state['offsets']) > 28:
            raise ValueError('Crossover pair cursors exceed archive bound.')
        keys = set()
        for item in state['offsets']:
            if type(item) is not dict or set(item) != {'parents', 'next'} or type(item['parents']) is not list or len(item['parents']) != 2:
                raise ValueError('Invalid crossover pair record.')
            left, right = [_board(self.problem, row) for row in item['parents']]
            key = left, right
            if left >= right or key in keys: raise ValueError('Repeated or noncanonical crossover pair.')
            keys.add(key)
            count = sum(any(left[cell] != right[cell] for cell in cycle) for cycle in relative_cycles(left, right))
            if not _integer(item['next'], 0, 2*count-1): raise ValueError('Invalid crossover cycle/direction cursor.')
        self._state = state  # Commit only after every record passes.

    def snapshot(self):
        self._check_scope()
        value = {'cycle_crossover_version': VERSION, 'binding': self.binding,
                 'policy': deepcopy(self.policy), 'state': deepcopy(self._state)}
        value['digest'] = _digest(value)
        if len(_encoded(value)) > MAX_SNAPSHOT_BYTES: raise ValueError('Crossover snapshot exceeds byte limit.')
        return value

    def record_injection(self):
        """Caller invokes this only after successful native board-write commit."""
        self._check_scope()
        if not self._state['injection_pending']:
            raise RuntimeError('No unrecorded crossover candidate is available for injection.')
        if self._state['stats']['cycle_injections'] >= 2**53-1:
            raise ValueError('Crossover injection counter limit reached.')
        self._state['stats']['cycle_injections'] += 1
        self._state['injection_pending'] = False

    def propose(self, archive):
        self._check_scope()
        if type(archive) is not list or len(archive) > 8:
            raise ValueError('Cycle crossover archive must contain at most eight rows.')
        from gpu_sampling import gold_score, score_board
        score_fn = gold_score if self.policy['gold_aware'] else score_board
        objective = 'gold_objective' if self.policy['gold_aware'] else 'objective'
        # Scores provided by an archive are hints, never ranking authority.
        parents = []
        for item in archive:
            if type(item) is not dict or 'codes' not in item: raise ValueError('Crossover archive row needs codes.')
            parents.append(_board(self.problem, item['codes']))
        unique = sorted(set(parents))
        metrics = {row: score_fn(self.problem, list(row)) for row in unique}
        state = deepcopy(self._state); stats = state['stats']
        stats['cycle_crossover_rounds'] += 1; stats['cycle_parent_scores'] += len(unique)
        stats['cycle_parent_duplicates'] += len(parents)-len(unique)
        state['injection_pending'] = False
        pairs = list(itertools.combinations(unique, 2)); pair_set = set(pairs)
        offsets = {(tuple(row['parents'][0]), tuple(row['parents'][1])): row['next'] for row in state['offsets']}
        stats['cycle_pair_cursor_evictions'] += len(set(offsets)-pair_set)
        offsets = {key: value for key, value in offsets.items() if key in pair_set}
        history = [tuple(row) for row in state['history']]; seen = set(history); seen_this_round = set(); parent_set = set(unique)
        candidate = candidate_metrics = solution = None
        candidate_rank = None; checked = 0
        start = state['pair_cursor'] % len(pairs) if pairs else 0
        for visit in range(min(self.policy['max_pairs'], len(pairs))):
            if checked >= self.policy['max_candidates']: break
            index = (start+visit)%len(pairs); left, right = pairs[index]
            state['pair_cursor'] = (index+1)%len(pairs); stats['cycle_parent_pairs'] += 1
            cycles = [cycle for cycle in relative_cycles(left, right) if any(left[cell] != right[cell] for cell in cycle)]
            stats['cycle_components'] += len(cycles)
            if not cycles: continue
            key = left, right; total = 2*len(cycles); begin = offsets.get(key, 0); walked = 0
            threshold = max(metrics[left][objective], metrics[right][objective])
            while walked < total and checked < self.policy['max_candidates']:
                slot = (begin+walked)%total; walked += 1
                cycle, direction = cycles[slot//2], slot%2
                base, donor = (left, right) if direction == 0 else (right, left)
                child = list(base)
                for cell in cycle: child[cell] = donor[cell]
                exact = tuple(child)
                if exact in parent_set or exact in seen or exact in seen_this_round:
                    stats['cycle_duplicate_skips'] += 1; continue
                # Limit ALL new child scorings, including domain-rejected hints.
                checked += 1; stats['cycle_candidates'] += 1
                child_metrics = score_fn(self.problem, child)
                history.append(exact); seen.add(exact); seen_this_round.add(exact)
                if len(history) > self.policy['history_limit']:
                    seen.remove(history.pop(0)); stats['cycle_history_evictions'] += 1
                if child_metrics['domain_violations']:
                    stats['cycle_domain_rejections'] += 1; continue
                is_gold = False
                if child_metrics['matched_edges'] == len(self.problem.edges):
                    is_gold = bool(self.problem.validate(child)['valid'])
                    if is_gold: stats['cycle_gold_solutions'] += 1
                improved = child_metrics[objective] > threshold
                if improved: stats['cycle_improvements'] += 1
                if not improved and not is_gold: continue
                rank = (is_gold, child_metrics[objective])
                if candidate is None or rank > candidate_rank or rank == candidate_rank and exact < tuple(candidate):
                    candidate, candidate_metrics, candidate_rank = child, child_metrics, rank
                if is_gold and (solution is None or exact < tuple(solution)): solution = child.copy()
            offsets[key] = (begin+walked)%total
        if not pairs: state['pair_cursor'] = 0
        state['offsets'] = [{'parents': [list(left), list(right)], 'next': offset}
                            for (left, right), offset in sorted(offsets.items())]
        state['history'] = [list(row) for row in history]
        if candidate is not None:
            state['injection_pending'] = True; stats['cycle_offers'] += 1
        if any(count > 2**53-1 for count in stats.values()): raise ValueError('Crossover counter limit reached.')
        self._check_scope()
        self._state = state
        return {'candidate': None if candidate is None else candidate.copy(),
                'metrics': None if candidate_metrics is None else dict(candidate_metrics),
                'solution': None if solution is None else solution.copy(), 'stats': dict(stats)}
