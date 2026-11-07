"""Independent unpacked CPU reference for bounded obstruction-guided proposals.

This module never constructs a GPU device. The proposal remains a heuristic:
local donor scores are not a bound, proof, or predicted full-board improvement.
"""
import random
import unittest

from constraint_lns import score
from search_problem import SearchProblem
from test_constraint_dfs import board, double_face, multiple_loops, simple_tiles, tetrahedron_tiles
from validator import reverse_mask


class XorShift:
    def __init__(self, state):
        if type(state) is not int or not 0 <= state < 2**32: raise ValueError('Invalid 32-bit RNG state.')
        self.state = state
    def next(self):
        value = self.state
        value ^= (value << 13) & 0xffffffff
        value ^= value >> 17
        value ^= (value << 5) & 0xffffffff
        self.state = value & 0xffffffff
        return self.state


def raw_objective(problem, codes):
    """Global domain/seam score, permitting hypothetical duplicate-piece trials."""
    legal = sum(code in problem.domains[cell] for cell, code in enumerate(codes))
    seams = sum(int(problem.masks[codes[a], sa]) == reverse_mask(problem.masks[codes[b], sb])
                for a, sa, b, sb in problem.edges)
    return legal*(len(problem.edges)+1)+seams


def local_fit(problem, board_codes, receiver, code):
    value = int(code in problem.domains[receiver])*(len(problem.edges)+1)
    for side, other in enumerate(problem.neighbors[receiver]):
        if int(other) >= 0:
            value += int(problem.masks[code, side]) == reverse_mask(problem.masks[board_codes[int(other)], int(problem.sides[receiver, side])])
    return int(value)


def obstructions(problem, codes, free_cells, *, gold_aware=False):
    metrics = score(problem, codes)
    closed = gold_aware and not metrics['domain_violations'] and metrics['matched_edges'] == len(problem.edges) and not problem.validate(codes)['valid']
    bad = set()
    for cell in free_cells:
        if codes[cell] not in problem.domains[cell]: bad.add(cell)
        for side, other in enumerate(problem.neighbors[cell]):
            if int(other) >= 0 and int(problem.masks[codes[cell], side]) != reverse_mask(problem.masks[codes[int(other)], int(problem.sides[cell, side])]):
                bad.add(cell)
        if closed and problem.tiles[codes[cell]//3]['segments']: bad.add(cell)
    return bad


def _inputs(problem, codes, free_cells):
    score(problem, codes)
    free = list(free_cells)
    if len(set(free)) != len(free) or any(type(cell) is not int or not 0<=cell<problem.n or cell in problem.fixed for cell in free):
        raise ValueError('Invalid movable-cell list.')
    return free


def guided_cycle_reference(problem, codes, free_cells, random_state, count, *, gold_aware=False):
    """Mirror only guided_cycle's RNG entry/exit; scores use unpacked puzzle data.

    The caller consumes the guided/fallback decision and chain-length draws.
    Returned trace records every eligible donor/rotation tested, in tie order.
    """
    free = _inputs(problem, codes, free_cells)
    if type(count) is not int or not 1<=count<=min(6, len(free)): raise ValueError('Invalid guided chain length.')
    rng = XorShift(random_state); offset = rng.next()%len(free)
    scan = free[offset:]+free[:offset]; bad = obstructions(problem, codes, free, gold_aware=gold_aware)
    first = next((cell for cell in scan if cell in bad), scan[0])
    cells, old, proposed, trace = [first], [codes[first]], [], []
    checks = 0
    for receiver_index in range(count-1):
        offset = rng.next()%len(free); rotation = rng.next()%3
        scan = free[offset:]+free[:offset]
        donors = [cell for cell in scan if cell not in cells][:16]
        trials = [(donor, 3*(codes[donor]//3)+(rotation+k)%3) for donor in donors for k in range(3)]
        scores = [local_fit(problem, codes, cells[receiver_index], code) for donor, code in trials]
        maximum = max(scores); best = scores.index(maximum)  # Strict greater-than keeps first tie.
        donor, code = trials[best]; checks += len(trials)
        trace.append({'receiver': cells[receiver_index], 'donors': donors, 'rotation': rotation,
                      'trials': [[cell, candidate, value] for (cell, candidate), value in zip(trials, scores)],
                      'chosen_donor': donor, 'chosen_code': code, 'chosen_score': maximum})
        proposed.append(code); cells.append(donor); old.append(codes[donor])
    rotation = rng.next()%3
    trials = [3*(old[0]//3)+(rotation+k)%3 for k in range(3)]
    values = [local_fit(problem, codes, cells[-1], code) for code in trials]
    chosen = values.index(max(values)); proposed.append(trials[chosen]); checks += 3
    trace.append({'receiver': cells[-1], 'donors': [cells[0]], 'rotation': rotation,
                  'trials': [[cells[0], candidate, value] for candidate, value in zip(trials, values)],
                  'chosen_donor': cells[0], 'chosen_code': trials[chosen], 'chosen_score': values[chosen]})
    result = list(codes)
    for cell, code in zip(cells, proposed): result[cell] = code
    # Independent complete permutation/fixed-placement check, not a proposal filter.
    score(problem, result)
    delta = raw_objective(problem, result)-raw_objective(problem, codes)
    return {'cells': cells, 'old': old, 'next': proposed, 'board': result, 'random_state': rng.state,
            'base_delta': delta, 'guided': True, 'trace': trace,
            'counters': {'guided_proposals': 1, 'guided_candidate_checks': checks,
                         'guided_obstruction_starts': int(first in bad), 'guided_base_improvements': int(delta > 0)}}


def proposal_reference(problem, codes, free_cells, random_state, flags, *, gold_aware=False):
    """Proposal generation only: no Metropolis/tabu draws or acceptance decision."""
    free = _inputs(problem, codes, free_cells)
    if type(flags) is not int or not 0<=flags<16: raise ValueError('Invalid sampling flags.')
    rng = XorShift(random_state); count = 1
    counters = {'guided_proposals': 0, 'guided_candidate_checks': 0,
                'guided_obstruction_starts': 0, 'guided_base_improvements': 0, 'chain_proposals': 0}
    if not free:
        return {'cells': [], 'old': [], 'next': [], 'board': list(codes), 'random_state': rng.state,
                'base_delta': 0, 'guided': False, 'trace': [], 'counters': counters}
    guided = bool(flags&8) and rng.next()%4 != 0
    if guided:
        count = 3+rng.next()%(min(6, len(free))-2) if len(free)>=3 else len(free)
        answer = guided_cycle_reference(problem, codes, free, rng.state, count, gold_aware=gold_aware)
        answer['counters']['chain_proposals'] = 1
        return answer
    if flags&1 and len(free)>=3 and rng.next()%4 != 0:
        count = 3+rng.next()%(min(6, len(free))-2); counters['chain_proposals'] += 1
    elif len(free)>=2:
        count = 1+rng.next()%2
    cells = []
    for index in range(count):
        chosen = free[rng.next()%len(free)]
        while chosen in cells: chosen = free[rng.next()%len(free)]
        cells.append(chosen)
    old = [codes[cell] for cell in cells]
    proposed = [3*(old[(i+1)%count]//3)+rng.next()%3 for i in range(count)]
    result = list(codes)
    for cell, code in zip(cells, proposed): result[cell] = code
    score(problem, result)
    return {'cells': cells, 'old': old, 'next': proposed, 'board': result, 'random_state': rng.state,
            'base_delta': raw_objective(problem, result)-raw_objective(problem, codes),
            'guided': False, 'trace': [], 'counters': counters}


class GuidedChainReferenceTests(unittest.TestCase):
    def test_small_proposals_preserve_permutation_fixed_cells_and_source(self):
        tiles, geometry = tetrahedron_tiles(); before = [0, 4, 6, 9]
        for fixed_count in range(4):
            fixed = {cell: before[cell] for cell in range(fixed_count)}
            problem = SearchProblem(tiles, geometry, fixed=fixed)
            free = [cell for cell in range(problem.n) if cell not in fixed]
            for seed in range(1, 65):
                for gold in (False, True):
                    result = proposal_reference(problem, before, free, seed, 9, gold_aware=gold)
                    self.assertEqual(sorted(code//3 for code in result['board']), list(range(4)))
                    self.assertEqual({cell: result['board'][cell] for cell in fixed}, fixed)
                    self.assertEqual(len(set(result['cells'])), len(result['cells']))
                    self.assertTrue(set(result['cells'])<=set(free))
                    for index, cell in enumerate(result['cells']):
                        self.assertEqual(result['next'][index]//3, result['old'][(index+1)%len(result['cells'])]//3)
                    self.assertEqual(before, [0, 4, 6, 9])
                    self.assertEqual(result['base_delta'], raw_objective(problem, result['board'])-raw_objective(problem, before))

    def test_local_candidates_are_maximal_in_checked_donor_family_and_ties_keep_first(self):
        problem = SearchProblem(*tetrahedron_tiles()); codes = [0, 4, 6, 9]
        for seed in range(1, 41):
            result = guided_cycle_reference(problem, codes, range(4), seed, 4)
            for entry in result['trace']:
                maximum = max(value for donor, code, value in entry['trials'])
                first = next(row for row in entry['trials'] if row[2] == maximum)
                self.assertEqual(first, [entry['chosen_donor'], entry['chosen_code'], entry['chosen_score']])
                for donor, code, value in entry['trials']:
                    trial = codes.copy(); receiver = entry['receiver']; trial[receiver] = code
                    self.assertEqual(value-local_fit(problem, codes, receiver, codes[receiver]),
                                     raw_objective(problem, trial)-raw_objective(problem, codes))
            self.assertEqual(result['counters']['guided_candidate_checks'], sum(len(entry['trials']) for entry in result['trace']))

    def test_actual_domain_or_seam_obstruction_is_selected(self):
        problem = SearchProblem(*tetrahedron_tiles()); codes = [0, 4, 6, 9]
        bad = obstructions(problem, codes, range(4))
        self.assertTrue(bad)
        for seed in range(1, 65):
            result = guided_cycle_reference(problem, codes, range(4), seed, 3)
            self.assertIn(result['cells'][0], bad)
            self.assertEqual(result['counters']['guided_obstruction_starts'], 1)
        n = 40
        problem = SearchProblem([{'id': i, 'segments': [[[0, 6], [1, 6]]]} for i in range(n)],
                                board(n, [(cell, 1, (cell+1)%n, 0) for cell in range(n)]))
        codes = [3*cell for cell in range(n)]; problem.domains[31] = [3*32]
        self.assertEqual(obstructions(problem, codes, range(n)), {31})
        for seed in range(1, 33):
            result = guided_cycle_reference(problem, codes, range(n), seed, 6)
            self.assertEqual(result['cells'][0], 31)
            self.assertEqual(result['counters']['guided_candidate_checks'], 5*16*3+3)
            self.assertEqual([len(entry['donors']) for entry in result['trace']], [16]*5+[1])

    def test_closed_loop_obstruction_requires_gold_and_does_not_invent_solved_obstruction(self):
        loops = SearchProblem(multiple_loops(), double_face()); codes = [0, 3]
        self.assertEqual(obstructions(loops, codes, [0, 1]), set())
        self.assertEqual(obstructions(loops, codes, [0, 1], gold_aware=True), {0, 1})
        for gold in (False, True):
            result = guided_cycle_reference(loops, codes, [0, 1], 11, 2, gold_aware=gold)
            self.assertEqual(result['counters']['guided_obstruction_starts'], int(gold))
        solved = SearchProblem(simple_tiles(), double_face())
        self.assertTrue(solved.validate(codes)['valid'])
        result = guided_cycle_reference(solved, codes, [0, 1], 11, 2, gold_aware=True)
        self.assertEqual(result['counters']['guided_obstruction_starts'], 0)

    def test_disabled_feature_keeps_exact_legacy_proposal_draw_sequence(self):
        problem = SearchProblem(*tetrahedron_tiles()); codes = [0, 4, 6, 9]; free = list(range(4))
        for flags in range(8):
            for seed in range(1, 65):
                rng = XorShift(seed); count = 1
                if flags&1 and rng.next()%4 != 0: count = 3+rng.next()%2
                else: count = 1+rng.next()%2
                selected = []
                while len(selected)<count:
                    cell = free[rng.next()%4]
                    if cell not in selected: selected.append(cell)
                expected = [3*(codes[selected[(i+1)%count]]//3)+rng.next()%3 for i in range(count)]
                result = proposal_reference(problem, codes, free, seed, flags)
                self.assertEqual(result['cells'], selected); self.assertEqual(result['next'], expected)
                self.assertEqual(result['random_state'], rng.state); self.assertFalse(result['guided'])
                self.assertEqual(result['counters']['guided_candidate_checks'], 0)

    def test_no_free_cells_and_bounded_guided_inputs(self):
        problem = SearchProblem(simple_tiles(), double_face(), fixed={0: 0, 1: 3})
        result = proposal_reference(problem, [0, 3], [], 17, 9)
        self.assertEqual(result['random_state'], 17); self.assertEqual(result['board'], [0, 3])
        with self.assertRaises(ValueError): guided_cycle_reference(problem, [0, 3], [], 17, 1)
        with self.assertRaises(ValueError): proposal_reference(problem, [0, 3], [0], 17, 9)
        fresh = SearchProblem(simple_tiles(), double_face())
        for count in (0, 3, True):
            with self.assertRaises(ValueError): guided_cycle_reference(fresh, [0, 3], [0, 1], 17, count)


if __name__ == '__main__': unittest.main()
