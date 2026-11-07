"""Pure integer/acceptance control-flow reference for lazy Gold scoring.

The optional float32 threshold emulates the expression structure, not a claim
that CUDA expf and every WebGPU exp implementation are bit-identical. Lazy and
eager paths use the SAME threshold function. Hardware trajectory comparisons
must separately run within each actual backend.
"""
import math
import random
import struct
import unittest

COUNTERS = ('old_base_scans_saved', 'old_loop_traces_saved', 'candidate_loop_traces',
            'tabu_traces_skipped', 'metropolis_traces_skipped', 'early_metropolis_draws')


def f32(value): return struct.unpack('<f', struct.pack('<f', value))[0]


def next_f32(value):
    bits = struct.unpack('<I', struct.pack('<f', value))[0]
    return struct.unpack('<f', struct.pack('<I', bits+1))[0]


def heat_reference(iteration, lane, exchange):
    if exchange: return f32(f32(.2)*f32(math.pow(f32(1.5), f32(lane%8))))
    return f32(f32(.25)+f32(f32(1.25)*f32(f32(1)-f32(f32(iteration%4096)/f32(4096)))))


def threshold_reference(acceptance, heat):
    return f32(f32(1000000)*f32(math.exp(f32(f32(acceptance)/heat))))


def decompose(score, total_segments):
    if type(score) is not int or score < 0 or type(total_segments) is not int or total_segments < 0:
        raise ValueError('Nonnegative integer objective and segment count required.')
    weight = total_segments+1
    return score//weight, total_segments-score%weight


class Tape:
    def __init__(self, state=1, values=None):
        self.state = state; self.values = None if values is None else list(values); self.draws = 0
    def next(self):
        if self.values is None:
            value = self.state
            value ^= (value << 13) & 0xffffffff
            value ^= value >> 17
            value ^= (value << 5) & 0xffffffff
            self.state = value & 0xffffffff
        else:
            if self.draws >= len(self.values): raise AssertionError('Unexpected extra random draw.')
            self.state = self.values[self.draws]
        self.draws += 1
        return self.state


def acceptance_reference(old_base, old_loops, candidate_base, candidate_loops, total_segments,
                         best_score, *, tabu=False, exchange=False, iteration=0, lane=0,
                         lazy=True, random_state=1, random_values=None, threshold=threshold_reference):
    """Compare one proposal after generation; includes the best-tie random draw."""
    if any(type(v) is not int or v < 0 for v in (old_base, old_loops, candidate_base, candidate_loops, total_segments, best_score)) or old_loops > total_segments or candidate_loops > total_segments:
        raise ValueError('Invalid finite objective components.')
    weight = total_segments+1
    current = old_base*weight+total_segments-old_loops
    counters = {key: 0 for key in COUNTERS}
    rng = Tape(random_state, random_values)
    if lazy:
        before, old_count = decompose(current, total_segments)
        counters['old_base_scans_saved'] = 1
        counters['old_loop_traces_saved'] = int(total_segments > 0)
    else:
        before, old_count = old_base, old_loops
    base_delta = candidate_base-before
    heat = heat_reference(iteration, lane, exchange)
    accept = skip = predrawn = False; delta = 0
    if lazy and tabu and candidate_base*weight+total_segments <= best_score:
        skip = True; counters['tabu_traces_skipped'] = int(total_segments > 0)
    elif lazy and not tabu and not exchange and base_delta < 0:
        predrawn = True; counters['early_metropolis_draws'] = 1
        accept = f32(rng.next()%1000000) < threshold(f32(base_delta), heat)
        if not accept:
            skip = True; counters['metropolis_traces_skipped'] = int(total_segments > 0)
    if not skip:
        delta = base_delta*weight+old_count-candidate_loops
        if lazy: counters['candidate_loop_traces'] = int(total_segments > 0)
        acceptance = f32(f32(delta)/f32(weight)) if exchange else f32(base_delta if base_delta else old_count-candidate_loops)
        if not predrawn:
            accept = (not tabu or current+delta > best_score) and (
                acceptance >= 0 or f32(rng.next()%1000000) < threshold(acceptance, heat))
    score_after = current+delta if accept else current
    refresh = bool(accept and (score_after > best_score or score_after == best_score and rng.next()%64 == 0))
    best_after = score_after if refresh else best_score
    return {'accepted': bool(accept), 'score': score_after, 'best_score': best_after,
            'best_refreshed': refresh, 'random_state': rng.state, 'random_draws': rng.draws,
            'tabu_rejected': bool(tabu and not accept), 'iteration': (iteration+1)&0xffffffff,
            'candidate_trace_calls': int(total_segments > 0 and not skip),
            'total_trace_calls': int(total_segments > 0 and not lazy)+int(total_segments > 0 and not skip),
            'counters': counters}


class LazyGoldScoringReferenceTests(unittest.TestCase):
    def compare(self, *args, **kwargs):
        eager = acceptance_reference(*args, lazy=False, **kwargs)
        lazy = acceptance_reference(*args, lazy=True, **kwargs)
        for key in ('accepted', 'score', 'best_score', 'best_refreshed', 'random_state', 'random_draws', 'tabu_rejected', 'iteration'):
            self.assertEqual(eager[key], lazy[key], (key, args, kwargs, eager, lazy))
        return eager, lazy

    def test_integer_score_decomposition_is_exact_above_float32_integer_precision(self):
        for total in (0, 1, 2, 7, 365, 2560):
            for base in (0, 1, 17, 38800):
                for loops in range(total+1):
                    score = base*(total+1)+total-loops
                    self.assertEqual(decompose(score, total), (base, loops))
        with self.assertRaises(ValueError): decompose(True, 1)
        with self.assertRaises(ValueError): decompose(1, False)

    def test_integer_upper_bound_includes_equality_and_every_possible_loop_count(self):
        for total in range(13):
            for base in range(9):
                upper = base*(total+1)+total
                for loops in range(total+1):
                    candidate = upper-loops
                    self.assertLessEqual(candidate, upper)
                    for best in (upper, upper+1): self.assertFalse(candidate > best)
        for loops in range(5):
            eager, lazy = self.compare(3, 3, 4, loops, 4, 24, tabu=True, random_values=[])
            self.assertFalse(lazy['accepted']); self.assertEqual(lazy['random_draws'], 0)
            self.assertEqual(eager['candidate_trace_calls'], 1); self.assertEqual(lazy['candidate_trace_calls'], 0)
            self.assertEqual(lazy['counters']['tabu_traces_skipped'], 1)

    def test_uncertain_tabu_aspiration_waits_for_exact_loops_and_keeps_short_circuit(self):
        # U=24>23, but actual score22 cannot aspirate. Neither path draws.
        _, lazy = self.compare(3, 3, 4, 2, 4, 23, tabu=True, random_values=[])
        self.assertFalse(lazy['accepted']); self.assertEqual(lazy['candidate_trace_calls'], 1)
        self.assertEqual(lazy['counters']['early_metropolis_draws'], 0)
        # U>best, actual equals best: strict aspiration still rejects.
        self.compare(3, 3, 4, 1, 4, 23, tabu=True, random_values=[])
        # Also check arbitrary stored-best values: do not assume aspiration
        # while computing a negative base Metropolis input.
        self.compare(5, 2, 4, 4, 4, 23, tabu=True, random_values=[])

    def test_negative_base_failed_and_successful_draws_are_consumed_exactly_once(self):
        for draw, boundary, expected in ((0, 0., False), (0, next_f32(0.), True),
                                         (1, 1., False), (1, next_f32(1.), True),
                                         (999999, 999999., False), (999999, 1000000., True)):
            callback = lambda acceptance, heat, b=boundary: b
            eager, lazy = self.compare(5, 2, 4, 0, 4, 27, random_values=[draw], threshold=callback)
            self.assertEqual(lazy['accepted'], expected); self.assertEqual(lazy['random_draws'], 1)
            self.assertEqual(lazy['counters']['early_metropolis_draws'], 1)
            self.assertEqual(lazy['candidate_trace_calls'], int(expected))
            self.assertEqual(lazy['counters']['metropolis_traces_skipped'], int(not expected))

    def test_zero_acceptance_has_no_metropolis_draw_but_best_tie_draw_is_preserved(self):
        self.compare(5, 2, 5, 2, 4, 28, random_values=[])
        for draw in (0, 1, 63, 64):
            eager, lazy = self.compare(5, 2, 5, 2, 4, 27, random_values=[draw])
            self.assertEqual(lazy['random_draws'], 1)
            self.assertEqual(lazy['best_refreshed'], draw%64 == 0)
            self.assertEqual(lazy['counters']['early_metropolis_draws'], 0)

    def test_exchange_mode_uses_full_candidate_trace_not_a_float_upper_bound(self):
        for lane in range(8):
            for draw in (0, 999999):
                _, lazy = self.compare(5, 2, 4, 0, 4, 27, exchange=True, lane=lane, random_values=[draw])
                self.assertEqual(lazy['candidate_trace_calls'], 1)
                self.assertEqual(lazy['counters']['early_metropolis_draws'], 0)
                self.assertEqual(lazy['counters']['metropolis_traces_skipped'], 0)

    def test_zero_segments_has_no_loop_counter_activity(self):
        for tabu in (False, True):
            for exchange in (False, True):
                _, lazy = self.compare(5, 0, 4, 0, 0, 5, tabu=tabu, exchange=exchange, random_values=[999999])
                self.assertEqual(lazy['total_trace_calls'], 0)
                for key in ('old_loop_traces_saved', 'candidate_loop_traces', 'tabu_traces_skipped', 'metropolis_traces_skipped'):
                    self.assertEqual(lazy['counters'][key], 0)
                self.assertEqual(lazy['counters']['old_base_scans_saved'], 1)

    def test_differential_control_flow_heat_wrap_and_all_objective_signs(self):
        rng = random.Random(2536)
        saved = skipped = 0
        for index in range(10000):
            total = rng.choice((0, 1, 2, 7, 365))
            old_base = rng.randrange(1, 100)
            new_base = max(0, old_base+rng.choice((-4, -1, 0, 1, 4)))
            old_loops, new_loops = rng.randrange(total+1), rng.randrange(total+1)
            old_score = old_base*(total+1)+total-old_loops
            upper = new_base*(total+1)+total
            best = max(old_score, rng.choice((old_score, upper-1, upper, upper+1)))
            eager, lazy = self.compare(old_base, old_loops, new_base, new_loops, total, best,
                                      tabu=bool(rng.randrange(2)), exchange=bool(rng.randrange(2)),
                                      lane=rng.randrange(8), iteration=rng.choice((0, 4095, 4096, 2**32-1)),
                                      random_state=rng.randrange(1, 2**32))
            saved += eager['total_trace_calls']-lazy['total_trace_calls']
            skipped += lazy['counters']['tabu_traces_skipped']+lazy['counters']['metropolis_traces_skipped']
        self.assertGreater(saved, 0); self.assertGreater(skipped, 0)
        self.assertEqual(heat_reference(0, 0, False), heat_reference(4096, 0, False))
        self.assertLess(heat_reference(4095, 0, False), heat_reference(0, 0, False))


if __name__ == '__main__': unittest.main()
