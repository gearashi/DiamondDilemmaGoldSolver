"""Pure CPU checks of bounded, resumable temperature redistribution.

These tests use no GPU, native solver, wall-time benchmark, or stochastic search.
The native same-backend trajectory/ladder checks are a separate integration gate.
"""
from copy import deepcopy
import hashlib
import json
import math
import random
import struct
import unittest
from unittest import mock

import sampling_temperatures_round25 as temperatures
from sampling_temperatures_round25 import AdaptiveTemperatures


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def native_ladder():
    return [f32(f32(.2)*f32(1.5**i)) for i in range(8)]


def feedback(controller, attempts=None, accepts=None):
    return {'temperatures': controller.temperatures or native_ladder(),
            'attempts': [100]*7 if attempts is None else list(attempts),
            'accepts': [10, 20, 30, 40, 50, 60, 70] if accepts is None else list(accepts)}


def resign(snapshot):
    snapshot['digest'] = hashlib.sha256(json.dumps(
        {key: value for key, value in snapshot.items() if key != 'digest'},
        sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')).hexdigest()
    return snapshot


class AdaptiveTemperaturesReferenceTests(unittest.TestCase):
    def test_initial_state_and_rate_redistribution_matches_independent_formula(self):
        controller = AdaptiveTemperatures()
        self.assertEqual(controller.last_dispatch, 0)
        self.assertIsNone(controller.temperatures)
        self.assertFalse(controller.frozen)
        self.assertTrue(all(value == 0 for value in controller.stats.values()))
        initial = feedback(controller)
        result = controller.observe(initial, 16)
        self.assertIsNotNone(result)
        self.assertEqual((result[0], result[-1]), (initial['temperatures'][0], initial['temperatures'][-1]))
        self.assertTrue(all(left < right for left, right in zip(result, result[1:])))
        self.assertTrue(all(f32(value) == value for value in result))
        # Independent uncentered weights give the same normalized formula:
        # subtraction of a common mean/target cannot impose an absolute rate.
        logs = list(map(math.log, initial['temperatures']))
        weights = [(logs[i+1]-logs[i])*math.exp(.5*initial['accepts'][i]/100) for i in range(7)]
        span = logs[-1]-logs[0]
        gaps = [span*weight/sum(weights) for weight in weights]
        expected = [initial['temperatures'][0]] + [f32(math.exp(logs[0]+sum(gaps[:i]))) for i in range(1, 7)] + [initial['temperatures'][-1]]
        self.assertEqual(result, expected)
        self.assertGreater(math.log(result[-1]/result[-2]), math.log(result[1]/result[0]))
        self.assertEqual(controller.stats['temperature_feedback_attempts'], 700)
        self.assertEqual(controller.stats['temperature_feedback_accepts'], 280)
        self.assertEqual(controller.stats['temperature_adjustments'], 1)
        self.assertEqual(controller.report(), controller.stats)

    def test_duplicate_requires_committed_ladder_and_changes_no_state(self):
        controller = AdaptiveTemperatures()
        original = feedback(controller)
        result = controller.observe(original, 16)
        self.assertNotEqual(result, original['temperatures'])
        before = controller.snapshot()
        current = dict(original, temperatures=controller.temperatures)
        self.assertIsNone(controller.observe(current, 16))
        self.assertEqual(before, controller.snapshot())
        with self.assertRaisesRegex(ValueError, 'ladder'):
            controller.observe(original, 16)
        changed = deepcopy(current); changed['attempts'][0] += 1
        with self.assertRaisesRegex(ValueError, 'changed counters'):
            controller.observe(changed, 16)
        self.assertEqual(before, controller.snapshot())

    def test_missing_edges_advance_baseline_without_imputed_zero_rate(self):
        controller = AdaptiveTemperatures()
        first = feedback(controller, [100]*6+[0], [50]*6+[0])
        self.assertIsNone(controller.observe(first, 16))
        self.assertEqual(controller.temperatures, native_ladder())
        self.assertEqual(controller.stats['temperature_empty_windows'], 1)
        second = feedback(controller, [110]*6+[10], [55]*6+[5])
        controller.observe(second, 32)
        self.assertEqual(controller.stats['temperature_eligible_windows'], 1)
        self.assertEqual(controller.stats['temperature_feedback_attempts'], 670)
        self.assertEqual(controller.stats['temperature_feedback_accepts'], 335)
        reference = AdaptiveTemperatures()
        reference.observe(feedback(reference, [10]*7, [5]*7), 16)
        self.assertEqual(reference.temperatures, controller.temperatures)

    def test_exact_cadence_and_malformed_feedback_are_atomic(self):
        controller = AdaptiveTemperatures()
        before = controller.snapshot()
        for dispatch in (0, 15, 17, 32, True, 16.0, 2**53):
            with self.subTest(dispatch=dispatch), self.assertRaises(ValueError):
                controller.observe(feedback(controller), dispatch)
            self.assertEqual(controller.snapshot(), before)
        malformed = []
        for value in (math.nan, math.inf, 0.0, -1.0, .21, True, 1):
            item = feedback(controller); item['temperatures'][3] = value; malformed.append(item)
        item = feedback(controller); item['temperatures'].pop(); malformed.append(item)
        item = feedback(controller); item['temperatures'][3] = item['temperatures'][2]; malformed.append(item)
        item = feedback(controller); item['extra'] = 0; malformed.append(item)
        for field in ('attempts', 'accepts'):
            for value in (True, -1, 2**32, 10.0):
                item = feedback(controller); item[field][0] = value; malformed.append(item)
        item = feedback(controller); item['accepts'][0] = 101; malformed.append(item)
        item = feedback(controller); item['attempts'][0] = 32769; malformed.append(item)
        for item in malformed:
            with self.subTest(item=item), self.assertRaises(ValueError):
                controller.observe(item, 16)
            self.assertEqual(controller.snapshot(), before)
        controller.observe(feedback(controller), 16)
        before = controller.snapshot()
        for dispatch in (0, 48):
            with self.assertRaises(ValueError): controller.observe(feedback(controller), dispatch)
            self.assertEqual(controller.snapshot(), before)

    def test_counter_wrap_uses_deltas_and_accepts_maximum_window(self):
        self.assertEqual(temperatures._counter_deltas([16]*7, [0xfffffff0]*7), [32]*7)
        self.assertEqual(temperatures._counter_deltas([32768]*7, [0]*7), [32768]*7)
        with self.assertRaises(ValueError): temperatures._counter_deltas([32769]*7, [0]*7)
        # Many possible empty windows permit six streams to approach wrap while
        # the seventh edge has never attempted. No history is a proof artifact.
        snapshot = AdaptiveTemperatures().snapshot(); state = snapshot['state']
        windows = 131072
        state.update(last_dispatch=16*windows, temperatures=native_ladder(),
                     endpoints=[native_ladder()[0], native_ladder()[-1]],
                     attempts=[0xfffffff0]*6+[0], accepts=[0xffffffe0]*6+[0])
        state['stats'].update(temperature_feedback_windows=windows,
                             temperature_empty_windows=windows,
                             temperature_feedback_attempts=6*0xfffffff0,
                             temperature_feedback_accepts=6*0xffffffe0)
        controller = AdaptiveTemperatures(resume=resign(snapshot))
        controller.observe(feedback(controller, [16]*6+[1], [0xfffffff0]*6+[1]), 16*(windows+1))
        self.assertEqual(controller.stats['temperature_feedback_attempts'], 6*(2**32+16)+1)
        self.assertEqual(controller.stats['temperature_feedback_accepts'], 6*0xfffffff0+1)
        self.assertEqual(controller.stats['temperature_eligible_windows'], 1)
        # Raw accepts > raw attempts after wrap is legitimate.
        self.assertGreater(controller.snapshot()['state']['accepts'][0], controller.snapshot()['state']['attempts'][0])
        controller = AdaptiveTemperatures()
        controller.observe(feedback(controller, [32768]*7, [32768]*7), 16)
        self.assertEqual(controller.stats['temperature_feedback_attempts'], 7*32768)

    def test_freezes_after_eight_eligible_windows_even_without_changes(self):
        controller = AdaptiveTemperatures()
        for index in range(1, 9):
            self.assertIsNone(controller.observe(feedback(controller, [100*index]*7, [50*index]*7), 16*index))
        self.assertTrue(controller.frozen)
        self.assertEqual(controller.stats['temperature_adjustments'], 0)
        self.assertEqual(controller.stats['temperature_eligible_windows'], 8)
        self.assertEqual(controller.stats['temperature_freezes'], 1)
        before = controller.snapshot()
        self.assertIsNone(controller.observe(feedback(controller, [800]*7, [400]*7), 128))
        self.assertEqual(controller.snapshot(), before)
        with self.assertRaisesRegex(ValueError, 'frozen'):
            controller.observe(feedback(controller, [900]*7, [450]*7), 144)
        self.assertEqual(controller.snapshot(), before)

    def test_floor_clamps_only_small_gaps_and_equal_ratios_preserve_bits(self):
        controller = AdaptiveTemperatures()
        initial = [f32(.2*math.exp(logvalue)) for logvalue in (0, .005, .01, .015, .02, .025, .03, 2.8)]
        result = controller.observe({'temperatures': initial, 'attempts': [100]*7,
                                     'accepts': [0, 100, 0, 100, 0, 100, 50]}, 16)
        gaps = [math.log(right/left) for left, right in zip(result, result[1:])]
        for gap in gaps[:6]: self.assertAlmostEqual(gap, .05, places=6)
        self.assertAlmostEqual(gaps[-1], math.log(initial[-1]/initial[0])-.3, places=6)
        self.assertEqual([result[0], result[-1]], [initial[0], initial[-1]])
        # Equal rational rates with different denominators are no new evidence.
        attempts = [102, 104, 106, 108, 110, 112, 114]
        accepts = [1, 102, 3, 104, 5, 106, 57]
        self.assertIsNone(controller.observe(feedback(controller, attempts, accepts), 32))
        self.assertEqual(controller.temperatures, result)
        self.assertEqual(controller.stats['temperature_eligible_windows'], 2)
        self.assertEqual(controller.stats['temperature_adjustments'], 1)

    def test_split_resume_matches_uninterrupted_and_detaches_values(self):
        rng = random.Random(37)
        for case in range(20):
            live = AdaptiveTemperatures(); resumed = AdaptiveTemperatures()
            attempts = [0]*7; accepts = [0]*7
            for index in range(1, 9):
                for edge in range(7):
                    delta = rng.randrange(1, 32769)
                    attempts[edge] += delta; accepts[edge] += rng.randrange(delta+1)
                left = live.observe(feedback(live, attempts, accepts), index*16)
                right = resumed.observe(feedback(resumed, attempts, accepts), index*16)
                self.assertEqual(left, right)
                self.assertEqual(live.snapshot(), resumed.snapshot())
                snapshot = json.loads(json.dumps(resumed.snapshot()))
                resumed = AdaptiveTemperatures(resume=snapshot)
                snapshot['state']['attempts'][0] = 0
                self.assertEqual(live.snapshot(), resumed.snapshot())
            self.assertTrue(live.frozen)
        saved = live.snapshot()
        ladder = live.temperatures; ladder[0] = 99.0
        stats = live.stats; stats['temperature_adjustments'] = 99
        saved['state']['temperatures'][0] = 99.0
        self.assertNotEqual(live.temperatures[0], 99.0)
        self.assertLessEqual(live.stats['temperature_adjustments'], 8)

    def test_checkpoint_rejects_binding_type_shape_and_counter_corruption(self):
        live = AdaptiveTemperatures(); live.observe(feedback(live), 16)
        original = live.snapshot()
        mutations = [lambda v: v.update(adaptive_temperatures_version=True),
                     lambda v: v.update(source_sha256='0'*64),
                     lambda v: v['policy'].update(initial_gain=0),
                     lambda v: v['policy'].update(max_eligible_windows=8.0),
                     lambda v: v['state'].update(last_dispatch=True),
                     lambda v: v['state'].update(last_dispatch=32),
                     lambda v: v['state']['stats'].update(temperature_eligible_windows=True),
                     lambda v: v['state']['stats'].update(temperature_empty_windows=1),
                     lambda v: v['state']['stats'].update(temperature_freezes=1),
                     lambda v: v['state']['stats'].update(temperature_adjustments=2),
                     lambda v: v['state']['stats'].update(temperature_feedback_attempts=701),
                     lambda v: v['state']['stats'].update(temperature_feedback_accepts=701),
                     lambda v: v['state']['attempts'].__setitem__(0, 2**32),
                     lambda v: v['state']['endpoints'].__setitem__(0, 1),
                     lambda v: v['state'].update(extra=0)]
        for change in mutations:
            modified = deepcopy(original); change(modified); resign(modified)
            with self.subTest(modified=modified), self.assertRaises(ValueError):
                AdaptiveTemperatures(resume=modified)
        modified = deepcopy(original); modified['state']['attempts'][0] += 1
        with self.assertRaisesRegex(ValueError, 'digest'):
            AdaptiveTemperatures(resume=modified)
        modified = deepcopy(original); modified['padding'] = 'x'*temperatures.MAX_SNAPSHOT_BYTES
        with self.assertRaisesRegex(ValueError, 'byte limit'):
            AdaptiveTemperatures(resume=modified)
        modified = deepcopy(original); modified['state']['temperatures'][1] = float('nan')
        with self.assertRaises(ValueError): AdaptiveTemperatures(resume=modified)
        self.assertEqual(live.snapshot(), original)

    def test_unobserved_checkpoint_roundtrip_and_late_source_failure_are_atomic(self):
        live = AdaptiveTemperatures(); snapshot = live.snapshot()
        self.assertEqual(AdaptiveTemperatures(resume=snapshot).snapshot(), snapshot)
        original = temperatures._source_digest()
        with mock.patch.object(temperatures, '_source_digest', side_effect=[original, 'different']):
            with self.assertRaisesRegex(ValueError, 'source changed'):
                live.observe({'temperatures': native_ladder(), 'attempts': [100]*7, 'accepts': [50]*7}, 16)
        self.assertEqual(live.snapshot(), snapshot)
        with mock.patch.object(temperatures, '_source_digest', return_value='different'):
            with self.assertRaisesRegex(ValueError, 'source changed'): live.snapshot()
            with self.assertRaisesRegex(ValueError, 'source changed'): _ = live.frozen
        invalid = deepcopy(snapshot); invalid['state']['temperatures'] = native_ladder()
        with self.assertRaises(ValueError): AdaptiveTemperatures(resume=resign(invalid))

    def test_native_endpoint_bits_authoritative_and_numerical_failure_atomic(self):
        controller = AdaptiveTemperatures()
        initial = native_ladder()
        bits = struct.unpack('<I', struct.pack('<f', initial[-1]))[0]
        initial[-1] = struct.unpack('<f', struct.pack('<I', bits+1))[0]
        controller.observe({'temperatures': initial, 'attempts': [100]*7, 'accepts': [0, 100, 0, 100, 0, 100, 0]}, 16)
        self.assertEqual(struct.pack('<f', controller.temperatures[-1]), struct.pack('<f', initial[-1]))
        before = controller.snapshot()
        with mock.patch.object(temperatures, '_redistribute', side_effect=ValueError('rounded ladder invalid')):
            with self.assertRaisesRegex(ValueError, 'rounded ladder invalid'):
                controller.observe(feedback(controller, [200]*7, [50, 150, 50, 150, 50, 150, 50]), 32)
        self.assertEqual(controller.snapshot(), before)


if __name__ == '__main__':
    unittest.main()
