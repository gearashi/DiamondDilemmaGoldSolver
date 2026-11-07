"""Bounded exchange-rate redistribution for the heuristic GPU sampler.

This is an RNG-free heuristic controller, not search coverage or a proof rule.
The first native float32 ladder is authoritative. Positive log-gap weights are
redistributed while retaining its endpoints. A common acceptance-rate target
would cancel in normalization, so there is deliberately no absolute-rate target.
Unequal rates trigger proportional gap redistribution with a clamped floor;
equal rates leave the ladder unchanged. Rounding to float32 may perturb the
mathematical log-gap floor slightly.

Snapshots are trusted local continuation, bound to this source and fixed policy.
The outer sampler must bind puzzle/native scope and install each returned ladder
before publishing a checkpoint. Failed native installation must poison that
session: observe() commits the host side at the successful observation boundary.
"""
from __future__ import annotations
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import struct

VERSION = 1
MAX_SNAPSHOT_BYTES = 16 * 1024
MAX_INTEGER = 2**53 - 1
UINT32_MASK = 2**32 - 1
STAT_KEYS = ('temperature_feedback_windows', 'temperature_eligible_windows',
             'temperature_adjustments', 'temperature_empty_windows',
             'temperature_freezes', 'temperature_feedback_attempts',
             'temperature_feedback_accepts')


def _policy():
    return {'window_dispatches': 16, 'max_eligible_windows': 8,
            'max_edge_delta': 32768, 'log_gap_floor': 0.05,
            'initial_gain': 0.5, 'ladder_size': 8}


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _source_digest():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _integer(value, low=0, high=MAX_INTEGER):
    return type(value) is int and low <= value <= high


def _f32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except (OverflowError, struct.error) as exc:
        raise ValueError('Temperature cannot be represented as finite float32.') from exc


def _ladder(value):
    if type(value) is not list or len(value) != 8 or any(
            type(v) is not float or not math.isfinite(v) or v <= 0 or _f32(v) != v
            for v in value):
        raise ValueError('Temperature ladder requires eight positive finite float32 values.')
    if any(left >= right for left, right in zip(value, value[1:])):
        raise ValueError('Temperature ladder must be strictly increasing.')
    if math.log(value[-1]) - math.log(value[0]) <= 7 * _policy()['log_gap_floor']:
        raise ValueError('Temperature ladder span is too small for the gap floor.')
    return list(value)


def _counts(value):
    if type(value) is not list or len(value) != 7 or any(
            not _integer(v, 0, UINT32_MASK) for v in value):
        raise ValueError('Exchange counters require seven uint32 integers.')
    return list(value)


def _counter_deltas(current, previous):
    """Exactly one bounded window; raw aggregate counters may wrap uint32."""
    current = _counts(current); previous = _counts(previous)
    result = [(new-old) & UINT32_MASK for new, old in zip(current, previous)]
    if any(value > _policy()['max_edge_delta'] for value in result):
        raise ValueError('Exchange counter delta exceeds one observation window.')
    return result


def _redistribute(temperatures, attempts, accepts, eligible_index):
    if all(accepted*attempts[0] == accepts[0]*attempted
           for attempted, accepted in zip(attempts, accepts)):
        return list(temperatures)  # No unequal-rate evidence; no ladder drift.
    logs = [math.log(value) for value in temperatures]
    gaps = [right-left for left, right in zip(logs, logs[1:])]
    rates = [accepted/attempted for attempted, accepted in zip(attempts, accepts)]
    mean_rate = math.fsum(rates)/7
    gain = _policy()['initial_gain']/math.sqrt(eligible_index+1)
    weights = [gap*math.exp(gain*(rate-mean_rate)) for gap, rate in zip(gaps, rates)]
    floor = _policy()['log_gap_floor']
    span = logs[-1]-logs[0]
    if span <= 7*floor or any(not math.isfinite(weight) or weight <= 0 for weight in weights):
        raise ValueError('Invalid temperature redistribution span or weights.')
    new_gaps = [0.0]*7
    active = list(range(7))
    for _ in range(7):
        remaining = span-(7-len(active))*floor
        total_weight = math.fsum(weights[index] for index in active)
        provisional = {index: remaining*weights[index]/total_weight for index in active}
        clamped = [index for index in active if provisional[index] < floor]
        if not clamped:
            for index in active: new_gaps[index] = provisional[index]
            break
        for index in clamped: new_gaps[index] = floor
        active = [index for index in active if index not in clamped]
        if not active: raise ValueError('Temperature floor consumed the complete span.')
    else:
        raise ValueError('Temperature floor normalization did not converge.')
    ladder = [temperatures[0]]
    for index in range(1, 7):
        ladder.append(_f32(math.exp(logs[0]+math.fsum(new_gaps[:index]))))
    ladder.append(temperatures[-1])
    return _ladder(ladder)


class AdaptiveTemperatures:
    def __init__(self, *, resume=None):
        self._source = _source_digest()
        self._state = {'last_dispatch': 0, 'temperatures': None, 'endpoints': None,
                       'attempts': [0]*7, 'accepts': [0]*7,
                       'stats': {key: 0 for key in STAT_KEYS}}
        if resume is not None:
            self._restore(resume)

    def _check_scope(self):
        if _source_digest() != self._source:
            raise ValueError('Adaptive temperature source changed during the session.')

    @property
    def last_dispatch(self):
        self._check_scope()
        return self._state['last_dispatch']

    @property
    def temperatures(self):
        self._check_scope()
        value = self._state['temperatures']
        return None if value is None else list(value)

    @property
    def frozen(self):
        self._check_scope()
        return self._state['stats']['temperature_eligible_windows'] == 8

    @property
    def stats(self):
        self._check_scope()
        return dict(self._state['stats'])

    def report(self):
        return self.stats

    def _validate_state(self, state):
        if type(state) is not dict or set(state) != set(self._state):
            raise ValueError('Invalid adaptive temperature state shape.')
        dispatch = state['last_dispatch']
        if not _integer(dispatch) or dispatch % 16:
            raise ValueError('Invalid temperature observation dispatch.')
        attempts = _counts(state['attempts']); accepts = _counts(state['accepts'])
        stats = state['stats']
        if type(stats) is not dict or set(stats) != set(STAT_KEYS) or any(
                not _integer(value) for value in stats.values()):
            raise ValueError('Invalid adaptive temperature counters.')
        windows = stats['temperature_feedback_windows']
        eligible = stats['temperature_eligible_windows']
        total_attempts = stats['temperature_feedback_attempts']
        total_accepts = stats['temperature_feedback_accepts']
        if windows != dispatch//16 or eligible > 8 or (
                eligible + stats['temperature_empty_windows'] != windows) or (
                stats['temperature_adjustments'] > eligible) or (
                stats['temperature_freezes'] != int(eligible == 8)):
            raise ValueError('Inconsistent temperature cadence or adjustment counts.')
        if not 7*eligible <= total_attempts <= windows*7*32768 or not 0 <= total_accepts <= total_attempts:
            raise ValueError('Inconsistent exchange work totals.')
        for counts, total in ((attempts, total_attempts), (accepts, total_accepts)):
            if any(value > windows*32768 for value in counts) or sum(counts) > total or (
                    total-sum(counts)) % (2**32):
                raise ValueError('Exchange totals disagree with modulo counters.')
        if dispatch == 0:
            if state['temperatures'] is not None or state['endpoints'] is not None or any(stats.values()):
                raise ValueError('Unobserved temperature state must be empty.')
        else:
            temperatures = _ladder(state['temperatures'])
            endpoints = state['endpoints']
            if type(endpoints) is not list or len(endpoints) != 2 or any(
                    type(value) is not float for value in endpoints) or endpoints != [temperatures[0], temperatures[-1]]:
                raise ValueError('Temperature endpoints changed or are malformed.')

    def _restore(self, value):
        try:
            encoded = _encoded(value)
            if len(encoded) > MAX_SNAPSHOT_BYTES:
                raise ValueError('Adaptive temperature checkpoint exceeds byte limit.')
            value = json.loads(encoded)
        except (TypeError, OverflowError, RecursionError) as exc:
            raise ValueError('Invalid adaptive temperature checkpoint JSON.') from exc
        if type(value) is not dict or set(value) != {
                'adaptive_temperatures_version', 'source_sha256', 'policy', 'state', 'digest'}:
            raise ValueError('Invalid adaptive temperature checkpoint envelope.')
        expected = _policy(); policy = value['policy']
        if type(value['adaptive_temperatures_version']) is not int or value['adaptive_temperatures_version'] != VERSION or (
                type(value['source_sha256']) is not str or value['source_sha256'] != self._source) or (
                type(policy) is not dict or set(policy) != set(expected)) or any(
                type(policy[key]) is not type(want) or policy[key] != want for key, want in expected.items()):
            raise ValueError('Adaptive temperature checkpoint source or policy differs.')
        if type(value['digest']) is not str or value['digest'] != _digest({
                key: data for key, data in value.items() if key != 'digest'}):
            raise ValueError('Adaptive temperature checkpoint digest mismatch.')
        self._validate_state(value['state'])
        self._check_scope()
        self._state = value['state']

    def snapshot(self):
        self._check_scope()
        value = {'adaptive_temperatures_version': VERSION, 'source_sha256': self._source,
                 'policy': _policy(), 'state': deepcopy(self._state)}
        value['digest'] = _digest(value)
        if len(_encoded(value)) > MAX_SNAPSHOT_BYTES:
            raise ValueError('Adaptive temperature checkpoint exceeds byte limit.')
        return value

    def observe(self, feedback, dispatches):
        """Consume one complete 16-dispatch window, atomically; no RNG calls.

        A duplicate observation must report the *committed* ladder (after the
        caller's native install), plus exactly the same cumulative counters.
        Missing-edge windows advance baselines but never impute an acceptance
        rate. New observations after eight eligible windows are rejected.
        """
        self._check_scope()
        if not _integer(dispatches, 16) or dispatches % 16:
            raise ValueError('Temperature feedback requires a positive 16-dispatch boundary.')
        if type(feedback) is not dict or set(feedback) != {'temperatures', 'attempts', 'accepts'}:
            raise ValueError('Invalid temperature feedback shape.')
        temperatures = _ladder(feedback['temperatures'])
        attempts = _counts(feedback['attempts']); accepts = _counts(feedback['accepts'])
        old = self._state
        if old['temperatures'] is not None and temperatures != old['temperatures']:
            raise ValueError('Native temperature ladder differs from the committed controller.')
        if dispatches == old['last_dispatch']:
            if attempts != old['attempts'] or accepts != old['accepts']:
                raise ValueError('Repeated temperature observation has changed counters.')
            self._check_scope()
            return None
        if old['stats']['temperature_eligible_windows'] == 8:
            raise ValueError('Adaptive temperature controller is frozen.')
        if dispatches != old['last_dispatch']+16:
            raise ValueError('Temperature observations must advance by exactly 16 dispatches.')
        delta_attempts = _counter_deltas(attempts, old['attempts'])
        delta_accepts = _counter_deltas(accepts, old['accepts'])
        if any(accepted > attempted for attempted, accepted in zip(delta_attempts, delta_accepts)):
            raise ValueError('Accepted exchanges exceed attempted exchanges in this window.')
        staged = deepcopy(old)
        staged.update(last_dispatch=dispatches, temperatures=temperatures,
                      attempts=attempts, accepts=accepts)
        if staged['endpoints'] is None:
            staged['endpoints'] = [temperatures[0], temperatures[-1]]
        stats = staged['stats']
        stats['temperature_feedback_windows'] += 1
        stats['temperature_feedback_attempts'] += sum(delta_attempts)
        stats['temperature_feedback_accepts'] += sum(delta_accepts)
        output = None
        if all(delta_attempts):
            updated = _redistribute(temperatures, delta_attempts, delta_accepts,
                                    stats['temperature_eligible_windows'])
            stats['temperature_eligible_windows'] += 1
            if updated != temperatures:
                stats['temperature_adjustments'] += 1
                staged['temperatures'] = updated
                output = list(updated)
            if stats['temperature_eligible_windows'] == 8:
                stats['temperature_freezes'] = 1
        else:
            stats['temperature_empty_windows'] += 1
        self._validate_state(staged)
        self._check_scope()
        self._state = staged
        return output
