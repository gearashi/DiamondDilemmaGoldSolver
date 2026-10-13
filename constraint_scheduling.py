"""Bounded scheduling hints for sound optional propagators.

A skipped call is never a deduction. Exact unchanged signatures are scoped to
one immutable puzzle/runtime; widening, backtracking and resume use the actual
new signature. Adaptive cooldowns retry periodically and never declare UNSAT.
"""
from collections import deque
from time import perf_counter_ns

STAT_KEYS = ('propagation_runs', 'propagation_unchanged_skips',
             'propagation_adaptive_skips', 'propagation_time_ns')


class PropagationScheduler:
    def __init__(self, *, incremental=True, adaptive=True, clock=perf_counter_ns):
        self.incremental, self.adaptive, self.clock = incremental, adaptive, clock
        self.records = {}

    def run(self, name, domains, action, stats, *, optional=False, complete=None, force=False, signature=None):
        record = self.records.setdefault(name, {'unchanged': deque(maxlen=8),
                                               'misses': 0, 'cooldown': 0})
        before = tuple(domains)
        signature = before if signature is None else signature
        if not force and self.incremental and signature in record['unchanged']:
            stats['propagation_unchanged_skips'] += 1
            return False
        if not force and optional and self.adaptive and record['cooldown']:
            record['cooldown'] -= 1
            stats['propagation_adaptive_skips'] += 1
            return False
        started = self.clock()
        stats['propagation_runs'] += 1
        try:
            action()
        except BaseException:
            # An interrupted call may have made sound reductions, but it is
            # not a completed observation and cannot justify memoization.
            stats['propagation_time_ns'] += max(0, self.clock()-started)
            raise
        elapsed = max(0, self.clock()-started)
        stats['propagation_time_ns'] += elapsed
        changed = before != tuple(domains)
        completed = complete is None or complete()
        if not changed and completed and self.incremental:
            record['unchanged'].append(signature)
        if changed:
            record['misses'], record['cooldown'] = 0, 0
        elif completed and elapsed >= 500_000:
            record['misses'] += 1
            if optional and self.adaptive and record['misses'] >= 3:
                record['cooldown'] = min(16, 1 << min(4, record['misses'] // 3))
        return changed
