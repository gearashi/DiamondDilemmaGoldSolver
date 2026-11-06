"""Bounded scheduling hints for sound optional propagators.

A skipped call is never a deduction. Exact unchanged signatures are scoped to
one immutable puzzle/runtime; widening, backtracking and resume use the actual
new signature. Adaptive cooldowns retry periodically and never declare UNSAT.
"""
from collections import deque
from time import perf_counter_ns

STAT_KEYS = ('propagation_runs', 'propagation_unchanged_skips',
             'propagation_adaptive_skips', 'propagation_time_ns',
             'propagation_values_removed','propagation_proofs_produced','propagation_yield_runs')


class PropagationScheduler:
    def __init__(self, *, incremental=True, adaptive=True, clock=perf_counter_ns):
        self.incremental, self.adaptive, self.clock = incremental, adaptive, clock
        self.records = {}

    def run(self, name, domains, action, stats, *, optional=False, complete=None, force=False, signature=None, proof_yield=None):
        record = self.records.setdefault(name, {'unchanged': deque(maxlen=8),
                                               'misses': 0, 'cooldown': 0, 'yields': deque(maxlen=16)})
        for key in STAT_KEYS:stats.setdefault(key,0)
        before = tuple(domains)
        signature = before if signature is None else signature
        if not force and self.incremental and signature in record['unchanged']:
            stats['propagation_unchanged_skips'] += 1
            return False
        if not force and optional and self.adaptive and record['cooldown']:
            record['cooldown'] -= 1
            stats['propagation_adaptive_skips'] += 1
            return False
        prior_proofs = proof_yield() if proof_yield is not None else 0
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
        removed=sum((a & ~b).bit_count() for a,b in zip(before,domains))
        proofs=max(0,(proof_yield() if proof_yield is not None else 0)-prior_proofs)
        stats['propagation_values_removed']+=removed
        stats['propagation_proofs_produced']+=proofs
        if completed:
            record['yields'].append(removed+proofs)
            if removed or proofs:stats['propagation_yield_runs']+=1
        if changed or proofs:
            record['misses'], record['cooldown'] = 0, 0
        elif completed and elapsed >= 500_000:
            record['misses'] += 1
            if optional and self.adaptive and record['misses'] >= 3:
                record['cooldown'] = min(16, 1 << min(4, record['misses'] // 3))
        return changed

    def budget(self,name,base,*,minimum=None,maximum=None):
        """Return a bounded scheduling hint from observed deductions/proofs.

        It changes effort only. Neither a low yield nor an omitted call proves
        infeasibility, and the first three observations retain the base budget.
        """
        if base<=0:return base
        minimum=base/4 if minimum is None else minimum
        maximum=base*4 if maximum is None else maximum
        if not 0<minimum<=maximum:raise ValueError('Invalid adaptive budget bounds.')
        observed=self.records.get(name,{}).get('yields',())
        if len(observed)<3:return min(maximum,max(minimum,base))
        productive=sum(bool(value) for value in observed)/len(observed)
        scale=.5 if productive==0 else (2 if productive>=.5 else 1)
        return min(maximum,max(minimum,base*scale))
