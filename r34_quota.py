"""Exact Lagrangian upper bound for a limited number of additional changed donors."""
from math import gcd

import numpy as np

from constraint_connectivity_round26 import assignment_upper_bound


def quota_price(weights, allowed, dirty, quota, hint=None):
    """Minimize max_matching(weight - penalty * dirty) + penalty * quota for penalty >= 0.

    Integer arithmetic locates the exact intersection of two active matching lines.
    The returned dual also certifies an upper bound for each forced donor arc.
    """
    n = len(weights)
    if not allowed.any(1).all() or not allowed.any(0).all():
        return None
    rows = np.arange(n)
    calls = 0

    def solve(num, den):
        nonlocal calls
        calls += 1
        scaled = den*weights-num*dirty
        shift = int(scaled[allowed].min())
        values = [[int(x)-shift if legal else None for x,legal in zip(line,mask)]
                  for line,mask in zip(scaled,allowed)]
        dual = assignment_upper_bound(values)
        matched = dual['assignment']
        if not allowed[rows,matched].all():
            return None
        cost = int(weights[rows,matched].sum())
        count = int(dirty[rows,matched].sum())
        # A matching containing (i,j) loses at least this dual slack.
        slack = np.array(dual['rows'])[:,None] + np.array(dual['columns']) - (scaled-shift)
        objective = den*cost + num*((quota or 0)-count)
        return dict(upper=objective//den, forced=(objective-slack)//den,
                    slope=(num,den), score=cost, changed=count, calls=calls)

    left = solve(0,1)
    if left is None or quota is None or left['changed']<=quota:
        return left
    right = None
    if hint and hint[0]>0:
        warm = solve(*hint)
        if warm['changed']==quota:
            return warm
        if warm['changed']<quota:
            right = warm
        else:
            left = warm
    if right is None:
        # A unit reduction in changed arcs outweighs every feasible score difference.
        span = int(weights[allowed].max()) - int(weights[allowed].min())
        right = solve(n*span+1,1)
        if right['changed']>quota:
            return None
    if right['changed']==quota:
        return right
    while True:
        dc = left['changed']-right['changed']
        dv = left['score']-right['score']
        divisor = gcd(dv,dc)
        num,den = dv//divisor,dc//divisor
        mid = solve(num,den)
        if (mid['score']-left['score'])*den == (mid['changed']-left['changed'])*num:
            return mid
        if mid['changed']==quota:
            return mid
        if mid['changed']>quota:
            left = mid
        else:
            right = mid
