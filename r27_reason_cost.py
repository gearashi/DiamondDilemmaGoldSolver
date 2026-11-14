"""Certified soft-seam bounds for a restricted placement domain."""
from collections import Counter
from dataclasses import dataclass

from r27_reason_maxsat import domain_rows
from validator import reverse_mask


@dataclass(frozen=True)
class DefectCores:
    fingerprint: str
    domains: tuple
    cores: tuple


def zero_consistent(problem, rows, edges, stats, check=lambda:None):
    """Mask-level arc consistency on zero-cost tuples, relaxing inventory."""
    live = [set(row) for row in rows]
    if not all(live):
        return False
    changed = True
    while changed:
        changed = False
        for index in edges:
            check()
            a, sa, b, sb = problem.edges[index]
            for c, s, d, t in ((a, sa, b, sb), (b, sb, a, sa)):
                stats['arc_visits'] += 1
                allowed = {reverse_mask(int(problem.masks[v, t])) for v in live[d]}
                keep = {v for v in live[c] if int(problem.masks[v, s]) in allowed}
                if not keep:
                    return False
                if len(keep) < len(live[c]):
                    live[c] = keep
                    changed = True
    return True


class SeamCostCache:
    """Reuse exact unary projections on unchanged seam endpoints in descendants."""
    def __init__(self,problem,limit=512):
        self.fingerprint = problem.fingerprint
        self.limit = limit
        self.projections = {}

    def project(self,problem,index,left,right,stats):
        if self.fingerprint != problem.fingerprint:
            raise ValueError('Cost projection cache source differs.')
        key = index,left,right
        if key in self.projections:
            stats['projection_hits'] += 1
            return self.projections[key]
        _,sa,_,sb = problem.edges[index]
        matrix = [[int(int(problem.masks[x,sa]) != reverse_mask(int(problem.masks[y,sb])))
                   for y in right] for x in left]
        stats['evaluated_pairs'] += len(left)*len(right)
        row_min = tuple(min(row) for row in matrix)
        col_min = tuple(min(row[j]-delta for row,delta in zip(matrix,row_min)) for j in range(len(right)))
        projection = row_min,col_min
        if self.limit:
            if len(self.projections) >= self.limit:
                del self.projections[next(iter(self.projections))]
            self.projections[key] = projection
        return projection


def cost_bound(problem, *, domains=None, inherited=None, probes=64, shift_pairs=25000, cache=None, check=lambda:None):
    rows = domain_rows(problem, domains)
    if not all(rows):
        return {'infeasible_domain': True, 'lower_bound': 0, 'stats': {}, 'certificate': None}
    if probes < 0 or shift_pairs < 0:
        raise ValueError('Bounds must be nonnegative.')
    stats = Counter()
    cores = []
    remaining = set(range(len(problem.edges)))
    if inherited is not None:
        if (not isinstance(inherited, DefectCores) or inherited.fingerprint != problem.fingerprint
                or len(inherited.domains) != problem.n
                or any(not set(row) <= set(old) for row, old in zip(rows, inherited.domains))):
            raise ValueError('Certificate source or descendant domains differ.')
        # Replay even supplied parent certificates. This permits serialized checkpoints.
        for given in inherited.cores:
            core = tuple(given)
            if not core or len(set(core)) != len(core) or not set(core) <= remaining or zero_consistent(problem, rows, core, stats, check):
                raise ValueError('Invalid inherited defect core.')
            remaining.difference_update(core)
            cores.append(core)
            stats['inherited'] += 1
    while remaining and stats['probes'] < probes:
        core = sorted(remaining)
        stats['probes'] += 1
        if zero_consistent(problem, rows, core, stats, check):
            break
        # The full set is already a certified unit deficit. Shrinking makes it
        # possible to charge disjoint deficits without counting one seam twice.
        for edge in tuple(core):
            if stats['probes'] >= probes:
                break
            trial = [x for x in core if x != edge]
            stats['probes'] += 1
            if not zero_consistent(problem, rows, trial, stats, check):
                core = trial
        cores.append(tuple(core))
        remaining.difference_update(core)

    # Project integral residual pair costs onto their endpoints, then project
    # unary minima onto the global constant. All moves preserve every board cost.
    projected = 0
    unary = [dict.fromkeys(row, 0) for row in rows]
    if cache is None:
        cache = SeamCostCache(problem,limit=0)
    for index,(a,sa,b,sb) in enumerate(problem.edges):
        check()
        size = len(rows[a]) * len(rows[b])
        if stats['factor_pairs'] + size > shift_pairs:
            continue
        stats['factor_pairs'] += size
        row_min,col_min = cache.project(problem,index,rows[a],rows[b],stats)
        for cell,projection in ((a,row_min),(b,col_min)):
            for code,delta in zip(rows[cell],projection):
                unary[cell][code] += delta
    for row in unary:
        projected += min(row.values())
    return {'lower_bound': max(projected, len(cores)), 'shift_bound': projected,
            'vac_bound': len(cores), 'stats': dict(stats), 'infeasible_domain': False,
            'certificate': DefectCores(problem.fingerprint, rows, tuple(cores))}

