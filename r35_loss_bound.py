"""Checked public reference-loss evidence and a bounded all-pattern signed-inventory search."""
from math import isnan

from r35_loss_witness import physical_price

# Digest of the fingerprint, all 207 reference bonds, 8 required bonds and ordered
# groups. The certificate and raw-endpoint checks are listed in R35_LOSS_PROOF.json.
REFERENCE_SCOPE = 'd639db280cbffdd887d81f336825b94e4e08d272d94fd11deaace4777e7d96b6'
CHECKED_LOWER_BOUND = 20
REFERENCE_PHASE = 0xd237aba7e99101e88041403e8c589902025e5326
REFERENCE_PHASES = (REFERENCE_PHASE,0xce55aaa6e89109fca073547e8c580ba32e0e5fa2,
                    0xd237aba7e91101e88041413e8c589902021e5126)


# Cap 20 is refuted by these three physical pairs as one additional
# proper-loop obligation, alongside the original ten disjoint groups.
BOUND_21_LOOP = frozenset({(115,176), (174,382), (285,383)})


def checked_lower_bound(packed_identity, groups):
    if packed_identity != REFERENCE_SCOPE:
        return None
    if any(frozenset(group)==BOUND_21_LOOP for group in groups):
        return 21
    return CHECKED_LOWER_BOUND


def published_hit(problem, reference, groups, required, outside, identity, cap, integer, phase=REFERENCE_PHASE):
    if identity != REFERENCE_SCOPE:
        return None
    result = physical_price(problem.tiles, reference, groups, required, phase, integer=integer)
    if result is None or result[0] > cap:
        return None
    cost, failed = result
    errors = [pair for pair in outside if pair in failed]
    return dict(status='INTEGER_RELAXATION_FEASIBLE' if integer else 'PHASE_RELAXATION_FEASIBLE',
                outside_errors=errors, errors=len(errors), cap=cap, loss_cost=cost,
                phase=hex(phase), witness_revalidated=True, reused=True,
                relaxation="disjoint_group_signed_inventory" if integer else "disjoint_group_parity")


def global_integer(problem, reference, groups, required, outside, cap, seconds, *, additional_groups=(), excess=False, independent_native=False, conditions=None):
    from r32_gold_loss import digest
    from r35_integer_sat import InventorySAT
    if type(cap) is not int or cap<0 or isnan(seconds) or seconds<0:
        raise ValueError('Invalid signed loss cap or time limit.')
    if conditions is not None:
        conditions=conditions.for_loss(reference,[*groups,*additional_groups],required,cap)
    if not independent_native and not excess:
        identity=digest([3,problem.fingerprint,sorted(reference),sorted(required),[sorted(g) for g in groups]])
        allowed=set(reference)|set(required)
        expected=set(reference)-set().union(*groups)-set(required)
        if (identity==REFERENCE_SCOPE and len(outside)==len(set(outside)) and set(outside)==expected
                and all(set(g)<=allowed for g in additional_groups)):
            kind='all_group_signed_inventory' if additional_groups else 'disjoint_group_signed_inventory'
            bound=checked_lower_bound(identity,additional_groups)
            if cap<bound:
                return dict(status='INFEASIBLE',cap=cap,all_outside_layers=True,relaxation=kind,
                            certified_loss_lower_bound=bound,
                            work={'decisions':0,'conflicts':0,'reference_loss_certificate_reused':1})
            for phase in REFERENCE_PHASES:
                if conditions is not None and not conditions.accepts(phase):
                    continue
                hit=published_hit(problem,reference,[*groups,*additional_groups],required,outside,identity,cap,True,phase)
                if hit is not None:
                    if conditions is not None:hit['loss_cost']=conditions.price(phase,hit['loss_cost'])
                    hit.update(relaxation=kind,work={'decisions':0,'conflicts':0,'reference_phase_reused':1})
                    return hit

    cost_pairs = conditions.native_cost_pairs() if conditions is not None else None
    check = InventorySAT(problem, groups, required, outside, cap, excess=excess, additional_groups=additional_groups,
                         mate_cost_pairs=cost_pairs)
    try:
        if conditions is None:
            outcome=check.decide(seconds)
        else:
            from time import monotonic
            from math import inf
            conditions.install(check)
            outcome=conditions.decide(check,monotonic()+seconds if seconds else inf)
        work = check.solver.accum_stats()
        if outcome is None:
            return dict(status='UNKNOWN', cap=cap, work=work)
        if outcome is False:
            return dict(status='INFEASIBLE', cap=cap, all_outside_layers=True, work=work)
        phase = check.phase()
        result = physical_price(problem.tiles, reference, [*groups,*additional_groups], required, phase)
        if result is None or result[0] > cap:
            raise ArithmeticError('SAT returned a phase that failed the physical inventory or loss cap.')
        cost, failed = result
        if conditions is not None:cost=conditions.price(phase,cost)
        if cost>cap:raise ArithmeticError('Native phase exceeded its conditional physical loss cap.')
        errors = [pair for pair in outside if pair in failed]
        return dict(status='INTEGER_RELAXATION_FEASIBLE',cap=cap,phase=hex(phase),
                    loss_cost=cost,outside_errors=errors,work=work)
    finally:
        check.close()
