"""Checked public reference-loss evidence and a bounded all-pattern signed-inventory search."""
from r35_loss_witness import physical_price

# Digest of the fingerprint, all 207 reference bonds, 8 required bonds and ordered
# groups. The certificate and raw-endpoint checks are listed in R35_LOSS_PROOF.json.
REFERENCE_SCOPE = 'd639db280cbffdd887d81f336825b94e4e08d272d94fd11deaace4777e7d96b6'
CHECKED_LOWER_BOUND = 19
REFERENCE_PHASE = 0xd237aba7e99101e88041403e8c589902025e5326


def published_hit(problem, reference, groups, required, outside, identity, cap, integer):
    if identity != REFERENCE_SCOPE:
        return None
    result = physical_price(problem.tiles, reference, groups, required, REFERENCE_PHASE, integer=integer)
    if result is None or result[0] > cap:
        return None
    cost, failed = result
    errors = [pair for pair in outside if pair in failed]
    return dict(status='INTEGER_RELAXATION_FEASIBLE' if integer else 'PHASE_RELAXATION_FEASIBLE',
                outside_errors=errors, errors=len(errors), cap=cap, loss_cost=cost,
                phase=hex(REFERENCE_PHASE), witness_revalidated=True, reused=True)


def global_integer(problem, reference, groups, required, outside, cap, seconds):
    from r35_integer_sat import InventorySAT

    check = InventorySAT(problem, groups, required, outside, cap, excess=True)
    try:
        outcome = check.decide(seconds)
        work = check.solver.accum_stats()
        if outcome is None:
            return dict(status='UNKNOWN', cap=cap, work=work)
        if outcome is False:
            return dict(status='INFEASIBLE', cap=cap, all_outside_layers=True, work=work)
        phase = check.phase()
        result = physical_price(problem.tiles, reference, groups, required, phase)
        if result is None or result[0] > cap:
            raise ArithmeticError('SAT returned a phase that failed the physical inventory or loss cap.')
        cost, failed = result
        errors = [pair for pair in outside if pair in failed]
        return dict(status='INTEGER_RELAXATION_FEASIBLE',cap=cap,phase=hex(phase),
                    loss_cost=cost,outside_errors=errors,work=work)
    finally:
        check.close()
