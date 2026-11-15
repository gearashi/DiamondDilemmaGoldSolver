"""Solve the unrestricted signed reference-loss relaxation with a physical-variable integer program."""
from math import ceil, inf
from time import monotonic

from r31_gold_connectivity import PhaseInterface
from r35_loss_witness import cover_missing, physical_price, raw_sides


def build_model(problem, reference, groups, required, cap, extra=()):
    from ortools.linear_solver import pywraplp

    engine = pywraplp.Solver.CreateSolver('SCIP')
    if engine is None:
        raise RuntimeError('This OR-Tools build does not provide SCIP.')
    engine.SetNumThreads(1)
    phase = [engine.BoolVar('tile_'+str(i)) for i in range(problem.n)]
    loss = {bond:engine.BoolVar(f'lost_{bond[0]}_{bond[1]}') for bond in sorted(reference-required)}
    engine.Add(phase[0] == 0)
    interface = PhaseInterface(problem)
    for j, parity in enumerate(interface.mod[:-1]):
        constant = sum(low[j] for low, high in interface.rows)
        row = constant + sum((high[j]-low[j]) * phase[i] for i, (low, high) in enumerate(interface.rows)
                             if high[j] != low[j])
        if parity:
            limit = 3 * problem.n
            engine.Add(row == 2 * engine.IntVar(-limit, limit, 'parity_'+str(j)))
        else:
            engine.Add(row == 0)

    sides = raw_sides(problem.tiles)
    for a, b in sorted(reference | required):
        x, y = phase[a//3], phase[b//3]
        slack = loss.get((a, b), 0)
        if sides[a][1] ^ sides[b][1] ^ (len(sides[a][0]) % 2):
            engine.Add(x+y+slack >= 1)
            engine.Add(x+y-slack <= 1)
        else:
            engine.Add(x-y <= slack)
            engine.Add(y-x <= slack)
    for group in [*groups, *extra]:
        engine.Add(sum(loss[b] for b in group-required) >= 1)
    engine.Add(sum(loss.values()) <= cap)
    engine.Minimize(sum(loss.values()))
    return engine, phase, loss


def checked_cycles(problem, reference, required):
    import json
    from r35_contact_certificate import CERTIFICATE, xor_conflict

    artifact = CERTIFICATE.parent/'R35_CONTACT_SIGNED_CYCLES.json'
    try:
        data = json.loads(artifact.read_bytes())
    except (OSError, ValueError):
        return []
    if data.get('fingerprint') != problem.fingerprint:
        return []
    allowed = reference | required
    result=[]
    for raw in data['required_xor']:
        group = frozenset(map(tuple, raw))
        if group <= allowed and group-required and xor_conflict(problem, group|required):
            result.append(group)
    return result


def decide(problem, reference, groups, required, outside, cap, seconds, *, conditions=None, hint=None, condition_factory=None, first_feasible=None):
    from ortools.linear_solver import pywraplp

    reference, required = set(reference), set(required)
    extra = checked_cycles(problem, reference, required)
    engine, phase, loss = build_model(problem, reference, groups, required, cap, extra)
    native_conditions = None
    if conditions is not None:
        from r35_mip_conditions import Conditions
        install = Conditions if condition_factory is None else condition_factory
        native_conditions = install(conditions, engine, phase, loss)
    if first_feasible is None:
        first_feasible = native_conditions is not None
    if first_feasible:
        engine.SetSolverSpecificParametersAsString('limits/solutions = 1')
    if hint is not None and 0 <= hint < 1 << problem.n:
        checked = physical_price(problem.tiles, reference, groups, required, hint)
        if checked is not None:
            omitted = (checked[1] & reference) | cover_missing(groups, required, checked[1])
            engine.SetHint([*phase,*loss.values()], [(hint >> i)&1 for i in range(problem.n)]
                           + [int(bond in omitted) for bond in loss])
    deadline = monotonic()+seconds if seconds and seconds != inf else inf
    exclusions = nodes = iterations = 0
    while True:
        if deadline != inf:
            remaining = deadline-monotonic()
            if remaining <= 0:
                answer = pywraplp.Solver.NOT_SOLVED
                break
            engine.SetTimeLimit(max(1, ceil(1000*remaining)))
        answer = engine.Solve()
        nodes += max(0, engine.nodes())
        iterations += max(0, engine.iterations())
        if answer not in (engine.OPTIMAL, engine.FEASIBLE):
            break
        witness = sum(int(v.solution_value() > .5) << i for i, v in enumerate(phase))
        price = physical_price(problem.tiles, reference, groups, required, witness)
        if price is None or price[0] > cap:
            raise ArithmeticError('Native integer witness failed raw physical inventory and loss replay.')
        cost = price[0]
        if native_conditions is not None:
            if native_conditions.reject(witness, cost, deadline):
                exclusions += 1
                continue
            cost = conditions.price(witness, cost)
        elif conditions is not None:
            if not conditions.accepts(witness) or conditions.price(witness, cost) > cap:
                # Only this complete phase was disproved. Other assignments keep their support.
                engine.Add(sum(1-v if (witness>>j)&1 else v for j,v in enumerate(phase)) >= 1)
                exclusions += 1
                continue
            cost = conditions.price(witness, cost)
        break
    work = dict(backend='SCIP', nodes=nodes, lp_iterations=iterations,
                checked_cycle_clauses=len(extra), rejected_conditional_phases=exclusions)
    if native_conditions is not None:
        work.update(native_conditions.work)
    if answer == pywraplp.Solver.INFEASIBLE:
        return dict(status='INFEASIBLE', cap=cap, all_outside_layers=True, work=work)
    if answer not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return dict(status='UNKNOWN', cap=cap, work=work)
    return dict(status='INTEGER_RELAXATION_FEASIBLE', cap=cap, loss_cost=cost, phase=hex(witness),
                outside_errors=[p for p in outside if p in price[1]], work=work)
