"""Minimize named reference departures in a single, physically paired circuit at one phase."""
from collections import defaultdict
from math import inf
from threading import Timer
from time import monotonic

from r27_structure_filters import domain_bits
from r30_gold import embedding
from r31_geographic import ribbon_genus
from r32_gold_bonds import endpoints, inventory, loop_groups
from r35_loss_witness import physical_price, raw_sides, reciprocal
from r35_mate_price import ClassPrices


def strand_components(problem, pairs, *, complete=True):
    """Independently count closed components using the raw named physical strands."""
    side_ends = {}
    count = 0
    for tile, data in enumerate(problem.tiles):
        for segment in data["segments"]:
            for side, slot in segment:
                side_ends[3 * tile + side, slot] = count
            count += 1
    graph = [[] for _ in range(count)]
    for a, b in pairs:
        for slot in range(1, 12):
            if (a, slot) in side_ends:
                u, v = side_ends[a, slot], side_ends[b, 12 - slot]
                graph[u].append(v)
                graph[v].append(u)
    if complete and any(len(neighbors) != 2 for neighbors in graph):
        raise ValueError("The pairing does not cover every labelled strand endpoint.")
    pending = set(range(count))
    sizes = []
    while pending:
        work = [pending.pop()]
        size = 0
        closed = True
        while work:
            u = work.pop()
            size += 1
            closed &= len(graph[u]) == 2
            for v in graph[u]:
                if v in pending:
                    pending.remove(v)
                    work.append(v)
        if closed:
            sizes.append(size)
    return sorted(sizes)


def _repair(problem, seed, allowed, required, reference, groups, deadline, max_trials):
    # Re-pair two seams without increasing the number of lost reference bonds.
    # An interrupted or unsuccessful neighborhood is only a failed witness probe.
    start = strand_components(problem, seed)
    page = [(seed, ())]
    visited = {seed}
    for depth in (1, 2):
        following = []
        for state, trail in page:
            choices = sorted(state - required)
            for index, (a, b) in enumerate(choices):
                if monotonic() >= deadline:
                    return None
                for c, d in choices[index + 1:]:
                    for x, y in ((c, d), (d, c)):
                        u, v = tuple(sorted((a, x))), tuple(sorted((b, y)))
                        if u not in allowed or v not in allowed:
                            continue
                        old = frozenset(((a, b), (c, d)))
                        if len(old & reference) > int(u in reference) + int(v in reference):
                            continue
                        child = (state - old) | {u, v}
                        if child in visited or any(group <= child for group in groups):
                            continue
                        if len(visited) >= max_trials:
                            return None
                        visited.add(child)
                        path = (*trail, (sorted(old), [u, v]))
                        cycles = strand_components(problem, child)
                        if cycles == [problem.total_segments]:
                            return child, dict(depth=depth, tested=len(visited), trades=path)
                        if depth == 1 and len(cycles) <= len(start):
                            following.append((child, path))
        page = following
    return None


def _lift(problem, mesh):
    table = [-1] * (3 * problem.n)
    for a, b in mesh:
        table[a], table[b] = b, a
    codes = embedding(problem, domain_bits(problem), table)
    if codes is not None and not problem.validate(codes)["valid"]:
        raise ArithmeticError("Geographic lift failed original-board validation.")
    return codes, ribbon_genus(problem, sorted(mesh))


def _checked_incumbent(problem, saved, allowed, reference, required, groups, upper):
    if not isinstance(saved, dict):
        return None
    rows = saved.get("pairs")
    if not isinstance(rows, (list, tuple)) or len(rows) != 3 * problem.n // 2:
        return None
    mesh = set()
    endpoints_seen = set()
    for row in rows:
        if (not isinstance(row, (list, tuple)) or len(row) != 2 or
                any(type(side) is not int for side in row)):
            return None
        edge = tuple(sorted(row))
        if edge not in allowed or endpoints_seen.intersection(edge):
            return None
        endpoints_seen.update(edge)
        mesh.add(edge)
    if (not required <= mesh or any(group <= mesh for group in groups) or
            len(reference - mesh) > upper or strand_components(problem, mesh) != [problem.total_segments]):
        return None
    return frozenset(mesh)


def _replayed_loop_cores(problem, saved, allowed):
    if not isinstance(saved, dict):
        return set()
    candidates = []
    prior = saved.get("replayed_loop_cores")
    if isinstance(prior, list):
        candidates.extend(prior)
    rounds = saved.get("rounds")
    if isinstance(rounds, list):
        for row in rounds:
            if isinstance(row, dict) and isinstance(row.get("new_loop_cores"), list):
                candidates.extend(row["new_loop_cores"])
    checked = set()
    for raw in candidates:
        if not isinstance(raw, (list, tuple)) or not 1 <= len(raw) <= 3 * problem.n // 2:
            continue
        if any(not isinstance(e, (list, tuple)) or len(e) != 2 or
               any(type(v) is not int for v in e) for e in raw):
            continue
        edges = frozenset(tuple(sorted(e)) for e in raw)
        if not edges <= allowed or len({side for e in edges for side in e}) != 2 * len(edges):
            continue
        if edges in checked:
            continue
        if any(size < problem.total_segments for size in strand_components(problem, edges, complete=False)):
            checked.add(edges)
    return checked


def _published_sewings(problem, phase):
    """Return possible hints; every bond, claimed optimum and lift is rechecked."""
    from pathlib import Path
    import json
    if problem.n != 160:
        return
    original = 0xd237aba7e91101e88041413e8c589902021e5126
    first = 0xcc57eae4e99109f8b073557e8c590b9202465766
    second = 0xcc778d82e99c517987f6b57ec15837c21207cf06
    mask = (1 << problem.n) - 1
    selected = {
        original: {"mip84-path-old-1"},
        first: {"mip84-29-combined-0", "mip84-path-new-1", "mip84-29-combined-19"},
        second: {"mip84-path-second-0"},
    }
    representative = next((p for p in selected if phase in (p, p ^ mask)), None)
    if representative is None:
        return
    docs = Path(__file__).resolve().parent / "docs"
    try:
        catalog = json.loads((docs / "R35_MIXED_MIP_DONOR.json").read_bytes())
    except (OSError, ValueError):
        catalog = None
    if isinstance(catalog, dict):
        others = catalog.get("others", [])
        if isinstance(others, list):
            for record in [catalog, *others]:
                if (isinstance(record, dict) and record.get("phase") == hex(representative)
                        and isinstance(record.get("name"), str) and record["name"] in selected[representative]):
                    yield {"pairs": record.get("pairs")}
    if representative != original:
        return
    for file in ("R35_NEAR_REFERENCE_TOURS.json", "R35_MATE_PORTFOLIOS.json"):
        try:
            data = json.loads((docs / file).read_bytes())
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        if file == "R35_NEAR_REFERENCE_TOURS.json":
            yield from data.values()
        else:
            for key, labels in (("portfolios", ("physical",)),
                                ("contact_cases", ("public", "sat_without_exchange", "independent_exchange")),
                                ("residual_cases", ("public",))):
                rows = data.get(key)
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            for label in labels:
                                yield row.get(label)


def search(problem, reference, groups, required, phase, upper, seconds, *, switch_trials=128, saved=None, publish=None, published=False):
    """Exact fixed-phase search up to the supplied loss cap; zero seconds is unlimited."""
    if (type(phase) is not int or not 0 <= phase < 1 << problem.n or
            type(upper) is not int or upper < 0 or type(switch_trials) is not int or
            switch_trials < 0 or not seconds >= 0):
        raise ValueError("Invalid phase, loss cap, switch budget, or duration.")
    deadline = monotonic() + seconds if seconds else inf
    reference, required = frozenset(reference), frozenset(required)
    groups = tuple(frozenset(group) for group in groups)
    result = dict(phase=hex(phase), fingerprint=problem.fingerprint, cap=upper,
                  status="UNKNOWN", rounds=[], named_bonds=len(reference))
    ordinary = physical_price(problem.tiles, reference, groups, required, phase)
    if ordinary is None:
        return dict(result, status="INFEASIBLE", signed_inventory_infeasible=True)
    lower = max(ordinary[0], ClassPrices(problem, reference, groups, required, limit=12).memo(phase))
    if lower == inf:
        return dict(result, status="INFEASIBLE", no_physical_mates=True)
    result["certified_loss_lower_bound"] = lower
    if lower > upper:
        return dict(result, status="INFEASIBLE")
    raw = raw_sides(problem.tiles)
    labels = [(slots, color ^ ((phase >> (i // 3)) & 1)) for i, (slots, color) in enumerate(raw)]
    allowed = {e for e in inventory(problem).possible if reciprocal(labels[e[0]]) == labels[e[1]]}
    if not required <= allowed:
        return dict(result, status="INFEASIBLE", no_physical_mates=True)
    replayed = _replayed_loop_cores(problem, saved, allowed)
    if replayed:
        result["replayed_loop_cores"] = [sorted(c) for c in sorted(replayed, key=lambda c: (len(c), sorted(c)))]
    incumbent = _checked_incumbent(problem, saved, allowed, reference, required, groups, upper)
    incumbent_loss = len(reference - incumbent) if incumbent is not None else None
    source = "saved_pairing_revalidated"
    if published and incumbent_loss != lower:
        for candidate in _published_sewings(problem, phase):
            mesh = _checked_incumbent(problem, candidate, allowed, reference, required, groups, upper)
            if mesh is None:
                continue
            cost = len(reference-mesh)
            if incumbent_loss is None or cost < incumbent_loss:
                incumbent, incumbent_loss = mesh, cost
                source = "published_pairing_revalidated"
                if cost == lower:
                    break
                replayed.update(_replayed_loop_cores(problem, candidate, allowed))
        if replayed:
            result["replayed_loop_cores"] = [sorted(c) for c in sorted(replayed, key=lambda c: (len(c), sorted(c)))]

    def finish_saved():
        codes, genus = _lift(problem, incumbent)
        return dict(result, status="GOLD" if codes is not None else "CONNECTED_SEWING",
                    loss=incumbent_loss, pairs=sorted(incumbent), codes=codes, genus=genus,
                    **{source: True})

    if incumbent is not None:
        if incumbent_loss < lower:
            raise ArithmeticError("Verified complete sewing contradicts the computed lower bound.")
        if incumbent_loss == lower:
            return finish_saved()
        result.update(pairs=sorted(incumbent), verified_incumbent_loss=incumbent_loss)
    if monotonic() >= deadline:
        return result

    from pysat.card import CardEnc, EncType, ITotalizer
    from pysat.formula import IDPool
    from pysat.solvers import Solver
    pool = IDPool()
    ids = {e: pool.id(e) for e in sorted(allowed)}
    by_side = defaultdict(list)
    for (a, b), literal in ids.items():
        by_side[a].append(literal)
        by_side[b].append(literal)
    clauses = [[ids[e]] for e in sorted(required)]
    for side in range(3 * problem.n):
        choices = by_side[side]
        clauses.append(choices)
        if len(choices) > 1:
            clauses.extend(CardEnc.atmost(lits=choices, bound=1, vpool=pool, encoding=EncType.seqcounter).clauses)
    for group in groups:
        if group <= allowed:
            clauses.append([-ids[e] for e in sorted(group)])
    fixed = len(reference - allowed)
    optional = [-ids[e] for e in sorted(reference & allowed)]
    ceiling = min(upper, len(reference), incumbent_loss if incumbent_loss is not None else upper)
    slots = [s for s, _ in raw]
    arcs = endpoints(problem)
    upper_extra = max(0, ceiling - fixed)
    # Totalizer outputs are shared by every successive loss cap. A final cap
    # that allows every remaining loss needs no assumption.
    from contextlib import ExitStack
    with ExitStack() as resources:
        tally = None
        if optional and lower - fixed < len(optional):
            tally = resources.enter_context(ITotalizer(lits=optional, ubound=upper_extra, top_id=pool.top))
            clauses.extend(tally.cnf.clauses)
        solver = resources.enter_context(Solver(name="g4", bootstrap_with=clauses))
        preferred = incumbent if incumbent is not None else reference
        solver.set_phases([v if e in preferred else -v for e, v in ids.items()])
        learned = set(replayed)
        for cut in result.get("replayed_loop_cores", ()):
            solver.add_clause([-ids[tuple(e)] for e in cut])
        for cap in range(lower, ceiling + 1):
            result["certified_loss_lower_bound"] = cap
            if cap == incumbent_loss:
                return finish_saved()
            pins = [-tally.rhs[cap-fixed]] if tally and cap-fixed < len(optional) else []
            while monotonic() < deadline:
                before = solver.accum_stats()["decisions"]
                if deadline == inf:
                    answer = solver.solve(assumptions=pins)
                else:
                    timer = Timer(max(.001, deadline - monotonic()), solver.interrupt)
                    timer.daemon = True
                    timer.start()
                    try:
                        answer = solver.solve_limited(assumptions=pins, expect_interrupt=True)
                    finally:
                        timer.cancel()
                        timer.join()
                row = dict(cap=cap, verdict=answer, decisions=solver.accum_stats()["decisions"] - before)
                result["rounds"].append(row)
                if answer is None:
                    return result
                if answer is False:
                    if publish is not None:
                        publish(result)
                    break
                model = set(solver.get_model())
                mesh = frozenset(e for e, v in ids.items() if v in model)
                if (len(mesh) != 3 * problem.n // 2 or not required <= mesh or
                        any(group <= mesh for group in groups) or len(reference - mesh) > cap):
                    raise ArithmeticError("Native matching failed independent physical obligations.")
                cycles = strand_components(problem, mesh)
                cuts = loop_groups(arcs, slots, mesh, problem.total_segments)
                if bool(cuts) == (cycles == [problem.total_segments]):
                    raise ArithmeticError("The two physical strand checks disagree.")
                row.update(loss=len(reference - mesh), cycles=cycles)
                result.setdefault("first_pairing", sorted(mesh))
                if cuts and switch_trials:
                    repaired = _repair(problem, mesh, allowed, required, reference, groups, deadline, switch_trials)
                    if repaired:
                        mesh, row["switch_repair"] = repaired
                        if len(reference-mesh) < lower or strand_components(problem,mesh) != [problem.total_segments]:
                            raise ArithmeticError("Exchanged pairing failed the fixed-phase certificate.")
                        cuts = set()
                if not cuts:
                    codes, genus = _lift(problem, mesh)
                    return dict(result, status="GOLD" if codes is not None else "CONNECTED_SEWING",
                                loss=len(reference - mesh), pairs=sorted(mesh), codes=codes, genus=genus)
                new = sorted(cuts - learned, key=lambda c: (len(c), sorted(c)))
                if not new:
                    raise ArithmeticError("Native matching repeated an excluded proper subloop.")
                for cut in new:
                    solver.add_clause([-ids[e] for e in sorted(cut)])
                row["new_loop_cores"] = [sorted(cut) for cut in new]
                learned.update(new)
                if publish is not None:
                    publish(result)
            else:
                return result
    return dict(result, status="INFEASIBLE", certified_loss_lower_bound=ceiling+1)
