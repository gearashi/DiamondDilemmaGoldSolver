"""Separate low-loss fixed phases using independently replayable geographic obstructions."""
import argparse
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope, verify_21
from io_utils import atomic_json
from r28_geo_sewing import PosePacking
from r32_gold_loss import read_json
from r35_loss_witness import cover_missing, physical_price


def minimize_obstruction(packing, bonds):
    """Every deletion requires another completed negative. UNKNOWN cannot remove a premise."""
    def impossible(pairs):
        return packing.check(pairs, max_steps=2000)['feasible'] is False

    core = sorted(bonds)
    if not impossible(core):
        return None
    width = (len(core) + 1) // 2
    while width:
        offset = 0
        while offset < len(core):
            smaller = core[:offset] + core[offset + width:]
            if impossible(smaller):
                core = smaller
            else:
                offset += width
        width //= 2
    return frozenset(core)


def connected_pose_count(problem, bonds):
    """Independent scalar check from raw edge incidence, with unrestricted root poses."""
    incidence = {}
    for a, sa, b, sb in problem.edges:
        incidence[a, sa] = b, sb
        incidence[b, sb] = a, sa
    graph = {}
    for a, b in bonds:
        for left, right in ((a, b), (b, a)):
            tile, side = divmod(left, 3)
            other, back = divmod(right, 3)
            graph.setdefault(tile, []).append((side, other, back))
    if not graph:
        return 1
    root = min(graph)
    reached = {root}
    fringe = [root]
    while fringe:
        for _, other, _ in graph[fringe.pop()]:
            if other not in reached:
                reached.add(other)
                fringe.append(other)
    if len(reached) != len(graph):
        return None
    count = 0
    for cell in range(problem.n):
        for rotation in range(3):
            assigned = {root: (cell, rotation)}
            cells = {cell}
            pending = [root]
            while pending:
                tile = pending.pop()
                at, turn = assigned[tile]
                for side, other, back in graph[tile]:
                    seam = incidence.get((at, (side + turn) % 3))
                    if seam is None:
                        break
                    dest, facing = seam
                    pose = (dest, (facing - back) % 3)
                    if other in assigned:
                        if assigned[other] != pose:
                            break
                    elif dest in cells:
                        break
                    else:
                        assigned[other] = pose
                        cells.add(dest)
                        pending.append(other)
                else:
                    continue
                break
            else:
                count += 1
    return count


def price_and_separate(problem, reference, groups, required, phase, cap, packing, cores, *, disconnected=False, max_rounds=None, checkpoint=None):
    rounds = []
    while True:
        projected = [core - required for core in cores]
        constraints = [*groups, *projected]
        result = physical_price(problem.tiles, reference, constraints, required, phase)
        if result is None:
            return dict(status='ineligible phase', rounds=rounds)
        price, failed = result
        lost = (reference & failed) | cover_missing(constraints, required, failed)
        if len(lost) != price:
            raise ArithmeticError('Loss price disagrees with the explicit minimizing departures.')
        if price > cap:
            return dict(status='all losses at cap excluded', minimum=price, rounds=rounds)
        if max_rounds is not None and len(rounds) >= max_rounds:
            return dict(status='separation limit reached', minimum=price, rounds=rounds)
        chosen = (reference - lost) | required
        feasibility = packing.check(chosen, max_steps=2000)['feasible']
        if feasibility is not False:
            return dict(status='pose packing feasible' if feasibility else 'pose packing undecided',
                        minimum=price, witness_departures=sorted(lost), rounds=rounds)
        core = minimize_obstruction(packing, chosen)
        from r35_geographic_packing_check import check_packing
        check=check_packing(problem,core) if (core is not None and disconnected) else None
        verified=(check['feasible'] is False if check is not None
                  else core is not None and connected_pose_count(problem,core)==0)
        if not verified:
            return dict(status='no independently checked connected obstruction', minimum=price, rounds=rounds)
        if not core - required or core in cores:
            raise ArithmeticError('The separating cut must exclude this assignment.')
        cores.append(core)
        row=dict(minimum=price,lost=sorted(lost),core=sorted(core))
        if check is not None:row['independent_packing']=check
        rounds.append(row)
        if checkpoint is not None:
            checkpoint(rounds)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles', required=True)
    cli.add_argument('--output', required=True, type=Path)
    args = cli.parse_args()
    p, ref, groups, required = problem_scope(args.tiles)
    phase21, _, _ = verify_21(p, ref, groups, required)
    data = read_json(Path(__file__).parent / 'docs/R35_OVERLAP_BOUNDARY.json')
    phase22 = int(data['disjoint20']['phase'], 16)
    pack = PosePacking(p)
    cores = []
    reports = []
    for w, cap in ((phase21, 21), (phase22, 22)):
        result = price_and_separate(p, ref, groups, required, w, cap, pack, cores)
        reports.append(dict(phase=hex(w), cap=cap, **result))
        atomic_json(args.output, dict(puzzle=p.fingerprint, reports=reports,
                    geographic_cores=[sorted(c) for c in cores]))
        print(cap, result['status'], result.get('minimum'), len(result['rounds']), flush=True)


if __name__ == '__main__':
    main()
