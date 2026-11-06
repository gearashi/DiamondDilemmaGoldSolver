"""Optional bounded CP-SAT repair of a GPU search candidate.

Only the tiles already in the selected region may move. Every seam touching
that region must match. INFEASIBLE therefore means this restricted repair is
impossible; UNKNOWN means no conclusion before the resource limit. Neither
proves anything about the complete puzzle. Edge matching does not certify
one connected gold loop: use validator.py on every resulting arrangement.
"""
from __future__ import annotations
import argparse
from collections import deque
import json
from pathlib import Path
import time
import numpy as np
from gpu_engine import legal_boards, reverse11, score_boards_cpu


def _mismatches(masks, neighbors, sides, board):
    result = []
    reversed_masks = reverse11(masks)
    for a in range(160):
        for sa in range(3):
            b, sb = int(neighbors[a, sa]), int(sides[a, sa])
            if a < b and masks[board[a], sa] != reversed_masks[board[b], sb]:
                result.append((a, b))
    return result


def select_region(masks, neighbors, sides, board, max_cells=64, seed=20261007):
    """Cover all defects when they fit; otherwise grow around a random defect."""
    mismatches = _mismatches(masks, neighbors, sides, board)
    if not mismatches:
        return []
    rng = np.random.default_rng(seed)
    endpoints = sorted({v for edge in mismatches for v in edge})
    if len(endpoints) <= max_cells:
        selected = set(endpoints)
    else:
        selected = set(mismatches[int(rng.integers(len(mismatches)))])
    degree = np.zeros(160, int)
    for a, b in mismatches:
        degree[a] += 1
        degree[b] += 1
    while len(selected) < max_cells:
        frontier = {int(b) for a in selected for b in neighbors[a]} - selected
        if not frontier:
            frontier = set(range(160)) - selected
        if not frontier:
            break
        values = []
        for cell in sorted(frontier):
            # Fill holes and include boundary defects early, with randomized ties.
            value = 4 * sum(int(nb) in selected for nb in neighbors[cell]) + degree[cell] + rng.random()
            values.append((value, cell))
        selected.add(max(values)[1])
    return sorted(selected)


def repair_local(masks, neighbors, neighbor_sides, board, max_cells=64,
                 seconds=10.0, workers=4, seed=20261007, region=None, require_change=True):
    """Return JSON-compatible local-repair evidence and a validated candidate if found."""
    started = time.monotonic()
    masks = np.asarray(masks)
    neighbors = np.asarray(neighbors)
    sides = np.asarray(neighbor_sides)
    if masks.shape != (480, 3) or masks.dtype.kind not in 'iu' or np.any(masks < 0) or np.any(masks > 2047):
        raise ValueError('Expected 480x3 masks with bit positions 0..10.')
    if neighbors.shape != (160, 3) or sides.shape != (160, 3):
        raise ValueError('Expected 160x3 neighbor and neighbor-side arrays.')
    if neighbors.dtype.kind not in 'iu' or sides.dtype.kind not in 'iu':
        raise ValueError('Neighbor arrays must contain integers.')
    if np.any(neighbors < 0) or np.any(neighbors >= 160) or np.any(sides < 0) or np.any(sides >= 3):
        raise ValueError('Invalid neighbor cell or side index.')
    for a in range(160):
        for sa in range(3):
            b, sb = int(neighbors[a, sa]), int(sides[a, sa])
            if a == b or neighbors[b, sb] != a or sides[b, sb] != sa:
                raise ValueError('Neighbor map must be reciprocal without self-edges.')
    if type(max_cells) is not int or not 2 <= max_cells <= 160:
        raise ValueError('max_cells must be an integer in 2..160.')
    if not np.isfinite(seconds) or not .001 <= seconds <= 3600:
        raise ValueError('seconds must be in .001..3600.')
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('workers must be an integer in 1..32.')
    codes = legal_boards(board)
    if len(codes) != 1:
        raise ValueError('Repair accepts exactly one arrangement.')
    codes = codes[0]
    before = int(score_boards_cpu(masks, neighbors, sides, codes)[0])
    if region is None:
        chosen = select_region(masks, neighbors, sides, codes, max_cells, seed)
    else:
        raw_region = np.asarray(region)
        if raw_region.ndim != 1 or raw_region.dtype.kind not in 'iu':
            raise ValueError('Region must be an integer sequence.')
        chosen = sorted(map(int, raw_region))
        if len(set(chosen)) != len(chosen) or len(chosen) > max_cells or any(c < 0 or c >= 160 for c in chosen):
            raise ValueError('Region contains duplicate, invalid, or too many cells.')
    result = {'status': 'pending', 'before_score': before, 'after_score': None,
              'improvement': None, 'board': None, 'region_cells': chosen,
              'time_limit_seconds': float(seconds), 'workers': workers, 'seed': int(seed),
              'scope': 'Only selected cells and the tiles already in those cells; all incident seams exact.',
              'single_loop_verified': False}
    def finish(status, **extra):
        result.update(status=status, wall_time_seconds=time.monotonic() - started, **extra)
        return result
    if not chosen:
        return finish('already_edge_perfect' if before == 240 else 'empty_region')
    selected = set(chosen)
    pieces = sorted(int(codes[c]) // 3 for c in chosen)
    pool = np.array([3 * piece + r for piece in pieces for r in range(3)], np.int16)
    reverse = reverse11(masks)
    domains = {}
    internal = []
    touching = 0
    for a in chosen:
        allowed = pool.copy()
        for sa in range(3):
            b, sb = int(neighbors[a, sa]), int(sides[a, sa])
            if b not in selected:
                allowed = allowed[masks[allowed, sa] == reverse[codes[b], sb]]
                touching += 1
            elif a < b:
                internal.append((a, sa, b, sb))
                touching += 1
        domains[a] = allowed
    result['affected_edges'] = touching
    # Sound arc consistency reduces the table model before the timed CP-SAT call.
    changed = True
    while changed:
        changed = False
        for a, sa, b, sb in internal:
            da, db = domains[a], domains[b]
            na = da[np.isin(masks[da, sa], reverse[db, sb])]
            nb = db[np.isin(reverse[db, sb], masks[na, sa])]
            if len(na) != len(da) or len(nb) != len(db):
                changed = True
                domains[a], domains[b] = na, nb
        if any(not len(v) for v in domains.values()):
            return finish('local_infeasible', solver_status='PREPROCESS_INFEASIBLE',
                          explanation='Boundary filtering or seam arc consistency exhausted a local domain.')
        forced = {}
        for cell, values in domains.items():
            tile_ids = np.unique(values // 3)
            if len(tile_ids) == 1:
                piece = int(tile_ids[0])
                if piece in forced and forced[piece] != cell:
                    return finish('local_infeasible', solver_status='PREPROCESS_INFEASIBLE',
                                  explanation='Two cells require the same available tile.')
                forced[piece] = cell
        for cell, values in list(domains.items()):
            forbidden = [piece for piece, location in forced.items() if location != cell]
            kept = values[~np.isin(values // 3, forbidden)]
            if len(kept) != len(values):
                changed = True
                domains[cell] = kept
        if any(not len(v) for v in domains.values()):
            return finish('local_infeasible', solver_status='PREPROCESS_INFEASIBLE',
                          explanation='Forced tile uniqueness exhausted a local domain.')
    try:
        from ortools.sat.python import cp_model
    except ImportError as exc:
        raise RuntimeError('Local exact repair requires the optional ortools package in this solver environment.') from exc
    model = cp_model.CpModel()
    variables, tile_variables, edge_vars, reverse_vars = {}, [], {}, {}
    for cell in chosen:
        allowed = domains[cell]
        cv = model.NewIntVarFromDomain(cp_model.Domain.FromValues([int(v) for v in allowed]), f'code_{cell}')
        pv = model.NewIntVarFromDomain(cp_model.Domain.FromValues([int(v) for v in np.unique(allowed // 3)]), f'tile_{cell}')
        ev = [model.NewIntVarFromDomain(cp_model.Domain.FromValues([int(v) for v in np.unique(masks[allowed, s])]), f'e_{cell}_{s}') for s in range(3)]
        rv = [model.NewIntVarFromDomain(cp_model.Domain.FromValues([int(v) for v in np.unique(reverse[allowed, s])]), f'r_{cell}_{s}') for s in range(3)]
        rows = [[int(code), int(code) // 3, *map(int, masks[code]), *map(int, reverse[code])] for code in allowed]
        model.AddAllowedAssignments([cv, pv, *ev, *rv], rows)
        variables[cell], edge_vars[cell], reverse_vars[cell] = cv, ev, rv
        tile_variables.append(pv)
        if int(codes[cell]) in allowed:
            model.AddHint(cv, int(codes[cell]))
    model.AddAllDifferent(tile_variables)
    for a, sa, b, sb in internal:
        model.Add(edge_vars[a][sa] == reverse_vars[b][sb])
    if require_change:
        model.AddForbiddenAssignments([variables[c] for c in chosen], [[int(codes[c]) for c in chosen]])
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(seconds)
    solver.parameters.num_search_workers = workers
    solver.parameters.random_seed = int(seed) % 2147483647
    status = solver.Solve(model)
    result.update(solver_status=solver.StatusName(status), solver_seconds=solver.WallTime(),
                  branches=solver.NumBranches(), conflicts=solver.NumConflicts(),
                  domain_rows=sum(len(v) for v in domains.values()))
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        repaired = codes.copy()
        for cell in chosen:
            repaired[cell] = solver.Value(variables[cell])
        # Rescore all 240 edges independently of the model's variables and tables.
        after = int(score_boards_cpu(masks, neighbors, sides, repaired)[0])
        if after < before:
            raise RuntimeError('Local exact repair unexpectedly lowered the independently checked score.')
        failures = _mismatches(masks, neighbors, sides, repaired)
        if any(a in selected or b in selected for a, b in failures):
            raise RuntimeError('CP-SAT result failed independent incident-seam validation.')
        outside = np.array(sorted(set(range(160)) - selected), dtype=np.int32)
        if not np.array_equal(codes[outside], repaired[outside]):
            raise RuntimeError('Local repair changed a tile outside its region.')
        return finish('repaired', board=repaired.astype(int).tolist(), after_score=after, improvement=after - before)
    if status == cp_model.INFEASIBLE:
        return finish('local_infeasible', explanation='This fixed-outside, fixed-tile-pool local model has no solution.')
    if status == cp_model.MODEL_INVALID:
        raise RuntimeError('Invalid CP-SAT model: ' + solver.ResponseStats())
    return finish('inconclusive', explanation='No solution or local impossibility proof found within the resource limit.')


def self_test():
    from test_gpu import fixture
    masks, neighbors, sides, reference = fixture()
    damaged = reference.copy()
    damaged[0], damaged[80] = damaged[80], damaged[0]
    result = repair_local(masks, neighbors, sides, damaged, max_cells=2, seconds=3, workers=1, region=[0, 80])
    assert result['status'] == 'repaired', result
    assert result['after_score'] == 240, result
    assert np.array_equal(np.array(result['board']) // 3, reference // 3)
    impossible = repair_local(masks, neighbors, sides, damaged, max_cells=2, seconds=3, workers=1, region=[0, 1])
    assert impossible['status'] == 'local_infeasible', impossible
    print(json.dumps({'status': 'passed', 'known_repair_score': result['after_score'],
                      'restricted_impossibility': impossible['solver_status'],
                      'solver_seconds': result['solver_seconds']}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board', type=Path)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'data' / 'tiles.json')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cells', type=int, default=64)
    parser.add_argument('--seconds', type=float, default=10)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.board is None:
        parser.error('--board is required unless --self-test is used')
    from geometry import build_board
    from validator import orientation_masks, validate_arrangement
    data = json.loads(args.data.read_text(encoding='utf-8-sig'))
    saved = json.loads(args.board.read_text(encoding='utf-8-sig'))
    codes = saved.get('codes', saved.get('board')) if isinstance(saved, dict) else saved
    geometry = build_board()
    result = repair_local(orientation_masks(data), geometry.neighbor_cells, geometry.neighbor_sides,
                          codes, max_cells=args.cells, seconds=args.seconds,
                          workers=args.workers, seed=args.seed)
    if result['board'] is not None:
        report = validate_arrangement(data, result['board'])
        result['validation'] = report
        result['single_loop_verified'] = bool(report['valid'])
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding='utf-8')
    print(rendered)


if __name__ == '__main__':
    main()
