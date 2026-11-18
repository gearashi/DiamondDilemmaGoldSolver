"""Exact engines on boards that are known to be solvable and look like the real puzzle.

Run from the repository root:
    python -B -m benchmarks.bench_r37_planted_inventory --data data/tiles.json --output PATH

Each test board comes from search_fixtures.planted_inventory: it has exactly
the side-signature inventory of the tile set in --data, and one full board of
matched seams is planted in it. --keep cells of that planted board are given
to the engine as fixed; the rest is searched. With --near-miss BOARD the
engines instead get --keep fully matched cells of that board on the real
tiles, which is the matching negative control.

The engine name `anneal` runs one targeted-annealing cooling run of --moves
moves on the whole planted board instead (--keep is ignored) and reports its
best seam score and how many cells agree with the planted board.

Results are timings on finite test instances for the edge target only. They
say nothing about whether the real puzzle has a Gold solution.
"""
import argparse
import json
import random
import time
from pathlib import Path

import constraint_dfs
import constraint_models
from anneal_core import DEFAULT_T_COLD, DEFAULT_T_HOT, AnnealModel, cooling_temperature
from search_fixtures import planted_inventory
from search_problem import SearchProblem

ENGINES = ('dfs', 'sat', 'sat-channel', 'sat-network', 'cp-sat', 'anneal')
COOLING_STEPS = 600


def instance(source, seed, keep, near_miss=None):
    """Return (problem, planted reference or None) with `keep` cells fixed."""
    rng = random.Random(seed)
    if near_miss is None:
        data, board, reference = planted_inventory(source, seed)
        cells = sorted(rng.sample(range(len(reference)), keep))
        return SearchProblem(data, board, fixed={cell: reference[cell] for cell in cells}), reference
    whole = SearchProblem(source)
    loose = {cell for edge in whole.validate(near_miss)['unmatched_edges'] for cell in (edge['cell_a'], edge['cell_b'])}
    cells = sorted(rng.sample([cell for cell in range(whole.n) if cell not in loose], keep))
    return SearchProblem(source, fixed={cell: near_miss[cell] for cell in cells}), None


def agreement(problem, codes, reference):
    """Cells with the planted tile and rotation, and cells that merely show the planted side signatures."""
    same = sum(int(a) == int(b) for a, b in zip(codes, reference))
    look = sum(tuple(problem.masks[int(a)]) == tuple(problem.masks[int(b)]) for a, b in zip(codes, reference))
    return {'cells_same_as_planted': same, 'cells_showing_planted_signatures': look}


def run(problem, engine, seconds):
    started = time.perf_counter()
    if engine == 'dfs':
        found = constraint_dfs.solve(problem, seconds=seconds, target='edges', options={'native': 'auto'})
    else:
        found = constraint_models.solve(problem, engine=engine, seconds=seconds, target='edges')
    elapsed = time.perf_counter() - started
    matched = None
    if found.get('codes'):
        report = problem.validate(found['codes'])
        matched = report['matched_edges'] == report['total_edges'] and report['unique_tiles']
    return {'status': found.get('status'), 'complete': bool(found.get('complete')), 'nodes': found.get('nodes'),
            'seconds': round(elapsed, 2), 'all_seams_matched': matched,
            'codes': [int(code) for code in found['codes']] if found.get('codes') else None}


def anneal(data, board, reference, seed, moves):
    model = AnnealModel(data, board)
    codes, state, best = model.random_board(random.Random(seed)), 1234567 + seed, None
    started = time.perf_counter()
    for step in range(COOLING_STEPS):
        temperature = cooling_temperature(step, COOLING_STEPS, DEFAULT_T_HOT, DEFAULT_T_COLD)
        codes, state, _, best_codes, best_score, _ = model.run(codes, state, max(1, moves // COOLING_STEPS), temperature, best=best)
        best = (best_codes, best_score)
        if best_score == model.total_edges:
            break
    report = model.validate(best[0])
    return {'status': 'edge_perfect' if best[1] == model.total_edges else 'score', 'matched_seams': best[1],
            'line_components': report['line_components'], **agreement(SearchProblem(data, board), best[0], reference),
            'moves': moves, 'kernel': model.backend, 'seconds': round(time.perf_counter() - started, 2)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--data', type=Path, default=Path('data/tiles.json'))
    parser.add_argument('--near-miss', type=Path, help='JSON file with the codes of a board on the real tiles')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--keep', type=int, nargs='+', default=[8])
    parser.add_argument('--engines', nargs='+', choices=ENGINES, default=['dfs', 'sat'])
    parser.add_argument('--seconds', type=float, default=60)
    parser.add_argument('--moves', type=float, default=1.5e9, help='annealing moves for the anneal engine')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    source = json.loads(args.data.read_text(encoding='utf-8-sig'))
    near_miss = json.loads(args.near_miss.read_text(encoding='utf-8'))['codes'] if args.near_miss else None
    rows = []
    if near_miss is not None and 'anneal' in args.engines:
        parser.error('anneal runs on planted boards only')
    for keep in args.keep:
        for seed in args.seeds:
            for engine in args.engines:
                if engine == 'anneal':
                    data, board, reference = planted_inventory(source, seed)
                    rows.append({'kind': 'planted', 'seed': seed, 'engine': engine, **anneal(data, board, reference, seed, int(args.moves))})
                    print(json.dumps(rows[-1]), flush=True)
                    continue
                problem, reference = instance(source, seed, keep, near_miss)
                row = {'kind': 'near_miss' if near_miss else 'planted', 'seed': seed, 'keep': keep, 'engine': engine,
                       **run(problem, engine, args.seconds)}
                codes = row.pop('codes')
                if reference is not None:
                    if codes:
                        row.update(agreement(problem, codes, reference))
                    # A planted board has a solution, so a refutation here is an engine fault.
                    if row['status'] == 'infeasible':
                        raise AssertionError(f'{engine} refuted a solvable board: {row}')
                rows.append(row)
                print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'data': str(args.data), 'seconds_limit': args.seconds, 'rows': rows}, indent=1) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
