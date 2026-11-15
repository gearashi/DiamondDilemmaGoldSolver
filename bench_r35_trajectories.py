"""Compare held-out bonded fresh-board trajectories using isolated deterministic prefixes."""
import argparse
import hashlib
from pathlib import Path
from unittest.mock import patch
from time import perf_counter

from io_utils import atomic_json
from r28_exchange import read_json
from r31_fresh import Engine, physical_bonds
from search_problem import SearchProblem


def digest(codes):
    return hashlib.sha256(b''.join(int(code).to_bytes(2, 'little') for code in codes)).hexdigest()


class RawSingles(Engine):
    """The v1.15.0 oracle charges full rotation products and packs only singles."""
    def repair(self,board,rng,step,stats,prices):
        from r34_pair_cycles import pair_repair

        def individual(model,board,cells,length,**kwargs):
            return pair_repair(model,board,cells,length,interact=False)

        with patch('r34_pair_cycles.pair_repair',individual):
            return super().repair(board,rng,step,stats,prices)


class EarlyContacts(Engine):
    """Held-out ablation: apply one local cycle repair before full-board matching."""
    def repair(self, board, rng, step, stats, prices):
        import r31_fresh
        from r34_pair_cycles import pair_repair
        full = r31_fresh.simultaneous_swaps
        choose = r31_fresh.forest_pool
        pool = None

        def early(model, board, allowed, price, **kwargs):
            nonlocal pool
            pool = choose(model, board, allowed, self.pool, rng)
            board,work = pair_repair(model,board,pool,self.pair_cycles,contact_group=self.contact_group)
            stats.update(work)
            return full(model,board,allowed,price,**kwargs)

        with patch('r31_fresh.simultaneous_swaps',early), patch('r31_fresh.forest_pool',lambda *args:pool), \
             patch('r34_pair_cycles.pair_repair',lambda model,board,*args,**kwargs:(board,{})):
            return super().repair(board,rng,step,stats,prices)


def engine_for(problem,policy):
    forest=policy.startswith(('forest-', 'early-'))
    engine=EarlyContacts if policy.startswith('early-') else Engine
    if policy=='singles':
        engine=RawSingles
    return engine(problem,bonds=True,joint=True,cycle_budget=3,pool=8 if forest else 10,forest=forest,
                  pair_cycles=0 if policy=='baseline' else 3,
                  contact_group={'singles':1,'positive':2,'triples':3}.get(policy.rsplit('-',1)[-1],0))


def trajectory(problem, folder, seed, policy, rounds, *, repeat=False):
    engine = engine_for(problem,policy)
    prefix = folder / f'{policy}-{seed}'
    result_file = prefix.with_name(prefix.name + '-results.json')
    cursor_file = prefix.with_name(prefix.name + '-cursor.json')
    snapshots = read_json(result_file) if result_file.exists() else {}
    saved = read_json(cursor_file) if cursor_file.exists() else None
    if saved and (saved['input'] != problem.fingerprint or saved['policy'] != policy or saved['seed'] != seed):
        raise ValueError('Foreign experiment cursor.')
    start_hash = saved['start'] if saved else None

    def save(state):
        nonlocal start_hash
        if state['cursor'] == 0:
            start_hash = digest(state['states'][0]['current']['codes'])
        atomic_json(cursor_file, dict(input=problem.fingerprint, policy=policy, seed=seed,
                                      start=start_hash, state=state))

    def receipt(row):
        codes = engine.model.legal_board(row['codes'])
        check = problem.validate(codes)
        bonds = physical_bonds(engine.model, codes)
        assert (check['matched_edges'], check['line_components'], bool(check['valid'])) == (
            row['score'], row['components'], row['gold'])
        assert engine.assemblies.bonds <= bonds and not engine.assemblies.forbidden & bonds
        return {**row, 'input': problem.fingerprint, 'start_hash': start_hash, 'final_hash': digest(codes)}

    for horizon in rounds:
        if str(horizon) in snapshots:
            continue
        state = saved['state'] if saved else None
        if state and state['cursor'] > horizon:
            raise ValueError('Earlier horizon missing from receipts.')
        start = perf_counter()
        row = receipt(engine.run(seed, rounds=horizon, saved=state, save=save))
        row['elapsed_seconds'] = perf_counter() - start
        row['resumed_round'] = state['cursor'] if state else 0
        snapshots[str(horizon)] = row
        atomic_json(result_file, snapshots)
        saved = read_json(cursor_file)
        print(f'{policy} seed {seed} round {horizon}: score {row["score"]}, components {row["components"]}', flush=True)

    if repeat and 'cold64' not in snapshots:
        reference = snapshots['64']
        cold = engine_for(problem,policy)
        raw = cold.run(seed, rounds=64)
        for key in ('codes', 'components', 'initial', 'score', 'gold', 'work'):
            if reference[key] != raw[key]:
                raise AssertionError(f'{policy} seed {seed}: cold 64 differs at {key}.')
        snapshots['cold64'] = dict(score=raw['score'], work_equal=True, codes_equal=True)
        atomic_json(result_file, snapshots)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    parser.add_argument('--policy', choices=('baseline','pairs','singles','factored-singles','forest-release','forest-positive','forest-triples',
                                                  'early-release','early-positive','early-triples'), required=True)
    parser.add_argument('--rounds', type=int, nargs='+', default=[16, 64, 320])
    parser.add_argument('--cold-seed', type=int, default=3500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    problem = SearchProblem(read_json(args.tiles))
    for horizon in args.rounds:
        for seed in args.seeds:
            trajectory(problem, args.output, seed, args.policy, [horizon],
                       repeat=horizon == 64 and seed == args.cold_seed)


if __name__ == '__main__':
    main()
