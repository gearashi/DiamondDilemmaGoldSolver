"""Targeted annealing runner: maximizes matched seams, with live status and exact resume.

Each worker thread cools one board from a random start. When a cooling run
ends, the worker starts the next seed. Chain k of W runs seed indices
k, k+W, k+2W, ..., and every chain owns its random stream, so a given
(seed, workers, moves, schedule, backend) reproduces the same boards whatever
the timing, and a checkpoint resumes each chain at the exact chunk it reached.

This is a score search, not a proof. A reported board counts only after
validator.py rescored it. Reaching 240 matched seams stops the run and saves
the board; it is Gold only if the validator also finds one closed loop.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import random
import signal
import sys
import threading
import time
from pathlib import Path

from anneal_core import (DEFAULT_T_COLD, DEFAULT_T_HOT, DEFAULT_TARGETED, AnnealModel,
                         completion_check, cooling_temperature)
from io_utils import atomic_json, read_json_shared

ROOT = Path(__file__).resolve().parent
CHECKPOINT_VERSION = 1
DEFAULT_MOVES = 1_500_000_000
COOLING_STEPS = 600


def chain_start(seed, index):
    """Deterministic start for one cooling run: a 64-bit stream state and a board seed."""
    digest = hashlib.sha256(f'diamond-anneal:{seed}:{index}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little') or 1, int.from_bytes(digest[8:16], 'little')


def validate_settings(seconds, seed, workers, moves, hot, cold, targeted):
    if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('Seconds must be finite and nonnegative; 0 means unlimited.')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('Seed must be a nonnegative 32-bit integer.')
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('Choose 1 to 64 annealing workers.')
    if type(moves) is not int or not COOLING_STEPS <= moves <= 10**13:
        raise ValueError(f'Moves per cooling run must be between {COOLING_STEPS} and 10^13.')
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (hot, cold)) or not 0 < cold <= hot <= 100:
        raise ValueError('Temperatures must satisfy 0 < cold <= hot <= 100.')
    if type(targeted) is not int or not 0 <= targeted <= 1024:
        raise ValueError('Targeted share must be an integer from 0 to 1024.')


class Chain:
    def __init__(self, model, seed, index):
        self.index = index
        self.rng_state, board_seed = chain_start(seed, index)
        self.codes = model.random_board(random.Random(board_seed))
        self.step = 0
        self.best_codes, self.best_score = list(self.codes), -1
        self.score = model.score(self.codes)

    def to_json(self):
        return {'index': self.index, 'step': self.step, 'rng_state': str(self.rng_state), 'codes': self.codes,
                'best_codes': self.best_codes, 'best_score': self.best_score}

    @classmethod
    def from_json(cls, model, payload):
        chain = cls.__new__(cls)
        chain.index, chain.step = payload['index'], payload['step']
        chain.rng_state = int(payload['rng_state'])
        chain.codes, chain.best_codes, chain.best_score = payload['codes'], payload['best_codes'], payload['best_score']
        if type(chain.index) is not int or chain.index < 0 or type(chain.step) is not int or not 0 <= chain.step < COOLING_STEPS:
            raise ValueError('Checkpoint chain position is out of range.')
        model.check_codes(chain.codes)
        model.check_codes(chain.best_codes)
        chain.score = model.score(chain.codes)
        if chain.best_score != -1 and model.score(chain.best_codes) != chain.best_score:
            raise ValueError('Checkpoint best board does not reproduce its score.')
        return chain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=3600, help='0 means unlimited.')
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--workers', type=int, default=4, help='Independent cooling runs in parallel, 1..64.')
    parser.add_argument('--moves', type=int, default=DEFAULT_MOVES, help='Moves in one cooling run.')
    parser.add_argument('--t-hot', type=float, default=DEFAULT_T_HOT)
    parser.add_argument('--t-cold', type=float, default=DEFAULT_T_COLD)
    parser.add_argument('--targeted', type=int, default=DEFAULT_TARGETED, help='Targeted swaps per 1024 swap moves.')
    parser.add_argument('--check-floor', type=int, default=200, help='Run the exact completion check on new best boards at or above this score.')
    parser.add_argument('--native', choices=('auto', 'compiled', 'python'), default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-file-initialized', action='store_true')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/tiles.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runtime')
    args = parser.parse_args(argv)
    try:
        validate_settings(args.seconds, args.seed, args.workers, args.moves, args.t_hot, args.t_cold, args.targeted)
    except ValueError as exc:
        parser.error(str(exc))

    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = run / 'run.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit('A solver lock exists; wait for the active solver to stop.')
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)

    started = time.monotonic()
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = run / 'searches' / 'anneal-edges'
    checkpoint_path = folder / 'checkpoint.json'
    evidence = run / 'runs' / run_id
    stopfile = run / 'stop.request'
    stop = threading.Event()
    previous_handlers = {}
    status = {'state': 'starting', 'pid': os.getpid(), 'run_id': run_id, 'method': 'anneal', 'algorithm': 'anneal',
              'target': 'edges', 'backend': 'cpu', 'workers': args.workers, 'replicas': args.workers,
              'seed': args.seed, 'time_limit_seconds': args.seconds, 'required_matched_edges': 240,
              'startup_phase': 'loading_input', 'phase_message': 'Loading puzzle data',
              'gpu': 'Targeted annealing | CPU', 'checkpoint': {}}

    def now():
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def publish():
        status.update(elapsed_seconds=round(time.monotonic() - started, 2), updated_at=now())
        atomic_json(run / 'status.json', status)

    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, lambda *_: stop.set())
        if not args.stop_file_initialized:
            stopfile.unlink(missing_ok=True)
        evidence.mkdir(parents=True, exist_ok=True)
        publish()

        input_bytes = args.data.read_bytes()
        input_sha = hashlib.sha256(input_bytes).hexdigest()
        data = json.loads(input_bytes.decode('utf-8-sig'))
        (evidence / 'tiles.json').write_bytes(input_bytes)
        model = AnnealModel(data, native=args.native)
        chunk = max(1, args.moves // COOLING_STEPS)
        settings = {'version': CHECKPOINT_VERSION, 'input_sha256': input_sha, 'seed': args.seed, 'workers': args.workers,
                    'moves': args.moves, 't_hot': args.t_hot, 't_cold': args.t_cold, 'targeted': args.targeted,
                    'backend': model.backend}
        status.update(native_backend=model.backend, moves_per_run=args.moves,
                      gpu=f'Targeted annealing | CPU, {model.backend} kernel, {args.workers} chains')
        if model.native_error:
            status['native_fallback_reason'] = model.native_error

        # -- incumbent and checkpoint ------------------------------------
        best = {'score': -1, 'codes': None, 'record': None}
        history, runs_done, moves_done, results = [], 0, 0, []
        previous = run / 'best.json'
        if previous.exists():
            try:
                prior = read_json_shared(previous)
                report = model.validate(prior['codes'])
                if report['unique_tiles'] and report['placement_count'] == model.n:
                    best.update(score=report['matched_edges'], codes=list(prior['codes']), record=prior)
                    history = [p for p in read_json_shared(run / 'history.json') if isinstance(p, dict)] if (run / 'history.json').exists() else []
            except (OSError, ValueError, KeyError, TypeError):
                pass
        chains, resumed = None, False
        if args.resume and checkpoint_path.exists():
            status.update(startup_phase='restoring_checkpoint', phase_message='Restoring saved annealing chains')
            publish()
            try:
                saved = read_json_shared(checkpoint_path)
                if saved.get('settings') == settings:
                    chains = [Chain.from_json(model, item) for item in saved['chains']]
                    if len(chains) != args.workers or sorted(c.index % args.workers for c in chains) != list(range(args.workers)):
                        raise ValueError('Checkpoint chains do not match the worker count.')
                    runs_done, moves_done = int(saved['runs_done']), int(saved['moves_done'])
                    results = [r for r in saved.get('results', []) if isinstance(r, dict)][-200:]
                    resumed = True
                else:
                    status['checkpoint_note'] = 'Saved annealing chains used different settings; started fresh chains and kept the best board.'
            except (OSError, ValueError, KeyError, TypeError) as exc:
                chains = None
                status['checkpoint_recovery'] = f'Ignored an unreadable checkpoint: {exc}'
        if chains is None:
            chains = [Chain(model, args.seed, k) for k in range(args.workers)]
        config = {'pid': os.getpid(), 'method': 'anneal', 'algorithm': 'anneal', 'target': 'edges', 'backend': 'cpu',
                  'replicas': args.workers, 'workers': args.workers, 'seed': args.seed, 'seconds': args.seconds,
                  'started_at': now(), 'run_id': run_id, 'resumed': resumed, 'settings': settings, 'command': sys.argv}
        atomic_json(run / 'run-config.json', config)
        atomic_json(evidence / 'run-config.json', config)

        state_lock = threading.Lock()
        pending_best = []            # (score, codes) candidates from workers, validated on the main thread
        errors = []

        def worker(slot):
            chain = chains[slot]
            try:
                while not stop.is_set():
                    temperature = cooling_temperature(chain.step, COOLING_STEPS, args.t_hot, args.t_cold)
                    codes, rng_state, score, best_codes, best_score, _ = model.run(
                        chain.codes, chain.rng_state, chunk, temperature, targeted=args.targeted,
                        best=(chain.best_codes, chain.best_score))
                    nonlocal moves_done, runs_done
                    with state_lock:
                        improved = best_score > chain.best_score
                        chain.codes, chain.rng_state, chain.score = codes, rng_state, score
                        chain.best_codes, chain.best_score = best_codes, best_score
                        chain.step += 1
                        moves_done += chunk
                        if improved and best_score > best['score']:
                            pending_best.append((best_score, list(best_codes)))
                        if chain.step >= COOLING_STEPS:
                            runs_done += 1
                            results.append({'seed_index': chain.index, 'best_matched_edges': chain.best_score})
                            del results[:-200]
                            chain = chains[slot] = Chain(model, args.seed, chain.index + args.workers)
            except BaseException as exc:          # surface kernel failures on the main thread
                errors.append(exc)
                stop.set()

        def write_checkpoint():
            status['checkpoint'] = {'phase': 'saving'}
            with state_lock:
                payload = {'settings': settings, 'runs_done': runs_done, 'moves_done': moves_done,
                           'results': list(results), 'saved_at': now(), 'run_id': run_id,
                           'chains': [chain.to_json() for chain in chains]}
            atomic_json(checkpoint_path, payload)
            status['checkpoint'] = {'phase': 'idle', 'saved_at': payload['saved_at'], 'runs_done': payload['runs_done']}

        def accept_best(score, codes):
            report = model.validate(codes)
            if report['matched_edges'] != score or not report['unique_tiles']:
                raise RuntimeError('Annealing score disagrees with the independent validator.')
            elapsed = round(time.monotonic() - started, 2)
            # The exact check is only informative near the top: a low-scoring
            # board frees most of the board and the search cannot finish.
            wanted = score >= args.check_floor and score < model.total_edges
            check = {'status': 'queued'} if wanted else {'status': 'not_run', 'reason': f'below score {args.check_floor}'} if score < model.total_edges else {'status': 'not_needed'}
            record = {'solved': bool(report['valid']), 'codes': list(codes), 'validation': report,
                      'elapsed_seconds': elapsed, 'seed': args.seed, 'run_id': run_id, 'method': 'anneal',
                      'algorithm': 'anneal', 'input_sha256': input_sha, 'gpu': status['gpu'],
                      'completion_check': check, 'updated_at': now()}
            best.update(score=score, codes=list(codes), record=record)
            atomic_json(run / 'best.json', record)
            atomic_json(evidence / 'best.json', record)
            history.append({'elapsed_seconds': elapsed, 'score': score, 'components': report['line_components']})
            atomic_json(run / 'history.json', history)
            if report['matched_edges'] == model.total_edges:
                atomic_json(run / 'edge-perfect.json', record)
            if report['valid']:
                atomic_json(run / 'solution.json', record)
            print(f'Best {score}/{model.total_edges} matched seams, {report["line_components"]} components', flush=True)
            if wanted:
                with check_wake:
                    check_job[:] = [list(codes)]
                    check_wake.notify()
            return report

        check_job, check_wake = [], threading.Condition()

        def checker():
            # Runs beside the search so a long exact check never delays Stop or the live view.
            while True:
                with check_wake:
                    while not check_job and not stop.is_set():
                        check_wake.wait(0.5)
                    if stop.is_set():
                        return
                    codes = check_job.pop()
                outcome = {}
                for rings in (0, 1):
                    found = completion_check(model, codes, rings=rings, node_limit=200000, seconds=20,
                                             cancelled=lambda: stop.is_set() or bool(check_job))
                    outcome[f'rings_{rings}'] = {k: v for k, v in found.items() if k not in ('codes', 'validation')}
                    if found['status'] == 'completion_found':
                        with state_lock:      # a full board: hand it to the main thread like any candidate
                            pending_best.append((found['validation']['matched_edges'], found['codes']))
                        break
                    if found['status'] == 'undetermined':
                        break
                with state_lock:
                    record = best['record']
                    if record is not None and record['codes'] == codes:
                        record['completion_check'] = outcome
                        atomic_json(run / 'best.json', record)
                        atomic_json(evidence / 'best.json', record)

        threads = [threading.Thread(target=worker, args=(slot,), daemon=True) for slot in range(args.workers)]
        threads.append(threading.Thread(target=checker, daemon=True))
        status.update(state='running', resumed=resumed)
        status.pop('startup_phase', None)
        status.pop('phase_message', None)
        for thread in threads:
            thread.start()
        final_state, last_checkpoint, last_live, rate_mark = None, time.monotonic(), 0.0, (time.monotonic(), moves_done)
        while final_state is None:
            stop.wait(0.25)
            if errors:
                raise errors[0]
            with state_lock:
                candidates, pending_best[:] = sorted(pending_best), []
                live_codes, live_step, live_index = list(chains[0].codes), chains[0].step, chains[0].index
                snapshot_moves, snapshot_runs = moves_done, runs_done
            for score, codes in candidates[-1:]:      # only the highest candidate of this tick
                if score > best['score']:
                    report = accept_best(score, codes)
                    if report['valid']:
                        final_state = 'solved'
                    elif report['matched_edges'] == model.total_edges:
                        final_state = 'edge_perfect'
            clock = time.monotonic()
            if clock - last_live >= 0.5:
                atomic_json(run / 'live.json', {'codes': live_codes, 'validation': model.validate(live_codes),
                                                 'replica': 0, 'method': 'anneal', 'algorithm': 'anneal',
                                                 'seed_index': live_index, 'cooling_step': live_step,
                                                 'elapsed_seconds': round(clock - started, 2), 'run_id': run_id,
                                                 'updated_at': now()})
                last_live = clock
                if clock - rate_mark[0] >= 2:
                    status['moves_per_second'] = round((snapshot_moves - rate_mark[1]) / (clock - rate_mark[0]))
                    rate_mark = (clock, snapshot_moves)
                status.update(best_matched_edges=best['score'] if best['score'] >= 0 else None,
                              best_components=(best['record'] or {}).get('validation', {}).get('line_components'),
                              candidates_checked=snapshot_runs, cooling_runs_completed=snapshot_runs,
                              moves_done=snapshot_moves, recent_run_scores=[r['best_matched_edges'] for r in results[-12:]])
                publish()
            if final_state is None and (stop.is_set() or stopfile.exists()):
                final_state = 'stopped'
            if final_state is None and args.seconds and clock - started >= args.seconds:
                final_state = 'time_limit'
            if final_state is None and clock - last_checkpoint >= 30:
                write_checkpoint()
                last_checkpoint = clock
        status['state'] = 'stopping'
        publish()
        stop.set()
        for thread in threads:
            thread.join()
        if errors:
            raise errors[0]
        for score, codes in sorted(pending_best)[-1:]:
            if score > best['score']:
                report = accept_best(score, codes)
                if report['valid']:
                    final_state = 'solved'
                elif report['matched_edges'] == model.total_edges and final_state != 'solved':
                    final_state = 'edge_perfect'
        write_checkpoint()
        status.update(state=final_state, finished_at=now(), best_matched_edges=best['score'] if best['score'] >= 0 else None,
                      best_components=(best['record'] or {}).get('validation', {}).get('line_components'),
                      candidates_checked=runs_done, cooling_runs_completed=runs_done, moves_done=moves_done,
                      coverage_note='Annealing is a score search. It proves nothing about boards it did not visit.')
        publish()
        atomic_json(evidence / 'status.json', status)
        print(f'Finished: {final_state}; best {best["score"]}/{model.total_edges}. Saved {run}', flush=True)
        return 0
    except BaseException as exc:
        stop.set()
        status.update(state='error', error=f'{type(exc).__name__}: {exc}')
        try:
            publish()
        except OSError:
            pass
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
