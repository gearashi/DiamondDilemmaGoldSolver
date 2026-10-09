"""Gold-only search runner with isolated per-algorithm state and live controls.

DFS saves exact pending branches. Native SAT/CP engines retain reported Gold
cuts/hints, but restart their internal search after a process restart.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import queue
import signal
import threading
import time

from io_utils import atomic_json, read_json_shared
from search_algorithms import ALGORITHMS, RECOMMENDED, LABELS
from search_problem import SearchProblem

ROOT = Path(__file__).resolve().parent
EVENT_BATCH = 32
EVENT_CAPACITY = 128


def terminal_digest(record):
    payload = {key: value for key, value in record.items() if key != 'sha256'}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def run_solver(problem, algorithm, **kwargs):
    if algorithm == 'dfs':
        from constraint_dfs import solve
    elif algorithm == 'hybrid':
        from gpu_sampling import solve
    else:
        from constraint_models import solve
        kwargs['engine'] = algorithm
    if algorithm != 'hybrid':
        kwargs.pop('replicas', None)
        kwargs.pop('backend', None)
    return solve(problem, **kwargs)


def drain_events(events, consume, *, limit=EVENT_BATCH, poll_stop=None):
    """Bound each UI batch so a busy producer cannot starve Stop polling."""
    consumed = 0
    for _ in range(limit):
        if poll_stop is not None:
            poll_stop()
        try:
            event = events.get_nowait()
        except queue.Empty:
            break
        consume(event)
        consumed += 1
    return consumed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--algorithm', choices=ALGORITHMS[1:], default=RECOMMENDED)
    parser.add_argument('--seconds', type=float, default=3600, help='0 means unlimited.')
    parser.add_argument('--seed', type=int, default=20261007)
    parser.add_argument('--replicas', type=int, default=4096, help='GPU sampling batch size, hybrid only.')
    parser.add_argument('--backend', choices=('auto', 'cuda', 'webgpu'), default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-file-initialized', action='store_true')
    parser.add_argument('--data', type=Path, default=ROOT/'data/tiles.json')
    parser.add_argument('--output', type=Path, default=ROOT/'runtime')
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds < 0 or not 128 <= args.replicas <= 131072 or not 0 <= args.seed < 2**32:
        parser.error('Use finite nonnegative seconds, 128..131072 replicas, and a 32-bit seed.')
    run = args.output.resolve()
    run.mkdir(parents=True, exist_ok=True)
    lock = run/'run.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit('A solver lock exists; wait for the active solver to stop.')

    started = time.monotonic()
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = run/'searches'/(args.algorithm+'-gold')
    checkpoint_path, live_path = folder/'checkpoint.json', run/'live.json'
    evidence = run/'runs'/run_id
    stopfile = run/'stop.request'
    cancel = threading.Event()
    events, result_queue = queue.Queue(maxsize=EVENT_CAPACITY), queue.Queue(maxsize=1)
    previous_handlers = {}
    thread = None
    binding = checkpoint = error = last_snapshot = None
    state = {
        'state': 'starting', 'phase': 'loading_input', 'phase_message': 'Loading puzzle data',
        'method': 'constraint', 'algorithm': args.algorithm, 'target': 'gold', 'pid': os.getpid(),
        'run_id': run_id, 'replicas': args.replicas if args.algorithm == 'hybrid' else 0,
        'backend': args.backend if args.algorithm == 'hybrid' else 'cpu',
        'gpu': LABELS[args.algorithm]+' · '+('GPU + CPU' if args.algorithm == 'hybrid' else 'CPU'),
        'time_limit_seconds': args.seconds, 'nodes_checked': 0, 'stats': {}, 'checkpoint': {},
        'resume_scope': ('Exact DFS branch stack' if args.algorithm in ('dfs', 'hybrid')
                         else 'Saved Gold cuts/hints; native internal search restarts'),
    }

    def publish():
        state.update(elapsed_seconds=round(time.monotonic()-started, 3),
                     updated_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        atomic_json(run/'status.json', state)

    def poll_stop():
        if stopfile.exists():
            cancel.set()
        if cancel.is_set():
            state.update(state='stopping', phase_message='Stopping and saving search progress')

    def save_checkpoint(payload):
        if payload is None:
            return
        envelope = {'version': 1, 'algorithm': args.algorithm, 'target': 'gold', 'binding': binding, 'payload': payload}
        atomic_json(checkpoint_path, envelope)
        state['checkpoint'] = {'phase': 'idle', 'saved_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                               'file_bytes': checkpoint_path.stat().st_size}

    def progress(event):
        # Backpressure bounds memory; Stop releases a producer blocked on UI I/O.
        # The final result independently carries its latest complete checkpoint.
        while not cancel.is_set():
            try:
                events.put(event, timeout=.05)
                return
            except queue.Full:
                continue

    def background():
        nonlocal checkpoint
        try:
            while True:
                remaining = 1e9 if args.seconds == 0 else max(0, args.seconds-(time.monotonic()-started))
                if remaining <= 0:
                    result_queue.put({'status': 'timeout', 'checkpoint': checkpoint,
                                      'nodes': state['nodes_checked'], 'stats': state['stats'], 'complete': False})
                    return
                budget = min(30, remaining) if args.algorithm in ('dfs', 'hybrid') else remaining
                result = run_solver(problem, args.algorithm, seconds=budget, seed=args.seed, target='gold', resume=checkpoint,
                                    stop=cancel.is_set, progress=progress, replicas=args.replicas, backend=args.backend)
                checkpoint = result.get('checkpoint', checkpoint)
                progress({'event': 'checkpoint', 'payload': checkpoint, 'nodes': result.get('nodes', 0),
                          'stats': result.get('stats', {}), 'gpu_sampling': result.get('gpu_sampling')})
                if (result['status'] != 'timeout' or args.algorithm not in ('dfs', 'hybrid') or cancel.is_set()
                        or (args.seconds and time.monotonic()-started >= args.seconds)):
                    result_queue.put(result)
                    return
        except Exception as exc:
            result_queue.put({'status': 'error', 'error': f'{type(exc).__name__}: {exc}'})

    def write_live(codes, *, kind='search', sample_metrics=None):
        nonlocal last_snapshot
        codes = [int(code) for code in codes]
        if len(codes) != problem.n or any(code < -1 or code >= 3*problem.n for code in codes):
            raise ValueError('Live search snapshot has invalid orientation codes.')
        assigned = sum(code >= 0 for code in codes)
        record = {'method': 'constraint', 'algorithm': args.algorithm, 'target': 'gold', 'run_id': run_id,
                  'codes': codes, 'is_partial': assigned < problem.n, 'assigned_tiles': assigned,
                  'snapshot_kind': kind, 'updated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'input_sha256': input_sha, 'input_snapshot': str(evidence/'tiles.json'),
                  'validation': problem.validate(codes) if assigned == problem.n else None}
        if sample_metrics is not None:
            record['sample_metrics'] = sample_metrics
        atomic_json(live_path, record)
        last_snapshot = record
        if kind == 'search':
            state['max_depth'] = max(state.get('max_depth', 0), assigned)
        if record['validation'] and record['validation']['matched_edges'] == len(problem.edges) and not record['validation']['valid']:
            state['rejected_multiloop'] = state.get('rejected_multiloop', 0)+1

    def sampling(report, *, show_sample=False):
        if report is None:
            return
        state['gpu_sampling'] = {key: value for key, value in report.items() if key != 'event'}
        if report.get('backend') in ('cuda', 'webgpu'):
            state['backend'] = report['backend']
        if report.get('device'):
            state['gpu'] = report['device']+' · GPU sampling + CPU DFS'
        if show_sample and report.get('top_board') is not None:
            write_live(report['top_board'], kind='gpu_sample', sample_metrics={
                key: report.get(key) for key in ('best_score', 'best_objective', 'domain_violations', 'samples_considered')})

    def consume(event):
        state['nodes_checked'] = event.get('nodes', state['nodes_checked'])
        state['stats'] = event.get('stats', state['stats'])
        phase = event.get('phase', event.get('event'))
        if phase and phase != 'checkpoint':
            state['phase'] = phase
            state['phase_message'] = {
                'building_model': 'Building the constraint model', 'model_ready': 'Constraint model ready',
                'searching': 'Searching for one closed Gold loop', 'gold_cut': 'Excluding a separate closed loop',
                'sampling': 'GPU sampling completed; preparing exact DFS', 'progress': 'Searching unfinished DFS branches',
                'candidate': 'Checking a complete candidate', 'finished': 'Finishing search',
            }.get(phase, phase.replace('_', ' ').capitalize())
        if event.get('event') == 'sampling':
            sampling(event, show_sample=True)
        elif event.get('gpu_sampling') is not None:
            sampling(event['gpu_sampling'], show_sample=last_snapshot is None)
        if event.get('event') == 'checkpoint':
            save_checkpoint(event.get('payload'))
            return
        if event.get('checkpoint') is not None:
            save_checkpoint(event['checkpoint'])
        if event.get('codes') is not None:
            write_live(event['codes'])

    def save_solution(codes, *, source_run_id=None):
        report = problem.validate(codes)
        if (not report['valid'] or len(codes) != problem.n
                or any(code not in problem.domains[cell] for cell, code in enumerate(codes))):
            raise RuntimeError('Search result failed independent single-loop Gold validation.')
        record = {'method': 'constraint', 'algorithm': args.algorithm, 'target': 'gold', 'run_id': run_id,
                  'codes': codes, 'validation': report, 'input_sha256': input_sha,
                  'input_snapshot': str(evidence/'tiles.json'), 'elapsed_seconds': time.monotonic()-started,
                  'solved': True, 'is_partial': False, 'assigned_tiles': problem.n, 'snapshot_kind': 'solution'}
        if source_run_id is not None:
            record['source_solution_run_id'] = source_run_id
        for path in (folder/'solution.json', run/'solution.json', run/'best.json', live_path):
            atomic_json(path, record)
        from render_board import render
        render(run/'best.html', data['tiles'] if isinstance(data, dict) else data, codes, report)

    try:
        try:
            os.write(fd, str(os.getpid()).encode())
        finally:
            os.close(fd)
        folder.mkdir(parents=True, exist_ok=True)
        evidence.mkdir(parents=True, exist_ok=True)
        if not args.stop_file_initialized:
            stopfile.unlink(missing_ok=True)
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, lambda *_: cancel.set())
        publish()
        raw = args.data.read_bytes()
        data = json.loads(raw.decode('utf-8-sig'))
        problem = SearchProblem(data)
        input_sha = hashlib.sha256(raw).hexdigest()
        binding = hashlib.sha256(raw+args.algorithm.encode()+b':gold:v1').hexdigest()
        (evidence/'tiles.json').write_bytes(raw)
        source_hashes = {}
        for name in ('constraint_search.py', 'search_problem.py', 'constraint_dfs.py', 'constraint_models.py',
                     'gpu_sampling.py', 'sampling_kernels.wgsl', 'search_algorithms.py', 'validator.py', 'geometry.py'):
            path = ROOT/name
            if path.exists():
                source = path.read_bytes()
                source_hashes[name] = hashlib.sha256(source).hexdigest()
                (evidence/name).write_bytes(source)
        atomic_json(run/'run-config.json', {'pid': os.getpid(), 'method': 'constraint', 'algorithm': args.algorithm,
                    'backend': state['backend'], 'replicas': state['replicas'], 'seconds': args.seconds,
                    'seed': args.seed, 'run_id': run_id, 'input_sha256': input_sha, 'input_snapshot': str(evidence/'tiles.json')})
        if checkpoint_path.exists():
            if not args.resume:
                raise ValueError('Saved algorithm state exists; resume it or choose a separate output directory.')
            saved = read_json_shared(checkpoint_path)
            if (saved.get('version') != 1 or saved.get('binding') != binding
                    or saved.get('algorithm') != args.algorithm or saved.get('target') != 'gold'):
                raise ValueError('Checkpoint belongs to different data, algorithm, or target.')
            checkpoint = saved['payload']
        terminal_path = folder/'completed-search.json'
        if terminal_path.exists():
            if not args.resume:
                raise ValueError('Completed algorithm state exists; resume it or choose a separate output directory.')
            terminal = read_json_shared(terminal_path)
            if (not isinstance(terminal, dict) or terminal.get('version') != 1
                    or terminal.get('binding') != binding or terminal.get('algorithm') != args.algorithm
                    or terminal.get('target') != 'gold' or terminal.get('source_sha256') != source_hashes
                    or terminal.get('sha256') != terminal_digest(terminal)):
                raise ValueError('Completed search evidence has changed or belongs to different input, algorithm, or code.')
            recorded = terminal.get('result')
            if (not isinstance(recorded, dict) or recorded.get('status') != 'infeasible'
                    or recorded.get('complete') is not True or recorded.get('codes') is not None):
                raise ValueError('Malformed completed Gold search result.')
            state.update(state='exhausted', phase='finished', phase_message='Previously exhausted Gold search restored',
                         complete=True, nodes_checked=recorded.get('nodes', 0), stats=recorded.get('stats', {}),
                         resumed_complete=True, source_result_run_id=terminal.get('run_id'),
                         terminal_evidence=str(terminal_path))
            sampling(recorded.get('gpu_sampling'), show_sample=True)
            return
        solution_path = folder/'solution.json'
        if solution_path.exists():
            previous = read_json_shared(solution_path)
            codes = previous.get('codes')
            if problem.validate(codes)['valid']:
                save_solution(codes, source_run_id=previous.get('run_id'))
                state.update(state='solved', phase='finished', phase_message='Gold solution verified', complete=True)
                return
        poll_stop()
        if not cancel.is_set():
            state.update(state='running', phase='searching', phase_message='Searching for one closed Gold loop')
        publish()
        thread = threading.Thread(target=background, name='gold-search', daemon=True)
        thread.start()
        last_publish = 0
        while True:
            poll_stop()
            drain_events(events, consume, poll_stop=poll_stop)
            try:
                result = result_queue.get(timeout=.1)
                break
            except queue.Empty:
                pass
            if time.monotonic()-last_publish >= 1:
                publish()
                last_publish = time.monotonic()
        thread.join()
        while drain_events(events, consume, poll_stop=poll_stop):
            pass
        state['nodes_checked'] = result.get('nodes', state['nodes_checked'])
        state['stats'] = result.get('stats', state['stats'])
        sampling(result.get('gpu_sampling'), show_sample=last_snapshot is None)
        save_checkpoint(result.get('checkpoint'))
        status = result['status']
        if status == 'solved':
            save_solution(result.get('codes'))
        elif status == 'error':
            error = result.get('error', 'Unknown search failure')
        elif status not in ('timeout', 'infeasible', 'stopped'):
            raise RuntimeError(f'Unexpected Gold search status: {status!r}')
        state['state'] = {'timeout': 'time_limit', 'infeasible': 'exhausted', 'stopped': 'stopped',
                          'solved': 'solved', 'error': 'error'}[status]
        if cancel.is_set() and status == 'timeout':
            state['state'] = 'stopped'
        state['complete'] = bool(result.get('complete', False))
        state['phase'] = 'finished'
        state['phase_message'] = {'time_limit': 'Time limit reached; progress saved', 'stopped': 'Stopped; progress saved',
                                  'solved': 'Gold solution verified', 'exhausted': 'Gold search exhausted',
                                  'error': 'Search failed'}[state['state']]
        atomic_json(folder/'last-result.json', result)
        if status == 'infeasible':
            if result.get('complete') is not True:
                raise RuntimeError('An infeasible result must represent completed search.')
            terminal = {'version': 1, 'binding': binding, 'algorithm': args.algorithm, 'target': 'gold',
                        'run_id': run_id, 'source_sha256': source_hashes, 'result': result}
            terminal['sha256'] = terminal_digest(terminal)
            atomic_json(terminal_path, terminal)
            state['terminal_evidence'] = str(terminal_path)
        if error:
            state['error'] = error
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        state.update(state='error', phase='failed', phase_message='Search failed', error=error)
    finally:
        # Even broken status storage or setup must release this process's lock.
        # Stop/join first: never advertise an unlocked runtime with live search.
        cancel.set()
        if thread is not None and thread.is_alive():
            thread.join()
        try:
            publish()
            if evidence.is_dir():
                atomic_json(evidence/'final-status.json', state)
        except Exception as exc:
            if error is None:
                error = f'{type(exc).__name__}: {exc}'
        finally:
            try:
                if lock.exists() and lock.read_text() in ('', str(os.getpid())):
                    lock.unlink()
            finally:
                for sig, handler in previous_handlers.items():
                    signal.signal(sig, handler)
        if error:
            raise SystemExit(error)


if __name__ == '__main__':
    main()
