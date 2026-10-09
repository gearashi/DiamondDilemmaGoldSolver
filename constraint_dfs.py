"""Exact CPU DFS with MRV, seam arc consistency, tile propagation, and loop cuts.

A checkpoint stores every unfinished sibling choice and each active domain
state. Resume may repeat propagation interrupted mid-pass, but never a completed
leaf. Timeouts and requested stops are explicitly incomplete search results.
The integrity digest detects accidental corruption; it is not a proof against
modified checkpoints whose digest has also been recomputed.
"""
from __future__ import annotations
from collections import deque
import hashlib
import json
import math
import random
import time
from validator import reverse_mask

VERSION = 2


def _checkpoint_digest(payload):
    body={key:value for key,value in payload.items() if key!='digest'}
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def _values(bits):
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def solve(problem, *, seconds=10, seed=0, target='gold', hints=None, stop=None, progress=None, resume=None):
    began = time.monotonic()
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative; zero gives an immediate timeout.')
    if target not in ('gold', 'edges'):
        raise ValueError('target must be gold or edges.')
    if type(seed) is not int:
        raise ValueError('seed must be an integer.')
    deadline = began + seconds
    n, count = problem.n, 3*problem.n
    initial = [sum(1 << code for code in domain) for domain in problem.domains]
    stats = {'propagations': 0, 'arc_revisions': 0, 'tile_prunes': 0, 'loop_prunes': 0, 'leaves': 0, 'backtracks': 0}
    nodes, previous_elapsed = 0, 0.0
    if resume is not None:
        if not isinstance(resume, dict) or resume.get('version') != VERSION or resume.get('fingerprint') != problem.fingerprint or resume.get('target') != target:
            raise ValueError('Checkpoint version, puzzle, fixed placements, or target differs.')
        if not isinstance(resume.get('digest'), str) or resume['digest'] != _checkpoint_digest(resume):
            raise ValueError('Checkpoint integrity digest differs; saved progress may be damaged.')
        if not {'seed', 'stats', 'nodes', 'stack'} <= resume.keys():
            raise ValueError('Checkpoint is missing search state.')
        seed = resume['seed']
        if type(seed) is not int or type(resume['nodes']) is not int or resume['nodes'] < 0:
            raise ValueError('Malformed checkpoint seed or node count.')
        if not isinstance(resume['stats'], dict) or set(resume['stats']) != set(stats) or any(type(value) is not int or value < 0 for value in resume['stats'].values()):
            raise ValueError('Malformed checkpoint counters.')
        hints = resume.get('hints')
        stats.update(resume['stats'])
        nodes = resume['nodes']
        previous_elapsed = resume.get('elapsed_seconds', 0)
        if isinstance(previous_elapsed, bool) or not isinstance(previous_elapsed, (int, float)) or not math.isfinite(previous_elapsed) or previous_elapsed < 0:
            raise ValueError('Malformed checkpoint elapsed time.')
        if not isinstance(resume['stack'], list) or not resume['stack']:
            raise ValueError('An incomplete checkpoint must retain an unfinished search stack.')
        stack = []
        for saved in resume['stack']:
            if not isinstance(saved, dict) or not {'domains', 'cell', 'choices', 'next'} <= saved.keys() or not isinstance(saved['domains'], list) or any(not isinstance(value, str) for value in saved['domains']):
                raise ValueError('Malformed checkpoint search frame.')
            domains = [int(value, 16) for value in saved['domains']]
            if len(domains) != n or any(value < 0 or value & ~initial[i] for i, value in enumerate(domains)):
                raise ValueError('Checkpoint contains invalid cell domains.')
            cell, choices, offset = saved['cell'], saved['choices'], saved['next']
            if type(cell) is not int or not -1 <= cell < n or not isinstance(choices, list) or type(offset) is not int or not 0 <= offset <= len(choices):
                raise ValueError('Malformed checkpoint search frame.')
            if cell == -1 and (choices or offset):
                raise ValueError('An unexpanded checkpoint frame has choices.')
            if any(type(code) is not int or not 0 <= code < count or cell < 0 or not domains[cell] & (1 << code) for code in choices) or len(set(choices)) != len(choices):
                raise ValueError('Malformed checkpoint sibling choices.')
            if cell >= 0 and (domains[cell].bit_count() <= 1 or set(choices) != set(_values(domains[cell]))):
                raise ValueError('Checkpoint does not retain all sibling choices.')
            if stack:
                parent = stack[-1]
                if parent['cell'] < 0 or parent['next'] == 0:
                    raise ValueError('Checkpoint child has no selected parent branch.')
                selected = parent['choices'][parent['next']-1]
                if domains[parent['cell']] != 1 << selected or any(value & ~old for value, old in zip(domains, parent['domains'])):
                    raise ValueError('Checkpoint child differs from its selected branch.')
            stack.append({'domains': domains, 'cell': cell, 'choices': choices.copy(), 'next': offset})
    else:
        stack = [{'domains': initial.copy(), 'cell': -1, 'choices': [], 'next': 0}]
    if hints is None:
        normalized_hints = {}
    elif isinstance(hints, dict):
        normalized_hints = {int(cell): ([codes] if type(codes) is int else list(codes)) for cell, codes in hints.items()}
    else:
        if len(hints) != n:
            raise ValueError('Hints must have one preferred orientation code per cell.')
        normalized_hints = {cell: [code] for cell, code in enumerate(hints)}
    for cell, codes in normalized_hints.items():
        if not 0 <= cell < n or any(type(code) is not int or not -1 <= code < count for code in codes):
            raise ValueError('Invalid hint cell or orientation code.')
    randomizer = random.Random(seed)
    priority = list(range(count)); randomizer.shuffle(priority)
    rank = {code: index for index, code in enumerate(priority)}
    cell_order = list(range(n)); randomizer.shuffle(cell_order)
    cell_rank = {cell: index for index, cell in enumerate(cell_order)}
    last_progress = began

    class Interrupted(Exception):
        def __init__(self, status): self.status = status

    def check():
        if stop is not None and stop():
            raise Interrupted('stopped')
        if time.monotonic() >= deadline:
            raise Interrupted('timeout')

    def snapshot():
        payload = {'version': VERSION, 'fingerprint': problem.fingerprint, 'target': target,
                'seed': seed, 'hints': {str(cell): codes for cell, codes in normalized_hints.items()},
                'nodes': nodes, 'stats': dict(stats), 'elapsed_seconds': previous_elapsed + time.monotonic()-began,
                'stack': [{'domains': [hex(value) for value in frame['domains']], 'cell': frame['cell'],
                           'choices': frame['choices'].copy(), 'next': frame['next']} for frame in stack]}
        payload['digest'] = _checkpoint_digest(payload)
        return payload

    def result(status, codes=None, report=None):
        value = {'status': status, 'codes': codes, 'elapsed_seconds': time.monotonic()-began,
                 'nodes': nodes, 'complete': status in ('solved', 'edge_perfect', 'infeasible'), 'stats': dict(stats)}
        if report is not None: value['validation'] = report
        if status in ('timeout', 'stopped'): value['checkpoint'] = snapshot()
        return value

    def notify(event, **extra):
        if progress is not None:
            progress({'event': event, 'nodes': nodes, 'elapsed_seconds': time.monotonic()-began,
                      'stats': dict(stats), **extra})

    try:
        check()
        tile_bits = [7 << (3*tile) for tile in range(n)]
        by_side = [{} for _ in range(3)]
        for code, masks in enumerate(problem.masks):
            for side, mask in enumerate(masks):
                by_side[side][int(mask)] = by_side[side].get(int(mask), 0) | (1 << code)
        compatible = [[[by_side[other].get(reverse_mask(problem.masks[code, side]), 0) & ~tile_bits[code//3]
                        for code in range(count)] for other in range(3)] for side in range(3)]
        arcs = [(a, sa, b, sb) for a, sa, b, sb in problem.edges] + [(b, sb, a, sa) for a, sa, b, sb in problem.edges]
        incoming = [[] for _ in range(n)]
        for arc in arcs: incoming[arc[2]].append(arc)

        def propagate(domains):
            stats['propagations'] += 1
            queue = deque(arcs)
            def narrow(cell, value):
                if value == domains[cell]: return False
                domains[cell] = value
                queue.extend(incoming[cell])
                return True
            while True:
                check()
                if any(value == 0 for value in domains): return False
                while queue:
                    a, sa, b, sb = queue.popleft()
                    support, keep = compatible[sa][sb], 0
                    for code in _values(domains[a]):
                        if support[code] & domains[b]: keep |= 1 << code
                    if keep != domains[a]:
                        stats['arc_revisions'] += 1
                        if not keep: domains[a] = 0; return False
                        narrow(a, keep)
                    if len(queue) % 32 == 0: check()
                # A cell whose candidates all use one tile owns that tile even
                # before its rotation is chosen. Also force each tile's sole cell.
                owners, occurrences = {}, [[] for _ in range(n)]
                for cell, domain in enumerate(domains):
                    possible = {code//3 for code in _values(domain)}
                    if len(possible) == 1:
                        tile = next(iter(possible))
                        if tile in owners: return False
                        owners[tile] = cell
                    for tile in possible: occurrences[tile].append(cell)
                    if cell % 8 == 0: check()
                for tile, cell in owners.items():
                    for other in occurrences[tile]:
                        if other != cell and narrow(other, domains[other] & ~tile_bits[tile]):
                            stats['tile_prunes'] += 1
                            if not domains[other]: return False
                for tile, cells in enumerate(occurrences):
                    if not cells: return False
                    if len(cells) == 1 and narrow(cells[0], domains[cells[0]] & tile_bits[tile]):
                        stats['tile_prunes'] += 1
                        if not domains[cells[0]]: return False
                if any(value == 0 for value in domains): return False
                if not queue: return True

        while stack:
            check()
            frame = stack[-1]
            if frame['cell'] == -1:
                nodes += 1
                domains = frame['domains']
                if not propagate(domains):
                    stack.pop(); stats['backtracks'] += 1; continue
                if all(value.bit_count() == 1 for value in domains):
                    codes = [value.bit_length()-1 for value in domains]
                    stack.pop()  # Commit leaf consumption before any callback can stop.
                    report = problem.validate(codes)
                    stats['leaves'] += 1
                    notify('candidate', codes=codes, validation=report)
                    if report['valid']: return result('solved', codes, report)
                    if target == 'edges' and report['unique_tiles'] and report['matched_edges'] == len(problem.edges):
                        return result('edge_perfect', codes, report)
                    continue
                assigned = [value.bit_length()-1 if value.bit_count() == 1 else -1 for value in domains]
                if target == 'gold' and problem.partial_loop_impossible(assigned):
                    stats['loop_prunes'] += 1; stack.pop(); continue
                cell = min((i for i, value in enumerate(domains) if value.bit_count() > 1),
                           key=lambda i: (domains[i].bit_count(), -sum(domains[int(nb)].bit_count() == 1 for nb in problem.neighbors[i] if nb >= 0), cell_rank[i]))
                hinted = {code: index for index, code in enumerate(normalized_hints.get(cell, []))}
                choices = sorted(_values(domains[cell]), key=lambda code: (0, hinted[code]) if code in hinted else (1, rank[code]))
                frame.update(cell=cell, choices=choices, next=0)
            if frame['next'] >= len(frame['choices']):
                stack.pop(); stats['backtracks'] += 1; continue
            code = frame['choices'][frame['next']]
            frame['next'] += 1
            child = frame['domains'].copy(); child[frame['cell']] = 1 << code
            stack.append({'domains': child, 'cell': -1, 'choices': [], 'next': 0})
            if time.monotonic() - last_progress >= 1:
                notify('progress', codes=[value.bit_length()-1 if value.bit_count() == 1 else -1 for value in child],
                       assigned_tiles=sum(value.bit_count() == 1 for value in child), is_partial=True)
                last_progress = time.monotonic()
        return result('infeasible')
    except Interrupted as exc:
        return result(exc.status)
