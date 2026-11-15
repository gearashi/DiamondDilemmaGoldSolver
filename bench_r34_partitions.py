"""Compare GPU domain partition policies on exact finite and open original residuals."""
import argparse
from contextlib import ExitStack, closing
from fractions import Fraction
import heapq
import json
import math
from pathlib import Path

from bench_r34_allocation import mathematical_answers, residuals
from constraint_work import WorkScope
from gpu_domain_round24 import DomainCUDA
from gpu_domain_runner import inventory_screen, split_cubes
from gpu_partitions_round34 import split_small_domains
from io_utils import atomic_json
from r28_exchange import read_json
from search_problem import SearchProblem
from test_gpu_domain_round25 import (fork_gold_fixture, independent_solutions,
    joint_hall_fixture, precise_reason_fixture, repeated_neighbor_fixture, word_boundary_fixture)


def volume(row):
    return math.prod(mask.bit_count() for mask in row)


def partition(cubes, count, policy):
    if policy == 'latest':
        return split_cubes(cubes, count)
    if policy == 'anchor':
        return split_small_domains(cubes, count)
    if policy not in ('oldest', 'volume'):
        raise ValueError('Unknown split priority')
    cubes = list(cubes)
    if len(cubes) >= count:
        return cubes
    heap = []
    ticket = 0
    def push(row):
        nonlocal ticket
        width, axis = max((mask.bit_count(), cell) for cell, mask in enumerate(row))
        priority = width if policy == 'oldest' else volume(row)
        heapq.heappush(heap, (-priority, ticket, width, axis, row))
        ticket += 1
    for row in cubes:
        push(row)
    while len(heap) < count and heap and heap[0][2] > 1:
        _, _, width, axis, row = heapq.heappop(heap)
        right, left = row[axis], 0
        for _ in range(width // 2):
            value = right & -right
            right ^= value
            left |= value
        for value in (left, right):
            new = list(row)
            new[axis] = value
            push(tuple(new))
    return [item[-1] for item in heap]


def cover(root, cubes):
    if sum(map(volume, cubes)) != volume(root):
        raise AssertionError('Incorrect total coverage')
    for index, row in enumerate(cubes):
        if not all(v and not v & ~a for v, a in zip(row, root)):
            raise AssertionError('Child escapes parent')
        if any(all(a & b for a, b in zip(row, other)) for other in cubes[:index]):
            raise AssertionError('Overlapping children')


COUNTERS = ('decisions','support_checks','domain_removals','backjumps','joint_support_checks',
            'event_inventory_checks','matching_edge_checks','strand_joins')
OPTIONS = dict(joint_support=True, event_propagation=True, precise_reasons=True,
               watched_clauses=True, global_matching=True, support_reuse=True, cross_lane=True)


def run(p, target, count, policy, fork, *, limited=False, resume=False, options=OPTIONS, quantum=1024):
    root = WorkScope.for_problem(p, target).root
    out = dict(status='open', splits=0, host_closed=0, gpu_lanes=0, solutions=0)
    work = dict.fromkeys(COUNTERS, 0)
    answers = set()
    with ExitStack() as owned:
        def track(engine):
            owned.callback(engine.close)
            return engine
        if fork:
            parent = track(DomainCUDA(p, [root], target=target, **options))
            for _ in range(1 if limited else 1024):
                states, ready = parent.propagate_root(1024 if limited else 65536)
                if ready:
                    break
            if ready and all(states == 2):
                engine = parent
                out['status'] = 'exhausted'
            elif ready:
                root, = parent.remaining_domain_cubes()
                cubes = partition([root], count, policy)
                cover(root, cubes)
                out['splits'] = len(cubes)
                if len(cubes) == 1:
                    engine = parent
                else:
                    old = parent.stats()
                    work.update({k: old.get(k, 0) for k in COUNTERS})
                    engine = track(parent.fork_root(cubes))
            elif limited:
                engine = parent
            else:
                raise AssertionError('Root propagation did not finish')
        else:
            cubes = partition([root], count, policy)
            cover(root, cubes)
            out['splits'] = len(cubes)
            if options.get('global_matching'):
                cubes, closed = inventory_screen(cubes, p.n)
                out['host_closed'] = len(closed)
            if not cubes:
                out.update(status='exhausted', work=work)
                return out, answers
            engine = track(DomainCUDA(p, cubes, target=target, **options))
        out['gpu_lanes'] = engine.lanes
        # A limited run consumes exactly one 1,024-primitive dispatch per
        # native lane, including the parent if preparing a propagated fork.
        for index in range(0 if limited and fork else 1 if limited else 4000):
            states = engine.step(quantum)
            pending = engine.candidates()
            for item in pending:
                row = tuple(item['codes'])
                if row in answers:
                    raise AssertionError('Duplicate enumerated placement')
                answers.add(row)
            if pending:
                if 'first_witness_work' not in out:
                    at = engine.stats()
                    out['first_witness_work'] = {k:work[k] + at.get(k,0) for k in COUNTERS}
                engine.acknowledge([item['lane'] for item in pending])
            if all(states == 2):
                out['status'] = 'exhausted'
                break
            if resume and index == 0:
                saved = engine.checkpoint()
                other = track(DomainCUDA(p, engine.cubes, target=target, **options))
                other.restore(saved)
                if other.stats() != engine.stats() or other.remaining_domain_cubes() != engine.remaining_domain_cubes():
                    raise AssertionError('Restored branch coverage or native counters differ')
                engine = other
                out['restored'] = True
        at = engine.stats()
        out['work'] = {k:work[k] + at.get(k,0) for k in COUNTERS}
        out['solutions'] = len(answers)
        if not limited and out['status'] != 'exhausted':
            raise AssertionError('Finite enumeration failed to finish')
    return out, answers


def original_matched(problem, count=128):
    """Equal aggregate native primitive slots across every surviving original lane."""
    root = WorkScope.for_problem(problem, 'gold').root
    live = {}
    for policy in ('latest','oldest','volume','anchor'):
        cubes = partition([root],count,policy)
        cover(root,cubes)
        live[policy],_ = inventory_screen(cubes,problem.n)
    block = math.lcm(*(len(cubes) for cubes in live.values())) * 8
    result = {}
    for policy,cubes in live.items():
        with closing(DomainCUDA(problem,cubes,target='gold',**OPTIONS)) as engine:
            issued = 0
            rows = []
            for multiple in (1,16,64,256,1024):
                per_lane = block * multiple // len(cubes)
                while issued < per_lane:
                    quantum = min(2048,per_lane-issued)
                    state = engine.step(quantum)
                    issued += quantum
                    if any(state == 1) or all(state == 2):
                        break
                stats = engine.stats()
                rows.append(dict(total_slots=issued*len(cubes), lanes=len(cubes), exhausted=int(sum(state==2)),
                                 validated=len(engine.candidates()),work={k:stats.get(k,0) for k in COUNTERS}))
                if any(state == 1) or all(state == 2):
                    break
            result[policy] = rows
    return result


def benchmark(original, output, phases):
    data = {'finite':{}, 'original':{}}
    def save():
        atomic_json(output, data)
    if 'finite' in phases:
        cases = [(name,p,'gold',128,OPTIONS) for name,p in residuals(original)]
        cases += [(name,factory(),target,count,opts) for name,factory,target in (
            ('fork',fork_gold_fixture,'gold'), ('hall',joint_hall_fixture,'edges'),
            ('word',word_boundary_fixture,'edges'), ('prism-wide',precise_reason_fixture,'edges'),
            ('parallel',repeated_neighbor_fixture,'edges'))
            for count in (4,8) for opts in ({},OPTIONS)]
        for i,(name,p,target,count,opts) in enumerate(cases):
            key = f'{name}-{count}-' + ('matching' if opts else 'basic')
            expected = mathematical_answers(p) if i < 16 else independent_solutions(p,target)
            row = {'expected':len(expected), 'trials':{}}
            for policy in ('latest','oldest','volume','anchor'):
                for fork in (False,True):
                    got,answers = run(p,target,count,policy,fork,options=opts)
                    if answers != expected:
                        raise AssertionError((key,policy,fork,len(expected),len(answers)))
                    row['trials'][policy+('-fork' if fork else '-raw')] = got
            data['finite'][key] = row
            save()
            print(key, flush=True)
    if 'original' in phases:
        root = WorkScope.for_problem(original,'gold').root
        for policy in ('latest','oldest','volume','anchor'):
            cubes = partition([root],128,policy)
            cover(root,cubes)
            live,closed = inventory_screen(cubes,original.n)
            row = {'split':len(cubes),'closed':len(closed),'min_fraction':str(Fraction(min(map(volume,cubes)),volume(root))),
                   'max_fraction':str(Fraction(max(map(volume,cubes)),volume(root))),
                   'closed_fraction':str(Fraction(sum(volume(c) for c,_ in closed),volume(root)))}
            for fork in (False,True):
                # Fork starts with the existing single parent. It has consumed
                # a different amount of total work from 128 concurrent lanes.
                row['fork' if fork else 'raw'] = run(original,'gold',128,policy,fork,limited=True)[0]
            data['original'][policy] = row
            save()
    if 'matched' in phases:
        data['matched'] = original_matched(original)
        save()
    return data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', type=Path, default=Path('../data/tiles.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phases', nargs='+', choices=['finite','original','matched'], default=['finite','original','matched'])
    args=parser.parse_args()
    benchmark(SearchProblem(read_json(args.tiles)), args.output, args.phases)
