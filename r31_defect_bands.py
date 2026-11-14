"""Exact geographical mismatch-budget search with disjoint, resumable SAT branches."""
import argparse
from collections import deque
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import Timer
import time

from r27_reason_maxsat import Budget, SeamFormula, checked_score, domain_rows
from validator import reverse_mask


def bands(problem, count):
    if not 1 <= count <= len(problem.edges):
        raise ValueError("Each band needs at least one seam")
    # Split consecutive breadth-first layers, with stable edge IDs for boundary ties.
    distance = {}
    for root in range(problem.n):
        if root in distance:
            continue
        distance[root] = len(distance)
        queue = deque([root])
        while queue:
            cell = queue.popleft()
            for neighbor in map(int, problem.neighbors[cell]):
                if neighbor >= 0 and neighbor not in distance:
                    distance[neighbor] = distance[cell] + 1
                    queue.append(neighbor)
    def key(i):
        a, _, b, _ = problem.edges[i]
        return (min(distance[a], distance[b]), max(distance[a], distance[b]), i)
    ordered = sorted(range(len(problem.edges)), key=key)
    return [ordered[len(ordered)*i//count:len(ordered)*(i+1)//count] for i in range(count)]


def seam_bounds(problem, rows, groups):
    ranges = []
    for edges in groups:
        forced = possible = 0
        for e in edges:
            a, sa, b, sb = problem.edges[e]
            left = set(int(problem.masks[c, sa]) for c in rows[a])
            right = set(reverse_mask(int(problem.masks[c, sb])) for c in rows[b])
            forced += left.isdisjoint(right)
            possible += len(left | right) != 1
        ranges.append((forced, possible))
    return ranges


def inventory_lower(problem, rows, group):
    """Price each seam end against relaxed neighbors, then require one copy per piece."""
    import numpy as np
    from r27_incumbent import assignment
    costs = [dict.fromkeys(row, 0) for row in rows]
    for e in group:
        a, sa, b, sb = problem.edges[e]
        for cell, side, other, mate in ((a,sa,b,sb), (b,sb,a,sa)):
            ids = {}
            for code in rows[other]:
                ids.setdefault(reverse_mask(int(problem.masks[code, mate])), set()).add(code//3)
            for code in rows[cell]:
                candidates = ids.get(int(problem.masks[code, side]))
                if not candidates or (len(candidates)==1 and code//3 in candidates):
                    costs[cell][code] += 1
    matrix = np.full((problem.n, problem.n), 2*len(group)+1, dtype=int)
    for cell, choices in enumerate(costs):
        for code, cost in choices.items():
            matrix[cell, code//3] = min(matrix[cell, code//3], cost)
    legal = matrix <= 2*len(group)
    try:
        mates = assignment(-matrix, legal)
    except ValueError:
        return len(group)+1
    return (int(matrix[np.arange(problem.n), mates].sum())+1)//2



def exact_thresholds(literals, limit, formula, solver, check):
    # Row j means at least j chosen literals. Encode BOTH implications since
    # lower and upper limits are used as branching assumptions.
    row = []
    for x in literals:
        check()
        prev, row = row, []
        for i in range(min(len(prev)+1, limit)):
            v = formula.pool.id()
            keep = prev[i] if i < len(prev) else None
            lower = prev[i-1] if i else None
            if keep:
                solver.add_clause([-keep, v])
            solver.add_clause(([-lower] if lower else []) + [-x, v])
            solver.add_clause(([keep] if keep else []) + [-v, x])
            if lower:
                solver.add_clause(([keep] if keep else []) + [-v, lower])
            row.append(v)
    return row


def solve(problem, database, *, target=208, domains=None, group_count=3, width=8,
          seconds=0, slice_conflicts=2000, max_calls=None, split=True):
    from pysat.card import ITotalizer
    from pysat.solvers import Glucose4
    rows = domain_rows(problem, domains)
    if not 0 <= target <= len(problem.edges) or width < 1 or seconds < 0 or slice_conflicts < 1:
        raise ValueError("Invalid threshold or search budget")
    if max_calls is not None and max_calls < 0:
        raise ValueError("Invalid call budget")
    groups = bands(problem, group_count)
    budget = len(problem.edges)-target
    limits = seam_bounds(problem, rows, groups)
    limits = [(lo, min(hi, budget)) for lo, hi in limits]
    key = hashlib.sha256(json.dumps([1, problem.fingerprint, rows, groups, target, width, split]).encode()).hexdigest()
    start = time.monotonic()
    deadline = start + seconds if seconds else float('inf')
    def check():
        if time.monotonic() >= deadline:
            raise Budget
    stats = dict(calls=0, conflicts=0, decisions=0, core_skips=0, budget_skips=0,
                 adaptive_splits=0, completed_branches=0, inventory_queries=0, inventory_tightened=0)
    with closing(sqlite3.connect(database, timeout=60)) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS r31_bands(scope TEXT PRIMARY KEY,
                   stack TEXT, witness TEXT, cores TEXT, generation INTEGER)""")
        with db:
            db.execute("INSERT OR IGNORE INTO r31_bands VALUES (?,?,NULL,'[]',0)",
                       (key, json.dumps([limits])))
        stack_text, witness_text, core_text, generation = db.execute(
            "SELECT stack,witness,cores,generation FROM r31_bands WHERE scope=?", (key,)).fetchone()
        stack = json.loads(stack_text)
        witness = json.loads(witness_text) if witness_text else None
        cores = [frozenset(map(tuple, c)) for c in json.loads(core_text)]
        def save():
            nonlocal generation
            with db:
                changed = db.execute("""UPDATE r31_bands SET stack=?,witness=?,cores=?,generation=generation+1
                         WHERE scope=? AND generation=?""", (json.dumps(stack), json.dumps(witness),
                         json.dumps([sorted(c) for c in cores]), key, generation))
                if not changed.rowcount:
                    raise RuntimeError("Another worker advanced this checkpoint. Reopen the saved state.")
            generation += 1
        if witness is not None:
            score, gold = checked_score(problem, rows, witness['codes'])
            if score != witness['score'] or gold != witness['gold'] or score < target:
                raise ValueError('Saved witness failed independent validation')
        def result():
            status = 'witness' if witness is not None else ('infeasible' if not stack else 'open')
            return dict(status=status, target=target, groups=groups, pending=len(stack), witness=witness,
                        **stats, seconds=time.monotonic()-start)
        if not stack or witness is not None or max_calls == 0:
            return result()
        if any(lo > hi for lo, hi in limits) or sum(lo for lo, _ in limits)>budget:
            stack.clear()
            stats['budget_skips'] += 1
            save()
            return result()
        priced = {}
        with Glucose4() as sat:
            try:
                formula = SeamFormula(problem, rows, sat, check, exact_defects=split)
                if budget < len(formula.defects):
                    with ITotalizer(lits=formula.defects, ubound=budget, top_id=formula.pool.top) as total:
                        sat.append_formula(total.cnf.clauses)
                        sat.add_clause([-total.rhs[budget]])
                        formula.pool.occupy(formula.pool.top+1, total.top_id)
                thresholds = [exact_thresholds([formula.defects[i] for i in group],
                               min(len(group), budget+1), formula, sat, check) for group in groups] if split else []
                while stack:
                    check()
                    if max_calls is not None and stats['calls'] >= max_calls:
                        break
                    box = stack[-1]
                    if split and box != [list(pair) for pair in limits]:
                        for k, (lo, hi) in enumerate(box):
                            if hi < limits[k][1] and k not in priced:
                                priced[k] = inventory_lower(problem, rows, groups[k])
                                stats["inventory_queries"] += 1
                                stats["inventory_tightened"] += int(priced[k] > limits[k][0])
                        box = [(max(lo, priced.get(k, 0)), hi) for k,(lo,hi) in enumerate(box)]
                    if any(lo>hi for lo,hi in box) or sum(lo for lo, _ in box) > budget:
                        stack.pop()
                        stats['budget_skips'] += 1
                        save()
                        continue
                    dim = max(range(len(box)), key=lambda j:box[j][1]-box[j][0])
                    low, high = box[dim]
                    def subdivide():
                        middle = (low+high)//2
                        first, second = list(box), list(box)
                        first[dim], second[dim] = (low,middle), (middle+1,high)
                        stack[-1:] = [second, first]
                    if split and high-low+1 > width:
                        subdivide()
                        continue
                    conditions = {}
                    if split:
                        for k, (lo, hi) in enumerate(box):
                            if lo > limits[k][0]:
                                conditions[thresholds[k][lo-1]] = (k, 1, lo)
                            if hi < limits[k][1]:
                                conditions[-thresholds[k][hi]] = (k, -1, hi)
                    def applies(core):
                        return all(box[k][0] >= v if sign == 1 else box[k][1] <= v
                                   for k,sign,v in core)
                    if any(applies(core) for core in cores):
                        stack.pop()
                        stats['core_skips'] += 1
                        save()
                        continue
                    sat.conf_budget(slice_conflicts)
                    timer = Timer(max(.001, deadline-time.monotonic()), sat.interrupt) if seconds else None
                    if timer:
                        timer.start()
                    try:
                        answer = sat.solve_limited(assumptions=list(conditions), expect_interrupt=True)
                    finally:
                        if timer:
                            timer.cancel()
                            timer.join()
                            sat.clear_interrupt()
                    stats['calls'] += 1
                    counters = sat.accum_stats()
                    stats.update(decisions=counters['decisions'], conflicts=counters['conflicts'])
                    if answer is True:
                        codes = formula.board(sat.get_model())
                        score, gold = checked_score(problem, rows, codes)
                        counts = [sum(int(problem.masks[codes[a],sa]) != reverse_mask(int(problem.masks[codes[b],sb]))
                            for a,sa,b,sb in (problem.edges[i] for i in group)) for group in groups]
                        if score < target or any(not lo<=v<=hi for (lo,hi),v in zip(box,counts)):
                            raise RuntimeError("Decoded board violated its geographic defect budget")
                        witness = dict(codes=codes, score=score, gold=gold, defects=counts)
                        save()
                        break
                    if answer is False:
                        core = frozenset(conditions[x] for x in sat.get_core() or [])
                        if not any(old <= core for old in cores):
                            cores = [old for old in cores if not core < old] + [core]
                        stack.pop()
                        stats['completed_branches'] += 1
                    elif split and high > low:
                        subdivide()
                        stats['adaptive_splits'] += 1
                    save()
            except Budget:
                pass
    return result()


def main():
    from r28_exchange import read_json
    from search_problem import SearchProblem
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles', default='data/tiles.json')
    parser.add_argument('--database', type=Path, required=True)
    parser.add_argument('--domains')
    parser.add_argument('--target', type=int, default=208)
    parser.add_argument('--bands', type=int, default=3)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--seconds', type=float, default=0)
    parser.add_argument('--slice-conflicts', type=int, default=2000)
    args = parser.parse_args()
    args.database.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps(solve(SearchProblem(read_json(args.tiles)), args.database, target=args.target,
          domains=read_json(args.domains) if args.domains else None, group_count=args.bands,
          width=args.width, seconds=args.seconds, slice_conflicts=args.slice_conflicts)))


if __name__ == '__main__':
    main()
