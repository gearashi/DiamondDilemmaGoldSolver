"""Exact geographical mismatch-budget search with disjoint, resumable SAT branches."""
import argparse
from collections import deque
from contextlib import ExitStack, closing
import hashlib
import json
from pathlib import Path
import sqlite3
from threading import Timer
import time

from r27_reason_maxsat import Budget, SeamFormula, checked_score, domain_rows
from validator import reverse_mask
from constraint_work import digest
import r34_band_cores as shared_cores
import r34_band_placements as placements


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
    from r34_query_ownership import LostQueryLease
    opts=dict(target=target,domains=domains,group_count=group_count,width=width,seconds=seconds,
              slice_conflicts=slice_conflicts,max_calls=max_calls,split=split)
    try:
        return _solve(problem,database,**opts)
    except LostQueryLease:
        # A successor owns the authoritative stack. Return only its committed state.
        return _solve(problem,database,**dict(opts,max_calls=0))


def _solve(problem,database,*,target,domains,group_count,width,seconds,slice_conflicts,max_calls,split):
    from r34_query_ownership import query_lease
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
    stats = dict(shared_core_checks=0, shared_core_skips=0, calls=0, conflicts=0, decisions=0, core_skips=0, budget_skips=0,
                 adaptive_splits=0, completed_branches=0, inventory_queries=0, inventory_tightened=0,
                 placement_splits=0, placement_local_unsat=0, placement_leaf_checks=0)
    with closing(sqlite3.connect(database, timeout=60)) as db, ExitStack() as leases:
        shared_cores.schema(db)
        db.execute("""CREATE TABLE IF NOT EXISTS r31_bands(scope TEXT PRIMARY KEY,
                   stack TEXT, witness TEXT, cores TEXT, generation INTEGER)""")
        with db:
            db.execute("INSERT OR IGNORE INTO r31_bands VALUES (?,?,NULL,'[]',0)",
                       (key, json.dumps([limits])))
        def read_state():
            return db.execute("SELECT stack,witness,cores,generation FROM r31_bands WHERE scope=?", (key,)).fetchone()
        stack_text,witness_text,core_text,generation=read_state()
        guard=None
        if max_calls!=0 and (witness_text is None or json.loads(witness_text) is None) and json.loads(stack_text):
            guard=leases.enter_context(query_lease(db,database,'defect-bands:'+key,deadline))
            stack_text,witness_text,core_text,generation=read_state()
        stack = json.loads(stack_text)
        witness = json.loads(witness_text) if witness_text else None
        cores = [frozenset(map(tuple, c)) for c in json.loads(core_text)]
        def save():
            nonlocal generation
            with db:
                db.execute("BEGIN IMMEDIATE")
                guard()
                changed = db.execute("""UPDATE r31_bands SET stack=?,witness=?,cores=?,generation=generation+1
                         WHERE scope=? AND generation=?""", (json.dumps(stack), json.dumps(witness) if witness is not None else None,
                         json.dumps([sorted(c) for c in cores]), key, generation))
                if not changed.rowcount:
                    raise RuntimeError("Another worker advanced this checkpoint. Reopen the saved state.")
            generation += 1
        def defect_counts(codes):
            return [sum(int(problem.masks[codes[a],sa]) != reverse_mask(int(problem.masks[codes[b],sb]))
                        for a,sa,b,sb in (problem.edges[i] for i in group)) for group in groups]
        if witness is not None:
            score, gold = checked_score(problem, rows, witness['codes'])
            if score != witness['score'] or gold != witness['gold'] or score < target or defect_counts(witness['codes'])!=witness['defects']:
                raise ValueError('Saved witness failed independent validation')
        def result():
            status = 'witness' if witness is not None else ('infeasible' if not stack else 'open')
            return dict(status=status, target=target, groups=groups, pending=len(stack), witness=witness,
                        **stats, seconds=time.monotonic()-start)
        if not stack or witness is not None or guard is None:
            return result()
        if any(lo > hi for lo, hi in limits) or sum(lo for lo, _ in limits)>budget:
            stack.clear()
            stats['budget_skips'] += 1
            save()
            return result()
        from r34_seam_cnf import encoded_scope,prepare
        structural=encoded_scope(problem,rows,split)
        core_scope=digest([structural,groups,'regional-cores-v1'])
        shared=shared_cores.candidates(db,core_scope,budget,key)
        def applies(core,box):
            return all(box[k][0]>=v if sign==1 else box[k][1]<=v for k,sign,v in core)
        retained=[node for node in stack if not any(verified and applies(core,placements.region(node))
                                                    for core,verified in shared.items())]
        if len(retained)<len(stack):
            stats['shared_core_skips']+=len(stack)-len(retained)
            stack=retained
            save()
            if not stack:return result()
        priced = {}
        with Glucose4() as sat:
            try:
                formula,thresholds=prepare(db,database,problem,rows,split,groups,budget,sat,check,deadline,stats,
                                           SeamFormula,exact_thresholds,structural)
                while stack:
                    check()
                    if max_calls is not None and stats['calls'] >= max_calls:
                        break
                    node = stack[-1]
                    box = placements.region(node)
                    pins = node['pins'] if isinstance(node,dict) else ()
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
                    if any(applies(core,box) for core in cores):
                        stack.pop()
                        stats['core_skips'] += 1
                        save()
                        continue
                    match=next((core for core in shared if applies(core,box)),None)
                    if match is not None and shared[match]:
                        stack.pop()
                        stats['shared_core_skips'] += 1
                        save()
                        continue
                    bounds = placements.intervals(rows,pins) if pins else None
                    if pins and all(hi-lo == 1 for lo,hi in bounds):
                        codes = [row[lo] for row,(lo,_) in zip(rows,bounds)]
                        stats['placement_leaf_checks'] += 1
                        if len({v//3 for v in codes}) == problem.n:
                            score,gold = checked_score(problem,rows,codes)
                            counts = defect_counts(codes)
                            if score >= target and all(lo <= v <= hi for (lo,hi),v in zip(box,counts)):
                                witness = dict(codes=codes,score=score,gold=gold,defects=counts)
                        if witness is None:
                            stack.pop()
                            stats['placement_local_unsat'] += 1
                            stats['completed_branches'] += 1
                        save()
                        if witness is not None:
                            break
                        continue
                    literals=list(conditions)
                    if pins:
                        literals.extend(placements.assumptions(rows,formula.literals,pins))
                    if match is not None:
                        # This new query has a fresh native solver. Verify the stored
                        # assumption subset independently before publishing an exclusion.
                        literals=shared_cores.assumptions(match,thresholds,budget)
                        stats['shared_core_checks'] += 1
                    check()
                    prior_search = stats['conflicts'] + stats['decisions']
                    sat.conf_budget(slice_conflicts)
                    timer = Timer(max(.001, deadline-time.monotonic()), sat.interrupt) if seconds else None
                    if timer:
                        timer.start()
                    try:
                        answer = sat.solve_limited(assumptions=literals, expect_interrupt=True)
                    finally:
                        if timer:
                            timer.cancel()
                            timer.join()
                            sat.clear_interrupt()
                    stats['calls'] += 1
                    counters = sat.accum_stats()
                    stats.update(decisions=counters['decisions'], conflicts=counters['conflicts'])
                    if match is not None:
                        if answer is False:
                            shared_cores.store(db,guard,core_scope,budget,key,match,True)
                            shared[match]=True
                            stack.pop()
                            stats['completed_branches'] += 1
                            save()
                        else:
                            # Never spend a later one-call slice retrying this hint.
                            # Only a verified exclusion may remove the actual box.
                            shared_cores.defer(db,guard,core_scope,key,match)
                            del shared[match]
                        continue
                    if answer is True:
                        codes = formula.board(sat.get_model())
                        score, gold = checked_score(problem, rows, codes)
                        counts = defect_counts(codes)
                        if (score < target or any(not lo<=v<=hi for (lo,hi),v in zip(box,counts))
                                or any(codes[cell] not in rows[cell][lo:hi] for cell,lo,hi in pins)):
                            raise RuntimeError("Decoded board violated its geographic defect budget")
                        witness = dict(codes=codes, score=score, gold=gold, defects=counts)
                        save()
                        break
                    if answer is False:
                        native_core = sat.get_core()
                        if pins and (native_core is None or any(x not in conditions for x in native_core)):
                            # This proof used a placement assumption. Its sibling
                            # or a different geographic query can still have a witness.
                            stats['placement_local_unsat'] += 1
                        else:
                            core = frozenset(conditions[x] for x in native_core or [])
                            if not any(old <= core for old in cores):
                                shared_cores.store(db,guard,core_scope,budget,key,core,False)
                                cores = [old for old in cores if not core < old] + [core]
                        stack.pop()
                        stats['completed_branches'] += 1
                    elif split and high > low:
                        subdivide()
                        stats['adaptive_splits'] += 1
                    elif ((max_calls is not None and stats['calls'] >= max_calls or time.monotonic() >= deadline)
                          and stats['conflicts'] + stats['decisions'] > prior_search):
                        # The working native solver keeps its learned clauses across
                        # conflict slices. Refine only when the solver is about to close.
                        children = placements.split_placement(box,bounds or placements.intervals(rows,()),pins)
                        if children:
                            stack[-1:] = children
                            stats['placement_splits'] += 1
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
