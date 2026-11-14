"""Anytime unweighted partial MaxSAT for the complete placement and seam objective."""
from __future__ import annotations

from collections import Counter
from threading import Timer
import time

from validator import reverse_mask


class Budget(Exception):
    """The experiment must return unknown, preserving any checked incumbent."""


def domain_rows(problem, domains=None):
    """Accept explicit placement code lists, preserving all original restrictions."""
    rows = problem.domains if domains is None else domains
    if len(rows) != problem.n:
        raise ValueError('Expected one placement domain per cell.')
    out = []
    for cell, row in enumerate(rows):
        original = set(problem.domains[cell])
        row = tuple(row)
        if any(type(code) is not int or code not in original for code in row):
            raise ValueError('Placement domain escapes the input scope.')
        out.append(tuple(sorted(set(row))))
    return tuple(out)


def checked_score(problem, rows, codes):
    if len(codes) != problem.n or any(type(code) is not int or code not in rows[cell]
                                      for cell, code in enumerate(codes)):
        raise ValueError('Candidate violates a cell domain.')
    if len({code // 3 for code in codes}) != problem.n:
        raise ValueError('Candidate repeats a physical tile.')
    result = problem.validate(list(codes))
    return result['matched_edges'], bool(result['valid'])


class SeamFormula:
    def __init__(self, problem, rows, solver, check):
        from pysat.card import CardEnc, EncType
        from pysat.formula import IDPool
        self.pool = IDPool()
        self.rows = rows
        self.literals = []
        self.defects = []
        self.stats = Counter()
        tile_rows = [[] for _ in range(problem.n)]

        def add(clause):
            solver.add_clause(clause)
            self.stats['clauses'] += 1
            if self.stats['clauses'] % 4096 == 0:
                check()

        def at_most_one(literals):
            if len(literals) > 1:
                encoding = CardEnc.atmost(lits=literals, bound=1, vpool=self.pool,
                                          encoding=EncType.seqcounter)
                for clause in encoding.clauses:
                    add(clause)

        for domain in rows:
            check()
            ids = {code: self.pool.id() for code in domain}
            add(list(ids.values()))
            at_most_one(list(ids.values()))
            self.literals.append(ids)
            for code, literal in ids.items():
                tile_rows[code // 3].append(literal)
        for literals in tile_rows:
            check()
            add(literals)
            at_most_one(literals)

        for a, sa, b, sb in problem.edges:
            check()
            defect = self.pool.id()
            self.defects.append(defect)
            symbols = {}
            for cell, side in ((a, sa), (b, sb)):
                for code, lit in self.literals[cell].items():
                    mask = int(problem.masks[code, side])
                    if cell == b:
                        mask = reverse_mask(mask)
                    if mask not in symbols:
                        symbols[mask] = self.pool.id()
                    # Relax only the second half of a seam. Both endpoints of a
                    # successful seam must select its single consistent symbol.
                    clause = [-lit, symbols[mask]]
                    if cell == b:
                        clause.append(defect)
                    add(clause)
            at_most_one(list(symbols.values()))
        self.stats['placement_variables'] = sum(map(len, self.literals))
        self.stats['base_variables'] = self.pool.top

    def board(self, model):
        positive = {x for x in model if x > 0}
        return [next(code for code, lit in row.items() if lit in positive) for row in self.literals]


def maximize_seams(problem, *, domains=None, seed_board=None, threshold=None, target='gold',
                   seconds=30, conflicts=None, stop=lambda: False, use_defect_bound=True):
    """Return a checked board and separate edge bound / Gold-existence status.

    An interrupted native decision has no authority to tighten a bound. Once
    Gold-only blockers are installed, an UNSAT result only rejects Gold.
    """
    from pysat.card import ITotalizer
    from pysat.solvers import Glucose4
    if target not in ('gold', 'edges') or seconds <= 0:
        raise ValueError('Choose a positive time budget and a supported target.')
    if conflicts is not None and (type(conflicts) is not int or conflicts < 1):
        raise ValueError('Conflict cap must be a positive integer.')
    rows = domain_rows(problem, domains)
    edges = len(problem.edges)
    if threshold is not None and (type(threshold) is not int or not 0 <= threshold <= edges):
        raise ValueError('Threshold is outside the seam objective.')
    start = time.monotonic()
    deadline = start + seconds
    best, best_score, best_gold = None, -1, False
    if seed_board is not None:
        best_score, best_gold = checked_score(problem, rows, seed_board)
        best = list(seed_board)
    result = {'status': 'unknown', 'target': target, 'edge_upper_bound': edges,
              'target_threshold': threshold, 'calls': 0, 'gold_cuts': 0,
              'scope': 'entire_board' if rows == domain_rows(problem) else 'restricted_domains'}
    def finish():
        result.update(codes=best, matched_edges=best_score, gold_valid=best_gold,
                      seconds=time.monotonic()-start)
        return result
    if best_gold or (target == 'edges' and best_score == edges):
        result['status'] = 'solved' if best_gold else 'edge_perfect'
        return finish()
    if stop() or time.monotonic() >= deadline:
        return finish()
    def check():
        if time.monotonic() >= deadline or stop():
            raise Budget
    if use_defect_bound:
        from r27_reason_cost import cost_bound
        try:
            bound = cost_bound(problem,domains=rows,probes=12,shift_pairs=25000,check=check)
        except Budget:
            return finish()
        result['defect_bound'] = {k:v for k,v in bound.items() if k!='certificate'}
        result['edge_upper_bound'] -= bound['lower_bound']
        if result['edge_upper_bound'] < best_score:
            raise RuntimeError('A defect lower bound contradicts the checked incumbent.')
        if threshold is not None and threshold > result['edge_upper_bound']:
            result['status'] = 'threshold_impossible'
            return finish()
        if threshold is None and best is not None and best_score == result['edge_upper_bound'] < edges:
            result['status'] = 'edge_optimum'
            return finish()
    if threshold is not None and best_score >= threshold and threshold < edges:
        result['status'] = 'threshold_met'
        return finish()
    need = threshold if threshold is not None else max(0, best_score+1)
    need = min(edges, need)
    with Glucose4() as solver:
        timer = None
        totalizer = None
        try:
            formula = SeamFormula(problem, rows, solver, check)
            result.update(formula.stats)
            if best is not None:
                solver.set_phases([formula.literals[i][code] for i, code in enumerate(best)])
            max_defects = edges - need
            if edges and max_defects:
                totalizer = ITotalizer(lits=formula.defects, ubound=min(edges-1,max_defects),
                                       top_id=formula.pool.top)
                solver.append_formula(totalizer.cnf.clauses)
                result['cardinality_clauses'] = len(totalizer.cnf.clauses)
            while True:
                check()
                allowed = edges - need
                if allowed == 0:
                    assumptions = [-lit for lit in formula.defects]
                elif allowed == edges:
                    assumptions = []
                else:
                    assumptions = [-totalizer.rhs[allowed]]
                # poll native search in interruptible slices so Stop is honored.
                remaining_conflicts = None if conflicts is None else conflicts-solver.accum_stats()['conflicts']
                if remaining_conflicts is not None and remaining_conflicts <= 0:
                    break
                if remaining_conflicts is not None:
                    solver.conf_budget(remaining_conflicts)
                timer = Timer(min(.5, max(.001, deadline-time.monotonic())),solver.interrupt)
                timer.start()
                try:
                    answer = solver.solve_limited(assumptions=assumptions, expect_interrupt=True)
                finally:
                    timer.cancel()
                    timer.join()
                    timer = None
                    solver.clear_interrupt()
                result['calls'] += 1
                if answer is None:
                    continue
                if not answer:
                    if result['gold_cuts']:
                        result['status'] = 'no_gold_in_scope'
                    else:
                        result['edge_upper_bound'] = min(result['edge_upper_bound'],need-1)
                        result['status'] = ('edge_optimum' if best_score == need-1 and best is not None
                                            else 'threshold_impossible')
                    break
                board = formula.board(solver.get_model())
                score, gold = checked_score(problem, rows, board)
                if score < need:
                    raise RuntimeError('Native SAT threshold witness failed independent validation.')
                if score > best_score or gold:
                    best_score, best_gold, best = score, gold, board
                if score == result['edge_upper_bound'] < edges and threshold is None:
                    result['status'] = 'edge_optimum'
                    break
                if gold or (target == 'edges' and score == edges):
                    result['status'] = 'solved' if gold else 'edge_perfect'
                    break
                if threshold is not None and threshold < edges:
                    result['status'] = 'threshold_met'
                    break
                if score == edges:
                    cut = problem.gold_nogood(board)
                    solver.add_clause([-formula.literals[c][code] for c, code in cut])
                    result['gold_cuts'] += 1
                    need = edges
                else:
                    need = score + 1
        except Budget:
            pass
        finally:
            if totalizer is not None:
                totalizer.delete()
            result['native'] = solver.accum_stats()
    return finish()

