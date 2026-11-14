"""One SAT owner with an interruptible two-sided, same-scope lookahead helper."""
from collections import Counter
from queue import Queue, Empty
from threading import Event, Thread, Timer
import time

from r27_reason_maxsat import Budget, SeamFormula, checked_score, domain_rows


def concurrent_sat(problem, *, domains=None, target='gold', seconds=10, pilots=6, probe_conflicts=12):
    from pysat.formula import CNF
    from pysat.solvers import Glucose4
    if target not in ('gold','edges') or pilots<0 or probe_conflicts<1:
        raise ValueError('Invalid concurrent SAT context.')
    rows = domain_rows(problem,domains)
    deadline = time.monotonic()+seconds
    stop = Event()
    messages = Queue()
    stats = Counter()
    def check():
        if stop.is_set() or time.monotonic()>=deadline:
            raise Budget
    class Sink:
        def __init__(self):
            self.base = CNF()
        def add_clause(self,clause):
            self.base.append(clause)
    sink = Sink()
    try:
        formula = SeamFormula(problem,rows,sink,check)
        for lit in formula.defects:
            sink.add_clause([-lit])
        check()
    except Budget:
        return {'status':'unknown','target':target,'codes':None,'gold_valid':False,'stats':dict(stats)}
    candidates = [next(iter(row.values())) for row in sorted(formula.literals,key=len) if len(row)>1][:pilots]
    def helper():
        try:
            with Glucose4(bootstrap_with=sink.base) as scout:
                for lit in candidates:
                    for side in (lit,-lit):
                        check()
                        scout.conf_budget(probe_conflicts)
                        # Bound native search if the owner finishes while a difficult pilot is active.
                        timer = Timer(min(.1,max(.001,deadline-time.monotonic())),scout.interrupt)
                        timer.start()
                        try:
                            answer = scout.solve_limited(assumptions=[side],expect_interrupt=True)
                        finally:
                            timer.cancel()
                            timer.join()
                            scout.clear_interrupt()
                        stats['pilot_calls'] += 1
                        stats['pilot_conflicts'] = scout.accum_stats()['conflicts']
                        if answer is False:
                            messages.put(('unit',-side))
                        elif answer is True:
                            messages.put(('board',formula.board(scout.get_model())))
                stats['pilot_completed_queue'] = 1
        except Budget:
            pass
        except Exception as error:
            messages.put(('error',error))

    thread = Thread(target=helper) if candidates else None
    result = {'status':'unknown','target':target,'codes':None,'gold_valid':False}
    if thread is not None:
        thread.start()
    try:
        with Glucose4(bootstrap_with=sink.base) as owner:
            known = set()
            def accept(board):
                score,gold = checked_score(problem,rows,board)
                if score != len(problem.edges):
                    raise RuntimeError('Invalid pilot or SAT witness.')
                if gold or target=='edges':
                    result.update(status='solved' if gold else 'edge_perfect',codes=board,gold_valid=gold)
                    return True
                cut = tuple(sorted(-formula.literals[c][v] for c,v in problem.gold_nogood(board)))
                if cut not in known:
                    owner.add_clause(list(cut))
                    known.add(cut)
                    stats['gold_cuts'] += 1
                return False
            while True:
                check()
                while not messages.empty():
                    try:
                        kind,value = messages.get_nowait()
                    except Empty:
                        break
                    if kind=='error':
                        raise value
                    if kind=='unit':
                        # The scout proved F & ~value UNSAT on this exact base.
                        owner.add_clause([value])
                        stats['imported_units'] += 1
                    elif accept(value):
                        break
                if result['codes'] is not None:
                    break
                timer = Timer(min(.1,max(.001,deadline-time.monotonic())),owner.interrupt)
                timer.start()
                try:
                    answer = owner.solve_limited(expect_interrupt=True)
                finally:
                    timer.cancel()
                    timer.join()
                    owner.clear_interrupt()
                stats['owner_calls'] += 1
                stats['owner_conflicts'] = owner.accum_stats()['conflicts']
                if answer is False:
                    result['status'] = 'infeasible'
                    break
                if answer is True and accept(formula.board(owner.get_model())):
                    break
    except Budget:
        pass
    finally:
        stop.set()
        if thread is not None:
            thread.join()
            stats['helper_formula_clauses'] = len(sink.base.clauses)
    return dict(result,stats=dict(stats))

