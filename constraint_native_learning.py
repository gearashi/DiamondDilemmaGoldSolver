"""Bounded independent replay of committed native seam/inventory failures.

Native fast rejection has already consumed its leaf before this code runs.
Captured domains describe the consumed leaf, not a proof: frontier replacement
can restrict a root to its remaining alternatives. Replay therefore starts from
the bound initial domains and active mask decisions, independently deriving all
basic consequences. Carried native reductions are deliberately not assumptions.
A budget exit, cancellation, or surviving fixpoint never produces a clause.
"""
from constraint_lcg import LCGState
from constraint_native_search import FastFailure
from constraint_pruning import _tile_choices, values

STAT_KEYS=('native_learning_replays','native_learning_checks',
           'native_learning_eliminated','native_learning_conflicts',
           'native_learning_clauses','native_learning_incomplete','native_learning_jumps')

class _ReplayLimit(Exception):
    pass


def replay_fast_failure(failure, *, initial, tables, fingerprint, target='gold',
                        stats=None, check=lambda:None, max_work=100000):
    """Return a root-bound first-UIP clause only after a completed replay proof.

    Work is counted separately from DFS nodes. The check callback may raise;
    callers retain the already-committed native frontier in that case.
    """
    if not isinstance(failure,FastFailure):raise ValueError('Expected a captured native failure.')
    if type(max_work)is not int or max_work<0:raise ValueError('Replay work limit must be nonnegative.')
    initial=tuple(initial);n=len(initial)
    if (not n or any(type(x)is not int or x<=0 or x.bit_length()>3*n for x in initial)
            or len(failure.domains)!=n
            or any(type(x)is not int or x<=0 or x&~root
                   for x,root in zip(failure.domains,initial))):
        raise ValueError('Native failure domains must be positive root-bounded masks.')
    state=LCGState(initial,fingerprint,target)
    for pair in failure.decisions:
        if not isinstance(pair,(tuple,list)) or len(pair)!=2:
            raise ValueError('Invalid native failure decision.')
        state=state.fork(*pair)
    if any(value&~root for value,root in zip(failure.domains,state.domains)):
        raise ValueError('Native failure domains escaped their active decisions.')
    domains=state.domains.copy()
    if stats is None:stats={}
    for key in STAT_KEYS:stats.setdefault(key,0)
    stats['native_learning_replays']+=1
    work=0
    def tick():
        nonlocal work
        if work>=max_work:raise _ReplayLimit()
        work+=1;stats['native_learning_checks']+=1
        if work%64==1:check()
    def narrow(cell,value,dependencies,rule):
        before=domains[cell]
        if before==value:return
        state.record_cell(cell,value,dependencies,rule)
        domains[cell]=value
        stats['native_learning_eliminated']+=before.bit_count()-value.bit_count()
    def conflict(cells=None):
        check()
        stats['native_learning_conflicts']+=1
        clause=state.first_uip(cells)
        if clause is None:stats['native_learning_incomplete']+=1
        else:stats['native_learning_clauses']+=1
        return clause
    try:
        check()
        while True:
            before=domains.copy()
            for a,sa,b,sb in tables.arcs:
                keep=0
                for code in values(domains[a]):
                    tick()
                    if tables.compatible[sa][sb][code]&domains[b]:keep|=1<<code
                if keep!=domains[a]:
                    narrow(a,keep,(b,),'replayed seam support')
                    if not keep:return conflict()
            owners={};occurrences=[[] for _ in range(n)]
            for cell,domain in enumerate(domains):
                tick();pieces=tuple(values(_tile_choices(domain)))
                if len(pieces)==1:
                    tile=pieces[0]
                    if tile in owners:return conflict((cell,owners[tile]))
                    owners[tile]=cell
                for tile in pieces:occurrences[tile].append(cell)
            for tile,cells in enumerate(occurrences):
                tick();bits=7<<(3*tile)
                if not cells:return conflict(range(n))
                if tile in owners:
                    for cell in cells:
                        tick()
                        if cell!=owners[tile]:
                            narrow(cell,domains[cell]&~bits,(owners[tile],),'replayed tile ownership')
                            if not domains[cell]:return conflict()
                if len(cells)==1:
                    # Every other cell excludes this tile, so its only host is
                    # forced. Include all current facts as conservative reasons.
                    narrow(cells[0],domains[cells[0]]&bits,range(n),'replayed sole tile host')
                    if not domains[cells[0]]:return conflict()
            if domains==before:
                stats['native_learning_incomplete']+=1
                return None
    except _ReplayLimit:
        stats['native_learning_incomplete']+=1
        return None
    except BaseException:
        stats['native_learning_incomplete']+=1
        raise
