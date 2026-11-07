"""Validated conversion between exact engine checkpoints and domain-box work.

A portable frontier represents unfinished work, not a proof that the original
puzzle is exhausted. Checksums protect local checkpoint integrity only.
"""
from constraint_work import WorkScope,checkpoint_boxes,frontier_snapshot,restore_frontier


def export_frontier(problem,checkpoint,target='gold',options=None):
    scope=WorkScope.for_problem(problem,target)
    if isinstance(checkpoint,dict) and checkpoint.get('work_version')==1:
        rows=restore_frontier(checkpoint,scope)
    elif isinstance(checkpoint,dict) and checkpoint.get('schema')=='diamond-sat-session-v1':
        from constraint_sat_session import PersistentSatSession
        session=PersistentSatSession.restore(problem,checkpoint,target)
        try:rows=tuple(session.boxes)
        finally:session.close()
    elif isinstance(checkpoint,dict) and 'parallel_version' in checkpoint:
        from constraint_parallel import solve
        checked=solve(problem,seconds=0,target=target,resume=checkpoint,options=options)
        rows=tuple(row for job in checked['checkpoint']['jobs'] for row in checkpoint_boxes(job['checkpoint']))
    else:
        from constraint_dfs import solve
        checked=solve(problem,seconds=0,target=target,resume=checkpoint,options=options)
        rows=checkpoint_boxes(checked['checkpoint'])
    return frontier_snapshot(scope,rows)


def dfs_roots(problem,payload,target='gold',seed=0,options=None):
    """Fresh metadata-free roots preserve the exact imported space, no new symmetry."""
    from constraint_dfs import _solve_core
    from constraint_parallel import _root_copy
    rows=restore_frontier(payload,WorkScope.for_problem(problem,target))
    template=_solve_core(problem,seconds=0,target=target,seed=seed,options=options)['checkpoint']
    return tuple(_root_copy(template,row) for row in rows)
