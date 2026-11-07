"""Portable symbolic work and validated adapters for existing exact engines.

Transfers replace ownership; callers must stop the old owner before dispatching
its exported remaining set elsewhere. Neither format is an external UNSAT proof.
"""
from constraint_work import WorkScope, digest, restore_frontier, checkpoint_boxes
from constraint_symbolic_frontier import OrderedSets, SymbolicFrontier


def portable_snapshot(frontier, check=lambda: None):
    pending = frontier.pending
    for _, root in frontier.leases.values():
        pending = frontier.sets.union(pending, root, check)
    body = {'work_version': 2, 'scope': frontier.scope.payload(),
            'graph': frontier.sets.snapshot((pending,), check)}
    body['digest'] = digest(body)
    return body


def restore_portable(payload, expected, check=lambda: None):
    if not isinstance(payload, dict):
        raise ValueError('Malformed portable work.')
    if payload.get('work_version') == 1:
        rows = restore_frontier(payload, expected, check=check)
        return SymbolicFrontier(WorkScope.restore(payload['scope']), rows)
    if set(payload) != {'work_version', 'scope', 'graph', 'digest'} or type(payload['work_version']) is not int or payload['work_version'] != 2:
        raise ValueError('Unknown symbolic portable work version.')
    if digest({k:v for k,v in payload.items() if k != 'digest'}) != payload['digest']:
        raise ValueError('Symbolic portable digest differs.')
    scope = WorkScope.restore(payload['scope'])
    if not expected.accepts(scope):
        raise ValueError('Symbolic work scope differs.')
    sets, roots = OrderedSets.restore(payload['graph'], check)
    if sets.root != scope.root or len(roots) != 1:
        raise ValueError('Malformed symbolic portable universe.')
    result = SymbolicFrontier(scope, (), order=sets.order, max_nodes=sets.graph.max_nodes)
    result.sets, result.pending = sets, roots[0]
    return result


def engine_remaining(problem, backend, checkpoint, *, target='gold', options=None):
    """Validate an owned native checkpoint and recover its pending products.

    GPU checkpoints already contain a host-exported frontier. Loading their
    device state for continued GPU execution additionally invokes that engine's
    validator; this conversion imports trusted local frontier data only.
    """
    expected = WorkScope.for_problem(problem, target)
    if backend == 'cpu':
        from constraint_dfs import solve
        checked = solve(problem, seconds=0, target=target, resume=checkpoint, options=options)
        payload = checked.get('checkpoint')
        if payload is None:
            if checked.get('status') == 'infeasible' and checked.get('complete'):
                return expected, ()
            raise ValueError('CPU checkpoint omitted its unfinished state.')
        rows = checkpoint_boxes(payload)
        root = tuple(int(x, 16) for x in payload.get('owned_root_domains', payload['stack'][0]['domains']))
        return WorkScope.for_problem(problem, target, root), rows
    if backend == 'sat':
        from constraint_sat_session import PersistentSatSession
        session = PersistentSatSession.restore(problem, checkpoint, target)
        try:
            return session.scope, tuple(session.boxes)
        finally:
            session.close()
    if backend == 'gpu':
        if (not isinstance(checkpoint, dict) or type(checkpoint.get('version')) is not int
                or checkpoint['version'] != 1 or 'frontier' not in checkpoint
                or digest({k:v for k,v in checkpoint.items() if k != 'digest'}) != checkpoint.get('digest')):
            raise ValueError('Malformed trusted GPU frontier checkpoint.')
        scope = WorkScope.restore(checkpoint['scope'])
        if not expected.accepts(scope):
            raise ValueError('GPU frontier scope differs.')
        rows = restore_frontier(checkpoint['frontier'], scope)
        return WorkScope.restore(checkpoint['frontier']['scope']), rows
    raise ValueError('Unknown exact engine adapter.')


def from_checkpoint(problem, checkpoint, *, target='gold', options=None):
    expected = WorkScope.for_problem(problem, target)
    if not isinstance(checkpoint, dict):
        raise ValueError('Expected an exact checkpoint.')
    if checkpoint.get('schema') == 'diamond-symbolic-coordinator-v1':
        from constraint_coordinator_round26 import Coordinator
        coordinator=Coordinator(problem,target=target,resume=checkpoint,options=options)
        try:return restore_portable(portable_snapshot(coordinator.frontier),expected)
        finally:coordinator.dispose()
    if 'work_version' in checkpoint:
        return restore_portable(checkpoint, expected)
    if 'symbolic_frontier_version' in checkpoint:
        return restore_portable(portable_snapshot(SymbolicFrontier.restore(checkpoint, expected)), expected)
    if checkpoint.get('schema') == 'diamond-sat-session-v1':
        scope, rows = engine_remaining(problem, 'sat', checkpoint, target=target, options=options)
    elif 'parallel_version' in checkpoint:
        from constraint_parallel import solve
        checked = solve(problem, seconds=0, target=target, resume=checkpoint, options=options)
        rows = tuple(row for job in checked['checkpoint']['jobs'] for row in checkpoint_boxes(job['checkpoint']))
        scope = expected
    elif 'native' in checkpoint and 'frontier' in checkpoint:
        scope, rows = engine_remaining(problem, 'gpu', checkpoint, target=target, options=options)
    else:
        scope, rows = engine_remaining(problem, 'cpu', checkpoint, target=target, options=options)
    return SymbolicFrontier(scope, rows)