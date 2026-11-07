"""Explicit parent ownership for the optional cross-process SAT service."""
import math
from search_options import normalize_options
from constraint_sat_service_round26 import SATServiceOwner,SATServiceClient

def run(delegate,problem,kwargs):
    options=normalize_options(kwargs.get('options'))
    state=kwargs.get('learning_state')
    if state is not None and not isinstance(state,dict):raise ValueError('SAT service state must be a trusted dictionary.')
    seconds=kwargs.get('seconds',10)
    if type(seconds) not in (int,float) or not math.isfinite(seconds) or seconds<0:raise ValueError('seconds must be a finite nonnegative budget.')
    if not options['shared_sat_service'] or seconds==0 or state is not None and '_sat_service' in state:
        return delegate(problem,**kwargs)
    stop=kwargs.get('stop')
    if stop is not None and (stop() if callable(stop) else stop.is_set()):return delegate(problem,**kwargs)
    if state is not None and not isinstance(state,dict):raise ValueError('SAT service state must be a trusted dictionary.')
    state={} if state is None else state
    with SATServiceOwner(problem,kwargs.get('target','gold'),directory=state.get('_reasoning_cache_dir'),instances=options['sat_service_instances']) as owner:
        client=owner.client();state['_sat_service']=client
        try:return delegate(problem,**{**kwargs,'learning_state':state})
        finally:
            client.close();state.pop('_sat_service',None)

def view(client,problem,target,initial):
    expected=(problem.fingerprint,target,tuple(sum(1<<c for c in set(row)) for row in problem.domains))
    if not isinstance(client,SATServiceClient) or client.binding!=expected or len(initial)!=problem.n or any(v&~root for v,root in zip(initial,client.initial)):
        raise ValueError('Shared SAT service differs from the original or effective root.')
    return SATServiceClient(client.proxy,client.binding,tuple(initial))
