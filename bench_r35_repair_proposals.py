"""Compare the actual Gold repair query and per-proposal local propagation on the same proven-optimal hit."""
import argparse
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack

from constraint_repair_round24 import core_guided_repair
from io_utils import atomic_json
from r28_exchange import read_json
from r35_repair_propagation import ProposedRepair
from r35_repair_seeds import active_catalog
from search_problem import SearchProblem

def measure(tiles):
    from ortools.sat.python import cp_model
    from bench_r35_repair_seeds import independently_allowed,PoseOracle
    from r35_repair_seeds import known_hint_conflicts
    p=SearchProblem(tiles);catalog=active_catalog(p)
    fixture=read_json(Path(__file__).parent/'docs/R35_REPAIR_PROPOSALS.json')
    if fixture['global_cuts']!=known_hint_conflicts():raise ValueError('Changed returned-hint controls.')
    upper=[set(catalog['hit'])]
    for section in ('single_exchanges','two_exchanges'):
        upper.extend({c for c in range(p.n) if int(source['hit'],16)>>c&1} for source in fixture[section]['sample'])
    oracle=PoseOracle(p,independently_allowed(p)[0])
    try:
        for k,(h,g) in enumerate(zip(upper,fixture['global_cuts'])):
            if (len(h)!=95 or not all(h.intersection(row) for row in catalog['rows']) or h.intersection(g)
                    or not oracle.replay(catalog['codes'],g)
                    or any(oracle.replay(catalog['codes'],set(g)-{c}) for c in g)):
                raise ValueError('Held-out optimal hit or independently minimized negative differs.')
            needed=fixture['global_cuts'][:1 if k in (1,2) else k]
            if not all(h.intersection(old) for old in needed):raise ValueError('A supposed surviving alternative is already excluded.')
    finally:oracle.close()
    real_solve=cp_model.CpSolver.solve
    runs=[]
    for mode in ('native','propagate'):
        state={'calls':[]}
        p=SearchProblem(tiles)
        archived=active_catalog(p)
        codes=archived['codes']
        def budget(solver,model,*args,**kwargs):
            solver.parameters.max_number_of_conflicts=128
            solver.parameters.max_deterministic_time=3
            status=real_solve(solver,model,*args,**kwargs)
            state['calls'].append(dict(status=solver.status_name(status),branches=int(solver.num_branches)))
            return status
        with ExitStack() as stack:
            # Isolate the actual hit that first justified these additional proofs.
            stack.enter_context(patch('r35_repair_seeds.known_hint_conflicts',return_value=(),create=True))
            if mode=='native':
                stack.enter_context(patch.object(ProposedRepair,'test',return_value=(None,None)))
            stack.enter_context(patch.object(cp_model.CpSolver,'solve',budget))
            result=core_guided_repair(p,codes,durable=True,max_queries=2,seconds=120)
            fields=('status','reason','derived_master_hits','propagated_cores','stats','certified_seed_distance')
            first={key:result[key] for key in fields if key in result}
            first['native_core_sizes']=[len(c) for c in result['cores']]
            if mode=='propagate':
                if not result['propagated_cores'] or result['exclusions']['cores']:
                    raise ValueError('A new checked propagation conflict was not retained separately.')
                confirmed=[]
                from constraint_repair_round24 import _minimum_hitting_set
                def used(rows,*args,**kw):
                    confirmed.append(all(frozenset(group) in rows for group in result['propagated_cores']))
                    return _minimum_hitting_set(rows,*args,**kw)
                with patch('constraint_repair_round24._minimum_hitting_set',used):
                    again=core_guided_repair(p,codes,resume=result['checkpoint'],max_queries=2,seconds=120)
                if not all(confirmed) or not confirmed:raise ValueError('Recovered negative did not reach the live master.')
                state['reopened']={k:again[k] for k in fields if k in again}
                state['reused']=len(confirmed)
        runs.append(dict(mode=mode,first=first,**state))
    return runs


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    atomic_json(args.output,measure(read_json(args.tiles)))
