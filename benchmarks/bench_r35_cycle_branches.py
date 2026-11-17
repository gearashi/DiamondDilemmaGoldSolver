"""Compare overlapping and disjoint circulation relaxations on the same legal physical boards."""
from collections import Counter
from contextlib import nullcontext
from unittest.mock import patch
import argparse
from pathlib import Path

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r28_exchange_patches import AssignmentPrices
from r30_fresh import fresh_start
from r31_fresh import forest_pool, pack_cycles, joint_repair
from search_problem import SearchProblem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=list(range(35360,35372)))
    parser.add_argument('--profiles',nargs='+',choices=('overlap','disjoint','neighbors','warm'),default=('overlap','disjoint','neighbors','warm'))
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--pool-budget',type=int)
    parser.add_argument('--audit-dual',action='store_true')
    parser.add_argument('--direct-cp',action='store_true')
    parser.add_argument('--identity',action='store_true')
    args=parser.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    model=BoardModel(problem)
    rows=[]
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        board=fresh_start(model,rng)
        if seed%2:
            for _ in range(8):
                board=model.block(board,model.independent_set(board,rng),rng)
        pools=[('random8',rng.choice(model.free,8,replace=False),512),
               ('forest12',forest_pool(model,board,model.free,12,rng),512),
               ('forest22',forest_pool(model,board,model.free,22,rng),256)]
        if args.full:
            pools=[('full3',model.free,3),('full24',model.free,24)]
        for scope,cells,budget in pools:
            if args.pool_budget and not args.full:
                budget=args.pool_budget
            data={}
            for name in args.profiles:
                work=Counter()
                from r35_restricted_assignment import matching
                def compare(weights,allowed,previous=None):
                    result,metrics=matching(weights,allowed,previous)
                    cold,baseline=matching(weights,allowed)
                    if (result is None)!=(cold is None):
                        raise ArithmeticError('Warm/cold assignment feasibility differs.')
                    if result is not None:
                        positions=np.arange(len(weights))
                        if weights[positions,result[0]].sum()!=weights[positions,cold[0]].sum():
                            raise ArithmeticError('Warm/cold assignment optima differ.')
                    work['counterfactual_cold_augmentations']+=baseline['augmentations']
                    work['counterfactual_cold_alternating_columns']+=baseline['alternating_columns']
                    work['paired_laps']+=1
                    return result,metrics
                audit=(patch('r35_restricted_assignment.matching',compare)
                       if name=='warm' and args.audit_dual else nullcontext())
                with audit:
                    candidate,report=pack_cycles(model,board,cells,AssignmentPrices(model),relaxations=budget,
                                                 partition=None if name=='overlap' else name,audit=work,identity_check=args.identity)
                score=problem.validate(model.legal_board(candidate).tolist())['matched_edges']
                data[name]=dict(score=score,**report,work=dict(work))
            completed={v['score'] for v in data.values() if v['complete']}
            if len(completed)>1 or any(v['score']>min(completed) for v in data.values() if completed):
                raise ArithmeticError('Physical optima differ between branchings.')
            direct=None
            if args.direct_cp:
                values,cp=joint_repair(model,board,cells)
                score=problem.validate(model.legal_board(values).tolist())['matched_edges']
                if completed and (score>min(completed) or (cp['status']=='OPTIMAL' and score!=min(completed))):
                    raise ArithmeticError('Regional and direct CP optima differ.')
                direct=dict(score=score,**cp)
            rows.append(dict(seed=seed,scope=scope,cells=cells.tolist(),budget=budget,direct_cp=direct,
                             initial=int(model.matches(board).sum()),profiles=data))
            atomic_json(args.output,dict(input=problem.fingerprint,records=rows))
            print(seed,scope,[(k,v['score'],v['relaxations'],v['complete']) for k,v in data.items()],flush=True)


if __name__=='__main__':
    main()
