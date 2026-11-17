"""Screen actual regional CP calls for identical or relabelled physical repair queries."""
import argparse
from collections import Counter
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import numpy as np

from io_utils import atomic_json
from r28_exchange import read_json
from r31_fresh import Engine,joint_repair
from search_problem import SearchProblem


def digest(item):
    return sha256(repr(item).encode('ascii')).hexdigest()


def keys(model,board,cells,options):
    cells=sorted(set(map(int,cells)))
    local=set(cells)
    tiles=board[cells]//3
    codes=3*tiles[:,None]+np.arange(3)
    forbidden=options.get('forbidden')
    barred=set()
    if forbidden is not None:
        barred={(c,v) for c,v,n in forbidden.blocked_codes(board) if c in local and n not in local}
    contacts=options.get('contacts')
    if contacts is None:
        contacts=-1 if options.get('disjoint') else int(options.get('adjacent',False))
    base=(contacts,bool(options.get('disjoint')),forbidden is not None)
    rows=[]
    degree=[]
    boundary=[]
    for i,c in enumerate(cells):
        gains=np.zeros(codes.shape,dtype=int)
        neighbors=model.problem.neighbors[c]
        degree.append(sum(int(n) in local for n in neighbors))
        allow=model.legal[c,codes].copy()
        for s,n in enumerate(neighbors):
            if n<0 or (contacts!=0 and n in local):continue
            d=model.problem.sides[c,s]
            gains+=(model.masks[codes,s]==model.reverse[model.masks[board[n],d]])
            boundary.append((c,s,int(board[n])))
        # Per-slot shifts do not change any optimum. The previous orientation
        # has to map to its old partner because the model uses changed flags.
        gains-=gains[i,board[c]%3]
        row=[]
        for j,t in enumerate(tiles):
            labels=tuple(sorted((int(gains[j,r]),bool(allow[j,r] and (c,int(codes[j,r])) not in barred),
                                 i==j and board[c]%3==r) for r in range(3)))
            row.append(labels)
        rows.append(row)
    # Both codes and coordinates may be relabelled. Keep only invariants under
    # such permutations; a different signature proves no equivalent residual.
    colors=[digest((degree[i],rows[i][i],sorted(rows[i]),sorted(row[i] for row in rows))) for i in range(len(cells))]
    flavors=[]
    for i,c in enumerate(cells):
        neighbor_colors=sorted(colors[cells.index(n)] for n in model.problem.neighbors[c] if n in local)
        values=sorted((colors[j],rows[i][j],rows[j][i]) for j in range(len(cells)))
        flavors.append(digest((colors[i],neighbor_colors,values)))
    physical=(base,tuple(cells),tuple(int(x) for x in board[cells]),tuple(boundary))
    inventory=(base,tuple(sorted(int(x) for x in tiles)))
    return digest(physical),digest(inventory),digest((base,sorted(flavors)))


def verify_resume(problem,seed,output):
    def prepare():
        return Engine(problem,bonds=True,joint=True,cycle_budget=3,exact_packing=True,forest=True,
                      pool=8,disjoint=True,joint_cache=True)
    states=[]
    start=prepare()
    initial=start.run(seed,rounds=36,save=lambda state:states.append(state))
    checkpoint=states[-1]
    continued=prepare().run(seed,rounds=64,saved=checkpoint)
    cold=prepare().run(seed,rounds=64)
    for name in ('codes','score','work','joint_residual_cache'):
        if continued[name]!=cold[name]:raise AssertionError(('nonreproducible whole residual',name))
    physical=problem.validate(continued['codes'])
    assert physical['matched_edges']==continued['score']
    result=dict(seed=seed,initial_score=initial['score'],final_score=continued['score'],
                checkpoint_round=checkpoint['cursor'],final_round=64,cold_equal=True,
                checkpoint_payload_characters=len(checkpoint['joint_residual_cache']['payload']),
                physical_hash=digest(continued['codes']),work=continued['joint_residual_cache'])
    atomic_json(output,result)
    print(result,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',nargs='+',type=int,default=[3500,3501,3502,3503])
    parser.add_argument('--rounds',type=int,default=96)
    parser.add_argument('--bound',choices=('dual','supports','exact'))
    parser.add_argument('--profiles',nargs='+',default=('random-contact','forest-disjoint','regional','guarded-contact'))
    parser.add_argument('--use-cache',action='store_true')
    parser.add_argument('--verify-resume',action='store_true')
    args=parser.parse_args()
    p=SearchProblem(read_json(args.tiles))
    if args.verify_resume:return verify_resume(p,args.seeds[0],args.output)
    cases=[]
    for profile in args.profiles:
        for seed in args.seeds:
            engine=Engine(p,bonds=True,joint=True,joint_bound=args.bound,cycle_budget=3,exact_packing=True,forest=profile=='forest-disjoint',
                          pool=8 if profile=='forest-disjoint' else 10,disjoint=profile=='forest-disjoint',
                          contacts=1 if profile.endswith('contact') else None,
                          regional_cycle_budget=2 if profile=='regional' else 0,
                          forbidden_repair='guarded' if profile=='guarded-contact' else None)
            records=[]
            from r35_joint_residual import JointResidualCache
            cache=JointResidualCache()
            def observed(model,values,cells,**kwargs):
                a,b,c=keys(model,values,cells,kwargs)
                answer,work=(cache.solve(joint_repair,model,values,cells,**kwargs) if args.use_cache else joint_repair(model,values,cells,**kwargs))
                records.append(dict(physical=a,inventory=b,invariant=c,status=work['status'],branches=work['branches'],gain=int(model.matches(answer).sum()-model.matches(values).sum())))
                return answer,work
            from bench_r35_verify_contact_bounds import independent
            from r35_contact_bound import no_contact_gain
            checks=Counter()
            def verify_bound(model,board,cells,**kw):
                closed,work=no_contact_gain(model,board,cells,**kw)
                if closed or kw['witness']:
                    _,optimum=independent(model,board,cells,kw['contacts'],kw['disjoint'],kw['forbidden'])
                    if optimum!=work.get('physical_gain',0):raise ArithmeticError('Independent region optimum differs.')
                    checks['certificates_checked']+=1
                    checks['positive_optima']+=int(optimum>0)
                return closed,work
            with patch('r31_fresh.joint_repair',observed),patch('r35_contact_bound.no_contact_gain',verify_bound):
                result=engine.run(seed,rounds=args.rounds)
            check=p.validate(result['codes'])
            assert result['score']==check['matched_edges']
            cases.append(dict(profile=profile,seed=seed,score=result['score'],codes=digest(result['codes']),cache=dict(cache.work),checks=dict(checks),bound=args.bound,
                              bound_work={k:v for k,v in result['work'].items() if k.startswith('contact_bound_') or k=='joint_CONTACT_OPTIMAL'},regional_proofs=result['work'].get('joint_CIRCULATION_OPTIMAL',0),records=records))
            atomic_json(args.output,dict(input=p.fingerprint,rounds=args.rounds,cases=cases))
            print(profile,seed,len(records),result['score'],flush=True)


if __name__=='__main__':
    main()
