"""Finite train/held-out ranking experiment; predictions never prove cuts.

Run serially. Disjoint predeclared fixture seeds prevent train/test leakage.
Work labels require a completed exact child search, not a censored timeout.
No weights from this experiment are automatically installed in the app.
"""
import argparse,hashlib,json,platform,time
from pathlib import Path
from unittest.mock import patch
from search_fixtures import planted_gold
from search_problem import SearchProblem
from constraint_dfs import solve,_solve_core
from constraint_parallel import _root_copy
from constraint_branching_round24 import LearnedOrdering,LEARNED_ORDERING_WEIGHTS,LEARNED_ORDERING_PROVENANCE

TRAIN_SEEDS=tuple(range(240310,240316))
TEST_SEEDS=tuple(range(240410,240422))
LARGE_SEEDS=tuple(range(240510,240516))
OPTIONS={'native':'python'}


def fixture(seed,size=1):
    import random
    data,board,known=planted_gold(seed,size=size,extra_edges=3*size)
    # Preserve the known path wiring while giving every occupied seam the
    # same two port positions. This removes unique mask fingerprints without
    # introducing duplicate ports or changing the known connected loop.
    for tile in data['tiles']:
        points={side:sorted({p for line in tile['segments'] for s,p in line if s==side}) for side in range(3)}
        tile['segments']=[[[s,(2,10)[points[s].index(p)]] for s,p in line] for line in tile['segments']]
    problem=SearchProblem(data,board);rng=random.Random(seed+900)
    for cell,row in enumerate(problem.domains):
        problem.domains[cell]=[code for code in row if code==known[cell] or rng.random()<.7]
    assert problem.validate(known)['valid']
    assert all(code in problem.domains[cell] for cell,code in enumerate(known))
    return problem


def checked(problem,result):
    if result['status']=='solved':assert problem.validate(result['codes'])['valid']
    return {'status':result['status'],'nodes':result['nodes'],
            'propagations':result['stats']['propagations'],'backtracks':result['stats']['backtracks'],
            'learned_order_calls':result['stats']['learned_order_calls']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=1.5)
    parser.add_argument('--evaluate-bundled',action='store_true',help='Evaluate the exact frozen production weights, without fitting a different model.')
    args=parser.parse_args();began=time.monotonic()
    files=('constraint_dfs.py','constraint_branching_round24.py','search_fixtures.py',Path(__file__).name)
    hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}
    rows=[];training=[];capped=0
    for seed in (() if args.evaluate_bundled else TRAIN_SEEDS):
        problem=fixture(seed);captured=[]
        def capture(self,cell,domains,choices,hinted,rank,check,stats):
            check()
            if len(captured)<2:captured.append((cell,tuple(domains),tuple(choices[:4])))
            return choices
        with patch.object(LearnedOrdering,'order',capture):
            result=solve(problem,seconds=args.seconds,seed=seed,options={**OPTIONS,'learned_ordering':True})
        training.append({'seed':seed,'baseline':checked(problem,result),'captured_states':len(captured)})
        model=LearnedOrdering(problem)
        base=_solve_core(problem,seconds=0,options=OPTIONS)['checkpoint']
        for state,(cell,domains,choices) in enumerate(captured):
            for code in choices:
                child=list(domains);child[cell]=1<<code
                result=solve(problem,seconds=args.seconds,seed=seed,resume=_root_copy(base,child),options=OPTIONS)
                checked(problem,result)
                if result['complete']:
                    rows.append({'seed':seed,'state':state,'cell':cell,'code':code,
                                 'features':model.features(cell,domains,code),'work':result['nodes'],
                                 'status':result['status']})
                else:capped+=1
        print(json.dumps({'stage':'train','seed':seed,'completed_labels':len(rows),'capped':capped}),flush=True)
    model=({'weights':list(LEARNED_ORDERING_WEIGHTS),'provenance':LEARNED_ORDERING_PROVENANCE} if args.evaluate_bundled else LearnedOrdering.fit(rows,ridge=1.0) if rows else None)
    heldout=[]
    for size,seed in [(1,seed) for seed in TEST_SEEDS]+[(2,seed) for seed in LARGE_SEEDS]:
        problem=fixture(seed,size)
        base=solve(problem,seconds=args.seconds,seed=seed,options=OPTIONS)
        if model is not None:
            factory=lambda problem,weights=None:LearnedOrdering(problem,model['weights'])
            with patch('constraint_dfs.LearnedOrdering',factory):
                result=solve(problem,seconds=args.seconds,seed=seed,options={**OPTIONS,'learned_ordering':True})
            row={'seed':seed,'cells':problem.n,'baseline':checked(problem,base),'learned':checked(problem,result)}
        else:row={'seed':seed,'cells':problem.n,'baseline':checked(problem,base),'learned':None}
        heldout.append(row);print(json.dumps({'stage':'held-out',**row}),flush=True)
    comparable=[r for r in heldout if r['learned'] and r['baseline']['status']=='solved' and r['learned']['status']=='solved']
    summary={'solved_pairs':len(comparable),'baseline_nodes':sum(r['baseline']['nodes'] for r in comparable),
             'learned_nodes':sum(r['learned']['nodes'] for r in comparable),
             'wins':sum(r['learned']['nodes']<r['baseline']['nodes'] for r in comparable),
             'ties':sum(r['learned']['nodes']==r['baseline']['nodes'] for r in comparable),
             'losses':sum(r['learned']['nodes']>r['baseline']['nodes'] for r in comparable)}
    after={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in files}
    report={'candidate':31,'scope':'10-cell training and held-out 10/40-cell synthetic Gold fixtures, compressed seam masks and planted feasible domains',
            'protocol':'ridge=1; minimize predicted log1p(completed child nodes); one frozen fit, held-out seeds never train',
            'train_seeds':TRAIN_SEEDS,'heldout_seeds':TEST_SEEDS,'large_heldout_seeds':LARGE_SEEDS,'model':model,'training_cases':training,
            'training_labels':rows,'censored_labels_excluded':capped,'heldout':heldout,'summary':summary,
            'seconds_per_query':args.seconds,'wall_seconds':time.monotonic()-began,'python':platform.python_version(),
            'source_hashes':hashes,'sources_unchanged':hashes==after,
            'non_claim':'Finite work comparison only. No original-puzzle generalization or proof of optimal ordering.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
