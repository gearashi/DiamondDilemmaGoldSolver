"""Measure sampler clocks, patch-query reuse, and cache admission on puzzle inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import tempfile

import numpy as np
from constraint_lns import _Control, _load_cp_model, _solve_patch
from constraint_work import WorkScope
from constraint_symbolic_frontier import SymbolicFrontier
from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_cache_policy import offline_bound, replay
from r28_patch_cache import exterior, transplant
from r28_sampling import discrepancy, jump, plateau_step
from r28_swap_walk import PairScores
from search_problem import SearchProblem


def policies(model, board):
    pairs = PairScores(model)
    grid = pairs.all(board)
    old_score = int(model.matches(board).sum())
    temperatures = {}
    for temp in (0., 0.05, 0.5):
        valid = grid >= 0
        rate = (grid>=old_score).sum()/grid.size if not temp else np.where(valid,np.exp(np.minimum(0,grid.astype(float)-old_score)/temp),0).mean()
        rng = np.random.default_rng(7)
        observations = []
        for _ in range(16):
            child,clock,accepted=jump(pairs,board,rng,temp,100000)
            observations.append((clock,int(accepted),int(model.matches(child).sum())))
        temperatures[str(temp)] = {'acceptance_probability':float(rate),'accepted':sum(r[1] for r in observations),
                  'proposals_on_clock':sum(r[0] for r in observations),'accepted_state_scores':[r[2] for r in observations if r[1]]}
    tie_rows = []
    for seed in range(6):
        for switch in (False,True):
            current = board.copy()
            best = old_score
            rng = np.random.default_rng(seed)
            for _ in range(40):
                current = plateau_step(pairs,current,rng,displacements=switch)
                best = max(best,int(model.matches(current).sum()))
            validation = model.problem.validate(current.tolist())
            tie_rows.append({'seed':seed,'displacement':switch,'best':best,'final':validation['matched_edges'],'components':validation['line_components']})
    return {'cold':temperatures,'ties':tie_rows}


def patch_trial(model, board):
    p = model.problem
    patch = [0,1,2,3]
    a = list(map(int,board))
    b = None
    key = exterior(p,a,patch)
    for i in range(4,p.n):
        c = a.copy()
        c[i] = 3*(a[i]//3)+(a[i]+1)%3
        if exterior(p,c,patch) == key:
            b = c
            break
    if b is None:
        raise RuntimeError('Benchmark requires distinct equivalent exterior')
    def query(center):
        diag={}
        result,status=_solve_patch(p,center,patch,cp_model=_load_cp_model(),control=_Control(30,None),seconds=15,seed=9,target='gold',diagnostics=diag)
        if status!='OPTIMAL' or result is None:
            raise RuntimeError('Incomplete patch comparison')
        return {'board':center,'patch':patch,'candidate':result,'status':status},diag.get('native_calls',0)
    prior,first = query(a)
    control,second = query(b)
    hit = transplant(p,{'board':b,'patch':patch},[prior],target='gold',loop_aware=True)
    if not hit or p.validate(hit['candidate'])['matched_edges'] != p.validate(control['candidate'])['matched_edges']:
        raise RuntimeError('Cached and cold patch optimum differ')
    return {'cached_native_calls':0,'cold_native_calls':second,'first_native_calls':first,
            'score':p.validate(hit['candidate'])['matched_edges'],'different_exterior_cell':i}


def cache_trace(model, board):
    # Record exact subproblems of the endpoint edit-distance DP. Each score
    # occupies a fixed-size trace entry; cost is the number of DP cells filled.
    trace=[]
    for ind in range(20):
        values=board.copy()
        values[ind]=values[ind]//3*3+(values[ind]+1)%3
        for a,x,b,y in model.problem.edges:
            l=int(model.masks[values[a],x]);r=int(model.reverse[model.masks[values[b],y]])
            if l==r:continue
            key=l.to_bytes(2,'little')+r.to_bytes(2,'little')
            trace.append((key,32,1+l.bit_count()*r.bit_count()))
    runs=[]
    with tempfile.TemporaryDirectory() as d:
        for size in (64,256,1024):
            for w in (False,True):
                for spill in (False,True):
                    runs.append(dict(capacity=size,weighted=w,spill=spill,**replay(trace,size,weighted=w,spill=Path(d)/'trial.db' if spill else None)))
    keys={r[0] for r in trace[:12]}
    part=[r for r in trace if r[0] in keys][:32]
    return {'requests':len(trace),'replay':runs,'oracle_on_first_32':offline_bound(part,64),
            'lru_first_32':replay(part,64),'weighted_first_32':replay(part,64,weighted=True)}


def factor_trace(model, board):
    import hashlib
    from r28_exchange_rotations import RotationOracle
    trace = []
    def observe(key, output_bytes, cost):
        # Include actual key payload bytes as well as the response, excluding
        # interpreter-specific dictionary and array headers.
        raw = repr(key).encode("ascii")
        trace.append((hashlib.sha256(raw).digest(), len(raw)+output_bytes, cost))
    oracle = RotationOracle(model,observe=observe)
    rng = np.random.default_rng(28)
    bounded = RotationOracle(model,cache_bytes=16384)
    for _ in range(24):
        trial=board.copy()
        a,b=rng.choice(model.n,2,replace=False)
        trial[a],trial[b]=board[b],board[a]
        expected=oracle.optimum(trial)
        actual=bounded.optimum(trial)
        if (expected is None)!=(actual is None) or expected is not None and expected[0]!=actual[0]:
            raise RuntimeError('Cache-policy comparison changed an optimum')
    return dict(live_bounded=bounded.work,requests=len(trace),unique=len(set(k for k,_,_ in trace)),
                capacities=[dict(capacity=size,weighted=flag,**replay(trace,size,weighted=flag))
                    for size in (16384,65536,262144,1048576) for flag in (False,True)])


def symbolic():
    # Split eight binary variables into disjoint boxed histories with a live lease.
    import itertools
    boxes=[tuple(1<<b for b in v) for v in itertools.product(range(2),repeat=8)
           if v[0]==v[5] and v[1]==v[6] and v[2]==v[7]]
    scope=WorkScope('r28-ordered-fixture','edges',(3,)*8,(3,)*8)
    f=SymbolicFrontier(scope,boxes)
    lease=f.claim('active')
    f.commit(f.claim('finished'),[])
    def held():
        roots=[f.pending,f.completed,f._lease(lease)]
        return [set(v for box in f.sets.iter_boxes(root) for v in itertools.product(*[[j for j in (0,1) if m>>j&1] for m in box])) for root in roots]
    before=held()
    results=[f.optimize(steps=150000) for _ in range(4)]
    if before!=held():raise RuntimeError('Reorder changed ownership')
    restored=SymbolicFrontier.restore(f.snapshot(),scope)
    restored.reclaim(lease)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('board',type=Path)
    parser.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args()
    model=BoardModel(SearchProblem(json.loads(a.tiles.read_text(encoding='utf-8-sig'))))
    board=model.legal_board(json.loads(a.board.read_text())['codes'])
    result={'patch':patch_trial(model,board),'sample':policies(model,board),'cache':cache_trace(model,board),'order':symbolic(),'factors':factor_trace(model,board)}
    atomic_json(a.output,result)
    print(json.dumps(result))
