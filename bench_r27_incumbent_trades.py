"""Look for 208 by chaining independent exact two-cell swaps and plateau ties."""
import argparse,heapq,itertools,json,time
from pathlib import Path
import numpy as np
from r27_incumbent import BoardModel
from search_problem import SearchProblem


def frontier(model,start,*,seed=0,max_boards=5000,seconds=300,target=208,emit=None):
    rng=np.random.default_rng(seed)
    mask=model.legal.copy()
    pair=np.triu(np.ones((model.n,model.n),dtype=bool),1)
    for a,_,b,_ in model.problem.edges:
        pair[a,b]=pair[b,a]=False
    pair[list(model.problem.fixed)]=False
    pair[:,list(model.problem.fixed)]=False
    row=np.arange(model.n)
    deadline=time.monotonic()+seconds
    board=model.legal_board(start)
    best=board.copy()
    best_pair=(int(model.matches(best).sum()),-model.problem.validate(best.tolist())['line_components'])
    queue=[]
    serial=itertools.count()
    seen={tuple(board)}
    heapq.heappush(queue,(-best_pair[0],0,next(serial),board))
    saved={}
    scored_pairs=0
    expanded=0
    while queue and expanded<max_boards and time.monotonic()<deadline:
        neg,_,_,board=heapq.heappop(queue)
        score=-neg
        expanded+=1
        choices=3*(board//3)[:,None]+np.arange(3)
        weights=np.zeros((model.n,model.n,3),dtype=np.int8)
        for s in range(3):
            nb=model.problem.neighbors[:,s]
            other=model.problem.sides[:,s]
            side=model.reverse[model.masks[board[nb],other]]
            weights+=(nb>=0)[:,None,None] & (model.masks[choices,s]==side[:,None,None])
        weights[~mask[row[:,None,None],choices[None,:,:]]]=-32
        cost=weights.max(axis=2)
        original=weights[row,row,board%3]
        delta=cost+cost.T-original[:,None]-original[None,:]
        eligible=pair & (delta>=max(-2,best_pair[0]-score-2))
        scored_pairs+=int(pair.sum())
        left,right=np.nonzero(eligible)
        order=rng.permutation(len(left))
        order=order[np.argsort(-delta[left[order],right[order]],kind='stable')]
        for j in order[:96]:
            a,b=int(left[j]),int(right[j])
            # Each tie is a separate physical orientation assignment.
            for ra,rb in itertools.product(np.flatnonzero(weights[a,b]==cost[a,b]),np.flatnonzero(weights[b,a]==cost[b,a])):
                child=board.copy()
                child[a],child[b]=choices[b,ra],choices[a,rb]
                key=tuple(child)
                if key in seen:
                    continue
                seen.add(key)
                s=score+int(delta[a,b])
                if s>=best_pair[0]:
                    report=model.problem.validate(child.tolist())
                    if report['matched_edges']!=s:
                        raise RuntimeError('Independent swap score mismatch.')
                    quality=(s,-report['line_components'])
                    if quality>best_pair:
                        best_pair=quality
                        best=child
                        if emit:
                            emit(child,report,expanded)
                if s>=target:
                    return best,dict(expanded=expanded,unique=len(seen),pair_scores=scored_pairs)
                # Keep a broad but bounded set at each attained score.
                if saved.get(s,0)<max_boards*3:
                    heapq.heappush(queue,(-s,rng.random(),next(serial),child))
                    saved[s]=saved.get(s,0)+1
    return best,dict(expanded=expanded,unique=len(seen),pair_scores=scored_pairs)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=180.)
    p.add_argument('--boards',type=int,default=4000)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--incumbent',default='audit/r27/incumbent.json')
    opts=p.parse_args()
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8-sig'))
    problem=SearchProblem(read('data/tiles.json'))
    model=BoardModel(problem)
    initial=read(opts.incumbent)['codes']
    def emit(board,info,step):
        opts.output.write_text(json.dumps(dict(codes=board.tolist(),score=info['matched_edges'],
                    components=info['line_components'],expanded=step),indent=2),encoding='utf-8')
        print('record',info['matched_edges'],info['line_components'],step,flush=True)
    begin=time.perf_counter()
    best,stats=frontier(model,initial,seed=opts.seed,max_boards=opts.boards,seconds=opts.seconds,emit=emit)
    result=problem.validate(best.tolist())
    opts.output.write_text(json.dumps(dict(codes=best.tolist(),score=result['matched_edges'],
        components=result['line_components'],**stats,seconds=time.perf_counter()-begin),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

