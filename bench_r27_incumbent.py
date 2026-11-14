"""Original-board, perturbed and independently randomized matched controls."""
import argparse,json,time
from pathlib import Path
import numpy as np
from r27_incumbent import BoardModel
from bench_r27_incumbent_population import construct, ensemble, geometric_children, project, propose, recombine
from search_problem import SearchProblem


def metrics(problem, model, values):
    board=model.legal_board(values)
    info=problem.validate(board.tolist())
    assert info['matched_edges']==int(model.matches(board).sum())
    return dict(score=info['matched_edges'],components=info['line_components'],gold=info['valid'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',type=int,default=4)
    parser.add_argument('--proposals',type=int,default=512)
    parser.add_argument('--rounds',type=int,default=24)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8-sig'))
    problem=SearchProblem(read('data/tiles.json'))
    model=BoardModel(problem)
    incumbent=model.legal_board(read('audit/r27/incumbent.json')['codes'])
    results=[]
    record=(0,0)
    def save(row):
        results.append(row)
        args.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    for seed in range(args.seeds):
        for start_name in ('incumbent','perturbed','fresh'):
            rng=np.random.default_rng(seed)
            start=incumbent.copy()
            if start_name=='fresh':
                start=project(model,rng.random(model.legal.shape),rng)
            if start_name=='perturbed':
                for _ in range(18):
                    start=propose(model,start,rng)
            initial=metrics(problem,model,start)
            for method in ('rotations','block','alternating','patch','exchange','tabu','penalty','archive','niche','population','flat','cross_entropy','belief','belief_decimation','fractional','geometric','independent_parent'):
                began=time.perf_counter()
                stats={}
                rng=np.random.default_rng(seed)
                best=start.copy()
                quality=initial
                def offer(values,count):
                    nonlocal best,quality
                    quality2=metrics(problem,model,values)
                    if (quality2['score'],-quality2['components'])>(quality['score'],-quality['components']):
                        quality,best=quality2,values.copy()
                        if 'first208' not in stats and quality['score']>=208:
                            stats['first208']=count
                if method in ('rotations','block','alternating'):
                    at=start.copy()
                    exact=0
                    entries=0
                    for j in range(args.rounds):
                        if method!='rotations':
                            cells=model.independent_set(at,rng)
                            at=model.block(at,cells,rng)
                            entries+=3*len(cells)**2
                        if method!='block':
                            at,info=model.orientations(at,max_entries=59049,rng=rng)
                            exact+=int(info['exact'])
                            entries+=info['table_entries']
                            stats['last_bound']=info['upper_bound']
                        offer(at,j+1)
                    stats.update(rounds=args.rounds,exact_orientation_rounds=exact,table_entries=entries)
                elif method=='patch':
                    from constraint_lns import repair
                    result=repair(problem,start.tolist(),seconds=5,per_attempt_seconds=.2,max_attempts=args.rounds,seed=seed)
                    offer(np.array(result['best_board']),result['stats']['attempts'])
                    stats.update(attempts=result['stats']['attempts'],patch_nodes=result['stats'].get('nodes'))
                elif method in ('belief','belief_decimation','fractional'):
                    tried,work=construct(model,start,rng,'belief' if method.startswith('belief') else method,
                                         damping=.15 if seed%2 else .65,decimate=method=='belief_decimation')
                    stats.update(work=work,constructed=metrics(problem,model,tried))
                    offer(tried,1)
                elif method in ('geometric','independent_parent'):
                    if method=='geometric':
                        children=geometric_children(model,start,rng,16)
                    else:
                        other=start.copy()
                        for _ in range(6):
                            other=propose(model,other,rng)
                        # Build a nearby independent parent from the same starting quality.
                        for _ in range(3):
                            other=model.block(other,model.independent_set(other,rng),rng)
                        stats['second_parent_score']=int(model.matches(other).sum())
                        children=[recombine(start,other,rng) for _ in range(16)]
                    stats.update(children=len(children),best_child=0,post_repair=0)
                    for i,child in enumerate(children):
                        stats['best_child']=max(stats['best_child'],int(model.matches(child).sum()))
                        child,_=model.orientations(child,max_entries=19683,rng=rng)
                        stats['post_repair']=max(stats['post_repair'],int(model.matches(child).sum()))
                        offer(child,i+1)
                else:
                    res=ensemble(model,start,seed,method,proposals=args.proposals)
                    offer(res['best'],res['steps'])
                    stats.update(proposals=res['steps'],first208=res['first208'],accepted=res['accepted'],extra=res['extra'])
                item=dict(seed=seed,start=start_name,method=method,initial=initial,
                          **quality,**stats,seconds=round(time.perf_counter()-began,5))
                if (quality['score'],-quality['components'])>record:
                    record=(quality['score'],-quality['components'])
                    (args.output.parent/'incumbent-record.json').write_text(json.dumps(dict(codes=best.tolist(),**item),indent=2),encoding='utf-8')
                save(item)
            print(seed,start_name,len(results),record,flush=True)


if __name__=='__main__':
    main()

