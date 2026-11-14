"""Repeated perturb-and-polish around the original 207; measured original objective."""
import argparse,json,time
from pathlib import Path
import numpy as np
from r27_incumbent import BoardModel
from bench_r27_incumbent_population import propose
from search_problem import SearchProblem


def run(problem,start,seed,epochs,method):
    model=BoardModel(problem)
    original=model.legal_board(start)
    best=original.copy()
    current=best.copy()
    rng=np.random.default_rng(seed)
    best_quality=(int(model.matches(best).sum()),-problem.validate(best.tolist())['line_components'])
    penalties=np.zeros(len(model.a))
    tabu=[]
    first208=None
    extra=0
    began=time.perf_counter()
    for step in range(epochs):
        bad=np.flatnonzero(~model.matches(current))
        if not len(bad):
            break
        weights=1+penalties[bad] if method=='penalty' else np.ones(len(bad))
        ids=rng.choice(bad,size=min(len(bad),int(rng.integers(2,6))),replace=False,p=weights/weights.sum())
        cells=np.unique(np.r_[model.a[ids],model.b[ids]])
        defective=np.intersect1d(cells,model.free)
        trial=current.copy()
        for _ in range(1+step%6):
            trial=propose(model,trial,rng,defective=defective)
        for _ in range(4):
            ids=model.independent_set(trial,rng)
            trial=model.block(trial,ids,rng)
            extra+=3*len(ids)**2
        match=model.matches(trial)
        primary=int(match.sum())
        quality=primary,-problem.validate(trial.tolist())['line_components']
        if quality>best_quality:
            best=trial.copy()
            best_quality=quality
            if primary>=208 and first208 is None:
                first208=step+1
        key=trial.astype(np.uint16).tobytes()
        delta=primary-int(model.matches(current).sum())
        if method=='penalty':
            delta+=float(np.dot(penalties,match.astype(float)-model.matches(current)))
            penalties=.96*penalties+.12*(~match)
        if key not in tabu and (delta>=0 or rng.random()<np.exp(max(-60,delta/.9))):
            current=trial
            tabu.append(key)
            tabu=tabu[-16:]
        if step%16==15:
            current=best.copy()
            if method=='penalty' and step%96==95:
                penalties.fill(0)
    return dict(seed=seed,method=method,epochs=epochs,score=best_quality[0],components=-best_quality[1],
                codes=best.tolist(),first208=first208,price_terms=extra,seconds=time.perf_counter()-began)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--epochs',type=int,default=600)
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--output',type=Path,required=True)
    o=p.parse_args()
    read=lambda path:json.loads(Path(path).read_text(encoding='utf-8-sig'))
    problem=SearchProblem(read('data/tiles.json'))
    codes=read('audit/r27/incumbent.json')['codes']
    results=[]
    for seed in range(o.seeds):
        for method in ('tabu','penalty'):
            row=run(problem,codes,seed,o.epochs,method)
            results.append(row)
            print(seed,method,row['score'],row['components'],flush=True)
            o.output.write_text(json.dumps(results,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

