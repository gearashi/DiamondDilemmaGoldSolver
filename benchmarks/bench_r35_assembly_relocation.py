"""Measure displaced-piece pricing and coordinated mandatory-assembly relocation on original boards."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np

from io_utils import atomic_json
from r28_exchange import read_json
from r30_fresh import fresh_start
from r31_fresh import Engine,physical_bonds
from r35_assembly_relocation import AssemblyRelocator
from search_problem import SearchProblem


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seeds',type=int,nargs='+',default=list(range(35380,35404)))
    args=ap.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    engine=Engine(problem,bonds=True,joint=True,cycle_budget=3)
    a=engine.assemblies
    m=engine.model
    relocator=AssemblyRelocator(a)
    rows=[]
    for seed in args.seeds:
        rng=np.random.default_rng(seed)
        board=a.seed(fresh_start(m,rng),rng)
        if seed%2:
            prices=engine.repair_prices()
            for step in range(8):
                board=np.asarray(engine.repair(board,rng,step,Counter(),prices)['codes'])
        for trials in (1,8):
            trial_seed=int(rng.integers(2**63))
            baseline=a.relocate(board,np.random.default_rng(trial_seed),trials=trials)
            results={'released':dict(score=int(m.matches(baseline).sum()))}
            for name in ('priced','reciprocal','mutual'):
                candidate,work=relocator.relocate(board,np.random.default_rng(trial_seed),name,trials=trials)
                bonds=physical_bonds(m,candidate)
                if not a.bonds<=bonds or a.forbidden&bonds:raise ArithmeticError('Physical mate violation.')
                score=problem.validate(m.legal_board(candidate).tolist())['matched_edges']
                if score<list(results.values())[-1]['score']:raise ArithmeticError('Lost a feasible placement.')
                results[name]=dict(score=score,work=work)
            rows.append(dict(seed=seed,polished=bool(seed%2),trials=trials,initial=int(m.matches(board).sum()),profiles=results))
            atomic_json(args.output,dict(input=problem.fingerprint,records=rows))
        print(seed,{k:v['score'] for k,v in results.items()},flush=True)


if __name__=='__main__':
    main()
