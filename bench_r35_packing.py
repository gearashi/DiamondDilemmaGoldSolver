"""Original 160-piece controls for fixed-code contact augmentation and conflict-graph structure."""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r30_fresh import fresh_start
from r31_fresh import AssignmentPrices,simultaneous_swaps
from search_problem import SearchProblem


def run(problem,path,starts,depth):
    records = []
    model = BoardModel(problem)
    for index in range(starts):
        seed = 35340+index
        rng = np.random.default_rng(seed)
        board = fresh_start(model,rng)
        if index%2:
            for _ in range(8):
                board = model.block(board,model.independent_set(board,rng),rng)
        for extent,allowed in [('board',model.free),('pool',rng.choice(model.free,30,replace=False))]:
            prices = AssignmentPrices(model)
            scores = {}
            costs = {}
            for label,touch,limit in [('released',None,0),('neutral','neutral',0),('priced','priced',0),
                                      ('separator',None,depth),('both','priced',depth),('complete','complete',0)]:
                work = Counter()
                candidate,count = simultaneous_swaps(model,board,allowed,prices,exact=True,touch=touch,
                                                     separator_depth=limit,packing_work=work)
                scores[label] = int(problem.validate(model.legal_board(candidate).tolist())['matched_edges'])
                costs[label] = dict(work)
            if scores['neutral'] < scores['released'] or scores['priced'] < scores['neutral']:
                raise ArithmeticError('Augmentation lost a released move.')
            if scores['separator'] < scores['released'] or scores['both'] < scores['separator']:
                raise ArithmeticError('Verified conflict solving lost a released solution.')
            records.append(dict(seed=seed,polished=bool(index%2),extent=extent,
                                initial=int(model.matches(board).sum()),candidates=count,
                                scores=scores,work=costs))
            atomic_json(path,dict(input=problem.fingerprint,records=records))
            print(f'{index+1}/{starts} {extent} {count} {scores}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--starts',type=int,default=24)
    parser.add_argument('--depth',type=int,default=3)
    args=parser.parse_args()
    run(SearchProblem(read_json(args.tiles)),args.output,args.starts,args.depth)


if __name__=='__main__':
    main()
