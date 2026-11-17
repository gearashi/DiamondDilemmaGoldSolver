"""Screen exact cycle interactions on geographic and scattered original-board pools."""
import argparse
from pathlib import Path

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r30_fresh import fresh_start
from r31_fresh import forest_pool
from r34_pair_cycles import cycle_prices, pair_repair
from r35_cycle_contacts import corrected_moves
from search_problem import SearchProblem


def benchmark(problem, count, output):
    model = BoardModel(problem)
    records = []
    for index in range(count):
        rng = np.random.default_rng(35320+index)
        board = fresh_start(model,rng)
        if index % 2:
            for _ in range(8):
                board = model.block(board,model.independent_set(board,rng),rng)
        for size,kind in ((6,'forest'),(8,'forest'),(10,'scattered')):
            cells = forest_pool(model,board,model.free,size,rng) if kind == 'forest' else rng.choice(model.free,size,replace=False)
            before,rows,singles,_ = cycle_prices(model,board,cells,3)
            released,old = pair_repair(model,board,cells,3)
            # The scored footprints of the released method come first, including
            # its fallback to positive single cycles.
            legacy,pred,_ = corrected_moves(model,rows,before,singles,1)
            expected = board.copy()
            for entry in pred:
                for c,v in entry.items():
                    expected[c] = v
            if not np.array_equal(expected,released):
                raise ArithmeticError('Compressed released policy changed its witness.')
            gain,changes,work = corrected_moves(model,rows,before,singles,3)
            result = board.copy()
            for changed in changes:
                for c,v in changed.items():
                    result[c] = v
            score = problem.validate(model.legal_board(result).tolist())['matched_edges']
            if int(before.sum())+gain != score or legacy > gain:
                raise ArithmeticError('Corrected witness failed independent scoring.')
            records.append(dict(seed=35320+index,polished=bool(index%2),pool=kind,cells=list(map(int,cells)),
                                score0=int(before.sum()),released_gain=legacy,score=score,released_work=old,work=work))
            atomic_json(output,dict(input=problem.fingerprint,records=records))
            print(f'{index+1}/{count} {kind} {size}: {legacy}/{work["gain_pairs"]}/{gain}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=16)
    args=parser.parse_args()
    benchmark(SearchProblem(read_json(args.tiles)),args.count,args.output)


if __name__=='__main__':
    main()
