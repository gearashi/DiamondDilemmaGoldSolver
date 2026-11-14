"""Reuse rotation elimination messages under a consistent cyclic frame at each cell."""
import argparse
import json

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import read_json
from r28_exchange_rotations import RotationOracle
from search_problem import SearchProblem


class CyclicRotationOracle(RotationOracle):
    def _eliminate(self, tiles, domains, edges, max_entries, ties, limit):
        incident=[[] for _ in domains]
        for a,b,table in edges:
            incident[a].append((b,table))
            incident[b].append((a,table.T))
        # All incidences of a cell use the same cyclic frame; ordering by
        # neighbor ID is stable under rotation relabeling of either endpoint.
        colors=[{r:0 for r in d} for d in domains]
        for _ in range(2):
            patterns=[]
            for c,d in enumerate(domains):
                patterns.append({r:tuple(tuple(sorted((colors[b][t],int(tab[r,t])) for t in domains[b]))
                                         for b,tab in sorted(incident[c],key=lambda pair:pair[0]))
                                 for r in d})
            ids={s:i for i,s in enumerate(sorted({s for pattern in patterns for s in pattern.values()}))}
            colors=[{r:ids[s] for r,s in pattern.items()} for pattern in patterns]
        mapped=[]
        for d,color in zip(domains,colors):
            d=tuple(d)
            mapped.append(min((d[i:]+d[:i] for i in range(len(d))),
                              key=lambda seq:tuple(color[r] for r in seq)))
        # Reconstruction in the existing oracle indexes these actual rotation labels.
        return super()._eliminate(tiles,mapped,edges,max_entries,ties,limit)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--boards',nargs='+',required=True)
    parser.add_argument('--max-entries',type=int,default=59049)
    args=parser.parse_args()
    model=BoardModel(SearchProblem(read_json(args.tiles)))
    solver=CyclicRotationOracle(model)
    for path in args.boards:
        board=read_json(path)['codes']
        answer=solver.optimum(board,max_entries=args.max_entries)
        if answer is None:
            print(json.dumps(dict(board=path,status='unknown',work=solver.work)))
            continue
        score,rows=answer
        chosen=next(rows)
        model.legal_board(chosen)
        if model.problem.validate(chosen)['matched_edges']!=score:
            raise ArithmeticError('Rotation score disagrees with independent validation.')
        print(json.dumps(dict(board=path,score=score,codes=chosen,work=solver.work)))


if __name__=='__main__':
    main()
