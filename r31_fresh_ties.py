"""Find the smallest component count within a fixed independent-set assignment score gap."""
import argparse
import json

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r28_exchange_patches import AssignmentPrices, joint_patch
from r31_fresh import result
from search_problem import SearchProblem


def optimize_ties(model,board,cells,max_calls=12,seconds=.5):
    """Challenge a large equality graph with exact threshold/forest feasibility checks."""
    if max_calls<1 or seconds<=0:
        raise ValueError('Positive per-layer call and time budgets required.')
    board=model.legal_board(board)
    cells=np.asarray(cells,dtype=int)
    prices=AssignmentPrices(model)
    prices.update(board)
    optimal=next(prices.independent(cells,limit=1))
    baseline=result(model,optimal)
    out=[]
    tiles=board[cells]//3
    grid=prices.prices[cells][:,tiles]
    dual=prices.dual
    for slack in range(3):
        rows={}
        for i,c in enumerate(cells):
            rows[int(c)]=[int(3*t+r) for j,t in enumerate(tiles) for r in range(3)
                          if int(grid[i,j,r])>=0 and dual['rows'][i]+dual['columns'][j]-int(grid[i,j,r])<=slack]
        best=baseline
        counts={}
        complete=False
        floor=baseline['score']-slack
        for _ in range(max_calls):
            if best['components']==1:
                complete=True
                break
            status,witness=joint_patch(model,best['codes'],rows,min_score=floor,
                                       max_components=best['components']-1,seconds=seconds)
            counts[status]=counts.get(status,0)+1
            if witness is None:
                complete=status=='INFEASIBLE'
                break
            best=result(model,witness)
        out.append(dict(slack=slack,score_floor=floor,first_components=baseline['components'],
                        best_components=best['components'],score=best['score'],closed=complete,calls=counts,codes=best['codes']))
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--board',required=True)
    p.add_argument('--cells',required=True,type=int,nargs='+',help='Nonadjacent editable cell indices.')
    p.add_argument('--seconds',type=float,default=.5,help='Time for each tighter component query.')
    p.add_argument('--calls',type=int,default=12,help='Maximum queries per score layer.')
    p.add_argument('--output')
    args=p.parse_args()
    model=BoardModel(SearchProblem(read_json(args.tiles)))
    report=optimize_ties(model,read_json(args.board)['codes'],args.cells,args.calls,args.seconds)
    if args.output:
        winner=min(report,key=lambda row:(row['best_components'],-row['score']))
        r=result(model,winner['codes'])
        atomic_json(args.output,dict(codes=r['codes'],matched_edges=r['score'],line_components=r['components'],valid=r['gold']))
    print(json.dumps([{k:v for k,v in row.items() if k!='codes'} for row in report]))


if __name__=='__main__':
    main()
