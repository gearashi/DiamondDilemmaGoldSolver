"""Compare complete tied response transport under small-component vertex relabeling."""
import argparse
from collections import Counter
from itertools import permutations
from pathlib import Path

import numpy as np

from bench_r35_rotation_components import cases
from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r28_exchange_rotations import RotationOracle
from r35_rotation_components import rotation_frame
from search_problem import SearchProblem


class VertexResponses(RotationOracle):
    def __init__(self,model,limit,ordered):
        super().__init__(model)
        self.component_cache.vertices=0
        self.limit,self.ordered=limit,ordered
        self.vertex={}
        self.extra=Counter()

    def _factor_trace(self,widths,factors,plan,cap):
        if len(plan)>self.limit or any(product>cap for _,_,product in plan):
            return super()._factor_trace(widths,factors,plan,cap)
        cells=[c for c,_,_ in plan]
        ind={c:i for i,c in enumerate(cells)}
        unary={s[0]:v for s,v in factors if len(s)==1}
        edges=[(ind[a],ind[b],v) for (a,b),v in ((s,v) for s,v in factors if len(s)==2)]
        offset=sum(int(u[0]) for u in unary.values())
        values=[tuple(map(int,unary[c]-unary[c][0])) for c in cells]
        if self.ordered:
            def signature(i):
                incident=[v if a==i else v.T for a,b,v in edges if i in (a,b)]
                return (tuple(sorted(values[i])),len(incident),
                        tuple(sorted(tuple(sorted(map(int,tab.ravel()))) for tab in incident)))
            options=[tuple(sorted(range(len(cells)),key=signature))]
        else:options=permutations(range(len(cells)))
        best,order,perms=None,None,None
        for proposed in options:
            self.extra['vertex_orders']+=1
            index={c:i for i,c in enumerate(proposed)}
            numer=[]
            for a,b,v in edges:
                a,b=index[a],index[b]
                if a>b:a,b,v=b,a,v.T
                numer.append((a,b,v.astype('<i4').tobytes()))
            candidate,ranks=rotation_frame((tuple(values[i] for i in proposed),tuple(sorted(numer)),()))
            if best is None or candidate<best:
                best,order,perms=candidate,proposed,ranks
        found=self.vertex.get(best)
        if found is None:
            score,rows=super()._factor_trace(widths,factors,plan,cap)
            chosen={}
            winners=[]
            def backward(i):
                if i<0:
                    winners.append(tuple(perms[j].index(chosen[cells[q]]) for j,q in enumerate(order)))
                    return
                c,rest,masks=rows[i]
                mask=int(masks[tuple(chosen[z] for z in rest)])
                for r in range(widths[c]):
                    if mask&(1<<r):
                        chosen[c]=r
                        backward(i-1)
            backward(len(rows)-1)
            self.extra['winners_encoded']+=len(winners)
            self.vertex[best]=(score-offset,winners)
            return score,rows
        score,tuples=found
        self.extra['vertex_hits']+=1
        self.extra['winners_decoded']+=len(tuples)
        rows=[(c,tuple(z for z in scope if z!=c),np.zeros(tuple(widths[z] for z in scope if z!=c),np.uint8))
              for c,scope,_ in plan]
        for win in tuples:
            bycell={cells[q]:perms[j][x] for j,(q,x) in enumerate(zip(order,win))}
            for c,rest,arr in rows:
                arr[tuple(bycell[z] for z in rest)]|=np.uint8(1<<bycell[c])
        return score+offset,rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    model=BoardModel(SearchProblem(read_json(args.tiles)))
    src=model.legal_board(read_json(args.board)['codes'])
    report={}
    for seed in (851,7407):
        for name,queries in cases(model,src,seed).items():
            out={}
            reference=None
            for limit,ordered in ((0,True),(2,True),(3,True),(3,False)):
                oracle=VertexResponses(model,limit,ordered)
                output=[]
                for board,cap in queries:
                    result=oracle.optimum(board,max_entries=cap,ties=True,limit=4)
                    output.append(None if result is None else (result[0],list(result[1])))
                if reference is None:reference=output
                elif output!=reference:raise ArithmeticError('Vertex response mismatch.')
                key=str(limit)+('-ordered' if ordered else '-permuted')
                out[key]=dict(oracle.extra,keys=oracle.work['numeric_keys'],entries=oracle.work['messages'])
            report[f'{seed}-{name}']=out
            atomic_json(args.output,dict(input=model.problem.fingerprint,groups=report))
            print(seed,name,out,flush=True)


if __name__=='__main__':main()
