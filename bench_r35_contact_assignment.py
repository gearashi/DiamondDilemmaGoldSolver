"""Test weighted matching of pure singleton support tails against exact subset layers."""
import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np

from io_utils import atomic_json
from r27_incumbent import assignment,BoardModel
from r28_exchange import read_json
from r32_gold_bonds import inventory
from r35_contact_bound import best_assignment,bounded_assignment,no_contact_gain
from r35_forbidden_mates import ForbiddenMates
from search_problem import SearchProblem


def solve(rows,units,one,pair,trace,target,prices,work,mode):
    singles=[u[0] for u in units if len(u)==1]
    paired=[u for u in units if len(u)==2]
    if len(singles)<4 or len(paired)>(1 if mode=='one-pair' else 0):
        return best_assignment(rows,units,one,pair,trace,target,prices)
    assignments=[(0,())]
    if paired:
        table=pair[paired[0]]
        assignments=[]
        for a,b in combinations(rows,2):
            d=(a,b) if table[a,b]>=table[b,a] else (b,a)
            if table[d]>-1000:assignments.append((int(table[d]),d))
    result=-1000
    winner=None
    target=max(target,0)
    for base,donors in assignments:
        remaining=[i for i in rows if i not in donors]
        matrix=one[np.ix_(singles,remaining)]
        legal=matrix>-1000
        bound=(matrix+prices[remaining][None,:]-prices[singles][:,None]).max(axis=1).sum()+base
        if donors:bound+=int(prices[list(donors)].sum()-prices[list(paired[0])].sum())
        if int(bound)<=max(target,result):continue
        work['weighted_problems']+=1
        work['weighted_rows']+=len(matrix)
        work['weighted_legal_arcs']+=int(legal.sum())
        try:perm=assignment(matrix,legal)
        except ValueError:continue
        score=base+int(matrix[np.arange(len(matrix)),perm].sum())
        if score>max(target,result):
            result=score
            if trace:
                winner={(i,):(remaining[int(j)],) for i,j in zip(singles,perm)}
                if paired:winner[paired[0]]=donors
    return result,0,winner


def memoized(rows,units,one,pair,trace,target,prices,work,cache):
    # All unit score tables are fixed within the same physical local query.
    # Remove an already assigned donor from subsequent subproblems; cache only
    # complete maxima, irrespective of the caller's current improvement target.
    def optimum(pending,available):
        if not pending:return 0
        key=pending,available
        if key in cache:
            work['exact_subproblem_hits']+=1
            return cache[key][0]
        unit=pending[0]
        rest=pending[1:]
        donors=[i for i in range(len(one)) if available>>i&1]
        if len(unit)==1:
            choices=[(int(one[unit[0],i]),(i,),1<<i) for i in donors]
        else:
            table=pair[unit]
            choices=[]
            for i,j in combinations(donors,2):
                placed=(i,j) if table[i,j]>=table[j,i] else (j,i)
                choices.append((int(table[placed]),placed,(1<<i)|(1<<j)))
        best=-1000
        winning=None
        for value,placed,mask in choices:
            if value<=-1000:continue
            tail=optimum(rest,available^mask)
            if tail<=-1000:continue
            gain=value+tail
            if gain>best:best,winning=gain,(placed,mask)
        cache[key]=best,winning
        work['exact_subproblems']+=1
        return best
    # Fix a global order so suffixes of distinct changed supports can coincide.
    units=tuple(sorted(units,key=lambda u:(-len(u),u)))
    mask=sum(1<<i for i in rows)
    score=optimum(units,mask)
    placement={}
    if score>target and trace:
        while units:
            placed,used=cache[units,mask][1]
            placement[units[0]]=placed
            units,mask=units[1:],mask^used
    return score,0,placement


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--queries',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    p=SearchProblem(read_json(args.tiles))
    model=BoardModel(p)
    inv=ForbiddenMates(model,inventory(p).forbidden)
    results=[]
    for case in read_json(args.queries)['cases']:
        counters={k:Counter() for k in ('subset','singles','one-pair','memoized','bounded-memo')}
        for r in case['rows']:
            b=model.legal_board(r['board'])
            refs=[]
            for mode,counts in counters.items():
                memo={}
                def run(*args):
                    if mode=='memoized':return memoized(*args,counts,memo)
                    if mode=='bounded-memo':return bounded_assignment(*args,counts,memo)
                    return best_assignment(*args) if mode=='subset' else solve(*args,counts,mode)
                forest=case['mode'] in ('207-forest','forest-disjoint')
                rules=inv if case['mode']=='guarded-contact' else None
                with patch('r35_contact_bound.best_assignment',run):
                    out=[]
                    proved,info=no_contact_gain(model,b,r['pool'],contacts=-1 if forest else 1,disjoint=forest,
                                                forbidden=rules,level='exact',witness=out,rank_supports='size-upper',memoized=False)
                result=info.get('physical_gain',0)
                counts['subset_states']+=info.get('subset_states',0)
                if out:assert int(model.matches(out[0]).sum()-model.matches(b).sum())==result
                refs.append((proved,result))
            assert len(set(refs))==1
        results.append(dict(mode=case['mode'],seed=case['seed'],totals={k:dict(x) for k,x in counters.items()}))
        atomic_json(args.output,dict(input=p.fingerprint,cases=results))


if __name__=='__main__':
    main()
