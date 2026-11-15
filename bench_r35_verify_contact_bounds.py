"""Independently verify observed local exclusions from direct scored physical component tables."""
import argparse
from collections import Counter
from itertools import combinations,permutations
from pathlib import Path

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r35_contact_bound import no_contact_gain
from r35_forbidden_mates import ForbiddenMates
from r32_gold_bonds import inventory
from search_problem import SearchProblem


def scores(model,board,cells,unit,rules):
    n=len(cells)
    donors=board[list(cells)]//3
    origin=model.matches(board)
    labels=(3*donors[:,None]+np.arange(3)).ravel()
    choices=np.array(np.meshgrid(*([labels]*len(unit)),indexing='ij')).reshape(len(unit),-1).T
    good=np.ones(len(choices),dtype=bool)
    raw=np.zeros(len(choices),dtype=int)
    slots=[cells[i] for i in unit]
    for k,c in enumerate(slots):
        good &= model.legal[c,choices[:,k]]&(choices[:,k]!=board[c])
    for edge,(a,s,b,t) in enumerate(model.problem.edges):
        if a not in slots and b not in slots:continue
        left=choices[:,slots.index(a)] if a in slots else board[a]
        right=choices[:,slots.index(b)] if b in slots else board[b]
        match=model.masks[left,s]==model.reverse[model.masks[right,t]]
        raw+=match.astype(int)-int(origin[edge])
        if rules is not None:
            dl=3*(left//3)+(s-left%3)%3
            dr=3*(right//3)+(t-right%3)%3
            good &= ~(match&rules.forbidden[dl,dr])
    ranked=np.where(good,raw,-1000).reshape((n,3)*len(unit)).max(axis=tuple(range(1,2*len(unit),2)))
    if len(unit)==2:np.fill_diagonal(ranked,-1000)
    return ranked


def independent(model,board,cells,limit,disjoint,rules):
    cells=tuple(map(int,cells))
    n=len(cells)
    edges=tuple((i,j) for i,j in combinations(range(n),2) if cells[j] in model.problem.neighbors[cells[i]])
    single={i:scores(model,board,cells,(i,),rules) for i in range(n)}
    pair={(i,j):scores(model,board,cells,(i,j),rules) for i,j in edges} if limit!=0 else {}
    # Construct dual labels from directly scored physical substitutions. A hit
    # certifies the bound for all supports, including every hidden rotation.
    prices=[0]*n
    for _ in range(n):
        for i in range(n):
            prices[i]=max(prices[i],*(int(v)+prices[j] for j,v in enumerate(single[i])))
    if all(v<=prices[i]-prices[j] for i,vs in single.items() for j,v in enumerate(vs)) and all(
        not np.any(t>prices[i]+prices[j]-np.add.outer(prices,prices)) for (i,j),t in pair.items()):
        return 'physical component dual',0
    best=0
    for size in range(1,n+1):
        for changed in combinations(range(n),size):
            groups=[(i,j) for i,j in edges if i in changed and j in changed]
            if limit>=0 and len(groups)>limit:continue
            grouped=[i for edge in groups for i in edge]
            if disjoint and len(grouped)!=len(set(grouped)):continue
            if len(grouped)!=len(set(grouped)):raise ValueError('Verifier requires separated changed contacting pairs.')
            units=[(i,) for i in changed if i not in grouped]+groups
            opts=[]
            for unit in units:
                if len(unit)==1:
                    vs=[(1<<j,int(single[unit[0]][j])) for j in changed]
                else:
                    t=pair[unit]
                    vs=[((1<<j)|(1<<k),int(t[j,k])) for j,k in permutations(changed,2)]
                opts.append(sorted([(m,v) for m,v in vs if v>-1000],key=lambda x:-x[1]))
            if any(not vs for vs in opts):continue
            opts.sort(key=len)
            suffix=[0]*(len(opts)+1)
            for k in reversed(range(len(opts))):suffix[k]=suffix[k+1]+opts[k][0][1]
            def enumerate_best(index,occupied,value):
                nonlocal best
                if value+suffix[index]<=best:return
                if index==len(opts):
                    best=value
                    return
                for mask,gain in opts[index]:
                    if not mask&occupied:enumerate_best(index+1,occupied|mask,value+gain)
            enumerate_best(0,0,0)
    return ('positive feasible placement' if best else 'physical support enumeration'),best


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--queries',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--rank-supports',choices=('colex','upper','size','size-upper'),default='size-upper')
    ap.add_argument('--no-memo',action='store_true')
    args=ap.parse_args()
    p=SearchProblem(read_json(args.tiles))
    model=BoardModel(p)
    inv=ForbiddenMates(model,inventory(p).forbidden)
    examples=read_json(args.queries)
    if examples['input']!=p.fingerprint:raise ValueError('Mismatched physical inventory.')
    results=[]
    for case in examples['cases']:
        counts=Counter()
        seen=set()
        for r in case['rows']:
            b=model.legal_board(r['board'])
            region=tuple(r['pool'])
            tag=(tuple(b),tuple(sorted(region)))
            counts['new_full_requests']+=int(tag not in seen)
            seen.add(tag)
            disjoint=case['mode'] in ('forest-disjoint','207-forest')
            limit=-1 if disjoint else 1
            rules=inv if case['mode']=='guarded-contact' else None
            winner=[]
            proved,stats=no_contact_gain(model,b,region,contacts=limit,disjoint=disjoint,forbidden=rules,level='exact',witness=winner,rank_supports=None if args.rank_supports=='colex' else args.rank_supports,memoized=not args.no_memo)
            independent_verdict,optimum=independent(model,b,region,limit,disjoint,rules)
            if proved != (optimum==0) or optimum!=stats.get('physical_gain',0):
                raise AssertionError((case['mode'],case['seed'],region,stats,independent_verdict))
            if winner:
                assert len(winner)==1 and int(model.matches(winner[0]).sum()-model.matches(b).sum())==optimum
                counts['positive_summed_seams']+=optimum
                counts['strict_gains_over_cp']+=int(optimum>r['gain'])
                counts['additional_seams_over_cp']+=optimum-r['gain']
            counts[independent_verdict]+=1
            counts['root_closed']+=stats.get('root_dual_proofs',0)
            if stats.get('support_proofs'):
                counts['exact_support_closed' if stats.get('refined_supports') else 'unit_upper_closed']+=1
            counts['matched_subset_states']+=stats.get('subset_states',0)
            counts['bounded_subproblems']+=stats.get('bounded_subproblems',0)
            counts['bounded_hits']+=stats.get('bounded_hits',0)
            counts['branches_on_closed_CP_queries']+=r['cp']['branches'] if proved else 0
            counts['cp_uncertain_closed']+=int(proved and r['cp']['status'] not in ('OPTIMAL','INFEASIBLE'))
        results.append(dict(mode=case['mode'],seed=case['seed'],counts=dict(counts)))
        atomic_json(args.output,dict(input=p.fingerprint,cases=results))
        print(case['mode'],case['seed'],counts,flush=True)


if __name__=='__main__':
    main()
