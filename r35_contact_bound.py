"""Bound and solve local donor/rotation repairs with separated changed contacts."""
from collections import Counter
from functools import lru_cache
from itertools import combinations
import numpy as np


@lru_cache(maxsize=13)
def donor_rows(n):
    return tuple(tuple(i for i in range(n) if mask>>i&1) for mask in range(1<<n))


@lru_cache(maxsize=16)
def supports(n,edges,limit):
    out=[]
    singletons=tuple((i,) for i in range(n))
    for mask,rows in enumerate(donor_rows(n)):
        if not mask:continue
        touched=[(i,j) for i,j in edges if mask>>i&1 and mask>>j&1]
        if limit>=0 and len(touched)>limit:continue
        owners={i for edge in touched for i in edge}
        if 2*len(touched)!=len(owners):continue
        units=tuple(singletons[i] for i in rows if i not in owners)+tuple(touched)
        out.append((rows,units))
    return out


def best_assignment(rows,units,one,pair,trace,target,prices):
    """Subset DP joins optimal local placements using each donor exactly once."""
    tables=[]
    for unit in units:
        i=unit[0]
        if len(unit)==1:
            options=[(1<<a,int(one[i,a]+prices[a]-prices[i]),(a,)) for a in rows if one[i,a]>-1000]
        else:
            table=pair[unit]
            options=[]
            for a,b in combinations(rows,2):
                donors=(a,b) if table[a,b]>=table[b,a] else (b,a)
                gain=int(table[donors])
                if gain>-1000:
                    adjusted=gain+int(prices[a]+prices[b]-prices[unit[0]]-prices[unit[1]])
                    options.append(((1<<a)|(1<<b),adjusted,donors))
        if not options:return -1000,0,None
        tables.append((unit,options))
    tables.sort(key=lambda item:len(item[1]))
    layers={0:0}
    states=0
    history=[]
    suffix=[0]*(len(tables)+1)
    for k in reversed(range(len(tables))):
        suffix[k]=suffix[k+1]+max(value for mask,value,donors in tables[k][1])
    for k,(unit,options) in enumerate(tables):
        next_layer={}
        parent={}
        for used,value in layers.items():
            for mask,gain,donors in options:
                state=used|mask
                if not used&mask and value+gain+suffix[k+1]>target and value+gain>next_layer.get(state,-10000):
                    next_layer[state]=value+gain
                    if trace:parent[state]=used,donors
        states+=len(next_layer)
        layers=next_layer
        if not layers:return -1000,states,None
        if trace:history.append((unit,parent))
    state,=layers
    value=layers[state]
    placement={}
    if value>0 and trace:
        for unit,parent in reversed(history):
            state,donors=parent[state]
            placement[unit]=donors
    return value,states,placement


def bounded_assignment(rows,units,one,pair,trace,target,prices,work,store):
    """Reuse exact subproblem maxima and explicitly marked threshold upper bounds."""
    tables=store.setdefault('tables',{})
    bounds=store.setdefault('bounds',{})
    memo=store.setdefault('memo',{})
    n=len(one)
    def options(unit):
        if unit not in tables:
            i=unit[0]
            values=[]
            if len(unit)==1:
                for a in range(n):
                    if one[i,a]>-1000:values.append((int(one[i,a]+prices[a]-prices[i]),1<<a,(a,)))
            else:
                j=unit[1]
                for a,b in combinations(range(n),2):
                    t=pair[unit]
                    ds=(a,b) if t[a,b]>=t[b,a] else (b,a)
                    if t[ds]>-1000:values.append((int(t[ds]+prices[a]+prices[b]-prices[i]-prices[j]),(1<<a)|(1<<b),ds))
            tables[unit]=sorted(values,key=lambda value:-value[0])
        return tables[unit]
    def upper(unit,mask):
        identity=unit,mask
        if identity not in bounds:
            bounds[identity]=next((v for v,occupied,ds in options(unit) if not occupied&~mask),-1000)
        return bounds[identity]
    def visit(pending,available,threshold):
        if not pending:return 0 if 0>threshold else -1000
        key=pending,available
        previous=memo.get(key)
        if previous is not None:
            bound,exact,placement=previous
            if exact or bound<=threshold:
                work['bounded_hits']+=1
                return bound if exact and bound>threshold else -1000
        work['bounded_subproblems']+=1
        ceiling=sum(upper(unit,available) for unit in pending)
        if ceiling<=threshold:
            memo[key]=ceiling,False,None
            return -1000
        unit=pending[0]
        rest=pending[1:]
        best=threshold
        winner=None
        for value,mask,ds in options(unit):
            if mask&~available:continue
            child=visit(rest,available^mask,best-value)
            if child>-1000 and child+value>best:
                best,winner=child+value,(ds,mask)
                if best==ceiling:break
        if winner is None:
            memo[key]=min(threshold,ceiling),False,None
            return -1000
        memo[key]=best,True,winner
        return best
    units=tuple(sorted(units,key=lambda u:(-len(u),u)))
    mask=sum(1<<i for i in rows)
    best=visit(units,mask,target)
    placement={}
    if trace and best>target:
        while units:
            ds,covered=memo[units,mask][2]
            placement[units[0]]=ds
            units,mask=units[1:],mask^covered
    return best,0,placement


def no_contact_gain(model,board,cells,*,contacts,disjoint,forbidden=None,level='supports',witness=None,rank_supports='size-upper',memoized=True):
    """Prove zero gain, or return an exact improving witness when requested.

    Changed contacts must form a matching on the induced geographic graph.
    A positive witness includes a full physical rescore before returning.
    """
    if type(contacts) is not int or contacts< -1 or level not in ('dual','supports','exact'):
        raise ValueError('Unknown local contact limit or bound policy.')
    cells=tuple(sorted(set(map(int,cells))))
    n=len(cells)
    if n>12 or not n:
        return False,{'outside_scope':1}
    board=model.legal_board(board)
    loc={c:i for i,c in enumerate(cells)}
    if any(not 0<=c<model.n or c in model.problem.fixed for c in cells):
        raise ValueError('Expected movable repair cells.')
    if forbidden is not None:
        incident=np.isin(model.a,cells)|np.isin(model.b,cells)
        if forbidden.violations(board)[incident].any():return False,{'incompatible_identity':1}
    tiles=board[list(cells)]//3
    labels=(3*tiles[:,None]+np.arange(3)).ravel()
    old=np.array([3*i+board[c]%3 for i,c in enumerate(cells)])
    gains=np.zeros((n,3*n),dtype=np.int16)
    hard=model.legal[np.ix_(cells,labels)]
    side_bans=np.zeros((n,3,3*n),dtype=bool)
    edges=[]
    for i,c in enumerate(cells):
        for s,nb in enumerate(model.problem.neighbors[c]):
            if nb<0:continue
            t=model.problem.sides[c,s]
            strengths=model.masks[labels,s]==model.reverse[model.masks[board[nb],t]]
            gains[i]+=strengths.astype(np.int16)-int(strengths[old[i]])
            if forbidden is not None:
                side_bans[i,s]=forbidden.edge_bans(labels,s,board[nb],t)
            if nb in loc and i<loc[nb]:edges.append((i,s,loc[nb],t))
    if not disjoint and contacts not in (0,1):
        endpoints=[i for i,s,j,t in edges]+[j for i,s,j,t in edges]
        if len(endpoints)!=len(set(endpoints)):return False,{'outside_scope':1}
    neg=-1000
    single=np.where(hard&~side_bans.any(axis=1),gains,neg)
    changed=single.copy()
    changed[np.arange(n),old]=neg
    one=changed.reshape(n,n,3).max(axis=2)
    full=single.reshape(n,n,3).max(axis=2)
    # Single changed cells form permutation cycles when they do not touch.
    # A dual price potential makes each weighted unilateral choice nonpositive.
    p=np.zeros(n,dtype=np.int16)
    for _ in range(n):
        next_p=(full+p[None,:]).max(axis=1)
        if np.array_equal(next_p,p):break
        p=next_p
    single_dual=bool(np.all(full<=p[:,None]-p[None,:]))
    work={'internal_seams':len(edges)}
    if single_dual and not edges:return True,{**work,'root_dual_proofs':1}
    pair={}
    pair_rotations={}
    all_pairs_safe=single_dual
    for i,s,j,t in edges:
        if contacts==0:continue
        # Two changed neighbors share one seam. Restore its exact value while
        # subtracting the two single-change prices already charged for it.
        m=model.masks[labels,s,None]==model.reverse[model.masks[labels,t][None,:]]
        corr=m.astype(np.int16)-m[:,old[j],None].astype(np.int16)-m[old[i],None,:].astype(np.int16)+int(m[old[i],old[j]])
        ok_i=hard[i]&~side_bans[i,[k for k in range(3) if k!=s]].any(axis=0)
        ok_j=hard[j]&~side_bans[j,[k for k in range(3) if k!=t]].any(axis=0)
        ok_i[old[i]]=False
        ok_j[old[j]]=False
        legal=ok_i[:,None]&ok_j[None,:]
        if forbidden is not None:legal &= ~forbidden.edge_bans(labels[:,None],s,labels[None,:],t)
        table=np.where(legal,gains[i,:,None]+gains[j,None,:]+corr,neg).reshape(n,3,n,3)
        value=table.max(axis=(1,3))
        if witness is not None:
            pair_rotations[i,j]=table.transpose(0,2,1,3).reshape(n,n,9).argmax(axis=2)
        np.fill_diagonal(value,neg)
        pair[i,j]=value
        if all_pairs_safe and np.any(value>p[i]+p[j]-p[:,None]-p[None,:]):
            all_pairs_safe=False
    if all_pairs_safe:return True,{**work,'root_dual_proofs':1}
    if level=='dual':return False,work
    graph=tuple((i,j) for i,s,j,t in edges)
    count=0
    subset_states=0
    refinements=0
    best=0
    winning=None
    store={}
    memo_work=Counter()
    share_subproblems=memoized and n>=10 and not disjoint and contacts==1
    def bounded_supports():
        for rows,units in supports(n,graph,contacts):
            labels_in=np.array(rows)
            upper=0
            for unit in units:
                i=unit[0]
                if len(unit)==1:
                    upper+=int(np.max(one[i,labels_in]+p[labels_in]-p[i]))
                else:
                    j=unit[1]
                    upper+=int(np.max(pair[i,j][np.ix_(labels_in,labels_in)]+p[labels_in,None]+p[None,labels_in]-p[i]-p[j]))
            yield upper,rows,units
    choices=bounded_supports()
    if rank_supports and level=='exact' and witness is not None:
        priorities={'upper':lambda item:-item[0], 'size':lambda item:len(item[1]),
                    'size-upper':lambda item:(len(item[1]),-item[0])}
        choices=sorted(choices,key=priorities[rank_supports])
    for upper,rows,units in choices:
        count+=1
        if upper>best:
            if level!='exact':
                return False,{**work,'supports_bounded':count,'loose_supports':1}
            if share_subproblems:
                value,visited,placement=bounded_assignment(rows,units,one,pair,witness is not None,best,p,memo_work,store)
            else:
                value,visited,placement=best_assignment(rows,units,one,pair,witness is not None,best,p)
            subset_states+=visited
            refinements+=1
            if value>best:
                if witness is None:
                    return False,{**work,**memo_work,'supports_bounded':count,'positive_supports':1,
                                  'subset_states':subset_states,'refined_supports':refinements}
                best,winning=value,placement
    work.update(memo_work)
    work.update(supports_bounded=count,subset_states=subset_states,refined_supports=refinements)
    if winning is None:
        return True,{**work,'support_proofs':1}
    result=board.copy()
    for unit,donors in winning.items():
        if len(unit)==1:
            i,=unit
            a,=donors
            rotation=int(changed[i].reshape(n,3)[a].argmax())
            result[cells[i]]=3*tiles[a]+rotation
        else:
            i,j=unit
            a,b=donors
            left,right=divmod(int(pair_rotations[unit][a,b]),3)
            result[cells[i]],result[cells[j]]=3*tiles[a]+left,3*tiles[b]+right
    model.legal_board(result)
    if int(model.matches(result).sum()-model.matches(board).sum())!=best:
        raise ArithmeticError('Contact support witness failed full physical rescoring.')
    if forbidden is not None and forbidden.violations(result)[incident].any():
        raise ArithmeticError('Contact support witness violates a physical mate rule.')
    witness.append(result)
    return False,{**work,'positive_optima':1,'physical_gain':best}
