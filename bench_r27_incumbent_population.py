"""Matched-budget, reproducible construction and repair experiments for round 27.

The common search proposes inventory-preserving, domain-legal edits. Each
variant can be replayed across a checkpoint via its Python/NumPy state.
"""
import math
import numpy as np
from r27_incumbent import assignment


def project(model, preferences, rng, legal=None):
    scores=np.where(model.legal if legal is None else legal,preferences,-np.inf).reshape(model.n,model.n,3)
    rotations=scores.argmax(axis=2)
    weights=scores.max(axis=2)
    legal=np.isfinite(weights)
    costs=np.rint(np.where(legal,weights,0)*1e6).astype(np.int64)
    tiles=assignment(costs,legal)
    return 3*tiles+rotations[np.arange(model.n),tiles]


def normalize(logits, legal):
    safe=np.where(legal,logits,-np.inf)
    mass=np.where(legal,np.exp(np.maximum(safe-safe.max(axis=1,keepdims=True),-70)),0.)
    return mass/mass.sum(axis=1,keepdims=True)


def support(model, probabilities, messages=None):
    output=np.zeros_like(probabilities)
    updated={}
    for a,sa,b,sb in model.problem.edges:
        for src,side,dst,other in ((a,sa,b,sb),(b,sb,a,sa)):
            density=probabilities[src]
            if messages is not None and (dst,src,other) in messages:
                density=density/np.maximum(messages[dst,src,other],1e-6)
                density=density/density.sum()
            target=np.bincount(model.masks[:,side],weights=density,minlength=2048)
            compat=target[model.reverse[model.masks[:,other]]]
            if messages is None:
                output[dst]+=compat
            else:
                message=.05+.95*compat
                updated[src,dst,side]=message
                output[dst]+=np.log(message)
    return output,updated


def construct(model, incumbent, rng, method, *, steps=6, damping=.5, decimate=False):
    initial=np.zeros((model.n,3*model.n))
    initial[np.arange(model.n),incumbent]=1.
    uniform=normalize(np.zeros_like(initial),model.legal)
    p=.35*initial+.65*uniform
    work=0
    if method=='belief':
        price=np.zeros(model.n)
        messages={}
        legal=model.legal.copy()
        for step in range(steps):
            field,new=support(model,p,messages)
            messages={key:(damping*messages.get(key,value)+(1-damping)*value) for key,value in new.items()}
            # Damped cavity messages, plus the price of excess physical-tile mass.
            if damping:
                field.fill(0)
                for (_,dst,_),value in messages.items():
                    field[dst]+=np.log(value)
            price+=.4*(p.reshape(model.n,model.n,3).sum(axis=(0,2))-1)
            target=normalize(field-np.repeat(price,3)[None,:]+initial,legal)
            p=damping*p+(1-damping)*target
            if decimate and step==steps//2:
                pinned=project(model,np.log(np.maximum(p,1e-50)),rng)
                confidence=p[np.arange(model.n),pinned]
                for c in np.argsort(-confidence)[:max(1,model.n//12)]:
                    tile=int(pinned[c]//3)
                    legal[:,3*tile:3*tile+3]=False
                    legal[c]=False
                    legal[c,pinned[c]]=True
                p=normalize(np.log(np.maximum(p,1e-50)),legal)
            work+=2*len(model.a)*p.shape[1]
    elif method=='fractional':
        # A convex mixture of feasible assignment vertices satisfies both
        # inventory and cell constraints even on restricted domains.
        alternate=project(model,rng.gumbel(size=p.shape),rng)
        p=.35*initial
        p[np.arange(model.n),alternate]+=.65
        for _ in range(steps):
            gradient,_=support(model,p)
            vertex=project(model,gradient+rng.random(gradient.shape)*1e-6,rng)
            d=-p.copy()
            d[np.arange(model.n),vertex]+=1
            gd,_=support(model,d)
            linear=float((gradient*d).sum())
            quadratic=float((gd*d).sum())/2
            # Pair energy is quadratic along a feasible search direction.
            choices=[0.,1.]
            if quadratic<0:
                choices.append(float(np.clip(-linear/(2*quadratic),0,1)))
            alpha=max(choices,key=lambda t:t*linear+t*t*quadratic)
            p+=alpha*d
            work+=4*len(model.a)*p.shape[1]
            if alpha<1e-10:
                break
    else:
        raise ValueError(method)
    rounded=project(model,p+rng.random(p.shape)*1e-7,rng,legal if method=='belief' else None)
    rounded,info=model.orientations(rounded,max_entries=19683,rng=rng)
    return rounded,work+info['table_entries']


def propose(model, board, rng, *, defective=None):
    child=board.copy()
    if not len(model.free):
        return child
    a=int(rng.choice(defective)) if defective is not None and len(defective) and rng.random()<.75 else int(rng.choice(model.free))
    b=a if rng.random()<.35 else int(rng.choice(model.free))
    for c,tile in ((a,board[b]//3),(b,board[a]//3)):
        choices=[3*tile+r for r in range(3) if model.legal[c,3*tile+r]]
        if not choices:
            return board.copy()
        child[c]=rng.choice(choices)
    return child


def category(model, board):
    validator=model.problem.validate(board.tolist())
    loops=len([1 for size,_ in model.problem._closed_components(board.tolist()) if size<model.problem.total_segments])
    edges=np.flatnonzero(~model.matches(board))
    sectors=set(int(model.a[e])//max(1,model.n//10) for e in edges)
    return (loops//2,(validator['line_components']-loops)//4,len(sectors)//2)


def recombine(left,right,rng):
    positions=np.argsort(left//3)
    other=positions[right//3]
    todo=set(range(len(left)))
    child=left.copy()
    while todo:
        cell=min(todo)
        exchange=bool(rng.integers(2))
        while cell in todo:
            todo.remove(cell)
            if exchange:
                child[cell]=right[cell]
            cell=int(other[cell])
    return child


def hamming_archive(rows, candidate, cap=8):
    from constraint_lns import diverse_archive
    records=[{'codes':v.tolist(),'objective':s} for s,v in [*rows,candidate]]
    kept=diverse_archive(records,capacity=cap)
    return [(row['objective'],np.asarray(row['codes'])) for row in kept]


def systematic(weights,rng):
    n=len(weights)
    return np.searchsorted(np.cumsum(weights/weights.sum()),(rng.random()+np.arange(n))/n).clip(0,n-1)


def ensemble(model,start,seed,method,*,proposals=1024,lanes=8,resume=None,interrupt=None):
    import copy
    rng=np.random.default_rng(seed)
    if resume is None:
        boards=[start.copy() for _ in range(lanes)]
        scores=np.array([int(model.matches(v).sum()) for v in boards])
        state=dict(boards=boards,scores=scores,archive=[(int(scores[0]),start.copy())],
                   best=start.copy(),best_score=int(scores[0]),hist={},penalty=np.zeros(len(model.a)),
                   distribution=np.zeros_like(model.legal,dtype=float),visits=np.zeros(len(model.a)),
                   accepted=0,round=0,steps=0,first208=None,tabu=[None]*lanes,extra=0)
        state['distribution'][np.arange(model.n),start]=1
    else:
        state=copy.deepcopy(resume)
        rng.bit_generator.state=state.pop('rng')
    while state['steps']<proposals and (interrupt is None or state['steps']<interrupt):
        boards,scores=state['boards'],state['scores']
        i=state['steps']%lanes
        sweep=state['steps']//lanes
        old=boards[i]
        matches=model.matches(old)
        bad=np.flatnonzero(~matches)
        defective=np.intersect1d(np.unique(np.r_[model.a[bad],model.b[bad]]),model.free)
        child=propose(model,old,rng,defective=defective)
        if method=='cross_entropy' and sweep%12==11:
            weights=state['distribution']
            child=project(model,np.log(.005+weights)+rng.gumbel(size=weights.shape)*.7,rng)
            state['extra']+=weights.size
        if method in ('archive','niche') and sweep%8==7 and len(state['archive'])>=2:
            archive=state['archive']
            ids=rng.choice(len(archive),size=2,replace=False)
            child=recombine(archive[int(ids[0])][1],archive[int(ids[1])][1],rng)
        valid=model.legal[np.arange(model.n),child].all()
        trial=model.matches(child)
        score=int(trial.sum())
        delta=score-scores[i]
        temp=.35*1.42**i
        if method=='penalty':
            delta+=float(np.dot(state['penalty'],trial.astype(float)-matches))
        elif method=='flat':
            delta+=.7*(state['hist'].get(int(scores[i]),0)-state['hist'].get(score,0))
        elif method=='population':
            temp=max(.14,1.8*(.92**(sweep//4)))
        signature=tuple(sorted((int(c),int(v)) for c,v in enumerate(child) if v!=old[c]))
        inverse=tuple((c,int(old[c])) for c,_ in signature)
        if valid and (delta>=0 or rng.random()<math.exp(max(-70,delta/temp))) and (method not in ('tabu','penalty') or signature!=state['tabu'][i]):
            boards[i]=child
            scores[i]=score
            state['accepted']+=1
            state['tabu'][i]=inverse
        if valid and score>state['best_score']:
            state['best_score']=score
            state['best']=child.copy()
            if score>=208 and state['first208'] is None:
                state['first208']=state['steps']+1
        if method=='flat':
            k=int(scores[i])
            state['hist'][k]=state['hist'].get(k,0)+1/(1+sweep/100)
        if method=='penalty':
            state['visits']+=~model.matches(boards[i])
        state['steps']+=1
        if i!=lanes-1:
            continue
        if method in ('exchange','archive','niche','tabu','penalty','cross_entropy'):
            for j in range(sweep%2,lanes-1,2):
                ratio=(1/(.35*1.42**j)-1/(.35*1.42**(j+1)))*(scores[j+1]-scores[j])
                if ratio>=0 or rng.random()<math.exp(max(-70,ratio)):
                    boards[j],boards[j+1]=boards[j+1],boards[j]
                    scores[j],scores[j+1]=scores[j+1],scores[j]
                    state['tabu'][j],state['tabu'][j+1]=state['tabu'][j+1],state['tabu'][j]
        if method=='population' and sweep%4==3:
            beta=1/max(.14,1.8*.92**(sweep//4))
            next_beta=1/max(.14,1.8*.92**(sweep//4+1))
            weights=np.exp(np.clip((next_beta-beta)*(scores-scores.max()),-60,0))
            ids=systematic(weights,rng)
            state['boards']=[boards[j].copy() for j in ids]
            state['scores']=scores[ids].copy()
        if method=='penalty' and sweep%6==5:
            state['penalty']=.7*state['penalty']+.45*state['visits']/(6*lanes)
            state['visits'].fill(0)
            if sweep%48==47:
                state['penalty'].fill(0)
        if method=='cross_entropy' and sweep%6==5:
            elite=np.argsort(-scores)[:max(2,lanes//4)]
            weights=state['distribution']
            weights*=.65
            for j in elite:
                weights[np.arange(model.n),boards[j]]+=.35/len(elite)
        if method in ('archive','niche') and sweep%4==3:
            for j in range(lanes):
                item=int(scores[j]),boards[j].copy()
                if method=='archive':
                    state['archive']=hamming_archive(state['archive'],item)
                else:
                    bins={}
                    for s,b in [*state['archive'],item]:
                        key=category(model,b)
                        if key not in bins or bins[key][0]<s:
                            bins[key]=s,b
                        state['extra']+=1
                    state['archive']=sorted(bins.values(),key=lambda row:-row[0])[:8]
    state['rng']=rng.bit_generator.state
    return state


def geometric_children(model,start,rng,count):
    from constraint_symmetry import proper_rotations
    rotations=proper_rotations(model.problem.board)
    identity=tuple(range(model.n))
    halves=[rot for rot in rotations if rot.cells!=identity and all(rot.cells[rot.cells[c]]==c for c in identity)]
    seen=set()
    result=[]
    for _ in range(count):
        rot=halves[int(rng.integers(len(halves)))]
        other=np.array(rot.apply(start.tolist()))
        orbits=[(i,j) for i,j in enumerate(rot.cells) if i<j]
        # Choose an irregular connected patch or independent cell orbits.
        if rng.integers(2):
            mask=set()
            border=[int(rng.integers(model.n))]
            while border and len(mask)<max(1,model.n//4):
                k=int(rng.integers(len(border)))
                c=border.pop(k)
                if c in mask:
                    continue
                mask.update((c,rot.cells[c]))
                border.extend(int(v) for v in model.problem.neighbors[c] if v>=0 and v not in mask)
        else:
            mask={c for pair in orbits if rng.random()<.25 for c in pair}
        child=start.copy()
        idx=np.array(sorted(mask),dtype=int)
        child[idx]=other[idx]
        if not model.legal[np.arange(model.n),child].all():
            continue
        key=min(tuple(g.apply(child.tolist())) for g in rotations)
        if key not in seen:
            seen.add(key)
            result.append(model.legal_board(child))
    return result

