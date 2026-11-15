"""Score legal displaced pieces and coordinated placements of mandatory assemblies."""
from collections import Counter,OrderedDict,defaultdict
from copy import deepcopy
from functools import lru_cache
from itertools import combinations,permutations

import numpy as np

from r35_forbidden_mates import ForbiddenMates


@lru_cache(maxsize=5)
def rotations(size):
    if size==0:
        return np.empty((1,0),dtype=int)
    return np.indices((3,)*size,dtype=int).reshape(size,-1).T


class AssemblyRelocator:
    def __init__(self,assemblies):
        self.assemblies=assemblies
        self.model=assemblies.model
        self.rules=ForbiddenMates(self.model,assemblies.forbidden)
        self.marked_darts=self.rules.forbidden.any(axis=1)
        self.at_footprint=[]
        self.at_cell=[]
        for _,poses in assemblies.groups:
            indexed=defaultdict(list)
            by_cell=defaultdict(list)
            for pose in poses:
                footprint=frozenset(c for c,_ in pose)
                indexed[footprint].append(pose)
                for c in footprint:
                    by_cell[c].append(pose)
            self.at_footprint.append(indexed)
            self.at_cell.append(by_cell)
        self.incident=[np.flatnonzero((self.model.a==c)|(self.model.b==c)) for c in range(self.model.n)]
        self.places=OrderedDict()
        self.layouts=OrderedDict()
        self.reciprocals=OrderedDict()
        self.cache_work=Counter()

    def boundary(self,board,affected_key):
        model=self.model
        layout=self.layouts.get(affected_key)
        if layout is None:
            affected=np.asarray(affected_key,dtype=int)
            edges=np.unique(np.concatenate([self.incident[c] for c in affected]))
            a,s,b,t=model.a[edges],model.sa[edges],model.b[edges],model.sb[edges]
            inside=np.zeros(model.n,dtype=bool)
            inside[affected]=True
            exterior=np.concatenate((a[~inside[a]],b[~inside[b]]))
            sides=np.concatenate((s[~inside[a]],t[~inside[b]]))
            layout=affected,edges,exterior,sides
            self.layouts[affected_key]=layout
            if len(self.layouts)>2048:self.layouts.popitem(last=False)
            self.cache_work['layouts_built']+=1
        else:
            self.layouts.move_to_end(affected_key)
            self.cache_work['layouts_reused']+=1
        affected,edges,exterior,sides=layout
        outside=board[exterior]
        darts=self.rules.dart(outside,sides)
        # An exterior affects score through its facing mask and forbidden
        # interactions through its physical dart, when that dart can be barred.
        boundary=np.asarray((model.masks[outside,sides],np.where(self.marked_darts[darts],darts,-1)),dtype='<i4')
        return affected,edges,(board[affected].astype('<i4').tobytes(),boundary.tobytes())

    def price(self,board,positions,pose,work):
        affected,edges,signature=self.boundary(board,tuple(sorted(positions|pose.keys())))
        key=(tuple(sorted(positions)),tuple(sorted(pose.items())),signature)
        cached=self.places.get(key)
        if cached is not None:
            self.places.move_to_end(key)
            gain,codes,logical=cached
            self.cache_work['price_hits']+=1
            self.cache_work['oriented_rows_reused']+=logical.get('legally_oriented_displacements',0)
        else:
            logical=Counter()
            result=self.score(board,positions,pose,logical,edges=edges)
            gain,codes=(None,None) if result is None else (result[0],result[1][affected].copy())
            self.places[key]=(gain,codes,dict(logical))
            if len(self.places)>4096:self.places.popitem(last=False)
            self.cache_work['price_misses']+=1
        work.update(logical)
        if gain is None:return None
        result=board.copy()
        result[affected]=codes
        return gain,result


    def score(self,board,positions,pose,work,*,edges=None):
        model=self.model
        targets=set(pose)
        empty=np.array(sorted(positions-targets),dtype=int)
        displaced=np.array(sorted(targets-positions),dtype=int)
        n=len(empty)
        if n!=len(displaced):
            raise ValueError('Assembly placement does not conserve physical pieces.')
        if n>4:
            work['outside_exact_displacement_limit']+=1
            return None
        if edges is None:
            edges=np.unique(np.concatenate([self.incident[c] for c in sorted(positions|targets)]))
        a,s,b,t=model.a[edges],model.sa[edges],model.b[edges],model.sb[edges]
        base=board.copy()
        for c,code in pose.items():
            base[c]=code
        free_index={c:i for i,c in enumerate(empty)}
        left=np.array([free_index.get(c,-1) for c in a])
        right=np.array([free_index.get(c,-1) for c in b])
        old=int(np.count_nonzero(model.masks[board[a],s]==model.reverse[model.masks[board[b],t]]))
        best_gain=None
        best_codes=None
        for order in permutations(board[displaced]//3):
            placed=3*np.asarray(order,dtype=int)[None,:]+rotations(n)
            physical=np.all(model.legal[empty,placed],axis=1)
            placed=placed[physical]
            if not len(placed):
                continue
            l=np.broadcast_to(base[a],(len(placed),len(a))).copy()
            r=np.broadcast_to(base[b],(len(placed),len(b))).copy()
            l[:,left>=0]=placed[:,left[left>=0]]
            r[:,right>=0]=placed[:,right[right>=0]]
            safe=~np.any(self.rules.edge_bans(l,s,r,t),axis=1)
            work['legally_oriented_displacements']+=len(placed)
            work['forbidden_displacements']+=int((~safe).sum())
            if not safe.any():
                continue
            score=np.count_nonzero(model.masks[l,s]==model.reverse[model.masks[r,t]],axis=1)-old
            score=np.where(safe,score,-100000)
            winner=int(score.argmax())
            if best_gain is None or score[winner]>best_gain:
                best_gain=int(score[winner])
                best_codes=placed[winner]
        if best_gain is None:
            return None
        base[empty]=best_codes
        return best_gain,base

    def reciprocal(self,board,g,h,old,work):
        positions=old[g]|old[h]
        slot_key=tuple(sorted(positions))
        affected,edges,signature=self.boundary(board,slot_key)
        key=(g,h,slot_key,signature)
        cached=self.reciprocals.get(key)
        if cached is not None:
            self.reciprocals.move_to_end(key)
            gain,codes,logical=cached
            self.cache_work['reciprocal_families_reused']+=1
            self.cache_work['reciprocal_poses_bypassed']+=logical.get('reciprocal_proposals',0)
            self.cache_work['oriented_rows_reused']+=logical.get('legally_oriented_displacements',0)
        else:
            logical=Counter()
            gain=None
            codes=None
            seen=set()
            for first in self.at_footprint[g].get(frozenset(old[h]),()):
                for second in self.at_footprint[h].get(frozenset(old[g]),()):
                    pose=dict((*first,*second))
                    identity=tuple(sorted(pose.items()))
                    if identity in seen:
                        logical['duplicate_proposals_skipped']+=1
                        continue
                    seen.add(identity)
                    logical['reciprocal_proposals']+=1
                    result=self.score(board,positions,pose,logical,edges=edges)
                    if result is None:continue
                    logical['reciprocal_legal']+=1
                    value,proposal=result
                    if gain is None or value>gain:
                        gain,codes=value,proposal[affected].copy()
            self.reciprocals[key]=(gain,codes,dict(logical))
            if len(self.reciprocals)>4096:self.reciprocals.popitem(last=False)
            self.cache_work['reciprocal_families_priced']+=1
        work.update(logical)
        if gain is None:return None
        result=board.copy()
        result[affected]=codes
        return gain,result

    def relocate(self,board,rng,mode,*,trials=8):
        if mode not in ('priced','reciprocal','mutual'):
            raise ValueError('Unknown assembly relocation policy.')
        groups=self.assemblies.groups
        if not groups:
            return board,{}
        sampler=deepcopy(rng)
        # The baseline keeps RNG progression and its witness on an equal score.
        best=self.assemblies.relocate(board,rng,trials=trials)
        initial=int(self.model.matches(board).sum())
        best_gain=int(self.model.matches(best).sum())-initial
        work=Counter(released_gain=best_gain)
        old=[set(np.flatnonzero(np.isin(board//3,list(tiles))).tolist()) for tiles,_ in groups]
        protected=set().union(*old)
        group=int(sampler.integers(len(groups)))
        candidates=groups[group][1]
        sampled=[candidates[i] for i in sampler.permutation(len(candidates))[:trials]]
        seen=set()
        def consider(ids,pose,kind):
            nonlocal best,best_gain
            key=(tuple(sorted(ids)),tuple(sorted(pose.items())))
            if key in seen:
                work['duplicate_proposals_skipped']+=1
                return
            seen.add(key)
            positions=set().union(*(old[i] for i in ids))
            if pose.keys() & (protected-positions):
                return
            work[kind+'_proposals']+=1
            result=self.price(board,positions,pose,work)
            if result is None:
                return
            work[kind+'_legal']+=1
            gain,answer=result
            if gain>best_gain:
                work[kind+'_incremental_gain']+=gain-best_gain
                best_gain,best=gain,answer
        for pose in sampled:
            consider((group,),dict(pose),'single')
        if mode in ('reciprocal','mutual'):
            for g,h in combinations(range(len(groups)),2):
                result=self.reciprocal(board,g,h,old,work)
                if result is not None and result[0]>best_gain:
                    work['reciprocal_incremental_gain']+=result[0]-best_gain
                    best_gain,best=result
        if mode=='mutual':
            for first in sampled:
                target=dict(first)
                obstruction=[g for g,positions in enumerate(old) if g!=group and target.keys() & positions]
                if len(obstruction)!=1:
                    continue
                other=obstruction[0]
                for cell in sorted(old[group]-target.keys()):
                    for pose in self.at_cell[other].get(cell,()):
                        second=dict(pose)
                        if target.keys()==old[other] and second.keys()==old[group]:
                            work['duplicate_proposals_skipped']+=1
                        elif not target.keys() & second.keys():
                            consider((group,other),target|second,'partial')
        work['best_gain']=best_gain
        if int(self.model.matches(self.model.legal_board(best)).sum())-initial!=best_gain:
            raise ArithmeticError('Assembly relocation differs from its exact incident score.')
        return best,dict(work)
