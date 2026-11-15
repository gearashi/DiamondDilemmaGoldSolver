"""Restrict local choices and reconcile full permutation cycles with physical Gold mate rules."""
from collections import Counter
from copy import copy

import numpy as np

from r28_exchange_patches import AssignmentPrices


class ForbiddenMates:
    def __init__(self,model,pairs):
        self.model=model
        self.pairs=frozenset(tuple(sorted(pair)) for pair in pairs)
        self.directed=sorted({(a,b) for pair in self.pairs for a,b in (pair,pair[::-1])})
        n=3*model.n
        self.forbidden=np.zeros((n,n),dtype=bool)
        for a,b in self.directed:
            self.forbidden[a,b]=True

        self.side_pairs={}
        for s in range(3):
            for t in range(3):
                rows=[]
                for a,b in self.directed:
                    left=3*(a//3)+(s-a%3)%3
                    right=3*(b//3)+(t-b%3)%3
                    if model.masks[left,s]==model.reverse[model.masks[right,t]]:
                        rows.append((left,right))
                self.side_pairs[s,t]=np.asarray(rows,dtype=int).reshape(-1,2)

    def local_pairs(self,labels,s,t):
        pairs=labels[self.side_pairs[s,t]]
        return sorted(pairs[(pairs>=0).all(axis=1)].tolist())

    @staticmethod
    def dart(code,side):
        code=np.asarray(code,dtype=int)
        return 3*(code//3)+(np.asarray(side,dtype=int)-code%3)%3

    def edge_bans(self,left,ls,right,rs):
        model=self.model
        return (self.forbidden[self.dart(left,ls),self.dart(right,rs)]
                & (model.masks[left,ls]==model.reverse[model.masks[right,rs]]))

    def violations(self,board):
        m=self.model
        board=np.asarray(board)
        return self.edge_bans(board[m.a],m.sa,board[m.b],m.sb)

    def blocked_codes(self,board):
        """Yield (destination, barred code, neighboring cell) for fixed neighbor poses."""
        model=self.model
        position=np.empty(model.n,dtype=int)
        position[board//3]=np.arange(model.n)
        for changed,fixed in self.directed:
            neighbor=position[fixed//3]
            facing=(fixed%3+board[neighbor]%3)%3
            c=model.problem.neighbors[neighbor,facing]
            if c<0:
                continue
            side=int(model.problem.sides[neighbor,facing])
            code=3*(changed//3)+(side-changed%3)%3
            if model.masks[code,side]==model.reverse[model.masks[board[neighbor],facing]]:
                yield int(c),int(code),int(neighbor)

    def block(self,board,cells,rng):
        model=self.model
        restricted=copy(model)
        restricted.legal=model.legal.copy()
        changed=set(map(int,cells))
        for c,code,_ in self.blocked_codes(board):
            if c in changed and code!=board[c]:
                restricted.legal[c,code]=False
        return type(model).block(restricted,board,cells,rng)

    def groups(self,old,trial):
        model=self.model
        origin=np.empty(model.n,dtype=int)
        origin[old//3]=np.arange(model.n)
        source=origin[trial//3]
        group=np.full(model.n,-1,dtype=int)
        members=[]
        for start in np.flatnonzero(old!=trial):
            if group[start]>=0:
                continue
            positions=[]
            cell=start
            while group[cell]<0:
                group[cell]=len(members)
                positions.append(cell)
                cell=source[cell]
            members.append(positions)
        return group,members

    def project(self,old,trial,*,limit=14):
        """Choose entire physical donor cycles from either legal permutation.

        Freeze trial choices with no possible forbidden interaction. Solve
        incident cycle groups as independent geographic components.
        """
        model=self.model
        if not self.violations(trial).any():
            return trial,{}
        if self.violations(old).any():
            raise ValueError('Rollback requires a physically compatible earlier board.')
        groups,members=self.groups(old,trial)
        ga,gb=groups[model.a],groups[model.b]
        can_ban=np.zeros(len(model.a),dtype=bool)
        for i,left in enumerate((old,trial)):
            for j,right in enumerate((old,trial)):
                possible=self.edge_bans(left[model.a],model.sa,right[model.b],model.sb)
                if i!=j:
                    possible &= (ga!=gb) | (ga<0)
                can_ban |= possible
        active=set(np.r_[ga[can_ban],gb[can_ban]].tolist())-{-1}
        adjacent={v:set() for v in active}
        for a,b in zip(ga,gb):
            if a!=b and a in active and b in active:
                adjacent[a].add(b)
                adjacent[b].add(a)
        answer=trial.copy()
        for v in active:
            answer[members[v]]=old[members[v]]
        if self.violations(answer).any():
            raise ArithmeticError('Fixed safe-cycle choices did not eliminate every physical collision.')
        pending=active.copy()
        work=Counter(active_cycles=len(active))
        while pending:
            component=[min(pending)]
            pending.remove(component[0])
            for v in component:
                for neighbor in sorted(adjacent[v]&pending):
                    pending.remove(neighbor)
                    component.append(neighbor)
            ids={v:i for i,v in enumerate(component)}
            edges=np.flatnonzero(np.isin(ga,component)|np.isin(gb,component))
            a,s,b,t=model.a[edges],model.sa[edges],model.b[edges],model.sb[edges]
            ia=np.array([ids.get(int(v),-1) for v in ga[edges]])
            ib=np.array([ids.get(int(v),-1) for v in gb[edges]])
            def score(mask):
                chosen=np.fromiter((bool(mask & (1<<i)) for i in range(len(component))),dtype=bool)
                use_a=(ia>=0)&chosen[np.maximum(ia,0)]
                use_b=(ib>=0)&chosen[np.maximum(ib,0)]
                left=np.where(use_a,trial[a],answer[a])
                right=np.where(use_b,trial[b],answer[b])
                if self.edge_bans(left,s,right,t).any():
                    return -1
                return int(np.count_nonzero(model.masks[left,s]==model.reverse[model.masks[right,t]]))
            n=len(component)
            if n<=limit:
                choices=range(1<<n)
                best=max(choices,key=lambda mask:(score(mask),mask.bit_count(),mask))
                work['exact_cycle_subsets']+=1<<n
            else:
                # Keep the known safe assignment and admit compatible cycles
                # only when they preserve or improve the local physical score.
                best=0
                optimum=score(0)
                for i in range(n):
                    proposal=best | (1<<i)
                    value=score(proposal)
                    if value>=optimum:
                        best,optimum=proposal,value
                work['bounded_components']+=1
            for i,v in enumerate(component):
                if best>>i&1:
                    answer[members[v]]=trial[members[v]]
        model.legal_board(answer)
        if self.violations(answer).any():
            raise ArithmeticError('Selected cycle projection still contains a forbidden physical mate.')
        if model.matches(answer).sum()<model.matches(old).sum():
            answer=old
        return answer,dict(work)


class GuardedPrices(AssignmentPrices):
    physical_neighbor_prices=True

    def __init__(self,model,rules):
        super().__init__(model)
        self.rules=rules

    def update(self,values):
        if self.rules.violations(values).any():
            # A comparison policy may enter with an already forbidden edge.
            # Recompute the unmasked diagonal of any newly current donor.
            self.board=None
        super().update(values)
        for c,code,_ in self.rules.blocked_codes(self.board):
            if code!=self.board[c]:
                self.prices[c,code//3,code%3]=-10
