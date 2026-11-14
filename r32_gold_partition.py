"""Exact reference-loss layers using disjoint physical conflicts and signed inventory."""
from collections import Counter
from itertools import combinations
import time

from r29_gold_phase import SignedParity


def maximize_disjoint(rows, limit=50000):
    """Exact within this candidate family when the bounded enumeration finishes."""
    rows=sorted(rows,key=lambda c:(-len(c),sorted(c)))
    best=[]
    work=0
    def visit(i,used,taken):
        nonlocal best,work
        if len(taken)>len(best):best=taken.copy()
        if i==len(rows) or len(taken)+len(rows)-i<=len(best) or work>=limit:return
        work+=1
        if not used & rows[i]:
            taken.append(rows[i])
            visit(i+1,used|rows[i],taken)
            taken.pop()
        visit(i+1,used,taken)
    visit(0,set(),[])
    return sorted(best,key=lambda c:(len(c),sorted(c)))


def insert(basis, row):
    # RHS occupies bit zero and is never a phase pivot.
    while row>1:
        pivot=row.bit_length()-1
        old=basis.get(pivot)
        if old is None:
            basis[pivot]=row
            return True
        row^=old
    return row==0


class ParityLoss:
    def __init__(self, problem, reference, groups, required=()):
        self.groups=maximize_disjoint(groups)
        self.outside=tuple(sorted(set(reference)-set().union(*self.groups)))
        self.base={}
        self.rows={}
        proof=SignedParity(problem)
        self.inconsistent = proof.base.conflict is not None
        for mask,rhs,_ in proof.base.pivots.values():
            if not insert(self.base,(mask<<1)|rhs):
                self.inconsistent=True
        for a,b in set(reference)|set(required):
            ma,ca=proof.sides[a]
            _,cb=proof.sides[b]
            eq=(1<<(a//3+1))^(1<<(b//3+1))^ca^cb^(ma.bit_count()&1)
            self.rows[a,b]=eq
        for p in required:
            if not insert(self.base,self.rows[p]):
                self.inconsistent=True
        self.required=set(required)

    def interval(self, upper, outside_count, start, stop, seconds, joint=True):
        """Lexicographic combinations identify every possible exact outside-loss set.

        Search all assignments to remaining per-group loss counts and identities when
        joint=True. Relaxed success is only an obstruction to proving a negative.
        """
        from math import comb
        if upper<0 or outside_count<0 or outside_count>len(self.outside) or seconds<=0:
            raise ValueError('Invalid loss-layer budget')
        n=len(self.outside)
        total=comb(n,outside_count)
        stop=total if stop is None else stop
        if not 0<=start<=stop<=total:raise ValueError('Invalid outside combination interval')
        expired=time.monotonic()+seconds
        work=Counter()
        def result(state,next_rank,**extra):
            return dict(status=state,start=start,stop=stop,next_rank=next_rank,total=total,
                        outside_count=outside_count,budget=upper,work=dict(work),**extra)
        if self.inconsistent or outside_count+len(self.groups)>upper:
            return result('INFEASIBLE',stop)
        def unrank(rank):
            chosen=[]
            j=0
            for rem in range(outside_count-1,-1,-1):
                while True:
                    width=comb(n-1-j,rem)
                    if rank<width:break
                    rank-=width
                    j+=1
                chosen.append(j)
                j+=1
            return chosen
        group_rows=[tuple((p,self.rows[p]) for p in sorted(c)) for c in self.groups]
        # Impose common stretches once per outside subset. Full bases are immutable to siblings.
        def extend(base, group, lost):
            b=base.copy()
            for p,r in group:
                if p not in lost and not insert(b,r):return None
            return b
        def joined(base, pending, allowance,chosen):
            if not pending:return chosen
            if time.monotonic()>=expired:raise TimeoutError
            group=pending[0]
            choices=[p for p,_ in group if p not in self.required]
            for size in range(1,min(len(choices),allowance-len(pending)+1)+1):
                for removed in combinations(choices,size):
                    work['joint_choices']+=1
                    narrowed=extend(base,group,removed)
                    if narrowed is None:continue
                    answer=joined(narrowed,pending[1:],allowance-size,chosen+list(removed))
                    if answer is not None:return answer
            return None
        for rank in range(start,stop):
            if time.monotonic()>=expired:return result('UNKNOWN',rank)
            deleted={self.outside[i] for i in unrank(rank)}
            work['outside_cases']+=1
            if deleted & self.required:continue
            base=self.base.copy()
            if any(not insert(base,self.rows[p]) for p in self.outside if p not in deleted):
                work['outside_parity_conflicts']+=1
                continue
            difficult=[]
            extra=upper-outside_count-len(self.groups)
            for group in group_rows:
                eligible=[p for p,r in group if p not in self.required]
                if not eligible:
                    difficult=None
                    break
                if not any(extend(base,group,{p}) is not None for p in eligible):
                    difficult.append(group)
                    if len(difficult)>extra:break
            if difficult is None or len(difficult)>extra:
                work['single_loss_groups_closed']+=1
                continue
            if not joint:return result('RELAXATION_OPEN',rank,witness_outside=sorted(deleted))
            remaining=difficult+[g for g in group_rows if g not in difficult]
            try:
                answer=joined(base,remaining,upper-outside_count,[])
            except TimeoutError:
                return result('UNKNOWN',rank)
            if answer is not None:
                return result('PHASE_RELAXATION_FEASIBLE',rank,
                              relaxed_losses=sorted(set(answer)|deleted))
            work['joint_closed']+=1
        return result('INFEASIBLE',stop)
