"""Project physical phase equations onto loss variables and enumerate sparse outside errors.

The flipped equations form a subset of the actually broken bonds. Paying for each
flip and at least one loss in each disjoint incompatible group is a lower bound.
"""
from bisect import bisect_left
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
import time
from math import comb

from r29_gold_phase import SignedParity
from r32_gold_partition import insert, maximize_disjoint


def projection(base, equations):
    """Bit j of each output column participates in independent consistency check j."""
    rows=dict(base)
    checks=[]
    for index,(mask,rhs) in enumerate(equations):
        symbol=(1<<(index+1))|rhs
        while mask:
            pivot=mask.bit_length()-1
            old=rows.get(pivot)
            if old is None:
                rows[pivot]=mask,symbol
                break
            prior,constant=old
            mask^=prior
            symbol^=constant
        else:
            if symbol:checks.append(symbol)
    col=[sum(1<<j for j,row in enumerate(checks) if row>>i&1)
         for i in range(len(equations)+1)]
    return col[0], col[1:]


@dataclass(frozen=True)
class _SuffixTable:
    anchor: tuple | None
    rhs: int
    columns: tuple
    prices: tuple
    buckets: dict


class OutsideSyndromes:
    @staticmethod
    def layout(reference, groups, required):
        packed = maximize_disjoint(groups)
        required = set(required)
        outside = tuple(sorted(set(reference) - set().union(*packed) - required))
        return packed, required, outside

    def __init__(self, problem, reference, groups, required=(), *, reuse_suffix_tables=True, seek_ordered_tails=True, use_bucket_price=True, layout=None):
        self.groups, self.required, self.outside = (
            self.layout(reference, groups, required) if layout is None else layout)
        # This index belongs to this fixed reference and projection. Only the
        # most recent layer survives, which bounds memory on multi-layer runs.
        self.reuse_suffix_tables = reuse_suffix_tables
        self.seek_ordered_tails = seek_ordered_tails
        self.use_bucket_price = use_bucket_price
        self._suffix_cache = None
        data=SignedParity(problem)
        # Eliminate phase columns only. Symbol bit zero is RHS; others represent errors.
        self.base={i:(m,b) for i,(m,b,_) in data.base.pivots.items()}
        self.inconsistent=data.base.conflict is not None
        self.eq={}
        for a,b in sorted(set(reference)|self.required):
            mask,ca=data.sides[a]
            _,cb=data.sides[b]
            self.eq[a,b]=(1<<(a//3))^(1<<(b//3)), ca^cb^(mask.bit_count()&1)
        for p in self.required:
            mask,rhs=self.eq[p]
            while mask:
                pivot=mask.bit_length()-1
                old=self.base.get(pivot)
                if old is None:
                    self.base[pivot]=mask,rhs
                    break
                mask^=old[0]
                rhs^=old[1]
            else:
                if rhs:self.inconsistent=True
        self.out_eq=[self.eq[p] for p in self.outside]
        self.tables=[self.compile([g]) for g in self.groups]
        rhs,cols=projection(self.base,self.out_eq)
        self.outside_check=rhs,tuple(cols)
        # Check constrained groups first, reducing per-branch projection work.
        self.tables.sort(key=lambda row:len(row[2]))

    def compile(self, groups):
        bonds=[sorted(g-self.required) for g in groups]
        all_pairs=[p for row in bonds for p in row]
        rhs,cols=projection(self.base,self.out_eq+[self.eq[p] for p in all_pairs])
        offset=len(self.outside)
        # Skipping an oversized group relaxation loses pruning only. The full
        # group remains in the final joint test. Avoid exponential tables on
        # unrelated reference loops with many bonds.
        rank_basis={}
        for value in cols[offset:]:
            while value:
                bit=value.bit_length()-1
                if bit not in rank_basis:
                    rank_basis[bit]=value
                    break
                value ^= rank_basis[bit]
        if len(rank_basis)>17:
            return 0,(0,)*offset,{0:len(groups)}
        sums={0:0}
        for row in bonds:
            within={0:0}
            for col in cols[offset:offset+len(row)]:
                old=within.copy()
                for s,v in old.items():
                    key=s^col
                    within[key]=min(within.get(key,v+1),v+1)
            offset+=len(row)
            if not row:return rhs,tuple(cols[:len(self.outside)]),{}
            combine={}
            for s,v in sums.items():
                for w,k in within.items():
                    key=s^w
                    cost=v+max(1,k)
                    combine[key]=min(combine.get(key,cost),cost)
            sums=combine
        return rhs,tuple(cols[:len(self.outside)]),sums

    def joint(self, flipped, budget, deadline):
        """Independent exact cross-group consistency test once the cheap projections pass."""
        basis={pivot+1:(row<<1)|const for pivot,(row,const) in self.base.items()}
        for i,(row,bit) in enumerate(self.out_eq):
            if not insert(basis,(row<<1)|(bit^(i in flipped))):return False
        def resolve(base,groups,left):
            if time.monotonic()>=deadline:raise TimeoutError
            if not groups:return True
            group=groups[0]
            available=sorted(group-self.required)
            for size in range(1,min(left-len(groups)+1,len(available))+1):
                for omitted in combinations(available,size):
                    child=base.copy()
                    if all(insert(child,(self.eq[p][0]<<1)|self.eq[p][1]) for p in group-set(omitted)):
                        if resolve(child,groups[1:],left-size):return True
            return False
        # Try the group with the fewest admissible single drops first.
        def difficulty(group):
            return sum(all(insert(child,(self.eq[p][0]<<1)|self.eq[p][1]) for p in group-{q})
                       for q in group-self.required for child in [basis.copy()])
        return resolve(basis,sorted(self.groups,key=difficulty),budget-len(flipped))

    def _build_suffix_table(self, cap, errors):
        n = len(self.outside)
        length = min(errors, 2)
        _, columns = self.outside_check
        # A projected group further subdivides suffixes. Their lookup
        # keys are exact XOR syndromes; omitted high-cost keys cannot fit cap.
        spare=cap-errors-len(self.groups)
        from operator import xor
        from functools import reduce
        from random import Random
        rng=Random(32)
        trials=[rng.sample(range(n),errors) for _ in range(96)]
        def estimated_hits(tab):
            val,cols,lookup=tab
            return sum(lookup.get(reduce(xor,(cols[i] for i in row),val),cap+2)<=1+spare
                       for row in trials),sum(v<=1+spare for v in lookup.values())
        anchor=min(self.tables,key=estimated_hits,default=None)
        if anchor is not None:
            group_rhs, group_cols, prices = anchor
            charges = tuple((syndrome, cost-1) for syndrome, cost in prices.items() if cost<=1+spare)
        else:
            group_rhs, group_cols, charges = 0, (0,)*n, ((0, 0),)
        buckets=defaultdict(list)
        for indices in combinations(range(n),length):
            value=0
            local=0
            for i in indices:
                value^=columns[i]
                local^=group_cols[i]
            buckets[value,local].append(indices)
        return _SuffixTable(anchor, group_rhs, group_cols, charges, dict(buckets))

    @staticmethod
    def _owned_tails(bucket, last, stats):
        # combinations() orders tuples lexicographically. All suffixes starting
        # at or before the final prefix index form one initial slice.
        if last < 0 or not bucket or bucket[0][0] > last:
            return bucket
        if bucket[-1][0] <= last:
            stats['unowned_tail_visits_skipped'] += len(bucket)
            return ()
        first = bisect_left(bucket, (last + 1,))
        stats['unowned_tail_visits_skipped'] += first
        return bucket[first:]

    def indexed(self, cap, errors, seconds=60, start=0, stop=None, *, accept_errors=None):
        """Index all suffixes of size at most two by their outside-only syndrome.

        The owned prefix ranks have a different scope from check()'s full-set ranks.
        Within a prefix, omitted indices increase, so every actual error set has one owner.
        """
        from itertools import chain
        n=len(self.outside)
        if errors<0 or errors>n or cap<0 or seconds<=0:
            raise ValueError('Invalid projected loss query')
        length=min(errors,2)
        prefix_size=errors-length
        total=comb(n-length,prefix_size)
        end=total if stop is None else stop
        if not 0<=start<=end<=total:raise ValueError('Invalid prefix interval')
        stats=Counter()
        def report(status,rank,**detail):
            return dict(status=status,start=start,stop=end,next_rank=rank,cap=cap,errors=errors,
                        total_prefixes=total,total_error_sets=comb(n,errors),work=dict(stats),**detail)
        if self.inconsistent or len(self.groups)+errors>cap:return report('INFEASIBLE',end)
        if start == end:
            return report('INFEASIBLE', end)
        rhs,columns=self.outside_check
        signature = (cap, errors)
        cached = self._suffix_cache
        if cached is not None and cached[0] == signature:
            table = cached[1]
            stats['suffix_table_reuses'] += 1
        else:
            # Release the previous layer before allocating its replacement.
            self._suffix_cache = None
            del cached
            table = self._build_suffix_table(cap, errors)
            if self.reuse_suffix_tables:
                self._suffix_cache = (signature, table)
            stats['suffix_rows'] = comb(n, length)
        # For the anchor group, a matched bucket already determines its exact
        # projected cost. The remaining group tests and the joint check still run.
        prepaid = self.use_bucket_price and table.anchor is not None
        checks = tuple(t for t in self.tables if t is not table.anchor) if prepaid else self.tables
        spare = cap - errors - len(self.groups)
        duration=time.monotonic()+seconds
        for rank,prefix in ranked_combinations(n-length,prefix_size,start,end):
            if rank%64==0 and time.monotonic()>=duration:return report('UNKNOWN',rank)
            stats['prefixes']+=1
            key=rhs
            local=table.rhs
            for i in prefix:
                key^=columns[i]
                local^=table.columns[i]
            base=[]
            for val,cols,lookup in checks:
                for i in prefix:val^=cols[i]
                base.append((val,cols,lookup))
            last=prefix[-1] if prefix else -1
            for syndrome, charge in table.prices:
                bucket = table.buckets.get((key, local ^ syndrome), ())
                if self.seek_ordered_tails:
                    bucket = self._owned_tails(bucket, last, stats)
                available = spare - charge if prepaid else spare
                if bucket and prepaid:
                    stats['bucket_prices_reused'] += 1
                for ntail,tail in enumerate(bucket):
                    if ntail%4096==0 and time.monotonic()>=duration:return report('UNKNOWN',rank)
                    if not self.seek_ordered_tails and tail and tail[0] <= last:
                        stats['unowned_tail_visits'] += 1
                        continue
                    stats['indexed_tails']+=1
                    slack = available
                    for val,cols,lookup in base:
                        for i in tail:val^=cols[i]
                        slack-=lookup.get(val,cap+2)-1
                        if slack<0:break
                    else:
                        stats['joint_queries']+=1
                        try:
                            errors_here = set(chain(prefix, tail))
                            if not self.joint(errors_here, cap, duration):
                                continue
                            status = 'PHASE_RELAXATION_FEASIBLE'
                            if accept_errors is not None:
                                stats['integer_checks'] += 1
                                if not accept_errors(errors_here, duration):
                                    stats['integer_rejections'] += 1
                                    continue
                                status = 'INTEGER_RELAXATION_FEASIBLE'
                            return report(status, rank, outside_errors=[self.outside[i] for i in chain(prefix, tail)])
                        except TimeoutError:return report('UNKNOWN',rank)
        return report('INFEASIBLE',end)

    def check(self, cap, errors, seconds=60, start=0, stop=None):
        if errors<0 or errors>len(self.outside) or cap<0 or seconds<=0:
            raise ValueError('Invalid projected loss interval')
        limit=comb(len(self.outside),errors)
        end=limit if stop is None else stop
        if not 0<=start<=end<=limit:raise ValueError('Invalid projected query interval')
        stats=Counter()
        def output(status,rank,**fields):
            return dict(status=status,start=start,stop=end,next_rank=rank,cap=cap,errors=errors,
                        total=limit,work=dict(stats),**fields)
        if self.inconsistent or len(self.groups)+errors>cap:return output('INFEASIBLE',end)
        # Direct lexicographic seek permits nonoverlapping worker intervals and restarts.
        deadline=time.monotonic()+seconds
        cost_limit=cap-errors
        for rank,choice in ranked_combinations(len(self.outside),errors,start,end):
            if rank%128==0 and time.monotonic()>=deadline:return output('UNKNOWN',rank)
            stats['error_sets']+=1
            slack=cost_limit-len(self.groups)
            for rhs,cols,lookup in self.tables:
                state=rhs
                for i in choice:state ^= cols[i]
                slack-=lookup.get(state,cap+2)-1
                if slack<0:break
            else:
                stats['joint_queries']+=1
                try:
                    if self.joint(set(choice),cap,deadline):
                        return output('PHASE_RELAXATION_FEASIBLE',rank,
                                      outside_errors=[self.outside[i] for i in choice])
                except TimeoutError:return output('UNKNOWN',rank)
        return output('INFEASIBLE',end)


def ranked_combinations(n,k,start,stop):
    """Seek directly to a lexicographic rank and stream that half-open range."""
    if start == stop:
        return
    j = 0
    choice = []
    index = start
    for rem in range(k-1,-1,-1):
        while True:
            width=comb(n-1-j,rem)
            if index < width:break
            index-=width
            j+=1
        choice.append(j)
        j+=1
    for rank in range(start,stop):
        yield rank,choice
        for i in range(k-1,-1,-1):
            if choice[i]<n-k+i:
                choice[i]+=1
                for j in range(i+1,k):choice[j]=choice[j-1]+1
                break
