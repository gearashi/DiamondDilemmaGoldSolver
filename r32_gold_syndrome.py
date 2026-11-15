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
from hashlib import sha256
import json

from r35_group_drops import GroupDropTimeout, reuse_drops
from r29_gold_phase import SignedParity
from r32_gold_partition import insert, maximize_disjoint


def eliminate_phases(base, equations):
    """Eliminate tile phases while keeping an affine RHS in the outside-error bits."""
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
    return rows, checks


def check_columns(checks, size):
    col=[sum(1<<j for j,row in enumerate(checks) if row>>i&1)
         for i in range(size+1)]
    return col[0], col[1:]


def projection(base, equations):
    """Bit j of each output column participates in independent consistency check j."""
    _, checks = eliminate_phases(base, equations)
    return check_columns(checks, len(equations))


@dataclass(frozen=True)
class _SuffixTable:
    anchor: tuple | None
    prices: tuple
    buckets: dict
    packed_tails: dict
    secondary: tuple | None
    split_buckets: dict
    order_key: str


class OutsideSyndromes:
    @staticmethod
    def layout(reference, groups, required):
        packed = maximize_disjoint(groups)
        required = set(required)
        outside = tuple(sorted(set(reference) - set().union(*packed) - required))
        return packed, required, outside

    def __init__(self, problem, reference, groups, required=(), *, reuse_suffix_tables=True, seek_ordered_tails=True, use_bucket_price=True, use_second_group=True, layout=None):
        self.groups, self.required, self.outside = (
            self.layout(reference, groups, required) if layout is None else layout)
        # This index belongs to this fixed reference and projection. Only the
        # most recent layer survives, which bounds memory on multi-layer runs.
        self._cursor_scope = (problem.fingerprint, self.outside, tuple(sorted(self.required)),
                              tuple(tuple(sorted(g)) for g in self.groups))
        self.resume_cursor = None
        self.checkpoint_progress = None
        self.reuse_suffix_tables = reuse_suffix_tables
        self.seek_ordered_tails = seek_ordered_tails
        self.use_bucket_price = use_bucket_price
        self.use_second_group = use_second_group
        self._suffix_cache = None
        self._catalogues = {}
        self._primary_index = None
        self._secondary_index = None
        self.index_work = Counter()
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
        phase_rows, dependencies = eliminate_phases(self.base, self.out_eq)
        # Quotient each group equation by this fixed echelon basis. The
        # remaining variables are free, and the RHS is affine in the errors.
        self._joint_bonds = {}
        pivots = sorted(phase_rows.items(), reverse=True)
        for bond in set().union(*self.groups):
            mask, symbol = self.eq[bond]
            for pivot, (row, constant) in pivots:
                if (mask >> pivot) & 1:
                    mask ^= row
                    symbol ^= constant
            self._joint_bonds[bond] = mask << 1, symbol
        self._joint_dependencies = tuple(dependencies)
        from r35_group_images import GroupImages
        self._joint_images = GroupImages.compile(self.groups, self.required, self._joint_bonds, len(self.outside))
        rhs,cols=check_columns(dependencies,len(self.out_eq))
        self.outside_check=rhs,tuple(cols)
        # Check constrained groups first, reducing per-branch projection work.
        self.tables.sort(key=lambda row:len(row[2]))
        # Put independent projections in nonoverlapping bit fields so that a
        # single XOR updates every residual at once.
        self._fields = {}
        self._packed_rhs = 0
        self._packed_cols = [0] * len(self.outside)
        offset = 0
        for table in (self.outside_check, *self.tables):
            rhs, cols = table[:2]
            width = max(rhs, max(cols, default=0)).bit_length()
            self._fields[id(table)] = (offset, (1 << width)-1)
            self._packed_rhs |= rhs << offset
            for i, col in enumerate(cols):
                self._packed_cols[i] |= col << offset
            offset += width

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

    def joint(self, flipped, budget, deadline, *, inner=(), checkpoint=None):
        """Independent exact cross-group consistency test once the cheap projections pass."""
        if time.monotonic() >= deadline:
            raise TimeoutError
        errors = 1 | sum(1 << (i+1) for i in flipped)
        if any((condition & errors).bit_count() & 1 for condition in self._joint_dependencies):
            return False
        if self.inconsistent or len(flipped) + len(self.groups) > budget:
            return False
        if self._joint_images is not None:
            return self._joint_images.minimum(flipped, deadline) <= budget - len(flipped)
        return reuse_drops(self, flipped, budget, deadline, Counter(), resume=inner, checkpoint=checkpoint)

    def _build_suffix_table(self, cap, errors):
        n = len(self.outside)
        length = min(errors, 2)
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
            _, _, prices = anchor
            charges = tuple((syndrome, cost-1) for syndrome, cost in prices.items() if cost<=1+spare)
        else:
            charges = ((0, 0),)
        catalogue = self._catalogues.get(length) if self.reuse_suffix_tables else None
        if catalogue is None:
            packed_tails = {}
            by_outside = defaultdict(list)
            offset, mask = self._fields[id(self.outside_check)]
            for indices in combinations(range(n),length):
                packed = 0
                for i in indices:
                    packed ^= self._packed_cols[i]
                packed_tails[indices] = packed
                by_outside[(packed >> offset) & mask].append(indices)
            self.index_work['catalogue_rows'] += len(packed_tails)
            catalogue = packed_tails, by_outside
            if self.reuse_suffix_tables:
                self._catalogues[length] = catalogue
        packed_tails, by_outside = catalogue
        anchor_key = (length,id(anchor))
        if self._primary_index is not None and self._primary_index[0] == anchor_key:
            buckets = self._primary_index[1]
        else:
            buckets = defaultdict(list)
            offset, mask = self._fields[id(anchor)] if anchor is not None else (0,0)
            for outside, tails in by_outside.items():
                for tail in tails:
                    buckets[outside,(packed_tails[tail] >> offset) & mask].append(tail)
            self.index_work['primary_rows'] += len(packed_tails)
            if self.reuse_suffix_tables:
                self._primary_index = anchor_key,buckets
        secondary = None
        split_buckets = {}
        if self.use_second_group and self.use_bucket_price and charges:
            # The first lookup already enforces outside consistency. Estimate
            # remaining selectivity over the reachable prices of each group.
            def permitted_fraction(table):
                prices = table[2]
                return sum(v <= spare+1 for v in prices.values()) / max(1, len(prices))

            candidates = [t for t in self.tables if t is not anchor and permitted_fraction(t)<0.5]
            secondary = min(candidates, key=permitted_fraction, default=None)
        secondary_key = (anchor_key,id(secondary))
        if secondary is not None and self._secondary_index is not None and self._secondary_index[0] == secondary_key:
            split_buckets = self._secondary_index[1]
        elif secondary is not None:
            offset, mask = self._fields[id(secondary)]
            for key, bucket in buckets.items():
                if len(bucket) < 8:
                    continue
                positions = defaultdict(int)
                for ordinal, tail in enumerate(bucket):
                    value = (packed_tails[tail] >> offset) & mask
                    positions[value] |= 1 << ordinal
                split_buckets[key] = dict(positions)
                self.index_work['secondary_rows'] += len(bucket)
            if self.reuse_suffix_tables:
                self._secondary_index = secondary_key,split_buckets
        # Tails are lexicographic within each first-anchor bucket. Bind the
        # precise bucket ordering to any later suffix resume token.
        order_key = sha256(repr((1,self._cursor_scope,cap,errors,self.outside_check,
                                anchor[:2] if anchor is not None else (),charges)).encode()).hexdigest()
        return _SuffixTable(anchor, charges, buckets,
                            packed_tails, secondary, split_buckets, order_key)

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

    @staticmethod
    def _secondary_tails(bucket, positions, prefix, prices, spare, last, stats, resume_tail=None):
        """Emit surviving tails in the original first-anchor and tuple order."""
        selected = 0
        for value, ordinals in positions.items():
            if prices.get(value ^ prefix, spare+2) <= spare+1:
                selected |= ordinals
        if selected and last >= 0:
            cut = bisect_left(bucket, (last+1,))
            unowned = selected & ((1 << cut)-1)
            stats['unowned_tail_visits_skipped'] += unowned.bit_count()
            selected ^= unowned
        if selected and resume_tail is not None:
            cut = bisect_left(bucket,resume_tail)
            selected &= -(1 << cut)
        while selected:
            first = selected & -selected
            yield bucket[first.bit_length()-1]
            selected ^= first

    def indexed(self, cap, errors, seconds=60, start=0, stop=None, *, accept_errors=None, reject_before_joint=None, cursor=None, checkpoint=None):
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
        cursor = self.resume_cursor if cursor is None else cursor
        checkpoint = self.checkpoint_progress if checkpoint is None else checkpoint
        refined = accept_errors is not None

        def make_cursor(rank, price, tail, stage, inner=None):
            payload = [table.order_key,refined,rank,price,list(tail)]
            if inner:
                payload.append([1,list(inner)])
            payload.append(stage)
            return [payload,sha256(json.dumps(payload,separators=(',',':')).encode()).hexdigest()]

        resume = None
        if isinstance(cursor,(tuple,list)) and len(cursor)==2:
            try:
                key,mode,rank,price,tail,*extra,stage = cursor[0]
                inner = ()
                if extra:
                    tag,inner = extra[0]
                    if (len(extra)!=1 or tag!=1 or stage!='joint' or not isinstance(inner,list)
                            or len(inner)>len(self.groups) or any(type(i) is not int or i<0 for i in inner)):
                        raise ValueError('Invalid group cursor')
                tail = tuple(tail)
                if (key==table.order_key and mode is refined and type(rank) is int and rank==start
                        and type(price) is int and 0<=price<len(table.prices)
                        and len(tail)==length and all(type(t) is int and 0<=t<n for t in tail)
                        and tuple(sorted(set(tail)))==tail and stage in ('tail','joint','integer')
                        and (stage!='integer' or refined)
                        and make_cursor(rank,price,tail,stage,inner)[1]==cursor[1]):
                    resume = price,tail,stage,inner
            except (TypeError,ValueError):
                pass
        next_flush = time.monotonic()+2.0

        def flush(rank, price, tail, stage, force=False, inner=None):
            nonlocal next_flush
            if checkpoint is not None and (force or time.monotonic()>=next_flush):
                checkpoint(rank,make_cursor(rank,price,tail,stage,inner))
                next_flush=time.monotonic()+2.0

        # For the anchor group, a matched bucket already determines its exact
        # projected cost. The remaining group tests and the joint check still run.
        prepaid = self.use_bucket_price and table.anchor is not None
        checks = tuple(t for t in self.tables if t is not table.anchor) if prepaid else self.tables
        packed_checks = tuple((*self._fields[id(t)], t[2]) for t in checks)
        outside_offset, outside_mask = self._fields[id(self.outside_check)]
        anchor_offset, anchor_mask = self._fields[id(table.anchor)] if table.anchor is not None else (0, 0)
        prefix_states = [self._packed_rhs] + [0]*prefix_size
        previous = [-1]*prefix_size
        spare = cap - errors - len(self.groups)
        duration=time.monotonic()+seconds
        for rank,prefix in ranked_combinations(n-length,prefix_size,start,end):
            if rank%64==0 and time.monotonic()>=duration:return report('UNKNOWN',rank)
            stats['prefixes']+=1
            # Lexicographic successors retain most leading indices. Rebuild
            # cumulative states starting at the first changed index.
            for position, index in enumerate(prefix):
                if previous[position] != index:
                    for j in range(position, prefix_size):
                        previous[j] = prefix[j]
                        prefix_states[j+1] = prefix_states[j] ^ self._packed_cols[prefix[j]]
                    break
            state = prefix_states[-1]
            key = (state >> outside_offset) & outside_mask
            local = (state >> anchor_offset) & anchor_mask
            last=prefix[-1] if prefix else -1
            start_price = 0
            if rank==start and resume is not None:
                price,tail,stage,_ = resume
                coordinate = (key,local ^ table.prices[price][0])
                bucket = table.buckets.get(coordinate,())
                i = bisect_left(bucket,tail)
                if i<len(bucket) and bucket[i]==tail and (not tail or tail[0]>last):
                    start_price=price
                    stats['suffix_cursors_resumed']+=1
                else:
                    resume=None
            for price_index in range(start_price,len(table.prices)):
                syndrome, charge = table.prices[price_index]
                resume_tail = resume[1] if resume is not None and rank==start and price_index==start_price else None
                bucket_key = (key, local ^ syndrome)
                bucket = table.buckets.get(bucket_key, ())
                available = spare - charge if prepaid else spare
                split = table.split_buckets.get(bucket_key)
                if split:
                    offset, mask = self._fields[id(table.secondary)]
                    prefix_value = (state >> offset) & mask
                    bucket = self._secondary_tails(bucket, split, prefix_value, table.secondary[2],
                                                   available, last if self.seek_ordered_tails else -1, stats, resume_tail)
                else:
                    if self.seek_ordered_tails:
                        bucket = self._owned_tails(bucket, last, stats)
                    if resume_tail is not None:
                        bucket = bucket[bisect_left(bucket,resume_tail):]
                for ntail,tail in enumerate(bucket):
                    if ntail == 0 and prepaid:
                        stats['bucket_prices_reused'] += 1
                    if not self.seek_ordered_tails and tail and tail[0] <= last:
                        stats['unowned_tail_visits'] += 1
                        continue
                    resume_integer = tail==resume_tail and resume[2]=='integer'
                    inner = resume[3] if tail==resume_tail and resume[2]=='joint' else ()
                    if ntail%4096==0:
                        if time.monotonic()>=duration:
                            stage='integer' if resume_integer else ('joint' if inner else 'tail')
                            return report('UNKNOWN',rank,cursor=make_cursor(rank,price_index,tail,stage,inner))
                        if checkpoint is not None:
                            flush(rank,price_index,tail,'integer' if resume_integer else ('joint' if inner else 'tail'),inner=inner)
                    stats['indexed_tails']+=1
                    slack = available
                    residual = state ^ table.packed_tails[tail]
                    for offset, mask, lookup in packed_checks:
                        price = lookup.get((residual >> offset) & mask, cap+2)
                        slack -= price - 1
                        if slack<0:break
                    else:
                        errors_here = set(chain(prefix, tail))
                        stage = 'integer' if resume_integer else 'joint'
                        try:
                            if reject_before_joint is not None and reject_before_joint(errors_here,duration):
                                stats['integer_core_joint_queries_avoided'] += 1
                                continue
                            if resume_integer:
                                stats['saved_joint_decisions_reused'] += 1
                            else:
                                if checkpoint is not None:
                                    flush(rank,price_index,tail,'joint',inner=inner)
                                stats['joint_queries']+=1
                                options = {}
                                if self._joint_images is None and (inner or checkpoint is not None):
                                    options['inner'] = inner
                                    if checkpoint is not None:
                                        options['checkpoint'] = lambda path: flush(rank,price_index,tail,'joint',inner=path)
                                if not self.joint(errors_here, cap, duration, **options):
                                    continue
                            status = 'PHASE_RELAXATION_FEASIBLE'
                            if accept_errors is not None:
                                stage='integer'
                                if checkpoint is not None:
                                    flush(rank,price_index,tail,stage,force=True)
                                stats['integer_checks'] += 1
                                if not accept_errors(errors_here, duration):
                                    stats['integer_rejections'] += 1
                                    continue
                                status = 'INTEGER_RELAXATION_FEASIBLE'
                            return report(status, rank, outside_errors=[self.outside[i] for i in chain(prefix, tail)])
                        except TimeoutError as exc:
                            saved = exc.path if isinstance(exc,GroupDropTimeout) else (inner if stage=='joint' else ())
                            return report('UNKNOWN',rank,cursor=make_cursor(rank,price_index,tail,stage,saved))
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
