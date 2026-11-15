"""Enumerate complete signed inventories when outside-bond equations leave few tile phases."""
from collections import Counter
from math import inf
from time import monotonic

from r29_gold_phase import SignedParity
from r31_gold_connectivity import PhaseInterface


class IntegerPhaseImages:
    @classmethod
    def compile(cls, problem, groups, required, outside, max_rank=10):
        from r32_gold_syndrome import eliminate_phases
        parity=SignedParity(problem)
        # All constraints are invariant under complementing every tile color.
        parity.pin(0,0)
        for a,b in sorted(required):
            parity.bond(a,b)
        if parity.base.conflict is not None:
            return None
        base={i:(row,rhs) for i,(row,rhs,why) in parity.base.pivots.items()}
        def relation(pair):
            a,b=pair
            mask,ca=parity.sides[a]
            _,cb=parity.sides[b]
            return (1<<(a//3))^(1<<(b//3)), ca^cb^(mask.bit_count()&1)
        rows,checks=eliminate_phases(base,[relation(b) for b in outside])
        if problem.n-len(rows)>max_rank:
            return None
        symbols=[]
        free=[]
        k=0
        for tile in range(problem.n):
            if tile not in rows:
                free.append(1<<k)
                symbols.append(0)
                k+=1
                continue
            row,symbol=rows[tile]
            parts=row^(1<<tile)
            effect=0
            while parts:
                bit=parts&-parts
                parts^=bit
                previous=bit.bit_length()-1
                effect^=free[previous]
                symbol^=symbols[previous]
            free.append(effect)
            symbols.append(symbol)
        vector=lambda bit,values:sum(1<<j for j,v in enumerate(values) if v>>bit&1)
        images=[0]
        for bit in range(k):
            column=vector(bit,free)
            images.extend([v^column for v in images])
        columns=[vector(bit,symbols) for bit in range(len(outside)+1)]
        data=PhaseInterface(problem)
        inventories=[]
        for j,mod2 in enumerate(data.mod[:-1]):
            base_sum=0
            terms={}
            support=0
            for tile,(low,high) in enumerate(data.rows):
                base_sum+=low[j]
                weight=high[j]-low[j]
                if mod2:
                    weight%=2
                if weight:
                    terms[weight]=terms.get(weight,0)|(1<<tile)
                    support|=symbols[tile]>>1
            inventories.append((support,mod2,base_sum,tuple(terms.items())))
        inventories.sort(key=lambda item:item[0].bit_count())
        groups=[(bool(group-required),tuple(relation(b) for b in group)) for group in groups]
        return cls(checks,columns,images,inventories,groups)

    def __init__(self,checks,columns,images,inventories,groups):
        self.checks=checks
        self.columns=columns
        self.images=images
        self.inventories=inventories
        self.groups=groups

    def minimum(self, flipped, deadline=inf, work=None, *, require_integer=True, price=None):
        """Return the attained integer cost and phase, or a sufficient inventory core."""
        if monotonic()>=deadline:
            raise TimeoutError
        value=sum(1<<i for i in flipped)
        errors=1|(value<<1)
        for row in self.checks:
            if (row&errors).bit_count()&1:
                return inf,None,row>>1
        if any(not removable for removable,group in self.groups):
            return inf,None,0
        origin=self.columns[0]
        for i in flipped:
            origin^=self.columns[i+1]
        best=inf
        witness=None
        sufficient=0
        for j,image in enumerate(self.images):
            if j%16==0 and monotonic()>=deadline:
                raise TimeoutError
            phase=origin^image
            if work is not None:
                label='integer_phase_assignments' if require_integer else 'parity_phase_assignments'
                work[label]=work.get(label,0)+1
            for support,mod2,base,terms in (self.inventories if require_integer else ()):
                balance=base+sum(weight*(phase&mask).bit_count() for weight,mask in terms)
                if (balance%2 if mod2 else balance):
                    sufficient|=support
                    break
            else:
                if price is None:
                    charge=len(flipped)
                    for _,group in self.groups:
                        mismatches=sum(((phase&mask).bit_count()&1)^rhs for mask,rhs in group)
                        charge+=max(1,mismatches)
                else:
                    charge=price(phase,flipped)
                if charge<best:
                    best=charge
                    witness=phase
        return best,witness,sufficient if witness is None else None



class EnumeratedPatternFilter:
    def __init__(self, image, outside, cap, store=None, work=None, published=None, *, transfer=None):
        self.transfer=transfer
        self.image=image
        self.outside=tuple(outside)
        self.cap=cap
        self.store=store
        self.work=Counter() if work is None else work
        self.verified=set()
        self.pending=set()
        full=(1<<len(self.outside))-1
        if store is not None:
            self.pending.update((mask,value) for mask,value in store.read()
                                if 0<=mask<=full and value>=0 and (value&mask)==value)
        if published is not None:
            self.pending.add(published)
        self._last=None
        self.last_witness=None

    def _increment(self,key):
        self.work[key]=self.work.get(key,0)+1

    def _evaluate(self,indices,value,deadline):
        if self._last is not None and self._last[0]==value:
            self.last_witness=self._last[2]
            return self._last[1]
        cost,phase,support=self.image.minimum(indices,deadline,self.work)
        self._increment('enumerated_integer_patterns')
        feasible=cost<=self.cap
        self.last_witness=(cost,phase) if feasible else None
        if not feasible:
            mask=(1<<len(self.outside))-1 if support is None else support
            subset=value&mask
            if not any((m&mask)==m and subset&m==v for m,v in self.verified):
                supersets={(m,v) for m,v in self.verified if (m&mask)==mask and (v&mask)==subset}
                self.verified.difference_update(supersets)
                self.verified.add((mask,subset))
                self._increment('integer_cores_learned')
                if support is not None and self.transfer is not None:
                    self.transfer.remember(mask,subset)
                if self.store is not None:
                    self.store.remember(mask,subset)
        self._last=value,feasible,self.last_witness
        return feasible

    def excludes(self,indices,deadline=inf):
        value=sum(1<<i for i in indices)
        if any((value&mask)==v for mask,v in self.verified):
            return True
        if self._check_transfer(value,deadline):
            return True
        matched=[(mask,v) for mask,v in self.pending if (value&mask)==v]
        if not matched:
            return False
        # A saved envelope is a hint. Compute the actual full pattern and
        # learn only the support our independently checked phase sum needs.
        answer=self._evaluate(indices,value,deadline)
        self.pending.difference_update(matched)
        self._increment('integer_core_replays')
        return not answer

    def _check_transfer(self,value,deadline):
        core = self.transfer.matched(value,deadline,self.work) if self.transfer is not None else None
        if core is None:
            return False
        self.verified.add(core)
        if self.store is not None:
            self.store.remember(*core)
        self.last_witness=None
        return True

    def close(self):
        if self.transfer is not None:
            self.transfer.close()

    def __call__(self,indices,deadline):
        indices=frozenset(indices)
        if any(type(i) is not int or not 0<=i<len(self.outside) for i in indices):
            raise ValueError('Outside error index out of range.')
        if not deadline>monotonic():
            raise TimeoutError
        value=sum(1<<i for i in indices)
        if any((value&mask)==v for mask,v in self.verified):
            self._increment('integer_native_calls_avoided')
            self.last_witness=None
            return False
        if self._check_transfer(value,deadline):
            return False
        return self._evaluate(indices,value,deadline)
