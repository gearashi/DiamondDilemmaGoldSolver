"""Exact prices of overlapping certified departure groups on one physical phase."""
from functools import lru_cache
from math import inf
from time import monotonic

from r29_gold_phase import EndpointCuts
from r35_integer_phases import IntegerPhaseImages


class OverlapPrices:
    def __init__(self, problem, packed, groups, required, outside):
        sides=EndpointCuts(problem).sides
        covers={}
        options=[]
        for i,group in enumerate(groups):
            possible=group-required
            options.append(possible)
            for pair in possible:
                covers[pair]=covers.get(pair,0)|(1<<i)
        self.all=(1<<len(groups))-1
        self.outside=tuple(covers.get(pair,0) for pair in outside)
        self.bonds=[]
        for a,b in sorted(set().union(*packed)-required):
            mask=(1<<(a//3))^(1<<(b//3))
            rhs=sides[a][1]^sides[b][1]^(sides[a][0].bit_count()%2)
            self.bonds.append((mask,rhs,covers.get((a,b),0)))
        self.options=tuple(tuple(sorted({covers[p] for p in row})) for row in options)
        self.more=lru_cache(maxsize=8192)(self._minimum)

    def _minimum(self, remaining):
        if not remaining:
            return 0
        i=min((j for j in range(len(self.options)) if remaining>>j & 1),key=lambda j:len(self.options[j]))
        return min((1+self.more(remaining & ~hit) for hit in self.options[i]),default=inf)

    def price(self, phase, flipped):
        hit=0
        for i in flipped:
            hit |= self.outside[i]
        lost=len(flipped)
        for mask,rhs,covered in self.bonds:
            if ((phase&mask).bit_count()%2)!=rhs:
                hit |= covered
                lost+=1
        return lost+self.more(self.all & ~hit)


class OverlapPattern:
    def __init__(self,problem,packed,groups,required,outside,cap,integer,work):
        self.image=IntegerPhaseImages.compile(problem,packed,required,outside)
        self.prices=OverlapPrices(problem,packed,groups,required,outside)
        self.integer=integer
        self.cap=cap
        self.work=work
        self.last_witness=None
        self.native=None
        if self.image is None:
            from r35_integer_sat import InventorySAT
            additional=[g for g in groups if g not in packed]
            self.native=InventorySAT(problem,packed,required,outside,cap,additional_groups=additional,
                                     parity_only=not integer)

    def __call__(self,flipped,deadline):
        self.last_witness=None
        if self.image is not None:
            cost,phase,_=self.image.minimum(flipped,deadline,self.work,
                                           require_integer=self.integer,price=self.prices.price)
            if cost<=self.cap:
                self.last_witness=cost,phase
                return True
            return False
        remaining=deadline-monotonic()
        if remaining<=0:
            raise TimeoutError
        pins=[v if i in flipped else -v for i,v in enumerate(self.native.errors)]
        answer=self.native.decide(remaining if remaining!=inf else 0,pins)
        if answer is None:
            raise TimeoutError
        if answer:
            phase=self.native.phase()
            cost=self.prices.price(phase,flipped)
            if cost>self.cap:
                raise ArithmeticError('Native overlapping-group witness exceeded its cap.')
            self.last_witness=cost,phase
        return answer

    def close(self):
        if self.native is not None:
            self.native.close()
