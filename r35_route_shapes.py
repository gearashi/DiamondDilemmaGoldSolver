"""Bound disconnected cycle-contact shapes using independent physical star contributions."""
from functools import lru_cache
from itertools import permutations, product
from collections import Counter

from r30_cycles import canonical_cycles


class ContactShapes:
    def __init__(self,stars):
        self.stars=stars
        self.maximum=lru_cache(maxsize=8192)(self._maximum)
        self.work=Counter()

    def _maximum(self,supports):
        cells=tuple(sorted(c for group in supports for c in group))
        position={c:i for i,c in enumerate(cells)}
        choices=[tuple((row[0],*rest) for rest in permutations(row[1:])) for row in supports]
        upper=None
        for directions in product(*choices):
            mapping={c:d for cycle in directions for c,d in zip(cycle,(*cycle[1:],cycle[0]))}
            order=tuple(position[mapping[c]] for c in cells)
            gain=self.stars.price(cells,order)-self.stars.bound.total
            upper=gain if upper is None else max(upper,gain)
            self.work['regional_prices']+=1
        return upper

    def groups(self,supports):
        pending=set(supports)
        pieces=[]
        while pending:
            group={pending.pop()}
            fringe=set(group)
            while fringe:
                adjacent={other for other in pending if any(self.stars.close[a,b]
                          for new in fringe for a in new for b in other)}
                pending-=adjacent
                group.update(adjacent)
                fringe=adjacent
            pieces.append(tuple(sorted(group)))
        return pieces

    def admissible(self,supports,target):
        pieces=self.groups(supports)
        if len(pieces)==1:
            return True
        self.work['disconnected_shapes']+=1
        ceiling=self.stars.bound.total+sum(self.maximum(part) for part in pieces)
        return ceiling>=2*target

    def allowed_orders(self,cells,orders,target):
        accepted={}
        for order in orders:
            shape=tuple(sorted(tuple(sorted(g)) for g in canonical_cycles(cells,order)))
            if shape not in accepted:
                accepted[shape]=self.admissible(shape,target)
            if accepted[shape]:
                yield order
            else:
                self.work['discarded_orders']+=1
