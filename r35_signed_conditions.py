"""Optional sphere Euler and sparse mate checks for exact signed-inventory searches."""
from functools import lru_cache
from itertools import product
from r31_gold_connectivity import PhaseInterface
from r32_gold_bonds import inventory
from r35_signed_mate_support import matching,raw_edges,reciprocal_classes,zero_supports


def euler_clauses(data,top):
    """A binary carry-save sum of the positive and complemented weighted phases."""
    rows=data.rows
    base=sum(lo[-1] for lo,_ in rows)
    differences=[hi[-1]-lo[-1] for lo,hi in rows]
    # Sum all integer weights without assuming a common factor of two on a new puzzle.
    from math import gcd
    scale=gcd(*differences) or 1
    offset=base+sum(min(0,w) for w in differences)
    weights=[w//scale for w in differences]
    allowed={(v-offset)//scale for v in data.euler if (v-offset)%scale==0}
    buckets=[[] for _ in range(sum(map(abs,weights)).bit_length()+1)]
    for i,w in enumerate(weights,1):
        for bit in range(abs(w).bit_length()):
            if abs(w)>>bit&1:buckets[bit].append(i if w>0 else -i)
    clauses=[];bits=[]
    for j,col in enumerate(buckets):
        while len(col)>1:
            operands=[col.pop() for _ in range(min(3,len(col)))]
            low=top+1;high=top+2;top+=2
            for case in product((0,1),repeat=len(operands)):
                deny=[-v if b else v for v,b in zip(operands,case)]
                count=sum(case)
                clauses.append([*deny,low if count&1 else -low])
                clauses.append([*deny,high if count>=2 else -high])
            col.append(low);buckets[j+1].append(high)
        if col:bits.append((j,col[0]))
    for mask in range(1<<len(bits)):
        value=sum(1<<weight for i,(weight,_) in enumerate(bits) if mask>>i&1)
        if value not in allowed:clauses.append([-v if mask>>i&1 else v for i,(_,v) in enumerate(bits)])
    return clauses,dict(offset=offset,scale=scale,raw_allowed=sorted(data.euler),sum_allowed=sorted(allowed),bits=bits)


class SignedConditions:
    def __init__(self,problem,*,euler=False,mates=False,mate_cost=False):
        self.problem=problem
        self._unprepared=problem
        self.use_euler=euler
        self.mates=mates or mate_cost
        self.mate_cost=mate_cost
        self._scope=None
        self._narrow_cost_scope=None
        self._prices=None
        self._price_rules={}
        self.euler=None
        self.rejecting=set()
        self.initial=set()
        self.families=[]
        self.cached_match=lru_cache(maxsize=8192)(self._match)

    def for_loss(self,reference,groups,required,cap):
        if not self.mate_cost:return self
        scoped=type(self)(self.problem,euler=self.use_euler,mates=True,mate_cost=True)
        scoped._scope=(frozenset(reference),tuple(frozenset(g) for g in groups),frozenset(required),cap)
        return scoped

    def _class_prices(self):
        if self._scope is None:
            raise ValueError("Mate cost requires a bound reference-loss scope.")
        if self._prices is None:
            from r35_mate_price import ClassPrices
            reference, groups, required, _ = self._scope
            # The width belongs to the whole physical scope. A partial proof
            # and all its full completions must price the same set of families.
            width = 8 if self._narrow_prices() else 12
            self._prices = ClassPrices(self.problem, reference, groups, required, limit=width)
        return self._prices

    def _narrow_prices(self):
        if self._narrow_cost_scope is None:
            from r35_mate_native import always_exact
            _, groups, required, _ = self._scope
            self._narrow_cost_scope = always_exact(self.problem, groups, required)
        return self._narrow_cost_scope

    def native_cost_pairs(self):
        if not self.mate_cost:
            return None
        from r35_mate_native import named_pairs
        if self._narrow_prices():
            return named_pairs(self._class_prices())
        return None

    def price(self,phase,ordinary):
        if not self.mate_cost:return ordinary
        strengthened=self._class_prices().memo(phase)
        return ordinary if strengthened is None else max(ordinary,strengthened)

    def _prepare(self):
        problem=self._unprepared
        if problem is None:return
        self.euler=PhaseInterface(problem) if self.use_euler else None
        if self.mates:
            labels,classes=reciprocal_classes(problem.tiles)
            allowed=inventory(problem).possible
            for group in classes:
                edges=raw_edges(labels,group,allowed)
                for pins in zero_supports(edges,group):
                    self.initial.add(self._clause(pins))
                if len(group)<=16:
                    tiles=tuple(sorted({side//3 for side in group}))
                    self.families.append((group,edges,tiles,sum(1<<i for i in tiles)))
        self._initial_masks=[(sum(1<<(abs(v)-1) for v in row),sum(1<<(-v-1) for v in row if v<0)) for row in self.initial]
        self._unprepared=None

    @staticmethod
    def _clause(pins):
        return tuple(-t-1 if value else t+1 for t,value in sorted(pins))

    def _match(self,index,projected):
        group,edges,tiles,_=self.families[index]
        pins={t:(projected>>t)&1 for t in tiles}
        if matching(group,edges,pins):return True
        # Forget a tile only if even the union over its phases has no matching.
        for t in tiles:
            v=pins.pop(t)
            if matching(group,edges,pins):pins[t]=v
        clause=self._clause(pins.items())
        self.rejecting.add(clause)
        self.rejecting.add(tuple(-v for v in clause))
        return False

    def accepts(self,phase):
        self._prepare()
        if any(phase & mask in blocked for mask,blocked in self._price_rules.items()):return False
        if self.euler:
            weight=sum(row[(phase>>i)&1][-1] for i,row in enumerate(self.euler.rows))
            if weight not in self.euler.euler:return False
        if any(phase&mask==bits for mask,bits in self._initial_masks):return False
        if not all(self.cached_match(j,phase&family[3]) for j,family in enumerate(self.families)):return False
        return not self.mate_cost or self.price(phase,0)<=self._scope[3]

    def install(self,model):
        self._prepare()
        if self.euler:
            rows,_=euler_clauses(self.euler,model.solver.nof_vars())
            model.solver.append_formula(rows)
        self._post(model, self.initial | self.rejecting)
        if self.mate_cost and getattr(model, "mate_cost_pairs", None) is not None:
            from r35_mate_native import clauses, named_pairs
            prices = self._class_prices()
            if model.mate_cost_pairs != named_pairs(prices):
                raise ValueError("Native mate costs require the matching physical bond scope.")
            model.solver.append_formula(clauses(prices, model.physical_losses, model.solver.nof_vars()))

    @staticmethod
    def _post(model, rows):
        gauge_fixed = getattr(model, "clauses", ())[:1] == [[-1]]
        for row in sorted(rows):
            if gauge_fixed and -1 in row:
                continue
            model.solver.add_clause([v for v in row if not (gauge_fixed and v == 1)])

    def _price_conflict(self, phase, deadline):
        mask = self._prices.conflict(phase, self._scope[3], deadline)
        blocked = phase & mask
        # Reversing every tile phase preserves all required and optional mate
        # compatibilities and loss prices. Both cylinders have the same bound.
        patterns = {blocked, blocked ^ mask}
        self._price_rules.setdefault(mask, set()).update(patterns)
        rows = {self._clause((t, (pattern >> t) & 1) for t in range(mask.bit_length())
                             if mask & (1 << t)) for pattern in patterns}
        self.rejecting.update(rows)
        return rows

    def decide(self,model,deadline,assumptions=()):
        from math import inf
        from time import monotonic
        while True:
            seconds=deadline-monotonic()
            if seconds<=0:return None
            verdict=model.decide(0 if seconds==inf else seconds,assumptions)
            if verdict is not True:return verdict
            phase=model.phase()
            if self.accepts(phase):return True
            # Another retained solver can discover a clause after this formula
            # was built. Install the actual violated clauses in this solver.
            learned=[row for row in self.rejecting if all(
                ((phase>>(abs(v)-1))&1)!=(v>0) for v in row)]
            if not learned:
                if self.mate_cost and self.price(phase,0)>self._scope[3]:
                    # for_loss gives each cap and physical scope its own rule store.
                    learned=self._price_conflict(phase,deadline)
                else:raise ArithmeticError('Native signed phase violates an installed constraint.')
            self._post(model,learned)
