"""Exact bounded integer-lattice relaxations of seam and piece inventory equations.

Choose one origin placement per cell. Every complete arrangement expresses the
required seam/inventory vector as those origins plus integer within-cell
differences. Nonmembership in their integer span is therefore a contradiction;
membership is only unknown. Caps never stand in for a completed membership test.
"""
from collections import OrderedDict
from constraint_relations import bits

STAT_KEYS=('lattice_contexts','lattice_builds','lattice_vectors','lattice_integer_ops','lattice_membership_checks',
           'lattice_trials','lattice_prunes','lattice_reuses','lattice_caps')


class LatticeLimit(Exception):pass


class _Budget:
    def __init__(self,check,stats,limit,max_bits):
        self.check,self.stats,self.limit,self.max_bits=check,stats,limit,max_bits;self.steps=0
    def tick(self,amount=1):
        self.steps+=amount;self.stats['lattice_integer_ops']+=amount;self.check()
        if self.steps>self.limit:raise LatticeLimit('Integer-operation bound reached.')
    def value(self,value):
        if abs(value).bit_length()>self.max_bits:raise LatticeLimit('Integer coefficient bound reached.')


def _vector(value,budget):
    if not isinstance(value,dict):raise ValueError('An exact sparse integer vector is required.')
    result={}
    for key,coefficient in value.items():
        budget.tick()
        if type(key)is not int or key<0 or type(coefficient)is not int:raise ValueError('Malformed exact integer vector.')
        budget.value(coefficient)
        if coefficient:result[key]=coefficient
    return result


def _subtract(left,right,multiplier,budget):
    for key,value in right.items():
        budget.tick();coefficient=left.get(key,0)-multiplier*value;budget.value(coefficient)
        if coefficient:left[key]=coefficient
        else:left.pop(key,None)


class IntegerLattice:
    """A complete triangular basis obtained only by unimodular integer operations."""
    def __init__(self,vectors,check=lambda:None,*,stats=None,max_operations=250000,max_bits=128):
        if (type(max_operations)is not int or max_operations<0 or type(max_bits)is not int or max_bits<1):
            raise ValueError('Invalid integer-lattice bounds.')
        self.stats=dict.fromkeys(STAT_KEYS,0) if stats is None else stats
        self.max_operations,self.max_bits=max_operations,max_bits
        budget=_Budget(check,self.stats,max_operations,max_bits);basis={}
        self.stats['lattice_builds']+=1
        for source in vectors:
            vector=_vector(source,budget);self.stats['lattice_vectors']+=1
            while vector:
                budget.tick();pivot=max(vector)
                if vector[pivot]<0:vector={key:-value for key,value in vector.items()}
                if pivot not in basis:basis[pivot]=vector;break
                prior=basis[pivot];quotient=vector[pivot]//prior[pivot]
                _subtract(vector,prior,quotient,budget)
                if pivot in vector:
                    # Euclidean remainder is positive and smaller than the old
                    # pivot. Swapping plus subtracting preserves integer span.
                    basis[pivot],vector=vector,prior
        check();self.basis=basis  # No partially built basis escapes on Stop/cap.

    def contains(self,source,check=lambda:None):
        budget=_Budget(check,self.stats,self.max_operations,self.max_bits)
        vector=_vector(source,budget);self.stats['lattice_membership_checks']+=1
        while vector:
            budget.tick();pivot=max(vector);row=self.basis.get(pivot)
            if row is None or vector[pivot]%row[pivot]:return False
            _subtract(vector,row,vector[pivot]//row[pivot],budget)
        check();return True


class IntegerLatticePruner:
    def __init__(self,problem,*,limit=4):
        if type(limit)is not int or not 1<=limit<=16:raise ValueError('Invalid lattice cache bound.')
        self.problem=problem;self.limit=limit;self.cache=OrderedDict();self.stats=dict.fromkeys(STAT_KEYS,0)
        self.original=tuple(sum(1<<code for code in set(row)) for row in problem.domains)
        self.binding=self._binding();self.incident=[[] for _ in range(problem.n)]
        for index,(a,sa,b,sb) in enumerate(problem.edges):
            self.incident[a].append((index,sa,1,False));self.incident[b].append((index,sb,-1,True))
        self.inventory_offset=11*len(problem.edges)

    def _binding(self):
        return (self.problem.fingerprint,tuple(tuple(row) for row in self.problem.domains),
                tuple(tuple(map(int,row)) for row in self.problem.masks),tuple(self.problem.edges))

    def _placement(self,cell,code):
        result={self.inventory_offset+code//3:1}
        for edge,side,sign,reverse in self.incident[cell]:
            for port in bits(int(self.problem.masks[code,side])):
                key=11*edge+(10-port if reverse else port);value=result.get(key,0)+sign
                if value:result[key]=value
                else:result.pop(key,None)
        return result

    def _model(self,domains,check,max_placements,max_dimensions):
        if sum(mask.bit_count() for mask in domains)>max_placements:raise LatticeLimit('Placement bound reached.')
        residual={self.inventory_offset+tile:1 for tile in range(self.problem.n)}
        origins=[];differences=[];active=set()
        for cell,domain in enumerate(domains):
            check();choices=list(bits(domain));origin=self._placement(cell,choices[0]);origins.append(origin)
            for key,value in origin.items():
                total=residual.get(key,0)-value
                if total:residual[key]=total
                else:residual.pop(key,None)
            rows={}
            for code in choices:
                check();column=self._placement(cell,code);difference={}
                for key in column.keys()|origin.keys():
                    value=column.get(key,0)-origin.get(key,0)
                    if value:difference[key]=value
                rows[code]=difference;active.update(difference)
            differences.append(rows)
        active.update(residual)
        if len(active)>max_dimensions:raise LatticeLimit('Active integer-equation bound reached.')
        return residual,differences

    def prune(self,domains,check=lambda:None,*,max_candidates=32,max_placements=4096,max_dimensions=512,max_operations=250000,max_bits=128):
        if self._binding()!=self.binding:raise ValueError('Integer-lattice puzzle binding changed.')
        if len(domains)!=len(self.original) or any(type(mask)is not int or mask<0 or mask&~old for mask,old in zip(domains,self.original)):
            raise ValueError('Integer-lattice domains escape the original root.')
        for value in (max_candidates,max_placements,max_dimensions,max_operations):
            if type(value)is not int or value<0:raise ValueError('Invalid integer-lattice work bound.')
        if type(max_bits)is not int or max_bits<1:raise ValueError('Invalid integer coefficient bound.')
        if not all(domains):return False
        check();key=(tuple(domains),max_placements,max_dimensions,max_operations,max_bits)
        session=self.cache.get(key)
        if session is None:
            self.stats['lattice_contexts']+=1
            try:
                residual,differences=self._model(domains,check,max_placements,max_dimensions)
                vectors=[dict(row) for rows in differences for row in sorted({tuple(sorted(v.items())) for v in rows.values()}) if row]
                basis=IntegerLattice(vectors,check,stats=self.stats,max_operations=max_operations,max_bits=max_bits)
                consistent=basis.contains(residual,check)
            except LatticeLimit:
                self.stats['lattice_caps']+=1;self.cache[key]={'limited':True}
                while len(self.cache)>self.limit:self.cache.popitem(last=False)
                return False
            candidates=[(c,code) for c in sorted(range(len(domains)),key=lambda c:(domains[c].bit_count(),c))
                        for code in bits(domains[c]) if domains[c].bit_count()>1]
            session={'residual':residual,'differences':differences,'consistent':consistent,
                     'without':{},'candidates':candidates,'cursor':0,'reduced':tuple(domains)}
            check();self.cache[key]=session
            while len(self.cache)>self.limit:self.cache.popitem(last=False)
        else:self.stats['lattice_reuses']+=1;self.cache.move_to_end(key)
        if session.get('limited'):return False
        if not session['consistent']:
            check();self.stats['lattice_prunes']+=domains[0].bit_count();domains[0]=0;return True
        current=list(session['reduced']);examined=0
        while session['cursor']<len(session['candidates']) and examined<max_candidates and all(current):
            check();cell,code=session['candidates'][session['cursor']]
            try:
                if cell not in session['without']:
                    vectors=[dict(row) for c,rows in enumerate(session['differences']) if c!=cell
                             for row in sorted({tuple(sorted(v.items())) for v in rows.values()}) if row]
                    try:lattice=IntegerLattice(vectors,check,stats=self.stats,max_operations=max_operations,max_bits=max_bits)
                    except LatticeLimit:
                        session['without'].clear();session['without'][cell]=None
                        raise
                    check();session['without'].clear();session['without'][cell]=lattice
                if session['without'][cell] is None:raise LatticeLimit('This fixed-budget basis was already capped.')
                trial=dict(session['residual'])
                for coord,value in session['differences'][cell][code].items():
                    changed=trial.get(coord,0)-value
                    if changed:trial[coord]=changed
                    else:trial.pop(coord,None)
                self.stats['lattice_trials']+=1
                if not session['without'][cell].contains(trial,check):current[cell]&=~(1<<code)
            except LatticeLimit:self.stats['lattice_caps']+=1
            # A completed negative membership removes a choice; a capped trial
            # only advances optional work and removes nothing.
            session['cursor']+=1;session['reduced']=tuple(current);examined+=1
        if session['cursor']==len(session['candidates']):session['without'].clear()
        check();self.stats['lattice_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed
