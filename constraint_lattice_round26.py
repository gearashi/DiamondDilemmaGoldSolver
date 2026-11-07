"""Reversible contribution-tree lattices and independently checked separators.

Subtree bases represent integer spans, not rational spans. A completed merge is
reusable only for exactly the same cell domains and arithmetic policy. A cached
basis never supplies authority for an imported reason: sparse modular/equality
functionals are checked against every actual allowed placement column.
"""
from collections import OrderedDict,deque
from fractions import Fraction
from math import gcd,lcm
from constraint_lattice import IntegerLattice,IntegerLatticePruner,LatticeLimit,_Budget,_vector,_subtract
from constraint_relations import bits

TREE_KEYS=('lattice_tree_nodes','lattice_tree_reuses','lattice_tree_merges','lattice_tree_without_reuses',
 'lattice_tree_source_vectors','lattice_separator_checks','lattice_separator_terms','lattice_separator_produced')


def separating_functional(lattice,source,check=lambda:None):
    budget=_Budget(check,lattice.stats,lattice.max_operations,lattice.max_bits);vector=_vector(source,budget)
    lattice.stats['lattice_membership_checks']+=1;pivot=None
    while vector:
        budget.tick();p=max(vector);row=lattice.basis.get(p)
        if row is None or vector[p]%row[p]:pivot=p;break
        _subtract(vector,row,vector[p]//row[p],budget)
    if pivot is None:return None
    # Eliminate higher pivot coordinates in the dual, in increasing order.
    # The resulting functional annihilates every other basis vector exactly.
    weights={pivot:Fraction(1)}
    for p in sorted(q for q in lattice.basis if q>pivot):
        row=lattice.basis[p];coefficient=Fraction(0)
        for coord,value in row.items():budget.tick();coefficient+=weights.get(coord,0)*value
        if coefficient:
            weights[p]=-coefficient/row[p]
            budget.value(weights[p].numerator);budget.value(weights[p].denominator)
    denominator=1
    for value in weights.values():denominator=lcm(denominator,value.denominator);budget.value(denominator)
    integers={coord:int(value*denominator) for coord,value in weights.items() if value}
    modulus=denominator*lattice.basis[pivot][pivot] if pivot in lattice.basis else 0
    common=abs(modulus)
    for value in integers.values():common=gcd(common,abs(value))
    if common>1:integers={c:v//common for c,v in integers.items()};modulus//=common
    for value in (*integers.values(),modulus):budget.value(value)
    check();return {'weights':[[c,v] for c,v in sorted(integers.items())],'modulus':modulus}


def verify_separator(problem,domains,certificate,check=lambda:None,stats=None):
    """Independent column replay; does not read any producer basis or origins."""
    stats={} if stats is None else stats
    for key in ('lattice_separator_checks','lattice_separator_terms'):stats.setdefault(key,0)
    if (not isinstance(domains,(tuple,list)) or len(domains)!=problem.n or
        any(type(mask)is not int or mask<=0 or mask&~sum(1<<v for v in set(row)) for mask,row in zip(domains,problem.domains))):raise ValueError('Invalid lattice reason domains.')
    if not isinstance(certificate,dict) or set(certificate)!={'weights','modulus'}:raise ValueError('Invalid lattice separator.')
    raw=certificate['weights'];modulus=certificate['modulus'];dimension=11*len(problem.edges)+problem.n
    if (not isinstance(raw,(tuple,list)) or len(raw)>dimension or type(modulus)is not int or modulus<0 or modulus.bit_length()>256):raise ValueError('Invalid lattice modulus.')
    weights={}
    for item in raw:
        if not isinstance(item,(tuple,list)) or len(item)!=2 or any(type(v)is not int for v in item):raise ValueError('Invalid lattice dual entry.')
        coord,value=item
        if not 0<=coord<dimension or not value or abs(value).bit_length()>256 or coord in weights:raise ValueError('Invalid lattice dual coordinate.')
        weights[coord]=value
    if not weights:raise ValueError('Empty lattice separator.')
    incident=[[] for _ in range(problem.n)];offset=11*len(problem.edges)
    for edge,(a,sa,b,sb) in enumerate(problem.edges):
        incident[a].append((edge,sa,1,False));incident[b].append((edge,sb,-1,True))
    desired=sum(weights.get(offset+tile,0) for tile in range(problem.n));origins=0
    for cell,mask in enumerate(domains):
        first=None
        for code in bits(mask):
            check();value=weights.get(offset+code//3,0);stats['lattice_separator_terms']+=1
            for edge,side,sign,reverse in incident[cell]:
                for port in bits(int(problem.masks[code,side])):
                    stats['lattice_separator_terms']+=1;value+=sign*weights.get(11*edge+(10-port if reverse else port),0)
            if first is None:first=value;origins+=value
            elif ((value-first)%modulus if modulus else value-first):return False
    check();stats['lattice_separator_checks']+=1;difference=desired-origins
    return bool(difference%modulus if modulus else difference)


class ContributionLatticePruner(IntegerLatticePruner):
    def __init__(self,problem,*,limit=4,node_limit=1024):
        super().__init__(problem,limit=limit)
        if type(node_limit)is not int or not 1<=node_limit<=8192:raise ValueError('Invalid lattice tree bound.')
        self.node_limit=node_limit;self.nodes=OrderedDict();self.stats.update(dict.fromkeys(TREE_KEYS,0));self.reasons=deque(maxlen=128)

    def drain_reasons(self):
        result=list(self.reasons);self.reasons.clear();return result

    def drain_deduction_chains(self,target='gold',limit=8):
        from copy import deepcopy
        if target not in ('gold','edges') or type(limit)is not int or not 0<=limit<=128:raise ValueError('Invalid lattice proof export policy.')
        from constraint_deduction_chain_round26 import FORMAT
        emitted=0;original=[hex(sum(1<<code for code in row)) for row in self.problem.domains]
        while self.reasons and emitted<limit:
            reason=self.reasons[0];decisions=[] if reason['cell'] is None else [[reason['cell'],hex(1<<reason['code'])]]
            yield deepcopy({'format':FORMAT,'fingerprint':self.problem.fingerprint,'target':target,'original':original,
              'root':[hex(mask) for mask in reason['domains']],'decisions':decisions,'events':[],
              'terminal':{'rule':'lattice','data':{'certificate':reason['certificate']}}})
            if self.reasons and self.reasons[0] is reason:self.reasons.popleft()
            emitted+=1

    def _tree(self,domains,differences,check,max_operations,max_bits):
        before=self.stats['lattice_integer_ops']
        def build(lo,hi):
            check();key=(lo,hi,tuple(domains[lo:hi]),max_operations,max_bits)
            if key in self.nodes:
                self.nodes.move_to_end(key);self.stats['lattice_tree_reuses']+=1;return self.nodes[key]
            if hi-lo==1:
                vectors=[dict(row) for row in sorted({tuple(sorted(v.items())) for v in differences[lo].values()}) if row];children=None
                self.stats['lattice_tree_source_vectors']+=len(vectors)
            else:
                middle=(lo+hi)//2;children=(build(lo,middle),build(middle,hi))
                vectors=[row for child in children for row in child['basis'].basis.values()];self.stats['lattice_tree_merges']+=1
            remaining=max_operations-(self.stats['lattice_integer_ops']-before)
            if remaining<0:raise LatticeLimit('Lattice tree operation bound.')
            basis=IntegerLattice(vectors,check,stats=self.stats,max_operations=remaining,max_bits=max_bits)
            # Future membership is separately bounded by the configured policy.
            basis.max_operations=max_operations
            node={'lo':lo,'hi':hi,'basis':basis,'children':children};check();self.nodes[key]=node;self.stats['lattice_tree_nodes']+=1
            while len(self.nodes)>self.node_limit:self.nodes.popitem(last=False)
            return node
        return build(0,len(domains))

    def _without(self,root,cell,check,max_operations,max_bits):
        nodes=[]
        def visit(node):
            check()
            if not node['lo']<=cell<node['hi']:nodes.append(node);return
            if node['children'] is not None:
                for child in node['children']:visit(child)
        visit(root);vectors=[row for node in nodes for row in node['basis'].basis.values()]
        self.stats['lattice_tree_merges']+=1
        return IntegerLattice(vectors,check,stats=self.stats,max_operations=max_operations,max_bits=max_bits)

    def prune(self,domains,check=lambda:None,*,max_candidates=32,max_placements=4096,max_dimensions=512,max_operations=250000,max_bits=128):
        if self._binding()!=self.binding:raise ValueError('Lattice tree source changed.')
        if len(domains)!=len(self.original) or any(type(mask)is not int or mask<0 or mask&~root for mask,root in zip(domains,self.original)):raise ValueError('Lattice tree domains escape root.')
        for value in (max_candidates,max_placements,max_dimensions,max_operations):
            if type(value)is not int or value<0:raise ValueError('Invalid lattice tree work cap.')
        if type(max_bits)is not int or max_bits<1:raise ValueError('Invalid lattice coefficient cap.')
        if not all(domains):return False
        key=(tuple(domains),max_placements,max_dimensions,max_operations,max_bits);session=self.cache.get(key);check()
        if session is None:
            self.stats['lattice_contexts']+=1
            try:
                residual,differences=self._model(domains,check,max_placements,max_dimensions)
                tree=self._tree(domains,differences,check,max_operations,max_bits)
                reason=separating_functional(tree['basis'],residual,check)
                if reason is not None and not verify_separator(self.problem,domains,reason,check,self.stats):raise RuntimeError('Lattice root separator failed replay.')
            except LatticeLimit:self.stats['lattice_caps']+=1;return False
            candidates=[(cell,code) for cell in sorted(range(len(domains)),key=lambda c:(domains[c].bit_count(),c)) if domains[cell].bit_count()>1 for code in bits(domains[cell])]
            session={'tree':tree,'residual':residual,'differences':differences,'root_reason':reason,'candidates':candidates,'cursor':0,'without':OrderedDict(),'reduced':tuple(domains)}
            check();self.cache[key]=session
            while len(self.cache)>self.limit:self.cache.popitem(last=False)
        else:self.stats['lattice_reuses']+=1;self.cache.move_to_end(key)
        if self._binding()!=self.binding:raise ValueError('Lattice source changed during construction.')
        if session['root_reason'] is not None:
            self.reasons.append({'domains':tuple(domains),'cell':None,'code':None,'certificate':session['root_reason']});self.stats['lattice_separator_produced']+=1
            self.stats['lattice_prunes']+=domains[0].bit_count();domains[0]=0;return True
        current=list(session['reduced']);examined=0
        while session['cursor']<len(session['candidates']) and examined<max_candidates:
            check();cell,code=session['candidates'][session['cursor']]
            try:
                if cell not in session['without']:
                    basis=self._without(session['tree'],cell,check,max_operations,max_bits);check();session['without'][cell]=basis
                    while len(session['without'])>8:session['without'].popitem(last=False)
                else:self.stats['lattice_tree_without_reuses']+=1
                target=dict(session['residual'])
                for coord,value in session['differences'][cell][code].items():
                    changed=target.get(coord,0)-value
                    if changed:target[coord]=changed
                    else:target.pop(coord,None)
                self.stats['lattice_trials']+=1;reason=separating_functional(session['without'][cell],target,check)
                if reason is not None:
                    trial=list(key[0]);trial[cell]=1<<code
                    if not verify_separator(self.problem,trial,reason,check,self.stats):raise RuntimeError('Lattice candidate separator failed replay.')
                    current[cell]&=~(1<<code);self.reasons.append({'domains':key[0],'cell':cell,'code':code,'certificate':reason});self.stats['lattice_separator_produced']+=1
            except LatticeLimit:self.stats['lattice_caps']+=1
            session['cursor']+=1;session['reduced']=tuple(current);examined+=1
        check()
        if self._binding()!=self.binding:raise ValueError('Lattice source changed during propagation.')
        self.stats['lattice_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,current));changed=current!=domains;domains[:]=current;return changed
