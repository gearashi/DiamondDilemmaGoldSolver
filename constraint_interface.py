"""Bounded exact interface formulas with replayable resolution derivations.

A regional CNF is an optimistic local relaxation: cell ownership, distinct
physical pieces, and seams wholly inside the selected cells. Davis--Putnam
elimination derives a formula over selected interface cells. Every published
clause is checked from the reconstructed regional clauses and resolution steps.
A cap means no newly published formula, never an infeasibility result.
"""
from collections import OrderedDict
from itertools import combinations
from constraint_relations import bits
from validator import reverse_mask

STAT_KEYS=('interface_queries', 'interface_completed', 'interface_unknown', 'interface_resolution_steps', 'interface_reuses', 'interface_prunes')


class ProjectionLimit(Exception):pass


def canonical_clause(literals):
    row=frozenset(literals)
    if any(-lit in row for lit in row):return None
    return tuple(sorted(row))


def regional_cnf(problem,scope):
    scope=tuple(scope);variables={};decode={};clauses=[]
    for cell,mask in scope:
        for code in bits(mask):
            var=len(variables)+1;variables[cell,code]=var;decode[var]=(cell,code)
    def clause(row):
        row=canonical_clause(row)
        if row is not None:clauses.append(row)
    for cell,mask in scope:
        row=[variables[cell,code] for code in bits(mask)]
        clause(row)
        for a,b in combinations(row,2):clause((-a,-b))
    links={}
    for a,sa,b,sb in problem.edges:
        links.setdefault((a,b),[]).append((sa,sb));links.setdefault((b,a),[]).append((sb,sa))
    for (a,left),(b,right) in combinations(scope,2):
        for x in bits(left):
            for y in bits(right):
                if x//3==y//3 or any(int(problem.masks[x,sa])!=reverse_mask(int(problem.masks[y,sb])) for sa,sb in links.get((a,b),())):
                    clause((-variables[a,x],-variables[b,y]))
    return tuple(dict.fromkeys(clauses)),variables,decode


def resolution_projection(clauses,eliminate,check=lambda:None,*,max_steps=10000,max_clauses=2048):
    clauses=tuple(canonical_clause(row) for row in clauses)
    if any(row is None for row in clauses):raise ValueError('Canonical nontautological CNF required.')
    proof=list(clauses);indices={row:i for i,row in enumerate(proof)};active=set(range(len(proof)));steps=[]
    remaining=set(eliminate);operations=0
    # Deleting a clause subsumed by another preserves the exact CNF. This
    # avoids generating the many supersets produced by pairwise ownership.
    def minimal(indices):
        kept=[]
        for ident in sorted(indices,key=lambda i:(len(proof[i]),proof[i])):
            candidate=set(proof[ident])
            if not any(set(proof[old])<=candidate for old in kept):kept.append(ident)
        return set(kept)
    active=minimal(active)
    while remaining:
        check()
        pivot=min(remaining,key=lambda v:(sum(v in proof[i] for i in active)*sum(-v in proof[i] for i in active),v))
        positive=[i for i in sorted(active) if pivot in proof[i]]
        negative=[i for i in sorted(active) if -pivot in proof[i]]
        additions=set();surviving=active-set(positive)-set(negative)
        for a in positive:
            for b in negative:
                check();operations+=1
                if operations>max_steps:raise ProjectionLimit('Resolution pair budget reached.')
                row=canonical_clause((set(proof[a])-{pivot})|(set(proof[b])-{-pivot}))
                if row is None:continue
                candidate=set(row)
                if any(set(proof[old])<=candidate for old in surviving|additions):continue
                if row not in indices:
                    if len(proof)>=max_clauses:raise ProjectionLimit('Resolution clause budget reached.')
                    indices[row]=len(proof);proof.append(row);steps.append((a,b,pivot,row))
                additions={old for old in additions if not candidate<set(proof[old])}
                surviving={old for old in surviving if not candidate<set(proof[old])}
                additions.add(indices[row])
        active=minimal(surviving|additions)
        remaining.remove(pivot)
    return {'steps':tuple(steps),'final':tuple(proof[i] for i in sorted(active)),
            'operations':operations,'base_count':len(clauses)}


def check_resolution(base,record):
    proof=list(base)
    try:
        if record['base_count']!=len(base):return False
        for a,b,pivot,claimed in record['steps']:
            if type(a)is not int or type(b)is not int or type(pivot)is not int or not 0<=a<len(proof) or not 0<=b<len(proof):return False
            if pivot<=0 or pivot not in proof[a] or -pivot not in proof[b]:return False
            row=canonical_clause((set(proof[a])-{pivot})|(set(proof[b])-{-pivot}))
            if row is None or tuple(claimed)!=row:return False
            proof.append(row)
        known=set(proof)
        return all(tuple(row) in known for row in record['final'])
    except (KeyError,TypeError,ValueError):return False


def apply_interface(domains,scope,decode,clauses,check=lambda:None):
    if any(domains[cell]&~mask for cell,mask in scope):return False
    roots=dict(scope);current=list(domains)
    for clause in clauses:
        check();forbidden={}
        for literal in clause:
            cell,code=decode[abs(literal)]
            mask=roots[cell]&~(1<<code) if literal>0 else 1<<code
            forbidden[cell]=forbidden.get(cell,roots[cell])&mask
        undecided=[]
        for cell,mask in forbidden.items():
            if not current[cell]&mask:break
            if current[cell]&~mask:undecided.append((cell,mask))
            if len(undecided)>1:break
        else:
            if not undecided:
                current[next(iter(roots))]=0;break
            if len(undecided)==1:
                cell,mask=undecided[0];current[cell]&=~mask
    check();changed=current!=domains;domains[:]=current;return changed


class InterfaceProjector:
    def __init__(self,problem,limit=32):
        self.problem=problem;self.limit=limit;self.records=OrderedDict()
        self.stats=dict.fromkeys(('interface_queries','interface_completed','interface_unknown',
                                 'interface_resolution_steps','interface_reuses','interface_prunes'),0)

    def compile(self,scope,boundary,check=lambda:None,**limits):
        scope=tuple(sorted(scope));boundary=tuple(sorted(boundary));key=(scope,boundary)
        if key in self.records:
            self.stats['interface_reuses']+=1;self.records.move_to_end(key);return self.records[key]
        self.stats['interface_queries']+=1
        base,variables,decode=regional_cnf(self.problem,scope)
        eliminate=[var for (cell,code),var in variables.items() if cell not in boundary]
        try:record=resolution_projection(base,eliminate,check,**limits)
        except ProjectionLimit:self.stats['interface_unknown']+=1;return None
        check()
        if not check_resolution(base,record):raise RuntimeError('Internal interface proof failed replay.')
        if any(decode[abs(lit)][0] not in boundary for row in record['final'] for lit in row):raise RuntimeError('Uneliminated private variable.')
        record={**record,'scope':scope,'decode':decode,'boundary':boundary}
        self.records[key]=record
        while len(self.records)>self.limit:self.records.popitem(last=False)
        self.stats['interface_completed']+=1;self.stats['interface_resolution_steps']+=record['operations']
        return record

    def prune(self,domains,check=lambda:None,*,max_cells=5,max_values=6):
        current=list(domains)
        for record in tuple(self.records.values()):
            apply_interface(current,record['scope'],record['decode'],record['final'],check)
            if not all(current):break
        eligible=sorted((c for c,d in enumerate(current) if 1<d.bit_count()<=max_values),key=lambda c:(current[c].bit_count(),c))
        if all(current) and len(eligible)>=3:
            chosen=[eligible[0]];remaining=set(eligible[1:])
            while remaining and len(chosen)<max_cells:
                # Include constraints crossing the growing region where possible.
                cell=min(remaining,key=lambda c:(-sum(int(n) in chosen for n in self.problem.neighbors[c]),current[c].bit_count(),c))
                chosen.append(cell);remaining.remove(cell)
            scope=tuple((c,current[c]) for c in sorted(chosen));boundary=(chosen[0],chosen[-1])
            record=self.compile(scope,boundary,check)
            if record is not None:apply_interface(current,scope,record['decode'],record['final'],check)
        check();self.stats['interface_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed
