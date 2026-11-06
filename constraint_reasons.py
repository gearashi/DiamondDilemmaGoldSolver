"""Trusted in-memory dependency explanations for domain deductions.

Each node means that the domain of one cell is contained in its recorded mask.
Rules link to earlier domain facts; decision nodes link to one placement literal.
Resolving this acyclic graph gives a sound decision conjunction, not necessarily
an inclusion-minimal one. Bulk native/global rules conservatively depend on all
pre-call domain facts. Nothing in this proof graph is deserialized or shared.
"""
from dataclasses import dataclass
STAT_KEYS=('reason_nodes','reason_conflicts','reason_literals_saved','reason_rebases','reason_direct_nodes')
@dataclass(frozen=True)
class Fact:
    cell:int
    domain:int
    rule:str
    parents:tuple
    causes:frozenset

class ReasonState:
    def __init__(self,domains,decisions=(),*,initial=None):
        self.domains=list(domains);self.facts=[]
        initial=list(domains) if initial is None else initial
        active=frozenset(decisions)
        self.current=[];self.failure=None
        for cell,value in enumerate(domains):
            causes=active if value!=initial[cell] else frozenset()
            self.current.append(len(self.facts));self.facts.append(Fact(cell,value,'initial' if not causes else 'resumed restrictions',(),causes))
    def fork(self,cell,code):
        # New frames share immutable facts; the parent has no later mutations
        # until its child has been discarded. Keep independent active pointers.
        child=object.__new__(ReasonState);child.domains=self.domains.copy()
        # DFS calls fork only after the previous descendant has been discarded.
        # All ancestors reference earlier indices; discard only that dead tail.
        del self.facts[max(self.current,default=-1)+1:]
        child.facts=self.facts;child.current=self.current.copy()
        child.failure=None
        old=child.current[cell];value=1<<code
        if not child.domains[cell]&value:raise ValueError('Decision is outside the reason domain.')
        causes=child.facts[old].causes|{(cell,code)}
        child.facts.append(Fact(cell,value,'decision',(old,),frozenset(causes)))
        child.current[cell]=len(child.facts)-1;child.domains[cell]=value
        return child
    def causes(self,cells):
        result=set()
        for cell in cells:result.update(self.facts[self.current[cell]].causes)
        return frozenset(result)
    def record_cell(self,cell,value,dependencies,rule):
        """Record one ordered deduction without scanning/copying other cells."""
        old=self.domains[cell]
        if value&~old:raise ValueError('Reason facts cannot widen domains.')
        if value==old:return 0
        refs=tuple(dict.fromkeys([*(self.current[c] for c in dependencies),self.current[cell]]))
        causes=frozenset().union(*(self.facts[i].causes for i in refs))
        self.facts.append(Fact(cell,value,rule,refs,causes));self.current[cell]=len(self.facts)-1
        self.domains[cell]=value;return 1
    def record(self,domains,dependencies,rule):
        """Record a sound completed or partial narrowing from pre-call facts.

        The caller supplies every domain read by the rule. Dependencies are
        captured BEFORE any output fact is added, avoiding circular reasons.
        """
        if len(domains)!=len(self.domains) or any(v&~old for v,old in zip(domains,self.domains)):
            raise ValueError('Reason facts cannot widen domains.')
        parents=tuple(self.current[cell] for cell in sorted(set(dependencies)))
        causes=frozenset().union(*(self.facts[i].causes for i in parents))
        changed=0
        for cell,(old,value) in enumerate(zip(self.domains,domains)):
            if value!=old:
                # The old fact also excludes previously removed values.
                prior=self.current[cell];refs=parents if prior in parents else (*parents,prior)
                resolved=causes|self.facts[prior].causes
                self.facts.append(Fact(cell,value,rule,refs,resolved));self.current[cell]=len(self.facts)-1;changed+=1
        self.domains=list(domains);return changed
    def rebase(self,decisions):
        replacement=ReasonState(self.domains,decisions,initial=[-1]*len(self.domains))
        self.facts,self.current=replacement.facts,replacement.current
        self.failure=None
    def contradiction(self,domains=None):
        domains=self.domains if domains is None else domains
        empty=[i for i,value in enumerate(domains) if value==0]
        return self.causes(empty) if empty else None
    def validate_graph(self):
        for i,fact in enumerate(self.facts):
            if any(type(p)is not int or not 0<=p<i for p in fact.parents):raise ValueError('Reason graph is not acyclic.')
            inherited=frozenset().union(*(self.facts[p].causes for p in fact.parents))
            if not inherited<=fact.causes:raise ValueError('Reason dropped an antecedent.')
        return True
