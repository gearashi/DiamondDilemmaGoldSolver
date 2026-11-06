"""Experimental native Gold conflicts and direct SAT connectivity.

The direct model adds a rooted acyclic parent forest over occupied endpoint
ports. Every occupied non-root port has a parent on a selected stroke/seam, and
parent ranks strictly decrease. Finite descent reaches the unique root, hence
all occupied ports are connected. Conversely a connected selected port graph
has a rooted spanning tree with ranks at most the number of ports minus one.
Tile endpoint degree two plus seam equality then makes one Gold cycle.

Crossing drawn strokes are not joined. Construction limits mean unsupported,
not UNSAT. The external propagator emits only independently checked proper-loop
witness clauses, including before a complete placement assignment exists.
"""
from collections import defaultdict,deque
import math
from constraint_knowledge import revalidate_witness


class SATConnectivityLimit(Exception):pass


def gold_propagator(problem,variables,*,run=None,interrupt=lambda:None):
    from pysat.engines import Propagator
    class GoldPropagator(Propagator):
        def __init__(self):
            super().__init__()
            self.decode={literal:(c,code) for c,row in enumerate(variables) for code,literal in row.items()}
            self.assignments={};self.fixed={};self.level=0;self.trail=[]
            self.pending=deque();self.known=set();self.dirty=True;self.error=None
            self.stats={'sat_external_checks':0,'sat_external_partial_conflicts':0,
                        'sat_external_model_conflicts':0,'sat_external_clauses':0}
        def on_assignment(self,lit,fixed=False):
            variable=abs(lit);old=self.assignments.get(variable)
            self.assignments[variable]=lit
            if fixed:self.fixed[variable]=lit
            elif old!=lit:self.trail.append((variable,old,self.level))
            self.dirty=True
        def on_new_level(self):self.level+=1
        def on_backtrack(self,to):
            while self.trail and self.trail[-1][2]>to:
                variable,old,level=self.trail.pop()
                if variable in self.fixed:self.assignments[variable]=self.fixed[variable]
                elif old is None:self.assignments.pop(variable,None)
                else:self.assignments[variable]=old
            self.level=to;self.dirty=True
        def _codes(self,selected):
            codes=[-1]*problem.n
            for literal in selected:
                if literal<=0 or literal not in self.decode:continue
                cell,code=self.decode[literal]
                if codes[cell]>=0 and codes[cell]!=code:return None
                codes[cell]=code
            chosen=[code//3 for code in codes if code>=0]
            # CNF owns tile exclusivity. A transient assignment that already
            # violates it need not enter the Gold witness verifier.
            return codes if len(chosen)==len(set(chosen)) else None
        def _enqueue(self,codes,*,complete=False):
            if codes is None:return False
            self.stats['sat_external_checks']+=1
            if not problem.partial_loop_impossible(codes):return False
            witness=revalidate_witness(problem,problem.gold_nogood(codes),target='gold')
            clause=tuple(-variables[cell][code] for cell,code in witness)
            if clause not in self.known or complete:
                self.pending.append(clause);self.known.add(clause)
                if len(self.known)>4096:self.known.pop()
                self.stats['sat_external_model_conflicts' if complete else 'sat_external_partial_conflicts']+=1
            return True
        def propagate(self):
            if not self.dirty:return []
            self.dirty=False
            try:
                if run is not None and run.cancelled():interrupt();return []
                self._enqueue(self._codes(self.assignments.values()))
            except BaseException as exc:
                self.error=exc
                if run is not None:run.error=exc
                interrupt()
            return []
        def check_model(self,model):
            try:return not self._enqueue(self._codes(model),complete=True)
            except BaseException as exc:
                self.error=exc
                if run is not None:run.error=exc
                interrupt();return True  # Caller raises the recorded exception.
        def decide(self):return 0
        def provide_reason(self,lit):
            raise RuntimeError('Gold propagator emits clauses, never unexplained literals.')
        def has_clause(self):return bool(self.pending)
        def add_clause(self):
            if not self.pending:return []
            self.stats['sat_external_clauses']+=1
            return list(self.pending.popleft())
    return GoldPropagator()


def add_reachability(problem,domains,variables,fresh,clause,*,check=lambda:None,
                     max_ports=6000,max_arcs=200000,max_gate_terms=4000000):
    """Append a complete, bounded optional-port rooted reachability encoding."""
    stats={'sat_reachability_ports':0,'sat_reachability_parent_arcs':0,
           'sat_reachability_gate_terms':0,'connectivity_encoding':'direct_ranked_port_reachability'}
    false=fresh();clause([-false]);true=-false
    def gate_and(literals):
        literals=tuple(dict.fromkeys(literals))
        if not literals:return true
        if len(literals)==1:return literals[0]
        stats['sat_reachability_gate_terms']+=len(literals)
        if stats['sat_reachability_gate_terms']>max_gate_terms:
            raise SATConnectivityLimit('SAT reachability linking-term cap exceeded.')
        output=fresh()
        for value in literals:clause([-output,value])
        clause([output,*(-value for value in literals)])
        return output
    def gate_or(literals):return -gate_and([-value for value in literals])
    def equal(a,b):return gate_or([gate_and([a,b]),gate_and([-a,-b])])
    def less(a,b):
        result=false
        # LSB first: each more significant unequal bit overrides lower bits.
        for x,y in zip(a,b):result=gate_or([gate_and([-x,y]),gate_and([equal(x,y),result])])
        return result
    segments=[]
    for tile in problem.tiles:
        for rotation in range(3):
            segments.append(tuple((((a[0]+rotation)%3)*11+a[1]-1,
                                   ((b[0]+rotation)%3)*11+b[1]-1)
                                  for a,b in tile['segments']))
    active={};roots={};local={};neighbors=defaultdict(list)
    for cell,row in enumerate(domains):
        check();ports=defaultdict(list);edges=defaultdict(list);root_support=defaultdict(list)
        for code in row:
            for a,b in segments[code]:
                ports[a].append(variables[cell][code]);ports[b].append(variables[cell][code])
                edges[tuple(sorted((a,b)))].append(variables[cell][code])
            if code//3==0:
                root_support[segments[code][0][0]].append(variables[cell][code])
        for port,literals in sorted(ports.items()):
            if len(active)>=max_ports:raise SATConnectivityLimit('SAT reachability port cap exceeded.')
            node=(cell,port);active[node]=gate_or(literals);roots[node]=gate_or(root_support[port])
        for (a,b),literals in edges.items():
            edge=gate_or(literals);left,right=(cell,a),(cell,b)
            neighbors[left].append((right,edge));neighbors[right].append((left,edge))
    for a,sa,b,sb in problem.edges:
        check()
        for p in range(11):
            left,right=(a,11*sa+p),(b,11*sb+10-p)
            if left not in active and right not in active:continue
            if left not in active:clause([-active[right]]);continue
            if right not in active:clause([-active[left]]);continue
            clause([-active[left],active[right]]);clause([-active[right],active[left]])
            neighbors[left].append((right,active[left]));neighbors[right].append((left,active[right]))
    for node,present in active.items():
        if int(problem.neighbors[node[0],node[1]//11])<0:clause([-present])
    if not active:clause([]);return stats
    width=max(1,(len(active)-1).bit_length());ranks={node:[fresh() for _ in range(width)] for node in active}
    for node,present in active.items():
        check();root=roots[node];parents=[]
        clause([-root,present])
        for bit in ranks[node]:clause([-root,-bit])
        for neighbor,edge in neighbors[node]:
            stats['sat_reachability_parent_arcs']+=1
            if stats['sat_reachability_parent_arcs']>max_arcs:raise SATConnectivityLimit('SAT reachability arc cap exceeded.')
            parent=fresh();parents.append(parent)
            clause([-parent,edge]);clause([-parent,present]);clause([-parent,active[neighbor]])
            clause([-parent,-root]);clause([-parent,less(ranks[neighbor],ranks[node])])
        clause([-present,root,*parents])
    stats['sat_reachability_ports']=len(active)
    stats['sat_reachability_rank_bits']=width
    return stats
