"""Exact alternative partitions and bounded ordering experiments for round 24.

Every returned branch partitions one existing domain. Scores are ordering only;
no approximate count or prediction removes a value.
"""
from collections import OrderedDict
from dataclasses import dataclass
import math
from constraint_branching import ConnectionBranch
from validator import reverse_mask

# Frozen ridge fit on completed exact child searches. Ordering only: these
# predictions are never used as feasibility evidence or to remove a value.
LEARNED_ORDERING_WEIGHTS=(0.06580072709342792,0.3365667764509336,
                         -0.18048129401887134,0.3179820653497499,-0.2102795764645618)
LEARNED_ORDERING_PROVENANCE={
    'version':1,'training_seeds':tuple(range(240310,240316)),
    'training_cells':10,'completed_labels':46,'ridge':1.0,
    'label':'log1p(completed_search_work)',
    'training_labels_sha256':'6f3b929e12b13acf5f6129b97f6f30caecfb02a6faae7f1b8fccc9e42c93e154',
    'scope':'ambiguous synthetic Gold fixtures; original-board generalization unproven',
    'script':'benchmark_learned_ordering_round24.py',
}

STAT_KEYS=('dual_piece_checks','dual_piece_branches','strand_branch_checks','strand_branches',
           'completion_order_calls','completion_order_states','completion_order_complete',
           'completion_order_capped','learned_order_calls')

def values(mask):
    while mask:
        bit=mask & -mask; mask-=bit; yield bit.bit_length()-1

class RoutingBranching:
    def __init__(self,problem):
        self.problem=problem
        self.partners=[]
        for tile in problem.tiles:
            for rot in range(3):
                row={}
                for a,b in tile['segments']:
                    a=((a[0]+rot)%3)*11+a[1]-1
                    b=((b[0]+rot)%3)*11+b[1]-1
                    row[a]=b;row[b]=a
                self.partners.append(row)
    def choose(self,domains,preferred,check,stats,*,dual_piece=False,strand=False):
        for key in STAT_KEYS:stats.setdefault(key,0)
        if strand:
            best=None
            for source,domain in enumerate(domains):
                if domain.bit_count()!=1:continue
                code=domain.bit_length()-1
                for port in self.partners[code]:
                    check();side,pos=divmod(port,11)
                    cell=int(self.problem.neighbors[source,side])
                    if cell<0 or domains[cell].bit_count()<2:continue
                    arrival=int(self.problem.sides[source,side])*11+10-pos
                    grouped={}
                    for candidate in values(domains[cell]):
                        partner=self.partners[candidate].get(arrival,-1)
                        grouped[partner]=grouped.get(partner,0)|(1<<candidate)
                    stats['strand_branch_checks']+=1
                    if len(grouped)<2:continue
                    largest=max(mask.bit_count() for mask in grouped.values())
                    key=(len(grouped),largest,domains[cell].bit_count(),cell!=preferred,cell,arrival)
                    if best is None or key<best[0]:
                        entries=tuple(sorted(grouped.items()))
                        best=(key,ConnectionBranch(cell,-2,tuple(k for k,_ in entries),tuple(v for _,v in entries)))
            if best is not None:
                stats['strand_branches']+=1;return best[1]
        if dual_piece:
            best=None
            for tile in range(self.problem.n):
                check();bits=7<<(3*tile)
                locations=[cell for cell,domain in enumerate(domains) if domain&bits]
                stats['dual_piece_checks']+=1
                if len(locations)<2:continue
                # A binary location decision forms an exact disjoint partition.
                # If absent, subsequent propagation/branching handles all other
                # locations without requiring a new checkpoint frame format.
                for cell in locations:
                    yes=domains[cell]&bits;no=domains[cell]&~bits
                    if not no:continue
                    key=(len(locations),yes.bit_count(),domains[cell].bit_count(),cell!=preferred,tile,cell)
                    if best is None or key<best[0]:best=(key,ConnectionBranch(cell,-3,(tile,-1),(yes,no)))
            if best is not None:
                stats['dual_piece_branches']+=1;return best[1]
        return None

class CompletionOrdering:
    """Count exact seam/inventory-compatible extensions in a bounded star.

The star is a relaxation of the full board. Counts order values only. A capped
count does not eliminate even a value whose current count happens to be zero.
"""
    def __init__(self,problem,max_cells=4,max_choices=24,max_states=4096,cache_size=128):
        self.problem=problem;self.max_cells=max_cells;self.max_choices=max_choices
        self.max_states=max_states;self.cache_size=cache_size;self.cache=OrderedDict()
        self.adj=[[] for _ in range(problem.n)]
        for a,sa,b,sb in problem.edges:
            self.adj[a].append((b,sa,sb));self.adj[b].append((a,sb,sa))
    def counts(self,cell,domains,choices,check,stats):
        for key in STAT_KEYS:stats.setdefault(key,0)
        stats['completion_order_calls']+=1
        pool=sorted({b for b,_,_ in self.adj[cell]},key=lambda c:(domains[c].bit_count(),c))
        selected=[cell]+[c for c in pool if domains[c].bit_count()<=self.max_choices][:self.max_cells-1]
        if len(selected)<2:return {}
        key=(tuple(selected),tuple(domains[c] for c in selected),tuple(sorted(choices)))
        if key in self.cache:return dict(self.cache[key])
        answer={};used_states=0;assignment={}
        def walk(at,used):
            nonlocal used_states
            check()
            if at==len(selected):return 1
            current=selected[at];total=0
            for code in values(domains[current]):
                check();used_states+=1;stats['completion_order_states']+=1
                if used_states>self.max_states:raise _CountCap
                if used&(1<<(code//3)):continue
                if any(other in assignment and int(self.problem.masks[code,side])!=reverse_mask(self.problem.masks[assignment[other],other_side]) for other,side,other_side in self.adj[current]):continue
                assignment[current]=code;total+=walk(at+1,used|(1<<(code//3)));del assignment[current]
            return total
        for code in choices:
            assignment.clear();assignment[cell]=code
            try:answer[code]=walk(1,1<<(code//3));stats['completion_order_complete']+=1
            except _CountCap:stats['completion_order_capped']+=1;break
        # Only completed counts survive; incomplete absence has no semantics.
        self.cache[key]=dict(answer)
        while len(self.cache)>self.cache_size:self.cache.popitem(last=False)
        return answer
    def order(self,cell,domains,choices,hinted,rank,check,stats):
        counts=self.counts(cell,domains,choices,check,stats)
        return sorted(choices,key=lambda code:(0,hinted[code],0,rank[code]) if code in hinted else
                      (1,0 if code in counts else 1,-counts.get(code,0),rank[code]))

class _CountCap(Exception):pass

class LearnedOrdering:
    """A fitted ranking model; predictions never alter feasible domains.

Training rows contain features and observed completed-subproblem work. The
caller controls train/held-out separation; no input is interpreted as a proof.
"""
    VERSION=1
    def __init__(self,problem,weights=None):
        self.problem=problem
        self.weights=tuple(weights) if weights is not None else None
        if self.weights is not None and (len(self.weights)!=5 or any(not math.isfinite(x) for x in self.weights)):
            raise ValueError('Ranking weights must be five finite values.')
    def features(self,cell,domains,code):
        tile=code//3;home_count=sum(bool(mask&(7<<(3*tile))) for mask in domains)
        counts=[]
        for side,other in enumerate(self.problem.neighbors[cell]):
            if other<0:continue
            other=int(other);other_side=int(self.problem.sides[cell,side])
            counts.append(sum(candidate//3!=tile and int(self.problem.masks[code,side])==reverse_mask(self.problem.masks[candidate,other_side]) for candidate in values(domains[other])))
        return (1.0,math.log1p(home_count),math.log1p(min(counts,default=0)),
                sum(math.log1p(value) for value in counts),math.log1p(domains[cell].bit_count()))
    @classmethod
    def fit(cls,examples,ridge=1.0):
        import numpy as np
        if ridge<=0 or not math.isfinite(ridge):raise ValueError('Positive finite ridge required.')
        rows=list(examples)
        if not rows:raise ValueError('Completed training examples required.')
        x=np.asarray([row['features'] for row in rows],dtype=float)
        y=np.asarray([math.log1p(row['work']) for row in rows],dtype=float)
        if x.shape!=(len(rows),5) or not np.isfinite(x).all() or not np.isfinite(y).all() or any(row['work']<0 for row in rows):
            raise ValueError('Malformed completed training examples.')
        weights=np.linalg.solve(x.T@x+np.eye(5)*ridge,x.T@y)
        return {'version':cls.VERSION,'weights':weights.tolist(),'examples':len(rows),'label':'log1p(completed_search_work)'}
    def order(self,cell,domains,choices,hinted,rank,check,stats):
        stats.setdefault('learned_order_calls',0);stats['learned_order_calls']+=1
        if self.weights is None:raise ValueError('Learned ordering requires fitted weights; no zero-score fallback is allowed.')
        predictions={}
        for code in choices:
            check();features=self.features(cell,domains,code)
            predictions[code]=sum(a*b for a,b in zip(self.weights,features))
        return sorted(choices,key=lambda code:(0,hinted[code],rank[code]) if code in hinted else (1,predictions[code],rank[code]))

