"""Complete search ordering policies: priorities never delete owned work."""
from math import prod
from constraint_pruning import values

STAT_KEYS=('impact_observations','impact_choices','frontier_reorders','portfolio_jobs','discrepancy_jobs','piece_symmetry_prunes','rotated_clauses')

class ImpactPolicy:
    def __init__(self,n):self.total=[0.0]*n;self.count=[0]*n
    def observe(self,cell,before,after,failed=False):
        if cell is None or not 0<=cell<len(self.count):return
        # A logarithm-free fraction measures removed choices in other cells.
        available=sum(v.bit_count() for c,v in enumerate(before) if c!=cell)
        remaining=sum(v.bit_count() for c,v in enumerate(after) if c!=cell)
        value=1.0 if failed else max(0,available-remaining)/max(1,available)
        self.total[cell]+=(value);self.count[cell]+=1
    def key(self,cell,size):
        mean=self.total[cell]/self.count[cell] if self.count[cell] else 0
        return size/(1+4*mean)


def worker_options(base,slot):
    result=dict(base)
    if not base.get('worker_portfolio'):return result
    # A worker owns a disjoint rectangle; variation changes ordering only.
    policy=slot%4
    result['ordering_goal']='conflict' if policy==1 else 'solution' if policy==2 else 'auto'
    result['objective_ordering']=policy in (1,2)
    result['impact_branching']=policy==3 or base.get('impact_branching',False)
    return result


def reorder_frontier(queue,jobs,preferred,*,discrepancy=False,epoch=0):
    """Reorder intact disjoint jobs; every eighth pick preserves oldest-first fairness.

    Discrepancy is the number of preferred cell values excluded by a rectangle,
    a lower bound on differing placements in every completion of that rectangle.
    Increasing this bound partitions the remaining finite frontier without
    dropping high-discrepancy jobs. It is a traversal hint, never a proof.
    """
    if len(queue)<2:return False
    old=tuple(queue)
    def key(job):
        row=[int(v,16) for v in jobs[job]['region']]
        cost=sum(not mask&(1<<code) for mask,code in zip(row,preferred))
        volume=sum(v.bit_count() for v in row)
        return (cost if discrepancy else 0,volume,job)
    queue.sort(key=key)
    if epoch%8==7:
        oldest=min(queue);queue.remove(oldest);queue.insert(0,oldest)
    return old!=tuple(queue)


class PieceSymmetry:
    def __init__(self,problem):
        from constraint_deep import orientation_equivalences
        self.rotations,self.classes=orientation_equivalences(problem)
        full=set(range(3*problem.n))
        # Arbitrary fixed/restricted inputs break physical-label interchange.
        self.classes=tuple(tuple(t for t in group if t!=0) for group in self.classes if sum(t!=0 for t in group)>1) if not problem.fixed and all(set(row)==full for row in problem.domains) else ()
    def prune(self,domains):
        before=list(domains)
        for c,domain in enumerate(domains):
            for group in self.rotations:
                present=[code for code in group if domain&(1<<code)]
                for code in present[1:]:domain&=~(1<<code)
            domains[c]=domain
        for group in self.classes:
            for lower,upper in zip(group,group[1:]):
                left=[c for c,v in enumerate(domains) if v&(7<<(3*lower))]
                right=[c for c,v in enumerate(domains) if v&(7<<(3*upper))]
                if not left or not right:domains[0]=0;return True
                for c in left:
                    if c>=max(right):domains[c]&=~(7<<(3*lower))
                for c in right:
                    if c<=min(left):domains[c]&=~(7<<(3*upper))
        return before!=domains
