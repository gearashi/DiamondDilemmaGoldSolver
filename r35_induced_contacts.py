"""Replay incompatible incidental mesh contacts implied by labelled rigid tile bonds."""
from collections import defaultdict
import numpy as np

from r35_geographic_replay import replay_for
from validator import reverse_mask


class InducedContacts:
    def __init__(self, problem):
        self.problem=problem
        self.poses=np.arange(3*problem.n)
        cells,rot=np.divmod(self.poses,3)
        self.step={}
        for s in range(3):
            side=(s+rot)%3
            nxt=problem.neighbors[cells,side];facing=problem.sides[cells,side]
            for t in range(3):
                self.step[s,t]=np.where(nxt<0,-1,3*nxt+(facing-t)%3)
        self.labels=problem.masks[::3]
        self.reversed=np.asarray([[reverse_mask(v) for v in row] for row in self.labels])
        self.cache={}

    def count(self, group):
        """Count geometric and all-internal-seams-matched root poses of a connected set."""
        graph=defaultdict(list)
        for a,b in sorted(group):
            u,s=divmod(a,3);v,t=divmod(b,3)
            graph[u].append((v,s,t));graph[v].append((u,t,s))
        root=min(graph)
        assigned={root:self.poses}
        occupied=[self.poses//3]
        valid=np.ones(len(self.poses),dtype=bool)
        stack=[root]
        while stack:
            u=stack.pop()
            for v,s,t in graph[u]:
                nxt=self.step[s,t][assigned[u]]
                valid &= nxt>=0
                if v in assigned:
                    valid &= nxt==assigned[v]
                    continue
                cell=nxt//3
                for other in occupied:valid &= cell!=other
                occupied.append(cell);assigned[v]=nxt;stack.append(v)
        if len(assigned)!=len(graph):
            raise ValueError('An induced-contact certificate must be connected.')
        poses=int(valid.sum())
        if not poses:return 0,0
        # Every pair of already placed tiles may share an additional edge.
        # Only physical side labels that disagree can disqualify this pose.
        members=list(assigned)
        for j,v in enumerate(members):
            for u in members[:j]:
                for s in range(3):
                    for t in range(3):
                        if self.labels[u,s]!=self.reversed[v,t]:
                            valid &= self.step[s,t][assigned[u]]!=assigned[v]
                if not valid.any():return poses,0
        return poses,int(valid.sum())

    def proves(self, group):
        if group in self.cache:return self.cache[group]
        try:geometric,matched=self.count(group)
        except ValueError:return False
        result=geometric>0 and matched==0
        if len(self.cache)>=4096:self.cache.clear()
        self.cache[group]=result
        return result


def verified_contacts(problem,hints,*,allowed=None):
    if not hints:return ()
    checker=getattr(problem,'_induced_contact_checker',None)
    if checker is None:problem._induced_contact_checker=checker=InducedContacts(problem)
    normalizer=replay_for(problem)
    checked=set()
    for hint in hints:
        group=normalizer.normalize(hint)
        if group and (allowed is None or group<=allowed) and checker.proves(group):
            checked.add(group)
    return tuple(sorted(checked,key=lambda g:(len(g),sorted(g))))


def contact_departures(problem, reference, required, hints):
    # Physical placement-only callers keep their separate zero-pose proof type.
    req=frozenset(required);allowed=frozenset(reference)|req
    groups={g-req for g in verified_contacts(problem,hints,allowed=allowed)}
    return tuple(sorted(groups,key=lambda g:(len(g),sorted(g))))
