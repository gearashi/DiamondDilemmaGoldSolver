"""Replay physical geographic obstructions and reuse them only on equally narrow domains."""
from collections import Counter

from r27_structure_filters import _vertex_links,domain_bits
from r28_geo_sewing import PosePacking
from r31_geographic import ribbon_genus
from validator import reverse_mask


_PROOF_LIMIT = 128
_BATCH_BOND_LIMIT = 32768


class GeographicReplay:
    def __init__(self,problem):
        self.problem=problem
        self.root=tuple(domain_bits(problem))
        self.spherical=_vertex_links(problem) is not None
        self.proofs=[]
        self.large_root_batch=None
        self.work=Counter()

    def normalize(self,hint):
        try:
            rows=frozenset(tuple(sorted(e)) for e in hint)
        except (ValueError,TypeError):
            return None
        if not rows or any(len(e)!=2 or any(type(d) is not int or not 0<=d<3*self.problem.n for d in e) for e in rows):
            return None
        if len({d for e in rows for d in e})!=2*len(rows):return None
        for a,b in rows:
            if a//3==b//3:return None
            if int(self.problem.masks[(a//3)*3,a%3])!=reverse_mask(int(self.problem.masks[(b//3)*3,b%3])):
                return None
        return rows

    def check(self,domains,hints,*,max_steps=2000,stop=lambda:None):
        ds=tuple(domains)
        if len(ds)!=len(self.root) or any(x<0 or x&~y for x,y in zip(ds,self.root)):
            raise ValueError('Geographic proof domains exceed the puzzle.')
        if max_steps<1 or not self.spherical:return ()
        verified=set(); packing=None
        for hint in hints:
            stop()
            row=self.normalize(hint)
            if not row:continue
            for scope,core in reversed(self.proofs):
                if core<=row and (scope is None or all(x&~y==0 for x,y in zip(ds,scope))):
                    self.work['proof_reuses']+=1;verified.add(core)
                    break
            else:
                self.work['replays']+=1
                if ribbon_genus(self.problem,row)>0:scope=None
                else:
                    if packing is None:packing=PosePacking(self.problem,ds)
                    result=packing.check(row,max_steps=max_steps)
                    if result['feasible'] is not False:continue
                    scope=None if ds==self.root else ds
                if len(self.proofs)==_PROOF_LIMIT:self.proofs.pop(0)
                self.proofs.append((scope,row));verified.add(row)
        return tuple(sorted(verified,key=lambda row:(len(row),sorted(row))))


def replay_for(problem):
    cache=getattr(problem,'_geographic_replay',None)
    if cache is None or cache.root!=tuple(domain_bits(problem)):
        problem._geographic_replay=GeographicReplay(problem)
    return problem._geographic_replay


def geographic_departures(problem, reference, required, hints):
    """A replayed obstruction loses a reference bond whenever its other bonds are required."""
    if not hints:
        return ()
    replay = replay_for(problem)
    required = frozenset(required)
    allowed = set(reference) | required
    candidates = []
    for hint in hints:
        core = replay.normalize(hint)
        if core and core <= allowed:
            candidates.append(core)
    candidate_set = frozenset(candidates)
    batchable = (len(candidate_set) > _PROOF_LIMIT and
                 sum(len(row) for row in candidate_set) <= _BATCH_BOND_LIMIT)
    if batchable and replay.large_root_batch is not None and candidate_set == replay.large_root_batch[0]:
        checked = replay.large_root_batch[1]
        replay.work['complete_root_batch_reuses'] += 1
    else:
        checked = replay.check(replay.root, candidates)
        if batchable and candidate_set == frozenset(checked):
            # Keep only one complete set of individually proved root obstructions.
            # An unknown, foreign, or unproved row will be retried on any later call.
            replay.large_root_batch = candidate_set, checked
    return tuple(sorted({core - required for core in checked}, key=lambda row: (len(row), sorted(row))))
