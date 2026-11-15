"""Replayable location-independent Gold clauses translated into placement restrictions."""
from r32_gold_loss import Certificates, pairset


def placed_bonds(problem, placements):
    loc = dict(placements)
    result = {}
    for a,s,b,t in problem.edges:
        if a not in loc or b not in loc:
            continue
        x,y=loc[a],loc[b]
        u,v=3*(x//3)+(s-x%3)%3,3*(y//3)+(t-y%3)%3
        result[min(u,v),max(u,v)]=((a,x),(b,y))
    return result


def verified_placement(problem, claims, placements):
    if len(dict(placements))!=len(placements) or any(not 0<=c<problem.n or v not in problem.domains[c]
                                                     for c,v in placements):
        return False
    rows=pairset(claims)
    if not rows<=placed_bonds(problem,placements).keys():return False
    return bool(Certificates(problem).conflicts(rows))


class BondWatcher:
    """Derive present physical mates and unit exclusions from every remaining owner pose."""
    def __init__(self, problem, claims, check=lambda:None):
        from collections import defaultdict
        self.problem = problem
        self.check = check
        cert = Certificates(problem)
        for row in claims:
            check()
            cert.add([row])
        self.clauses = tuple(sorted(cert.rows,key=lambda row:(len(row),sorted(row))))
        self.by_tile = defaultdict(set)
        self.by_pair = defaultdict(set)
        for i,clause in enumerate(self.clauses):
            for pair in clause:
                self.by_pair[pair].add(i)
                for dart in pair:
                    self.by_tile[dart//3].add(pair)
        self.support = {}
        self.status = {}
        self.actions = {}
        self.previous = None
        self.applied = self.removed = self.conflicts = self.pair_checks = 0

    def _forced_poses(self, domains, dart, mate):
        from r27_structure_filters import bits
        from validator import reverse_mask
        p = self.problem
        tile,side = divmod(dart,3)
        other,physical_side = divmod(mate,3)
        target = reverse_mask(int(p.masks[3*tile,side]))
        forced = []
        all_forced = True
        for c,mask in enumerate(domains):
            for rot in bits((mask>>(3*tile)) & 7):
                s = (side+rot)%3
                neighbor,opposite = int(p.neighbors[c,s]),int(p.sides[c,s])
                if neighbor < 0:
                    all_forced = False
                    continue
                key = tile,target,opposite
                if key not in self.support:
                    self.support[key] = sum(1<<v for v in range(3*p.n)
                                            if v//3!=tile and int(p.masks[v,opposite])==target)
                options = domains[neighbor] & self.support[key]
                if not options:
                    # This pose cannot occur in any edge-perfect arrangement.
                    continue
                expected = 1 << (3*other+(opposite-physical_side)%3)
                if options == expected:
                    forced.append((c,3*tile+rot))
                else:
                    all_forced = False
        return bool(forced) and all_forced,forced

    def prune(self, domains):
        from r27_structure_filters import bits
        if not self.clauses:
            return False
        original = tuple(domains)
        if self.previous is None:
            pairs = self.by_pair.keys()
        else:
            dirty = {c for c,(before,after) in enumerate(zip(self.previous,original)) if before!=after}
            touched = dirty | {int(n) for c in dirty for n in self.problem.neighbors[c] if n>=0}
            owners = {v//3 for c in touched for v in bits(self.previous[c] | original[c])}
            pairs = {pair for owner in owners for pair in self.by_tile.get(owner,())}
        changed = set()
        for pair in pairs:
            self.check()
            left,left_poses = self._forced_poses(original,*pair)
            right,right_poses = self._forced_poses(original,*reversed(pair))
            self.status[pair] = left or right, (*left_poses,*right_poses)
            changed.update(self.by_pair[pair])
            self.pair_checks += 1
        self.previous = original
        for i in changed:
            missing = [pair for pair in self.clauses[i] if not self.status[pair][0]]
            if len(missing)<2:
                self.actions[i] = missing
            else:
                self.actions.pop(i,None)
        for missing in self.actions.values():
            if not missing:
                domains[0] = 0
                self.conflicts += 1
                break
            for c,v in self.status[missing[0]][1]:
                bit = 1<<v
                if domains[c] & bit:
                    domains[c] ^= bit
                    self.applied += 1
        removed = sum((a^b).bit_count() for a,b in zip(original,domains))
        self.removed += removed
        return bool(removed)
