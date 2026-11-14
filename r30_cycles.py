"""Certified cycle geometry and predecessor comparisons for a four-swap endpoint."""
import itertools
import json

import numpy as np

from r29_exchange_route import ROOT, domain_binding, physical_hash


def canonical_cycles(cells, order):
    unseen = set(range(len(cells)))
    result = []
    while unseen:
        start = min(unseen,key=lambda i: cells[i])
        i,cycle = start,[]
        while i in unseen:
            unseen.remove(i)
            cycle.append(int(cells[i]))
            i=order[i]
        if len(cycle)>1:
            result.append(tuple(cycle))
    return tuple(sorted(result,key=lambda cycle:(len(cycle),cycle)))


FOUR_FAMILIES=((2,4),(3,3),(2,2,3),(2,2,2,2))


def _contact_rules(bound,target):
    if target<208:return {}
    try:
        center=physical_hash(bound.tiles)
        domain=domain_binding(bound.model)
        archive=json.loads((ROOT/'docs/R29_EXCHANGE_CENTERS.json').read_text())
        three=json.loads((ROOT/'docs/R29_EXCHANGE_RESULTS.json').read_text())['triples']
        contact=json.loads((ROOT/'docs/R30_CONTACT_RESULTS.json').read_text())
        valid=(archive['target']==208 and archive['radius']>=5 and domain==archive['domain']
               and any(c['sha256']==center and physical_hash(c['pieces'])==center for c in archive['centers'])
               and any(c['center']==center and c['objective']=='matched_edges' and c['swap_distance']==3
                       and c['moved']==6 and c['target']==208 and c['complete'] is True and c['hits']==0 for c in three)
               and any(c['center']==center and c['cross_cycle_required'] is True and c['star']==bound.total
                       for c in contact))
    except (OSError,ValueError,TypeError,KeyError):
        return {}
    if not valid:return {}
    rules=dict.fromkeys(FOUR_FAMILIES,1)
    try:
        second=json.loads((ROOT/'docs/R30_SINGLE_CONTACT.json').read_text())
        if second['domain']==domain and second['target']==208:
            for c in second['proofs']:
                family=tuple(sorted(c['family']))
                if (family in ((2,2,3),(2,2,2,2)) and c['center']==center and
                    c['second_contact_required'] is True and all(s['score']<208 for s in c['star_survivors'])):
                    rules[family]=2
    except (OSError,ValueError,TypeError,KeyError):
        pass
    return rules


def _group_rules(bound):
    try:
        groups=json.loads((ROOT/'docs/R30_CYCLE_GROUPS.json').read_text())
        if groups['domain']!=domain_binding(bound.model) or groups['target']!=208:return False,False
        center=physical_hash(bound.tiles)
        row=next((c for c in groups['proofs'] if c['center']==center and not c['hits']
                  and c['minimum_connected_cycles']==3),None)
        if row:
            return True,(row.get('four_swaps_all_connected') is True and '2222_three_plus_one' in row['cases'])
    except (OSError,TypeError,KeyError,ValueError):
        pass
    return False,False


def certified_contacts(bound,target,families):
    """Minimum cell pairs within distance two across distinct cycles in every supplied family."""
    rules=_contact_rules(bound,target)
    if rules and _group_rules(bound)[1]:
        rules[2,2,2,2]=3
    return min((rules.get(tuple(sorted(f)),0) for f in families),default=0)


class Predecessors:
    """The earlier closed distance-three ball supports a strict-gain test."""
    def __init__(self,bound,target):
        model=bound.model
        self.bound=bound
        self.near=[set(map(int,row))-{bound.n} for row in bound.neighbors[:bound.n]]
        self.close=[{v for j in ({i}|adj) for v in ({j}|self.near[j])} for i,adj in enumerate(self.near)]
        self.contact_rules=_contact_rules(bound,target)
        self.certified=bool(self.contact_rules)
        self.group_three,self.group_four=_group_rules(bound) if self.certified else (False,False)
        if self.group_four:self.contact_rules[2,2,2,2]=3
        self.gain=None
        if self.certified:
            self.gain=self.local_gains()

    def local_gains(self):
        """Upper on changing one identity with each exterior rotation held equal."""
        n=self.bound.n
        m=self.bound.model
        tiles=self.bound.tiles
        gains=np.empty((n,n),dtype=np.int16)
        rr=np.array(list(itertools.product(range(3),repeat=3)))
        for cell in range(n):
            value=np.zeros((n,27,3),dtype=np.int16)
            for k,other in enumerate(self.bound.neighbors[cell]):
                if other>=n:continue
                right=m.reverse[m.masks[3*tiles[other]+rr[:,k],m.problem.sides[cell,k]]]
                value+=(m.masks[:,k].reshape(n,1,3)==right.reshape(1,27,1))
            best=value.max(2)
            gains[cell]=(best[tiles]-best[tiles[cell]]).max(1)
        return gains

    def reject(self,cycles,deep=False):
        """Return the first sound exclusion, or None. All four open families."""
        family=tuple(sorted(map(len,cycles)))
        if not self.certified or family not in ((2,4),(3,3),(2,2,3),(2,2,2,2)):
            return None
        contacts=0
        connections=[0]*len(cycles)
        for i,first in enumerate(cycles):
            for j,second in enumerate(cycles[:i]):
                count=sum(b in self.close[a] for a in first for b in second)
                if count:
                    contacts+=count
                    connections[i]+=1
                    connections[j]+=1
        if contacts<self.contact_rules[family]:return 'cross_cycle'
        # A simple cycle-contact graph has a component of size >=3 exactly
        # when some vertex has degree >=2.
        if self.group_three and len(cycles)>=3 and max(connections)<2:return 'cycle_component'
        # With four vertices and a component of >=3, disconnection means
        # exactly one vertex is isolated.
        if self.group_four and len(cycles)==4 and min(connections)==0:return 'cycle_component'
        change={a:cycle[(i+1)%len(cycle)] for cycle in cycles for i,a in enumerate(cycle)}
        for cycle in cycles:
            if not any(self.near[c].intersection(change) for c in cycle):
                if sum(int(self.gain[c,change[c]]) for c in cycle)<=0:
                    return 'undo_cycle'
        if deep and not self.consistent_splits(cycles,change):
            return 'undo_split'
        return None

    def consistent_splits(self,cycles,change,*,budget=6000):
        """Test splitting transpositions over a shared exterior rotation assignment.

        Keep every other physical placement and its rotation fixed in each
        predecessor. The necessary test is existential over the endpoint's
        rotations. Work exhaustion leaves the candidate in play.
        """
        model=self.bound.model
        donors=self.bound.tiles.copy()
        for c,j in change.items():
            donors[c]=self.bound.tiles[j]
        masks=model.masks
        reverse=model.reverse
        constraints=[]
        for cycle in cycles:
            for a,b in itertools.combinations(cycle,2):
                incident=[(x,s,y,t) for x,s,y,t in model.problem.edges if x in (a,b) or y in (a,b)]
                outside=sorted({z for x,_,y,_ in incident for z in (x,y)}-{a,b})
                if 3**len(outside)>budget:
                    continue
                # Precompute all nine rotations for the two changed placements
                # in both boards and broadcast each comparison over the exterior.
                assignments=np.asarray(list(itertools.product(range(3),repeat=len(outside))),dtype=np.int16)
                rotations=np.asarray(list(itertools.product(range(3),repeat=2)),dtype=np.int16)
                rots={z:assignments[:,i,None] for i,z in enumerate(outside)}
                rots.update({a:rotations[None,:,0],b:rotations[None,:,1]})
                def scores(swapped):
                    tot=np.zeros((len(assignments),9),dtype=np.int8)
                    for x,s,y,t in incident:
                        def own(z):
                            return donors[b if swapped and z==a else a if swapped and z==b else z]
                        tot+=masks[3*own(x)+rots[x],s]==reverse[masks[3*own(y)+rots[y],t]]
                    return tot
                feasible=scores(False)>scores(True).max(1)[:,None]
                if not np.any(feasible):return False
                if np.all(feasible):continue
                rows,cols=np.nonzero(feasible)
                constraints.append((tuple(outside+[a,b]),np.column_stack((assignments[rows],rotations[cols]))))
        if not constraints:return True
        # Intersect supported values until stable. Exact bounded natural joins
        # enforce shared rotations when the residual has few enough rows.
        domains={c:{0,1,2} for scope,_ in constraints for c in scope}
        changed=True
        while changed:
            changed=False
            for k,(scope,rows) in enumerate(constraints):
                keep=np.ones(len(rows),bool)
                for j,c in enumerate(scope):
                    if len(domains[c])<3:keep &= np.isin(rows[:,j],list(domains[c]))
                rows=rows[keep]
                if not len(rows):return False
                constraints[k]=(scope,rows)
                for j,c in enumerate(scope):
                    allowed=set(map(int,rows[:,j]))
                    if domains[c]!=allowed:
                        domains[c]=allowed
                        changed=True
        constraints.sort(key=lambda row:len(row[1]))
        retained=[{}]
        for scope,rows in constraints:
            if len(retained)*len(rows)>budget:return True
            out=[]
            for old in retained:
                for row in rows:
                    if all(c not in old or old[c]==v for c,v in zip(scope,row)):
                        out.append(dict(old)|dict(zip(scope,map(int,row))))
            if not out:return False
            retained=out
        return True
