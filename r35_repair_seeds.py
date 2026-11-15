"""Necessary sets of seed cells that must change in every geographic Gold repair."""
from collections import Counter,defaultdict
from itertools import combinations,product
from pathlib import Path
import json

from constraint_lns import score
from r32_gold_bonds import inventory


class Repairs:
    def __init__(self,problem,codes,check):
        self.p,self.codes,self.check=problem,list(codes),check
        self.movable={i for i,row in enumerate(problem.domains)
                      if i not in problem.fixed and any(v!=codes[i] for v in row)}
        self.owners={v//3:i for i,v in enumerate(codes)}
        self.rows=set()
        self.by_cell=defaultdict(set)
        self.work=Counter()
        self.joined={}
        self.guarantees={}
        self.inventory=inventory(problem)
        self.mates=defaultdict(set)
        for a,b in self.inventory.possible:
            self.mates[a].add(b)
            self.mates[b].add(a)
        self.domains=[set(row) for row in problem.domains]

    def covered(self,row):
        if len(row)<=4:
            return any(frozenset(part) in self.rows for k in range(len(row)+1) for part in combinations(row,k))
        return any(old<=row for old in self.rows)

    def add(self,row,kind):
        row=frozenset(row)&self.movable
        if self.covered(row):return
        candidates=min((self.by_cell[c] for c in row),key=len) if row else self.rows
        supersets=[old for old in candidates if row<old]
        for old in supersets:
            self.rows.remove(old)
            for c in old:self.by_cell[c].remove(old)
        self.rows.add(row)
        for c in row:self.by_cell[c].add(row)
        self.work[kind]+=1

    def physical(self,cell,side,code=None):
        code=self.codes[cell] if code is None else code
        return 3*(code//3)+(side-code%3)%3

    def neighbors(self,*,exact,owners,hall,joined_stars):
        p=self.p
        incoming=defaultdict(list)
        for c,code in enumerate(self.codes):
            self.check()
            if code not in p.domains[c]:
                self.add((c,),'outside_domain')
                continue
            neighbors=defaultdict(list)
            for side,other in enumerate(p.neighbors[c]):
                if other>=0:neighbors[int(other)].append((side,int(p.sides[c,side])))
            choices=[]
            changed=[]
            for b,sides in neighbors.items():
                if not exact:continue
                pool=None
                for s,t in sides:
                    permitted={3*(dart//3)+(t-dart%3)%3 for dart in self.mates[self.physical(c,s)]}
                    pool=permitted & (self.domains[b] if pool is None else pool)
                incoming[b].append((c,pool))
                if not pool:
                    self.add((c,),'unsupported_seed_pose')
                    continue
                tile_pool={v//3 for v in pool}
                choices.append(tile_pool)
                if self.codes[b] not in pool:
                    changed.append(tile_pool)
                    if owners:self.add([c,*(self.owners[t] for t in tile_pool)],'required_donor')
                else:
                    for s,t in sides:
                        edge=tuple(sorted((self.physical(c,s),self.physical(b,t))))
                        if len(pool)==1:
                            self.guarantees.setdefault(edge,set()).add(c)
            if hall:
                for k in range(2,len(choices)+1):
                    if any(len(set.union(*group))<k for group in combinations(choices,k)):
                        self.add((c,),'neighbor_inventory')
                        break
                for k in range(2,len(changed)+1):
                    for group in combinations(changed,k):
                        donor={self.owners[t] for t in set.union(*group)}&self.movable
                        if len(donor)<k:self.add((c,),'required_distinct_donors')
                        elif len(donor)<=12:
                            for spare in combinations(donor,k-1):
                                self.add({c}|(donor-set(spare)),'required_distinct_donors')

        if joined_stars:
            # Independently supported neighbors may demand incompatible poses
            # in the same cell. All held anchors must share one actual code.
            for b,anchors in incoming.items():
                for k in range(2,len(anchors)+1):
                    for group in combinations(anchors,k):
                        cells={c for c,_ in group}
                        if self.covered(cells):continue
                        pool=set.intersection(*(q for _,q in group))
                        if not pool:self.add(cells,'joint_neighbor_pose')
                        elif self.codes[b] not in pool:
                            self.add(cells|{b},'joint_replacement')
                            if owners:self.add(cells|{self.owners[v//3] for v in pool},'joint_donor')

    def mandatory(self):
        p=self.p
        for a,b in self.inventory.required:
            if (a,b) in self.joined:continue
            ca,cb=self.owners[a//3],self.owners[b//3]
            self.add((ca,cb),'compulsory_pair')
            for dart,mate,home in ((a,b,ca),(b,a,cb)):
                side=(dart%3+self.codes[home]%3)%3
                neighbor=int(p.neighbors[home,side])
                if neighbor<0:
                    self.add((home,),'unsupported_compulsory_pose')
                    continue
                facing=int(p.sides[home,side])
                needed=3*(mate//3)+(facing-mate%3)%3
                if needed not in p.domains[neighbor]:
                    self.add((home,),'unsupported_compulsory_pose')
                elif self.codes[neighbor]!=needed:self.add((home,neighbor),'compulsory_neighbor')

    def projected(self,bonds):
        # Keeping both seed endpoint cells keeps a present bond. When keeping
        # either individual endpoint forces its mate, that premise alone suffices.
        options=[]
        for e in sorted(bonds-self.inventory.required):
            if e not in self.joined:return ()
            forced=self.guarantees.get(e,())
            options.append([(c,) for c in sorted(forced)] if forced else [self.joined[e]])
        # A bounded subset of these independently valid clauses is still sound.
        result=[]
        for i,parts in enumerate(product(*options)):
            if i>=64:break
            result.append(frozenset(c for row in parts for c in row)&self.movable)
        return tuple(result)

    def corpus(self,hints):
        from r35_contact_certificate import xor_conflict
        from r35_geographic_replay import replay_for
        from r35_induced_contacts import verified_contacts
        from r32_gold_loss import Certificates
        geographic=replay_for(self.p)
        cert=None
        ranked=[]
        for kind,terms in hints.items():
            for candidate in terms:
                group=geographic.normalize(candidate)
                if group is None:continue
                projected=self.projected(group)
                if projected:ranked.append((min(map(len,projected)),kind,tuple(sorted(group)),projected))
        for _,kind,bonds,projected in sorted(ranked,key=lambda r:r[:3]):
            self.check()
            if all(self.covered(new) for new in projected):continue
            self.work['verified_'+kind]+=1
            group=frozenset(bonds)
            if kind=='physical':
                if cert is None:cert=Certificates(self.p)
                valid=bool(cert.conflicts(group))
            elif kind=='required_xor':
                # This hint is supplied only for the published spherical input.
                valid=geographic.spherical and xor_conflict(self.p,group|self.inventory.required)
            elif kind=='geographic':
                valid=bool(geographic.check(geographic.root,[group]))
            elif kind=='edge_perfect':
                valid=bool(verified_contacts(self.p,[group]))
            else:raise ValueError('Unknown Gold repair evidence type.')
            if valid:
                for row in projected:self.add(row,kind)


def on_original_mesh(problem):
    from hashlib import sha256
    # Fixed placements narrow the same published puzzle. Authenticate tiles
    # and mesh without fixed placements; every candidate is then rechecked.
    image=json.dumps({'tiles':problem.tiles,'edges':problem.edges,'fixed':[]},
                     sort_keys=True,separators=(',',':')).encode()
    return sha256(image).hexdigest()=='e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853'


def original_hints(problem,mode='all'):
    if not on_original_mesh(problem):return {}
    docs=Path(__file__).resolve().parent/'docs'
    def read(name,field):
        try:
            data=json.loads((docs/name).read_bytes())
        except (OSError,ValueError):
            return []
        return data.get(field,[])
    hints=defaultdict(list)
    for item in read('R35_CONTACT_LOSS_CERTIFICATE.json','weighted_cores'):
        if isinstance(item,dict) and item.get('kind') in ('physical','required_xor','geographic','edge_perfect'):
            hints[item['kind']].append(item.get('bonds',[]))
    if mode=='all':
        hints['required_xor'].extend(read('R35_INTEGER_DEPARTURES.json','groups'))
        hints['geographic'].extend(read('R35_GEOGRAPHIC_RESIDUALS.json','geographic_cores'))
        hints['edge_perfect'].extend(read('R35_EDGE_CONTACTS.json','edge_perfect_cores'))
        # The original physical reference also contains checked closed circuits.
        record=read('R31_GOLD_RESULTS.json','departures')
        if isinstance(record,dict):hints['physical'].extend(record.get('center',{}).get('witnesses',[]))
    return hints


def checked_catalog(problem):
    """Return only the exact mesh and catalog covered by the checked 88-change refutation."""
    from hashlib import sha256
    if not on_original_mesh(problem):return
    try:body=(Path(__file__).resolve().parent/'docs/R35_REPAIR_PLACEMENTS.json').read_bytes().strip()
    except OSError:return
    if sha256(body).hexdigest()!='d72d6c147d3af5798f9f53a96832adf1ac82ae7e4fdc9f009c48700ddf020818':return
    return json.loads(body)


def regional_catalog():
    """Return the independently replayed local refutations for the authenticated reference repair."""
    from hashlib import sha256
    try:body=(Path(__file__).resolve().parent/'docs/R35_REPAIR_REGIONAL.json').read_bytes().strip()
    except OSError:return None
    if sha256(body).hexdigest()!='5ae4ed6a4af821eb01d653e11d202aa5122ce73334f1c15114745c17768d4c61':return None
    return json.loads(body)


def active_catalog(problem):
    record=checked_catalog(problem)
    if record is None or record.get('puzzle')!='e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853':
        return record
    from hashlib import sha256
    try:body=(Path(__file__).resolve().parent/'docs/R35_REPAIR_REFINED.json').read_bytes().strip()
    except OSError:return record
    if sha256(body).hexdigest()!='3369433c96e41f6af984ba4d8fed42998d26b62538f7920395b1df7a16719397':return record
    refined=json.loads(body)
    rows=record['rows']+refined['extra']+refined['additional']
    regional=regional_catalog()
    if regional is None:return dict(record,rows=rows,hit=refined['hit'])
    return dict(record,rows=rows+regional['small']+regional['larger'],
                lower_bound=regional['proved_lower'],hit=regional['hit'])


def certified_distance(problem,codes):
    """Transfer the exact reference lower bound using the best legal distance change per cell."""
    record=active_catalog(problem)
    if record is None:return 0,None
    reference=record['codes']
    hint=set(record['hit'])
    lower=record.get('lower_bound',len(hint))
    for c,(actual,other) in enumerate(zip(codes,reference)):
        if actual==other:continue
        domain=problem.domains[c]
        if actual in domain:
            lower-=1
            hint.discard(c)
        elif domain and all(v==other for v in domain):
            lower+=1
            hint.add(c)
    return max(0,lower),hint


def certified_choice(lower,hint,rows,cells):
    if hint is not None and len(hint)==lower and hint<=set(cells) and all(hint & row for row in rows):
        return set(hint)
    return None


def archive_rows(state):
    """A verified impossible conjunction remains impossible on every seed preserving its named poses."""
    record=active_catalog(state.p)
    if record is None:return
    agreement={c for c,(v,w) in enumerate(zip(state.codes,record['codes'])) if v==w}
    for i,row in enumerate(record['rows']):
        if i%128==0:state.check()
        if agreement.issuperset(row):state.add(row,'saved_pose_conflict')


def seed_rows(problem,codes,*,check=lambda:None,mode='all',loops=True,mandatory=True,exact=True,owners=True,hall=True,joined_stars=True,anchor_radius=None,unique=True,max_domain=40,hints=None,archive=False):
    """Build an antichain of changed-cell obligations; empty row proves this restricted repair infeasible."""
    check()
    metric=score(problem,codes)
    state=Repairs(problem,codes,check)
    mismatch=set(map(tuple,metric['unmatched_edges']))
    for a,s,b,t in problem.edges:
        edge=tuple(sorted((state.physical(a,s),state.physical(b,t))))
        if (a,s,b,t) in mismatch:
            state.add((a,b),'mismatch')
        else:
            state.joined[edge]=(a,b)
            if edge not in state.inventory.possible:state.add((a,b),'forbidden_mate')
    if loops:
        for size,cells in problem._closed_components(list(codes)):
            if size<problem.total_segments:state.add(cells,'proper_circuit')
    if mandatory:state.mandatory()
    state.neighbors(exact=exact,owners=owners,hall=hall,joined_stars=joined_stars)
    if archive:
        if certified_distance(problem,codes)[0]>len(state.movable):
            state.add((),'distance_exceeds_movable')
        else:archive_rows(state)
    if anchor_radius is not None and frozenset() not in state.rows:
        from r35_repair_propagation import RepairPropagation
        inferred=RepairPropagation(problem,state.mates)
        for anchors in inferred.anchor_sets(anchor_radius):
            check()
            inferred.project(state,anchors,max_domain=max_domain,unique=unique)
    if mode!='none':state.corpus(original_hints(problem,mode) if hints is None else hints)
    rows=sorted(state.rows,key=lambda r:(len(r),sorted(r)))
    return rows,dict(state.work)


def repair_rows(problem,codes,check=lambda:None):
    """Reuse only an identical puzzle, seed, and current domain snapshot in this process."""
    check()
    ds=tuple(tuple(row) for row in problem.domains)
    semantic=json.dumps([problem.tiles,problem.edges,problem.board.edges],separators=(',',':')).encode()
    key=(semantic,problem.masks.tobytes(),problem.neighbors.tobytes(),problem.sides.tobytes(),
         tuple(codes),ds,tuple(sorted(problem.fixed.items())),tuple(proof_sources().items()))
    known=getattr(problem,'_repair_seed_rows',None)
    if known is not None and known[0]==key:return known[1]
    rows,_=seed_rows(problem,codes,check=check,mode='compact',anchor_radius=2,max_domain=64,archive=True)
    rows=tuple(rows)
    problem._repair_seed_rows=key,rows
    return rows


def proof_sources():
    from hashlib import sha256
    root=Path(__file__).resolve().parent
    names=('r35_repair_seeds.py','r35_repair_propagation.py','r32_gold_bonds.py','r27_structure_filters.py',
           'r35_contact_certificate.py','r35_loss_witness.py','r35_geographic_replay.py',
           'r35_induced_contacts.py','r32_gold_loss.py','r31_geographic.py','r28_geo_sewing.py',
           'r28_geo_color.py','r29_gold_phase.py','docs/R35_CONTACT_LOSS_CERTIFICATE.json',
           'docs/R35_REPAIR_PLACEMENTS.json','docs/R35_REPAIR_REFINED.json','docs/R35_REPAIR_REGIONAL.json')
    return {name:(sha256((root/name).read_bytes()).hexdigest() if (root/name).is_file() else 'absent')
            for name in names}

def master_rows(derived,earned,limit=8192):
    """Preserve all native cores and fit derived rows within the retained master's input cap."""
    earned=[frozenset(row) for row in earned]
    if any(not row for row in earned):return [frozenset()]
    extras=[row for row in derived if not any(old<=row for old in earned)]
    return [*earned,*extras[:max(0,limit-len(earned))]]
