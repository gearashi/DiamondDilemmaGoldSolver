"""Exact bounded reasoning and portable, independently checked certificates.

All support tables are relaxations of the current domains. A budget expiration
retains only already completed deductions. A cached proof applies only to a
subset of its explicit scope and an identical puzzle and objective.
"""
from collections import Counter, OrderedDict
from itertools import combinations
import hashlib
import json
from constraint_pruning import _tile_choices, values
from validator import reverse_mask

STAT_KEYS=('path_checks','path_pairs_removed','path_prunes','conservation_checks',
 'conservation_prunes','hall_certificate_hits','hall_certificates',
 'certificate_hits','certificate_checks','explanation_tests','explanation_cells_removed',
 'relation_records','relation_hits','relation_prunes','joint_conservation_checks',
 'joint_conservation_prunes','capacity_edges')


def hall_certificate(rows, check=lambda:None):
    """Alternating-forest Hall witness, independently checkable in linear size."""
    owners={}
    for start in range(len(rows)):
        queue=[start]; seen={start}; parent={}; found=None
        for cell in queue:
            check()
            for tile in values(rows[cell]):
                if tile in parent:continue
                parent[tile]=cell
                if tile not in owners:found=tile;break
                other=owners[tile]
                if other not in seen:seen.add(other);queue.append(other)
            if found is not None:break
        if found is None:
            union=0
            for cell in seen:union |= rows[cell]
            assert union.bit_count()<len(seen)
            return tuple(sorted(seen)),union
        tile=found
        while True:
            cell=parent[tile]
            previous=next((old for old,owner in owners.items() if owner==cell),None)
            owners[tile]=cell
            if previous is None:break
            del owners[previous];tile=previous
    return None


class HallCertificateCache:
    def __init__(self,limit=128):self.limit=limit;self.records=[];self.hits=0
    def impossible(self,rows,check=lambda:None):
        for cells,tiles in self.records:
            check()
            if all(cell<len(rows) and not rows[cell]&~tiles for cell in cells):self.hits+=1;return True
        return False
    def remember(self,rows,check=lambda:None):
        witness=hall_certificate(rows,check)
        if witness is not None and witness not in self.records:
            self.records.append(witness);self.records=self.records[-self.limit:]
        return witness


def minimize_failure_scope(before,initial,oracle,check=lambda:None,max_tests=8):
    """Release assumptions only after a COMPLETE independent failure proof.

    oracle returns False for proved impossible, True for maybe possible and None
    for unknown. The returned cells therefore suffice relative to initial.
    """
    current=list(before);active=[i for i,(a,b) in enumerate(zip(current,initial)) if a!=b]
    tests=0
    # Do not assume the supplied conclusion: establish it in this oracle first.
    if oracle(current) is not False:return None,0
    for cell in tuple(active):
        if tests>=max_tests:break
        check();trial=current.copy();trial[cell]=initial[cell];tests+=1
        answer=oracle(trial);check()
        if answer is False:current=trial;active.remove(cell)
    return tuple(active),tests


class DeepPruner:
    def __init__(self,problem,*,retain_relations=False):
        self.problem=problem;self.stats=dict.fromkeys(STAT_KEYS,0)
        from constraint_relations import ScopedRelations
        self.relations=ScopedRelations() if retain_relations else None
        self.masks=tuple(tuple(map(int,row)) for row in problem.masks)
        self.links={}
        for a,sa,b,sb in problem.edges:
            self.links.setdefault((a,b),[]).append((sa,sb))
            self.links.setdefault((b,a),[]).append((sb,sa))
        self.closed=all(all(int(x)>=0 for x in row) for row in problem.neighbors)
        counts=Counter()
        if self.closed:
            for tile in range(problem.n):
                for mask in self.masks[3*tile]:counts[min(mask,reverse_mask(mask))]+=1
        self.orbit_counts=counts
        self.cache=OrderedDict()
    def compatible(self,a,x,b,y):
        return x//3!=y//3 and all(self.masks[x][sa]==reverse_mask(self.masks[y][sb]) for sa,sb in self.links.get((a,b),()))
    def paths(self,domains,check=lambda:None,max_cells=8,max_values=8,max_sweeps=4):
        """Restricted path consistency on complete binary relations of a subset."""
        selected=sorted((i for i,v in enumerate(domains) if 1<v.bit_count()<=max_values),key=lambda c:(domains[c].bit_count(),c))[:max_cells]
        if len(selected)<3:return False
        source=tuple(domains[c] for c in selected);key=(tuple(selected),source)
        self.stats['path_checks']+=1
        cached=self.cache.get(key)
        if cached is not None:
            changed=False
            for c,v in zip(selected,cached):changed |= domains[c]!=v;domains[c]=v
            return changed
        codes={c:tuple(values(domains[c])) for c in selected}
        relation={(a,b):{x:{y for y in codes[b] if self.compatible(a,x,b,y)} for x in codes[a]} for a in selected for b in selected if a!=b}
        for sweep in range(max_sweeps):
            changed=False
            for a,b in combinations(selected,2):
                for c in selected:
                    if c in (a,b):continue
                    for x in codes[a]:
                        check()
                        for y in tuple(relation[a,b][x]):
                            if not relation[a,c][x]&relation[b,c][y]:
                                relation[a,b][x].remove(y);relation[b,a][y].discard(x)
                                self.stats['path_pairs_removed']+=1;changed=True
            if not changed:break
        result=[]
        for a in selected:
            keep=sum(1<<x for x in codes[a] if all(relation[a,b][x] for b in selected if a!=b));result.append(keep)
        check()
        if self.relations is not None:
            scope=tuple(zip(selected,source))
            for a,b in combinations(selected,2):
                self.relations.remember(scope,a,b,((x,y) for x,ys in relation[a,b].items() for y in ys))
            self.stats.update(self.relations.stats)
        self.cache[key]=tuple(result)
        while len(self.cache)>128:self.cache.popitem(last=False)
        changed=False
        for c,v in zip(selected,result):
            self.stats['path_prunes']+=(domains[c]&~v).bit_count();changed |= domains[c]!=v;domains[c]=v
        return changed
    def retained_relations(self,domains,check=lambda:None):
        if self.relations is None:return False
        changed=self.relations.prune(domains,check);self.stats.update(self.relations.stats)
        return changed

    def joint_conservation(self,domains,check=lambda:None):
        """Enforce all unoriented seam-signature capacities simultaneously."""
        if not self.closed or not all(domains):return False
        from constraint_relations import capacity_supports
        self.stats['joint_conservation_checks']+=1
        capacities={orbit:count//2 for orbit,count in self.orbit_counts.items()}
        if any(count%2 for count in self.orbit_counts.values()):domains[0]=0;return True
        masks=[];choices=[]
        for a,sa,b,sb in self.problem.edges:
            check()
            common={self.masks[c][sa] for c in values(domains[a])}&{reverse_mask(self.masks[c][sb]) for c in values(domains[b])}
            masks.append(common);choices.append({min(m,reverse_mask(m)) for m in common})
        supported=capacity_supports(choices,capacities,check,self.stats)
        if supported is None:domains[0]=0;return True
        current=list(domains)
        for i,(a,sa,b,sb) in enumerate(self.problem.edges):
            check();allowed={m for m in masks[i] if min(m,reverse_mask(m)) in supported[i]}
            current[a]&=sum(1<<c for c in values(domains[a]) if self.masks[c][sa] in allowed)
            current[b]&=sum(1<<c for c in values(domains[b]) if reverse_mask(self.masks[c][sb]) in allowed)
        check();self.stats['joint_conservation_prunes']+=sum((a&~b).bit_count() for a,b in zip(domains,current))
        changed=current!=domains;domains[:]=current;return changed

    def conservation(self,domains,check=lambda:None):
        """Global seam-mask multiplicities and total strand count, beyond parity."""
        if not self.closed or not all(domains):return False
        self.stats['conservation_checks']+=1
        edge_masks=[]
        for a,sa,b,sb in self.problem.edges:
            check();left={self.masks[c][sa] for c in values(domains[a])};right={reverse_mask(self.masks[c][sb]) for c in values(domains[b])}
            common=left&right
            if not common:domains[a]=0;return True
            edge_masks.append(common)
        allowed=[set(row) for row in edge_masks]
        # Every unoriented mask incidence is paired exactly once on a closed board.
        for orbit,total in self.orbit_counts.items():
            check()
            if total%2:domains[0]=0;return True
            target=total//2
            forced=[i for i,row in enumerate(allowed) if all(min(m,reverse_mask(m))==orbit for m in row)]
            possible=[i for i,row in enumerate(allowed) if any(min(m,reverse_mask(m))==orbit for m in row)]
            if len(forced)>target or len(possible)<target:domains[0]=0;return True
            if len(forced)==target:
                fixed=set(forced)
                for i in possible:
                    if i not in fixed:allowed[i]={m for m in allowed[i] if min(m,reverse_mask(m))!=orbit}
            elif len(possible)==target:
                for i in possible:allowed[i]={m for m in allowed[i] if min(m,reverse_mask(m))==orbit}
        # Exact subset-sum support for sum(seam endpoint counts)=all tile segments.
        counts=[{m.bit_count() for m in row} for row in allowed];target=self.problem.total_segments
        prefix=[{0}]
        for row in counts:
            check();prefix.append({a+b for a in prefix[-1] for b in row if a+b<=target})
        if target not in prefix[-1]:domains[0]=0;return True
        suffix=[set() for _ in range(len(counts)+1)];suffix[-1]={0}
        for i in range(len(counts)-1,-1,-1):
            check();suffix[i]={a+b for a in suffix[i+1] for b in counts[i] if a+b<=target}
        for i,row in enumerate(allowed):
            check();keep_counts={v for v in counts[i] if any(target-v-a in suffix[i+1] for a in prefix[i])}
            allowed[i]={m for m in row if m.bit_count() in keep_counts}
        narrowed=list(domains)
        for i,(a,sa,b,sb) in enumerate(self.problem.edges):
            check();narrowed[a]&=sum(1<<c for c in values(domains[a]) if self.masks[c][sa] in allowed[i])
            narrowed[b]&=sum(1<<c for c in values(domains[b]) if reverse_mask(self.masks[c][sb]) in allowed[i])
        changed=narrowed!=domains
        self.stats['conservation_prunes']+=sum((old&~new).bit_count() for old,new in zip(domains,narrowed))
        domains[:]=narrowed;return changed


def orientation_equivalences(problem):
    """Full strand pairing equality, never just equal three-side edge masks."""
    groups=[];piece_classes={}
    for tile in range(problem.n):
        signatures=[]
        for rotation in range(3):
            signatures.append(tuple(sorted(tuple(sorted((((a[0]+rotation)%3,a[1]),((b[0]+rotation)%3,b[1])))) for a,b in problem.tiles[tile]['segments'])))
        by={}
        for r,s in enumerate(signatures):by.setdefault(s,[]).append(3*tile+r)
        groups.extend(tuple(row) for row in by.values() if len(row)>1)
        piece_classes.setdefault(min(signatures),[]).append(tile)
    return tuple(groups),tuple(tuple(v) for v in piece_classes.values() if len(v)>1)


def rotate_gold_clause(problem,placements,check=lambda:None):
    """Transport only independently validated proper-loop witnesses."""
    from constraint_symmetry import proper_rotations
    partial=[-1]*problem.n
    for cell,code in placements:partial[cell]=code
    if not problem.partial_loop_impossible(partial):return ()
    try:rotations=proper_rotations(problem.board,check)
    except ValueError:return ()
    if tuple(problem.edges)!=tuple(problem.board.edges):return ()
    result=set()
    for transform in rotations:
        check();mapped=tuple(sorted((transform.cells[cell],3*(code//3)+(code%3+transform.side_offsets[cell])%3) for cell,code in placements))
        witness=[-1]*problem.n
        for cell,code in mapped:witness[cell]=code
        # Independent semantic validation also detects a mismatched rotation convention.
        if problem.partial_loop_impossible(witness):result.add(mapped)
    return tuple(sorted(result))


class CertificateStore:
    """Portable Hall and proper-loop certificates; no solver needed on import."""
    def __init__(self,problem,target='gold',limit=128):
        self.problem,self.target,self.limit=problem,target,limit;self.records=[];self.hits=0
        self.root=tuple(sum(1<<c for c in set(row)) for row in problem.domains)
    def verify(self,record):
        if not isinstance(record,dict) or record.get('kind') not in ('hall','loop'):return False
        try:scope=tuple(int(v,16) for v in record['scope'])
        except (ValueError,TypeError,KeyError):return False
        if len(scope)!=len(self.root) or any(v<0 or v&~r for v,r in zip(scope,self.root)):return False
        cells=record.get('cells')
        if not isinstance(cells,list) or not cells or any(type(c)is not int or not 0<=c<len(scope) for c in cells) or len(set(cells))!=len(cells):return False
        if record['kind']=='hall':
            union=0
            for c in cells:union|=_tile_choices(scope[c])
            return union.bit_count()<len(cells)
        if self.target!='gold' or any(scope[c].bit_count()!=1 for c in cells):return False
        partial=[scope[c].bit_length()-1 if c in cells else -1 for c in range(len(scope))]
        return self.problem.partial_loop_impossible(partial)
    def offer(self,record):
        if not self.verify(record):return False
        clean=json.loads(json.dumps(record))
        if clean not in self.records:self.records.append(clean);self.records=self.records[-self.limit:]
        return True
    def capture(self,domains,check=lambda:None):
        witness=hall_certificate(tuple(_tile_choices(v) for v in domains),check)
        if witness is not None:return self.offer({'kind':'hall','scope':[hex(v) for v in domains],'cells':list(witness[0])})
        if self.target=='gold':
            assigned=[v.bit_length()-1 if v.bit_count()==1 else -1 for v in domains]
            if self.problem.partial_loop_impossible(assigned):return self.offer({'kind':'loop','scope':[hex(v) for v in domains],'cells':[c for c,v in enumerate(assigned) if v>=0]})
        return False
    def prune(self,domains,check=lambda:None):
        for record in self.records:
            check()
            if all(not value&~int(old,16) for value,old in zip(domains,record['scope'])):
                self.hits+=1;domains[0]=0;return True
        return False
    def snapshot(self):return {'version':1,'fingerprint':self.problem.fingerprint,'target':self.target,'root':[hex(v) for v in self.root],'records':json.loads(json.dumps(self.records))}
    def load_snapshot(self,payload):
        if (not isinstance(payload,dict) or payload.get('version')!=1 or payload.get('fingerprint')!=self.problem.fingerprint or payload.get('target')!=self.target or payload.get('root')!=[hex(v) for v in self.root]):raise ValueError('Certificate scope differs.')
        records=payload.get('records')
        if not isinstance(records,list) or len(records)>self.limit:raise ValueError('Certificate list exceeds bounds.')
        return sum(self.offer(record) for record in records)
