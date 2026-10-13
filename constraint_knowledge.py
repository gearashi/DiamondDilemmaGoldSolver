"""Bounded, revalidated global Gold-loop witnesses; hashes are not proofs.

Only a conjunction of legal placements containing a closed proper Gold
component is reusable globally. Generic failures under a restricted DFS/job
root are deliberately unsupported. Every novel imported witness is proved
from original tile segments and seam incidence. Only exact records still
retained in this bound store can reuse their earlier independent proof. DFS
caches, claimed scores, and serialized checksums are never witness proofs.
"""
from __future__ import annotations
import hashlib
import json

VERSION=1
MAX_RECORDS=256
MAX_BYTES=262144


def _domains_hash(problem):
    return hashlib.sha256(json.dumps([sorted(set(row)) for row in problem.domains],separators=(',',':')).encode()).hexdigest()


def _disk_bytes(payload):
    # io_utils.atomic_json writes indent=2 through text mode. Reserve the larger
    # Windows CRLF representation so a file written on any host reloads there.
    return len(json.dumps(payload,indent=2,allow_nan=False).replace('\n','\r\n').encode('utf-8'))


def _digest(payload):
    return hashlib.sha256(json.dumps({k:v for k,v in payload.items() if k!='digest'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def revalidate_witness(problem,placements,*,target='gold'):
    """Prove and minimize one proper closed-loop placement conjunction.

    Returns canonical (cell,code) pairs for the cells of one independently
    reconstructed closed component. Raises ValueError for malformed records,
    impossible ownership, open paths, a sole full Gold loop, or an edge target.
    """
    if target!='gold':raise ValueError('Closed Gold-loop witnesses apply only to the Gold target.')
    if not isinstance(placements,(list,tuple)) or not 1<=len(placements)<=problem.n:
        raise ValueError('A witness needs a bounded nonempty list of placements.')
    placed={};used=set()
    for pair in placements:
        if not isinstance(pair,(list,tuple)) or len(pair)!=2:raise ValueError('Malformed witness placement.')
        cell,code=pair
        if type(cell) is not int or type(code) is not int or not 0<=cell<problem.n or not 0<=code<3*problem.n:
            raise ValueError('Witness placement is out of range.')
        if cell in placed or code//3 in used:raise ValueError('Witness repeats a cell or physical tile.')
        if code not in problem.domains[cell]:raise ValueError('Witness placement violates an original cell domain.')
        placed[cell]=code;used.add(code//3)
    parents=[];degree=[];cells=[];endpoints={}
    for cell,code in sorted(placed.items()):
        tile,rotation=divmod(code,3)
        for segment in problem.tiles[tile]['segments']:
            index=len(parents);parents.append(index);degree.append(0);cells.append(cell)
            for side,position in segment:
                point=(cell,(side+rotation)%3,position)
                if point in endpoints:raise ValueError('Original tile endpoints are not unique.')
                endpoints[point]=index
    def find(index):
        while parents[index]!=index:
            parents[index]=parents[parents[index]];index=parents[index]
        return index
    # Visit each occupied endpoint. An actual seam contributes one degree at
    # each of its two ends; union duplication is harmless and exact.
    for (cell,side,position),index in endpoints.items():
        neighbor=int(problem.neighbors[cell,side]);opposite=int(problem.sides[cell,side])
        if neighbor<0:continue
        other=endpoints.get((neighbor,opposite,12-position))
        if other is not None:
            degree[index]+=1;left,right=find(index),find(other)
            if left!=right:parents[right]=left
    groups={}
    for index in range(len(parents)):groups.setdefault(find(index),[]).append(index)
    total=sum(len(tile['segments']) for tile in problem.tiles)
    clauses=[]
    for group in groups.values():
        if len(group)<total and all(degree[index]==2 for index in group):
            clause=tuple((cell,placed[cell]) for cell in sorted({cells[index] for index in group}))
            clauses.append(clause)
    if not clauses:raise ValueError('Placements do not prove a closed proper Gold component.')
    return min(clauses,key=lambda clause:(len(clause),clause))


class KnowledgeStore:
    def __init__(self,problem,*,target='gold',limit=MAX_RECORDS):
        if target not in ('gold','edges'):raise ValueError('Unknown knowledge target.')
        if type(limit) is not int or not 1<=limit<=MAX_RECORDS:raise ValueError('Invalid knowledge record limit.')
        self.problem=problem;self.fingerprint=problem.fingerprint;self.target=target
        self._domain_rows=tuple(tuple(row) for row in problem.domains)
        self._canonical_domains=tuple(tuple(sorted(set(row))) for row in self._domain_rows)
        self.domains_sha256=_domains_hash(problem);self.limit=limit;self._records=[];self._record_set=set();self._snapshot_json=None
        self.stats={'verified':0,'proof_reuses':0,'added':0,'duplicates':0,'evictions':0}
    def matches(self,problem,target='gold'):
        if self.fingerprint!=problem.fingerprint or self.target!=target or len(problem.domains)!=len(self._domain_rows):return False
        # Exact row comparisons avoid repeatedly serializing up to 76,800 codes.
        # Do not trust object identity: callers may mutate original domains.
        for row,original,canonical in zip(problem.domains,self._domain_rows,self._canonical_domains):
            current=tuple(row)
            if current!=original and tuple(sorted(set(current)))!=canonical:return False
        return True
    def __len__(self):return len(self._records)
    def _retained_exact(self,placements):
        # Shape/type checks matter: Python bool equals int, but is not a legal
        # serialized placement. Equality is to an actual retained proof, never
        # merely a digest, prefix, subset, or caller-provided provenance flag.
        if (isinstance(placements,(list,tuple)) and 1<=len(placements)<=self.problem.n
                and all(isinstance(pair,(list,tuple)) and len(pair)==2 and
                        type(pair[0]) is int and type(pair[1]) is int for pair in placements)):
            candidate=tuple(sorted(tuple(pair) for pair in placements))
            if candidate in self._record_set:return candidate
        return None
    def add(self,placements):
        if not self.matches(self.problem,self.target):raise ValueError('Knowledge source domains changed.')
        if self._retained_exact(placements) is not None:
            self.stats['duplicates']+=1;self.stats['proof_reuses']+=1;return False
        clause=revalidate_witness(self.problem,placements,target=self.target);self.stats['verified']+=1
        return self._insert_verified(clause)
    def _insert_verified(self,clause):
        if clause in self._record_set:self.stats['duplicates']+=1;return False
        values=frozenset(clause)
        if any(frozenset(old)<=values for old in self._records):self.stats['duplicates']+=1;return False
        self._records=[old for old in self._records if not values<frozenset(old)]
        self._records.append(clause);self.stats['added']+=1;self._snapshot_json=None
        def exceeds_disk_bound():
            payload=self._payload();payload['digest']='0'*64
            return _disk_bytes(payload)>MAX_BYTES
        while len(self._records)>self.limit or exceeds_disk_bound():
            # Short witnesses forbid more completions; evict the oldest longest.
            victim=max(range(len(self._records)),key=lambda index:(len(self._records[index]),-index))
            self._records.pop(victim);self.stats['evictions']+=1
        self._record_set=set(self._records)
        return clause in self._record_set
    def merge(self,records):
        if not self.matches(self.problem,self.target):raise ValueError('Knowledge source domains changed.')
        if not isinstance(records,(list,tuple)) or len(records)>MAX_RECORDS:raise ValueError('Knowledge records exceed the bounded format.')
        # Validate the complete incoming batch before changing trusted state.
        normalized=[];reused=0
        for record in records:
            retained=self._retained_exact(record)
            if retained is None:normalized.append(revalidate_witness(self.problem,record,target=self.target))
            else:normalized.append(retained);reused+=1
        # Commit counters and trusted records only after every incoming record
        # has passed. A malformed later record cannot partly change the store.
        self.stats['verified']+=len(normalized)-reused;self.stats['proof_reuses']+=reused
        return sum(self._insert_verified(record) for record in normalized)
    def merge_snapshot(self,payload):
        """Validate and independently prove an imported snapshot against this binding.

        Full shape, integrity and original-domain binding checks run each time.
        Novel or evicted clauses are independently proved. Exact retained
        clauses reuse only this store's prior proof, never the supplied hash.
        """
        _validate_payload(self,payload)
        return self.merge(payload['records'])
    def merge_store(self,other):
        """Merge a locally verified store without proving its records twice.

        Serialized/untrusted input must go through load(), never this method.
        """
        if not isinstance(other,KnowledgeStore) or not self.matches(self.problem,self.target) or not other.matches(self.problem,self.target):
            raise ValueError('Knowledge stores have different original inputs.')
        return sum(self._insert_verified(record) for record in tuple(other._records))
    def records(self):return [[list(pair) for pair in record] for record in self._records]
    def _payload(self):
        return {'knowledge_version':VERSION,'fingerprint':self.fingerprint,'domains_sha256':self.domains_sha256,
                'target':self.target,'records':self.records()}
    def snapshot(self):
        if not self.matches(self.problem,self.target):raise ValueError('Knowledge source domains changed.')
        if self._snapshot_json is None:
            result=self._payload();result['digest']=_digest(result)
            self._snapshot_json=json.dumps(result,separators=(',',':'))
        # Every caller owns its copy, including background JSON writers.
        return json.loads(self._snapshot_json)


def _validate_payload(store,payload):
    if not store.matches(store.problem,store.target):raise ValueError('Knowledge source domains changed.')
    if not isinstance(payload,dict) or set(payload)!={'knowledge_version','fingerprint','domains_sha256','target','records','digest'}:
        raise ValueError('Malformed knowledge payload.')
    if type(payload['knowledge_version']) is not int or payload['knowledge_version']!=VERSION:
        raise ValueError('Unsupported knowledge version.')
    if payload['fingerprint']!=store.fingerprint or payload['target']!=store.target or payload['domains_sha256']!=store.domains_sha256:
        raise ValueError('Knowledge puzzle, original domains or target differs.')
    if (len(json.dumps(payload,separators=(',',':'),allow_nan=False).encode())>MAX_BYTES
            or _disk_bytes(payload)>MAX_BYTES):
        raise ValueError('Knowledge payload exceeds its byte limit.')
    if not isinstance(payload['digest'],str) or payload['digest']!=_digest(payload):
        raise ValueError('Knowledge integrity digest differs.')


def load(problem,payload,*,target='gold',limit=MAX_RECORDS):
    """Load a bounded payload; checksum validation never replaces witness proof."""
    store=KnowledgeStore(problem,target=target,limit=limit)
    store.merge_snapshot(payload)
    return store
