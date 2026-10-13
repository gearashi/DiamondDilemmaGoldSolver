"""Bounded global conflicts, independently proved under original puzzle domains.

An offered or imported conjunction is only a claim. The private verification
SAT model includes ORIGINAL cell domains, physical-tile inventory and seam
constraints, without a job prefix, root symmetry or previously imported cuts.
Only a completed UNSAT query publishes its assumption core. Serialized hashes
check integrity, never proof. Unknown and cancelled work remain untrusted.
"""
from collections import OrderedDict
import hashlib,json,math,time
from constraint_sat_assist import SatAssist

VERSION=1
MAX_RECORDS=128
MAX_PENDING=128
MAX_LITERALS=32
MAX_BYTES=131072
MAX_ATTEMPTS=3
MAX_CALLS_PER_ADVANCE=4
MAX_SECONDS_PER_ADVANCE=.25
STAT_KEYS=('shared_conflicts_offered','shared_conflicts_verified','shared_conflicts_rejected',
           'shared_conflicts_unknown','shared_conflicts_duplicates','shared_conflicts_evictions',
           'shared_conflicts_pending_dropped','shared_conflicts_calls','shared_conflicts_entries',
           'shared_conflicts_pending','shared_conflicts_clauses')
GAUGE_KEYS=('shared_conflicts_entries','shared_conflicts_pending','shared_conflicts_clauses')


def _digest(payload):
    return hashlib.sha256(json.dumps({key:value for key,value in payload.items() if key!='digest'},
                                    sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _disk_bytes(payload):
    # atomic_json uses indentation and Windows may translate newlines to CRLF.
    return len(json.dumps(payload,indent=2,allow_nan=False).replace('\n','\r\n').encode('utf-8'))


class VerifiedConflictStore:
    def __init__(self,problem,target='gold',*,limit=MAX_RECORDS,factory=None):
        if target not in ('gold','edges'):raise ValueError('Unknown shared conflict target.')
        if type(limit)is not int or not 1<=limit<=MAX_RECORDS:raise ValueError('Invalid shared conflict limit.')
        self.problem,self.target,self.limit=problem,target,limit
        self.fingerprint=problem.fingerprint
        self._domain_rows=tuple(tuple(row) for row in problem.domains)
        self._domains=tuple(tuple(sorted(set(row))) for row in self._domain_rows)
        self._initial=tuple(sum(1<<code for code in row) for row in self._domains)
        self.domains_sha256=hashlib.sha256(json.dumps(self._domains,separators=(',',':')).encode()).hexdigest()
        self._records=[];self._pending=OrderedDict();self._discarded=OrderedDict()
        self._factory=factory;self._oracle=None;self._snapshot_json=None;self._closed=False
        self.stats=dict.fromkeys(STAT_KEYS,0);self.status='idle';self.reason=None
    def matches(self,problem,target='gold'):
        return (self.fingerprint==problem.fingerprint and self.target==target
                and len(problem.domains)==len(self._domains)
                and all(tuple(row)==original or tuple(sorted(set(row)))==canonical
                        for row,original,canonical in zip(problem.domains,self._domain_rows,self._domains)))
    def _assert_binding(self):
        if not self.matches(self.problem,self.target):raise ValueError('Shared conflict original domains changed.')
    def _normalize(self,placements):
        if not isinstance(placements,(list,tuple)) or len(placements)>self.problem.n:
            raise ValueError('A conflict needs a bounded placement list.')
        pairs=[];cells=set()
        for pair in placements:
            if not isinstance(pair,(list,tuple)) or len(pair)!=2:raise ValueError('Malformed conflict placement.')
            cell,code=pair
            if type(cell)is not int or type(code)is not int or not 0<=cell<self.problem.n or code not in self._domains[cell]:
                raise ValueError('Conflict placement violates original domains.')
            if cell in cells:raise ValueError('A conflict repeats a cell.')
            cells.add(cell);pairs.append((cell,code))
        return tuple(sorted(pairs))
    def _refresh(self):
        self.stats['shared_conflicts_entries']=len(self._records)
        self.stats['shared_conflicts_pending']=len(self._pending)
        if self._oracle is not None:
            self.stats['shared_conflicts_calls']=self._oracle.stats['sat_assist_calls']
            self.stats['shared_conflicts_clauses']=self._oracle.stats['sat_assist_clauses']
    @property
    def pending_count(self):return len(self._pending)
    def __len__(self):return len(self._records)
    def offer(self,placements):
        """Queue a claim; False means duplicate or bounded work was discarded."""
        self._assert_binding();record=self._normalize(placements)
        self.stats['shared_conflicts_offered']+=1
        if len(record)>MAX_LITERALS:
            self.stats['shared_conflicts_pending_dropped']+=1;return False
        values=frozenset(record)
        if (record in self._pending or record in self._discarded
                or any(frozenset(old)<=values for old in self._records)):
            self.stats['shared_conflicts_duplicates']+=1;return False
        self._pending[record]=0
        while len(self._pending)>MAX_PENDING:
            self._pending.popitem(last=False);self.stats['shared_conflicts_pending_dropped']+=1
        self._refresh();return record in self._pending
    def _remember_discard(self,record):
        self._discarded[record]=None
        while len(self._discarded)>MAX_PENDING:self._discarded.popitem(last=False)
    def _insert_verified(self,record):
        # Internal only: callers cannot turn serialized claims into proofs.
        values=frozenset(record)
        if any(frozenset(old)<=values for old in self._records):
            self.stats['shared_conflicts_duplicates']+=1;return False
        self._records=[old for old in self._records if not values<frozenset(old)]
        self._records.append(record);self._snapshot_json=None
        while len(self._records)>self.limit or _disk_bytes({**self._payload(),'digest':'0'*64})>MAX_BYTES:
            victim=max(range(len(self._records)),key=lambda index:(len(self._records[index]),-index))
            self._records.pop(victim);self.stats['shared_conflicts_evictions']+=1
        if record in self._records:
            self.stats['shared_conflicts_verified']+=1;return True
        return False
    def advance(self,check=lambda:None,*,seconds=.01,attempts=2,conflicts=128):
        """Prepare/query bounded incremental work, returning new verified count.

        No pending record is exposed by records()/snapshot(). A timed-out query
        can be retried at most MAX_ATTEMPTS times; it never becomes a clause.
        """
        if (isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds)
                or not 0<=seconds<=MAX_SECONDS_PER_ADVANCE):raise ValueError('Invalid shared verification time budget.')
        if type(attempts)is not int or not 1<=attempts<=MAX_CALLS_PER_ADVANCE:raise ValueError('Invalid shared verification query count.')
        if type(conflicts)is not int or not 1<=conflicts<=10000:raise ValueError('Invalid shared verification conflict budget.')
        self._assert_binding();check()
        if self._closed or not self._pending or seconds==0:return 0
        deadline=time.monotonic()+seconds;added=0
        if self._oracle is None:self._oracle=SatAssist(self.problem,self._initial,self.target,factory=self._factory)
        try:
            if not self._oracle.prepare(check,seconds=max(0,deadline-time.monotonic())):
                self.status=self._oracle.status;self.reason=self._oracle.reason;return 0
            self._assert_binding();self.status='ready'
            for _ in range(attempts):
                check();remaining=deadline-time.monotonic()
                if not self._pending or remaining<=0:break
                record,tries=next(iter(self._pending.items()))
                core=self._oracle.query(record,check,seconds=remaining,conflicts=conflicts)
                check();self._assert_binding()
                # Dequeue only after complete cooperative work: a cancellation
                # leaves the original unverified claim available for retry.
                self._pending.pop(record)
                if core is not None:
                    if self._oracle.last_outcome!='unsat' or not set(core)<=set(record):
                        raise RuntimeError('Global verifier returned an invalid assumption core.')
                    added+=self._insert_verified(core)
                elif self._oracle.last_outcome=='sat':
                    self.stats['shared_conflicts_rejected']+=1;self._remember_discard(record)
                else:
                    self.stats['shared_conflicts_unknown']+=1;tries+=1
                    if tries<MAX_ATTEMPTS:self._pending[record]=tries
                    else:self.stats['shared_conflicts_pending_dropped']+=1;self._remember_discard(record)
            return added
        finally:self._refresh()
    def records(self):
        self._assert_binding();return [[list(pair) for pair in record] for record in self._records]
    def _payload(self):
        return {'shared_conflicts_version':VERSION,'fingerprint':self.fingerprint,
                'domains_sha256':self.domains_sha256,'target':self.target,
                'records':[[list(pair) for pair in record] for record in self._records]}
    def snapshot(self):
        self._assert_binding()
        if self._snapshot_json is None:
            payload=self._payload();payload['digest']=_digest(payload)
            self._snapshot_json=json.dumps(payload,separators=(',',':'))
        return json.loads(self._snapshot_json)
    def import_snapshot(self,payload):
        """Validate the transport, then queue EVERY novel record for new proof."""
        self._assert_binding()
        if not isinstance(payload,dict) or set(payload)!={'shared_conflicts_version','fingerprint','domains_sha256','target','records','digest'}:
            raise ValueError('Malformed shared conflict payload.')
        if type(payload['shared_conflicts_version'])is not int or payload['shared_conflicts_version']!=VERSION:
            raise ValueError('Unsupported shared conflict version.')
        if payload['fingerprint']!=self.fingerprint or payload['domains_sha256']!=self.domains_sha256 or payload['target']!=self.target:
            raise ValueError('Shared conflict input, original domains or target differs.')
        records=payload['records']
        if not isinstance(records,list) or len(records)>MAX_RECORDS:raise ValueError('Shared conflict record limit exceeded.')
        normalized=[self._normalize(record) for record in records]
        if any(len(record)>MAX_LITERALS for record in normalized):raise ValueError('Shared conflict literal limit exceeded.')
        if _disk_bytes(payload)>MAX_BYTES:raise ValueError('Shared conflict byte limit exceeded.')
        if not isinstance(payload['digest'],str) or payload['digest']!=_digest(payload):raise ValueError('Shared conflict integrity digest differs.')
        return sum(self.offer(record) for record in normalized)
    def close(self):
        if self._oracle is not None:self._oracle.close()
        self._oracle=None;self._closed=True;self.status='closed';self.stats['shared_conflicts_clauses']=0
    def __del__(self):
        try:self.close()
        except Exception:pass
