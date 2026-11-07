"""Portable mask-conflict claims with original-root or job-root revalidation.

Persisted bytes and worker messages never become proof authority. Every imported
claim is checked under the exact bound root before it can prune any domains.
This reuses discovered conflicts across restarts without replaying their search
subtrees; limited/unknown verification remains queued, never exclusion evidence.
"""
from collections import OrderedDict
import hashlib,json,math,time
from pathlib import Path
from constraint_sat_assist import SatAssist

VERSION=1
MAX_RECORDS=128
MAX_LITERALS=64
MAX_BYTES=1048576
STAT_KEYS=('mask_conflicts_offered','mask_conflicts_verified','mask_conflicts_rejected',
           'mask_conflicts_unknown','mask_conflicts_duplicates','mask_conflicts_prunes',
           'mask_conflicts_hits','mask_conflicts_calls','mask_conflicts_entries',
           'mask_conflicts_pending','mask_conflicts_loaded','mask_proofs_verified',
           'mask_proofs_generated','mask_proofs_rejected','mask_proofs_unknown')
GAUGE_KEYS=('mask_conflicts_entries','mask_conflicts_pending')


def _digest(payload):
    body={key:value for key,value in payload.items() if key!='digest'}
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _subsumes(left,right):
    other=dict(right)
    return all(cell in other and not other[cell]&~mask for cell,mask in left)


class VerifiedMaskConflictStore:
    def __init__(self,problem,target='gold',*,initial=None,factory=None,proof_transport=False):
        if target not in ('gold','edges'):raise ValueError('Unknown mask-conflict target.')
        self.problem,self.target=problem,target
        self.original=tuple(sum(1<<code for code in sorted(set(row))) for row in problem.domains)
        self.initial=self.original if initial is None else tuple(initial)
        if (len(self.initial)!=problem.n or any(type(mask)is not int or mask<=0 or mask&~old
                                              for mask,old in zip(self.initial,self.original))):
            raise ValueError('Invalid mask-conflict effective root.')
        self.binding=(problem.fingerprint,target,self.original,self.initial)
        self._records=[];self._pending=OrderedDict();self._discarded=OrderedDict()
        self._oracle=None;self._factory=factory;self._closed=False
        self.proof_transport=bool(proof_transport);self._proof_context=None
        self._certificates=OrderedDict();self._certificate_pending=OrderedDict();self._proof_queue=OrderedDict()
        self.stats=dict.fromkeys(STAT_KEYS,0);self.status='idle';self.reason=None

    def matches(self,problem,target='gold',initial=None):
        source=tuple(sum(1<<code for code in sorted(set(row))) for row in problem.domains)
        root=source if initial is None else tuple(initial)
        return self.binding==(problem.fingerprint,target,source,root)

    def _check(self):
        if not self.matches(self.problem,self.target,self.initial):raise ValueError('Mask-conflict binding changed.')

    def _normalize(self,record):
        if not isinstance(record,(list,tuple)) or len(record)>self.problem.n:
            raise ValueError('Malformed mask-conflict record.')
        cells=set();result=[]
        for item in record:
            if not isinstance(item,(list,tuple)) or len(item)!=2:raise ValueError('Malformed mask-conflict literal.')
            cell,mask=item
            if (type(cell)is not int or not 0<=cell<self.problem.n or cell in cells
                    or type(mask)is not int or mask<=0 or mask&~self.initial[cell]):
                raise ValueError('Mask-conflict literal escapes its effective root.')
            cells.add(cell)
            if mask!=self.initial[cell]:result.append((cell,mask))
        return tuple(sorted(result))

    def _refresh(self):
        self.stats['mask_conflicts_entries']=len(self._records)
        self.stats['mask_conflicts_pending']=len(self._pending)

    def __len__(self):return len(self._records)
    @property
    def pending_count(self):return len(self._pending)

    def offer(self,record):
        self._check();record=self._normalize(record);self.stats['mask_conflicts_offered']+=1
        if len(record)>MAX_LITERALS:return False
        if record in self._pending or record in self._discarded or any(_subsumes(old,record) for old in self._records):
            self.stats['mask_conflicts_duplicates']+=1;return False
        self._pending[record]=0
        while len(self._pending)>MAX_RECORDS:self._pending.popitem(last=False)
        self._refresh();return record in self._pending

    def _proofs(self):
        if self._proof_context is None:
            from constraint_proof import MaskProofContext
            self._proof_context=MaskProofContext(self.problem,self.initial,self.target)
        return self._proof_context

    def _admit_verified(self,core):
        if any(_subsumes(old,core) for old in self._records):return False
        self._records=[old for old in self._records if not _subsumes(core,old)]
        self._records.append(core)
        while len(self._records)>MAX_RECORDS:self._records.pop(0)
        for old in tuple(self._certificates):
            if old not in self._records:del self._certificates[old]
        self.stats['mask_conflicts_verified']+=1
        return True

    def offer_certificate(self,certificate):
        """Queue untrusted proof bytes; only independent verification admits them."""
        from constraint_proof import MAX_PROOF_BYTES
        self._check()
        if not self.proof_transport:return False
        if not isinstance(certificate,dict):return False
        try:
            if len(json.dumps(certificate))>MAX_PROOF_BYTES:return False
            record=self._normalize([(c,int(mask,16)) for c,mask in certificate['record']])
        except (KeyError,ValueError,TypeError):return False
        if (certificate.get('fingerprint')!=self.problem.fingerprint or certificate.get('target')!=self.target
                or certificate.get('initial')!=[hex(mask) for mask in self.initial]):return False
        if len(record)>MAX_LITERALS:return False
        if record in self._certificates:return False
        self.offer(record)
        self._certificate_pending[record]=(json.loads(json.dumps(certificate)),0)
        while len(self._certificate_pending)>8:self._certificate_pending.popitem(last=False)
        return True

    def advance(self,check=lambda:None,*,seconds=.01,attempts=2,conflicts=256):
        if (isinstance(seconds,bool) or not isinstance(seconds,(float,int)) or not math.isfinite(seconds)
                or not 0<=seconds<=1 or type(attempts)is not int or not 1<=attempts<=16
                or type(conflicts)is not int or not 1<=conflicts<=100000):
            raise ValueError('Invalid mask verification budget.')
        self._check();check()
        if self._closed or not seconds or not (self._pending or self._proof_queue or self._certificate_pending):return 0
        deadline=time.monotonic()+seconds;added=0
        try:
            if self.proof_transport and self._certificate_pending:
                from constraint_proof import ProofLimit
                context=self._proofs()
                if not context.prepare(check,seconds=max(0,deadline-time.monotonic())):
                    self.status='building_proof_context';return 0
                record,(certificate,tries)=next(iter(self._certificate_pending.items()))
                def proof_check():
                    check()
                    if time.monotonic()>=deadline:raise ProofLimit('Import proof time budget reached.')
                try:valid=context.verify(certificate,proof_check)
                except ProofLimit:
                    self.stats['mask_proofs_unknown']+=1
                    # A large proof may be unsuitable for tiny work leases.
                    # Keep its ordinary claim so it can use existing SAT verification.
                    if tries<2:self._certificate_pending[record]=(certificate,tries+1)
                    else:self._certificate_pending.pop(record,None)
                    return 0
                self._certificate_pending.pop(record,None)
                if valid:
                    added+=self._admit_verified(record);self._pending.pop(record,None)
                    if record in self._records:self._certificates[record]=certificate
                    self.stats['mask_proofs_verified']+=1
                else:self.stats['mask_proofs_rejected']+=1
            if self._pending and time.monotonic()<deadline:
                if self._oracle is None:self._oracle=SatAssist(self.problem,self.initial,self.target,factory=self._factory)
                if not self._oracle.prepare(check,seconds=max(0,deadline-time.monotonic())):
                    self.status=self._oracle.status;self.reason=self._oracle.reason
                    if self.proof_transport and self.status=='unavailable' and time.monotonic()<deadline:
                        record=next(iter(self._pending));certificate=self._proofs().prove(record,check,seconds=deadline-time.monotonic(),conflicts=conflicts)
                        if certificate is not None:
                            added+=self._admit_verified(record);self._pending.pop(record,None)
                            self._certificates[record]=certificate;self.stats['mask_proofs_generated']+=1
                    return added
                self.status='ready'
                for _ in range(attempts):
                    check();remaining=deadline-time.monotonic()
                    if remaining<=0 or not self._pending:break
                    record,tries=next(iter(self._pending.items()))
                    core=self._oracle.query_masks(record,check,seconds=remaining,conflicts=conflicts)
                    check();self._check();self._pending.pop(record)
                    if core is not None:
                        core=self._normalize(core)
                        if self._oracle.last_outcome!='unsat' or not _subsumes(core,record):raise RuntimeError('Verifier returned an unrelated mask core.')
                        if self._admit_verified(core):
                            added+=1
                            if self.proof_transport:self._proof_queue[core]=0
                    elif self._oracle.last_outcome=='sat':
                        self.stats['mask_conflicts_rejected']+=1;self._discarded[record]=None
                    else:
                        self.stats['mask_conflicts_unknown']+=1;self._pending[record]=tries+1
                    while len(self._discarded)>MAX_RECORDS:self._discarded.popitem(last=False)
            if self.proof_transport and self._proof_queue and time.monotonic()<deadline:
                record,tries=next(iter(self._proof_queue.items()));context=self._proofs()
                certificate=context.prove(record,check,seconds=max(0,deadline-time.monotonic()),conflicts=conflicts)
                if certificate is not None:
                    self._certificates[record]=certificate;self._proof_queue.pop(record,None)
                    self.stats['mask_proofs_generated']+=1
                elif context.base is not None:
                    self.stats['mask_proofs_unknown']+=1
                    if tries<2:self._proof_queue[record]=tries+1
                    else:self._proof_queue.pop(record,None)
            return added
        finally:
            if self._oracle is not None:self.stats['mask_conflicts_calls']=self._oracle.stats['sat_assist_mask_calls']
            self._refresh()

    def records(self):
        self._check();return tuple(self._records)

    def prune(self,domains,check=lambda:None,reason=None):
        self._check()
        if len(domains)!=len(self.initial) or any(type(mask)is not int or mask<0 or mask&~old for mask,old in zip(domains,self.initial)):
            raise ValueError('Mask conflict query escapes its bound root.')
        changed=False
        for clause in self._records:
            check();unknown=[]
            for cell,mask in clause:
                if not domains[cell]&mask:break
                if domains[cell]&~mask:unknown.append((cell,mask))
                if len(unknown)>1:break
            else:
                if not unknown:
                    domains[0]=0;self.stats['mask_conflicts_hits']+=1
                    if reason is not None:reason(0,0,tuple(c for c,_ in clause))
                    return True
                cell,mask=unknown[0];old=domains[cell];domains[cell]&=~mask;changed=True
                self.stats['mask_conflicts_prunes']+=(old&~domains[cell]).bit_count()
                if reason is not None:reason(cell,domains[cell],tuple(c for c,_ in clause if c!=cell))
        return changed

    def snapshot(self,*,include_pending=True):
        self._check()
        # Pending claims are deliberately retained for local resume. Receivers
        # independently verify BOTH categories; the label is never authority.
        records=list(self._records)
        if include_pending:records.extend(record for record in self._pending if record not in records)
        records=records[:MAX_RECORDS]
        payload={'mask_conflicts_version':2 if self.proof_transport else VERSION,'fingerprint':self.binding[0],'target':self.target,
                 'original':[hex(mask) for mask in self.original],'initial':[hex(mask) for mask in self.initial],
                 'records':[[[cell,hex(mask)] for cell,mask in record] for record in records]}
        # Respect the durable reader's byte limit even with indented Windows
        # CRLF JSON. A full 128-by-64 mask set can otherwise exceed 1 MiB and
        # reject its own checkpoint. Omitting claims only loses optional reuse.
        if self.proof_transport:
            payload['certificates']=[self._certificates[record] for record in records if record in self._certificates][:8]
        payload['digest']='0'*64
        while True:
            size=len(json.dumps(payload,indent=2).replace('\n','\r\n').encode('utf-8'))
            if size<=MAX_BYTES:break
            if payload.get('certificates'):
                payload['certificates'].pop();continue
            count=len(payload['records'])
            if not count:raise ValueError('Mask-conflict root exceeds the snapshot byte bound.')
            drop=max(1,(count*(size-MAX_BYTES)+size-1)//size)
            del payload['records'][max(0,count-drop):]
        payload['digest']=_digest(payload)
        return payload

    def import_snapshot(self,payload):
        self._check()
        keys={'mask_conflicts_version','fingerprint','target','original','initial','records','digest'}
        if not isinstance(payload,dict):raise ValueError('Malformed mask-conflict payload.')
        if payload.get('mask_conflicts_version')==2:keys.add('certificates')
        if set(payload)!=keys or type(payload.get('mask_conflicts_version'))is not int or payload['mask_conflicts_version'] not in (1,2):
            raise ValueError('Malformed mask-conflict payload.')
        if len(json.dumps(payload))>MAX_BYTES:raise ValueError('Mask-conflict payload too large.')
        expected=self.snapshot()
        for key in ('fingerprint','target','original','initial'):
            if type(payload[key])is not type(expected[key]) or payload[key]!=expected[key]:
                raise ValueError('Mask-conflict input, target or effective root differs.')
        if payload['digest']!=_digest(payload):raise ValueError('Mask-conflict digest differs.')
        records=payload['records']
        if not isinstance(records,list) or len(records)>MAX_RECORDS:raise ValueError('Too many mask-conflict records.')
        normalized=[]
        for record in records:
            if not isinstance(record,list) or len(record)>MAX_LITERALS:raise ValueError('Malformed serialized mask conflict.')
            converted=[]
            for item in record:
                if (not isinstance(item,list) or len(item)!=2 or not isinstance(item[1],str)
                        or len(item[1])>2+(3*self.problem.n+3)//4 or not item[1].startswith('0x')):
                    raise ValueError('Malformed serialized mask literal.')
                try:mask=int(item[1],16)
                except ValueError as exc:raise ValueError('Malformed serialized mask literal.') from exc
                converted.append((item[0],mask))
            normalized.append(self._normalize(converted))
        certificates=payload.get('certificates',[])
        if not isinstance(certificates,list) or len(certificates)>8:raise ValueError('Too many mask proofs.')
        admitted=sum(self.offer(record) for record in normalized)
        if self.proof_transport:
            for certificate in certificates:self.offer_certificate(certificate)
        self.stats['mask_conflicts_loaded']+=admitted;return admitted

    def save(self,path):
        from io_utils import atomic_json
        atomic_json(Path(path),self.snapshot())

    def load(self,path):
        with Path(path).open('rb') as stream:raw=stream.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES:raise ValueError('Mask-conflict file too large.')
        return self.import_snapshot(json.loads(raw))

    def close(self):
        if self._oracle is not None:self._oracle.close()
        if self._proof_context is not None:self._proof_context.close()
        self._oracle=None;self._proof_context=None;self._closed=True;self.status='closed'

    def __del__(self):
        try:self.close()
        except Exception:pass
