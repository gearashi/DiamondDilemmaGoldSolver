"""Bounded verified RUP clause-subset persistence for tested native SAT builds.

Only edge-CNF consequences are proof authority. Native Gold clauses, temporary
selectors, assumptions, file hashes and native success claims are never axioms.
This preserves selected clauses, not the native decision stack or heuristics.
"""
from __future__ import annotations
from collections import deque
import hashlib,importlib.metadata,json,os,platform,sys
from constraint_proof_cursor import ResumableRUPChecker,_ValidatedRUPBase

MAX_LOG_BYTES=4*1024*1024
READ_BYTES=16384
MAX_PENDING_BYTES=65536
MAX_LINE_BYTES=16384
MAX_PROOF_BYTES=262144
MAX_SNAPSHOT_BYTES=524288
MAX_STEPS=1024
MAX_PROJECTED=256
MAX_VISITS=2000000
STAT_KEYS=('session_persist_read_bytes','session_persist_reads','session_persist_log_peak_bytes',
 'session_persist_verification_visits','session_persist_verification_propagations',
 'session_persist_verified_clauses','session_persist_rejected_clauses','session_persist_unknown_clauses',
 'session_persist_imported_clauses','session_persist_model_rebuilds','session_persist_limit_hits',
 'session_persist_retained_bytes','session_persist_retained_clauses','session_persist_projected_clauses',
 'session_persist_reusable_clauses','session_persist_auxiliary_clauses')
GAUGES={'session_persist_log_peak_bytes','session_persist_retained_bytes','session_persist_retained_clauses','session_persist_projected_clauses','session_persist_reusable_clauses','session_persist_auxiliary_clauses'}


def capability():
    """No native module import or logger creation occurs during this check."""
    if sys.platform not in ('linux','darwin'):
        return False,'Native SAT proof persistence is disabled on Windows and untested operating systems.'
    machine=platform.machine().lower()
    if ((sys.platform=='linux' and machine not in ('x86_64','amd64')) or
            (sys.platform=='darwin' and machine not in ('x86_64','arm64','aarch64'))):
        return False,'Native SAT proof persistence requires a tested Linux x86-64 or macOS x86-64/ARM64 build.'
    if platform.python_implementation()!='CPython' or sys.version_info[:2]!=(3,12):
        return False,'Native SAT proof persistence is validated only with CPython 3.12.'
    try:version=importlib.metadata.version('python-sat')
    except importlib.metadata.PackageNotFoundError:return False,'Native SAT proof persistence requires python-sat 1.9.dev15.'
    if version!='1.9.dev15':return False,'Native SAT proof persistence requires the tested python-sat 1.9.dev15 package.'
    return True,''


def require_capability():
    available,reason=capability()
    if not available:raise ValueError(reason)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class VerifiedNativeClauses:
    """Pure checker; testable on Windows without loading a proof-enabled solver."""
    def __init__(self,clauses,variables,*,fingerprint,target,initial,max_visits=MAX_VISITS):
        self.base=_ValidatedRUPBase(tuple(tuple(row) for row in clauses))
        self.variables=tuple(tuple(sorted(row.items())) if isinstance(row,dict) else tuple(row) for row in variables)
        self.projection=frozenset(lit for row in self.variables for _,lit in row)
        self.base_vars=frozenset(abs(lit) for row in self.base.clauses for lit in row)
        if (not self.projection<=self.base_vars or any(type(lit)is not int or lit<=0 for lit in self.projection)
                or len(self.projection)!=sum(len(row) for row in self.variables)):
            raise ValueError('Native proof variable mapping is invalid.')
        self.binding={'fingerprint':fingerprint,'target':target,'initial':[hex(mask) for mask in initial],
                      'base_sha256':_digest(self.base.clauses),'variables_sha256':_digest(self.variables)}
        self.max_visits=max_visits;self.stats=dict.fromkeys(STAT_KEYS,0)
        self.proof=[];self.projected=[];self.reusable=[];self.known=set(self.base.clauses)
        self.pending=deque();self.pending_bytes=0;self.buffer=b'';self.offset=0;self.discard_line=False
        self.cursor=None;self.candidate=None;self.importing=False;self.disable_reason=None;self._claims_after_import=[]
        self._proof_bytes=0

    def disable(self,reason):
        if self.disable_reason is None:self.disable_reason=reason;self.stats['session_persist_limit_hits']+=1

    def queue_import(self,payload):
        if (not isinstance(payload,dict) or not {'version','binding','proof'}<=set(payload)<= {'version','binding','proof','claims'}
                or type(payload['version'])is not int or payload['version']!=1 or payload['binding']!=self.binding):
            raise ValueError('Native clause checkpoint root, CNF or variable mapping differs.')
        try:encoded=json.dumps(payload,separators=(',',':'),allow_nan=False)
        except (ValueError,TypeError):raise ValueError('Malformed native clause checkpoint.') from None
        if len(encoded.encode())>MAX_SNAPSHOT_BYTES:raise ValueError('Native clause checkpoint exceeds its byte bound.')
        rows=payload['proof']
        if not isinstance(rows,list) or len(rows)>MAX_STEPS:raise ValueError('Native clause checkpoint exceeds its step bound.')
        validated=[];size=0
        for row in rows:
            clause=self._clause(row)
            if not clause:raise ValueError('Empty clauses are not a persisted learning subset.')
            size+=len(json.dumps(clause,separators=(',',':')))+1
            validated.append(clause)
        if size>MAX_PROOF_BYTES:raise ValueError('Native clause derivation exceeds its byte bound.')
        if self.proof or self.pending or self.cursor is not None:raise ValueError('Import requires a fresh proof checker.')
        claims=payload.get('claims',[])
        if not isinstance(claims,list) or len(claims)>MAX_STEPS:raise ValueError('Too many unverified native claims.')
        claims=[self._clause(row) for row in claims]
        if any(not row for row in claims):raise ValueError('Malformed unverified native claim.')
        claim_bytes=sum(len(json.dumps(row,separators=(',',':')))+1 for row in claims)
        if claim_bytes>MAX_PENDING_BYTES:raise ValueError('Unverified native claims exceed their byte bound.')
        self.pending.extend(validated);self.pending_bytes=size;self.importing=bool(validated)
        if self.importing:self._claims_after_import=claims
        else:self.pending.extend(claims);self.pending_bytes+=claim_bytes

    def _clause(self,row):
        if (not isinstance(row,(list,tuple)) or len(row)>4096
                or any(type(lit)is not int or lit==0 or abs(lit) not in self.base_vars for lit in row)):
            raise ValueError('Malformed or unknown native proof literal.')
        return tuple(dict.fromkeys(row))

    def monitor_log(self,solver):
        """Poll the native temporary file; caller interrupts then retires logger."""
        stream=getattr(solver,'prfile',None)
        if stream is None:raise RuntimeError('The selected native backend exposes no proof file.')
        size=os.fstat(stream.fileno()).st_size
        self.stats['session_persist_log_peak_bytes']=max(self.stats['session_persist_log_peak_bytes'],size)
        if size>=MAX_LOG_BYTES:self.disable('native proof file reached its monitored byte threshold')
        return self.disable_reason is not None

    def harvest(self,solver,check=lambda:None):
        check();self.monitor_log(solver)
        # pread does not move the file position shared with the native writer.
        data=os.pread(solver.prfile.fileno(),READ_BYTES,self.offset)
        self.offset+=len(data);self.stats['session_persist_reads']+=1;self.stats['session_persist_read_bytes']+=len(data)
        self.buffer+=data
        while b'\n' in self.buffer:
            check();line,self.buffer=self.buffer.split(b'\n',1)
            if self.discard_line:self.discard_line=False;continue
            if len(line)>MAX_LINE_BYTES:self.stats['session_persist_rejected_clauses']+=1;continue
            tokens=line.split()
            if not tokens or tokens[0]==b'd':continue
            try:
                row=[int(word) for word in tokens]
                if not row or row[-1]!=0 or 0 in row[:-1]:raise ValueError()
                clause=self._clause(row[:-1])
            except (ValueError,TypeError):self.stats['session_persist_rejected_clauses']+=1;continue
            if not clause or clause in self.known or clause==self.candidate or clause in self.pending:continue
            cost=len(json.dumps(clause,separators=(',',':')))+1
            candidate_bytes=0 if self.candidate is None else len(json.dumps(self.candidate,separators=(',',':')))+1
            if self.pending_bytes+candidate_bytes+cost>MAX_PENDING_BYTES or len(self.pending)+int(self.candidate is not None)>=MAX_STEPS:
                self.disable('unverified native proof queue reached its byte limit');continue
            self.pending.append(clause);self.pending_bytes+=cost
        if len(self.buffer)>MAX_LINE_BYTES:
            self.buffer=b'';self.discard_line=True;self.stats['session_persist_rejected_clauses']+=1

    def advance(self,check=lambda:None,*,ticks=4096):
        if type(ticks)is not int or ticks<0:raise ValueError('Invalid native proof verification budget.')
        while ticks and (self.cursor is not None or self.pending or self._claims_after_import):
            check()
            if self.importing and self.cursor is None and not self.pending:
                self.importing=False;self.pending.extend(self._claims_after_import)
                self.pending_bytes+=sum(len(json.dumps(row,separators=(',',':')))+1 for row in self._claims_after_import)
                self._claims_after_import=[]
            if self.cursor is None:
                self.candidate=self.pending.popleft()
                self.pending_bytes-=len(json.dumps(self.candidate,separators=(',',':')))+1
                if self.candidate in self.known:continue
                # UP contradiction under negated candidate is exactly RUP.
                extras=tuple(self.proof)+tuple((-lit,) for lit in self.candidate)
                self.cursor=ResumableRUPChecker(self.base,['0'],extra_clauses=extras,max_visits=self.max_visits)
            before=dict(self.cursor.stats);portion=min(ticks,256)
            try:answer=self.cursor.advance(check,ticks=portion)
            finally:
                self.stats['session_persist_verification_visits']+=self.cursor.stats['proof_clause_visits']-before['proof_clause_visits']
                self.stats['session_persist_verification_propagations']+=self.cursor.stats['proof_propagations']-before['proof_propagations']
            ticks-=portion
            if not self.cursor.finished:continue
            candidate=self.candidate;limited=self.cursor.limited
            self.cursor=None;self.candidate=None
            if answer is True:self._retain(candidate)
            elif self.importing:
                raise ValueError('Native clause derivation is unverified or exceeds its independent verification bound.')
            else:self.stats['session_persist_unknown_clauses' if limited else 'session_persist_rejected_clauses']+=1
        if self.importing and not self.pending and self.cursor is None and not self._claims_after_import:self.importing=False
        return self.cursor is None and not self.pending and not self._claims_after_import

    def _retain(self,clause):
        cost=len(json.dumps(clause,separators=(',',':')))+1
        if len(self.proof)>=MAX_STEPS or self._proof_bytes+cost>MAX_PROOF_BYTES:
            self.disable('verified derivation reached its retention bound')
            if self.importing:raise ValueError('Imported native derivation exceeds retention limits.')
            return
        self.proof.append(clause);self.known.add(clause);self._proof_bytes+=cost
        self.stats['session_persist_verified_clauses']+=1
        if len(self.projected)<MAX_PROJECTED and all(abs(lit) in self.projection for lit in clause):self.projected.append(clause)
        # Original auxiliary IDs are safe only with this exact CNF/map binding.
        # Solver-created guards remain excluded by _clause before verification.
        if len(self.reusable)<MAX_PROJECTED:self.reusable.append(clause)
        self.stats.update(session_persist_retained_bytes=self._proof_bytes,session_persist_retained_clauses=len(self.proof),session_persist_projected_clauses=len(self.projected),
                          session_persist_reusable_clauses=len(self.reusable),
                          session_persist_auxiliary_clauses=sum(any(abs(lit) not in self.projection for lit in row) for row in self.reusable))
        if len(self.proof)>=MAX_STEPS:self.disable('verified derivation reached its retention bound')

    def apply(self,solver):
        if self.importing:raise ValueError('Cannot import a partially verified native proof.')
        for clause in self.reusable:solver.add_clause(clause)
        self.stats['session_persist_imported_clauses']+=len(self.reusable)

    def snapshot(self):
        if self.importing:raise ValueError('Unverified imports must retain their original checkpoint bytes.')
        claims=([self.candidate] if self.cursor is not None else [])+list(self.pending)
        payload={'version':1,'binding':dict(self.binding),'proof':[list(row) for row in self.proof]}
        if claims:payload['claims']=[list(row) for row in claims]
        if len(json.dumps(payload,separators=(',',':')).encode())>MAX_SNAPSHOT_BYTES:raise ValueError('Native proof snapshot exceeded its bound.')
        return payload

    def create_solver(self,clauses,*,logging=True):
        if logging:require_capability()  # Before importing/constructing a native logger.
        from pysat.solvers import Glucose4
        solver=Glucose4(bootstrap_with=clauses,with_proof=bool(logging))
        try:
            if logging and getattr(solver,'prfile',None) is None:raise RuntimeError('Glucose4 proof logging is unavailable.')
            self.apply(solver)
            return solver
        except BaseException:solver.delete();raise
