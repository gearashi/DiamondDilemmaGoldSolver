"""Independent RUP checking for scoped finite-domain mask exclusions.

The checker reconstructs the seam/inventory CNF from the actual puzzle. It does
not trust clauses supplied as axioms by a worker. The first implementation uses
this necessary edge theory: its UNSAT proofs are valid for both edge and Gold
objectives. Gold-only claims without such a proof retain ordinary revalidation.
"""
from __future__ import annotations
import hashlib,json,time
from collections import OrderedDict
from constraint_sat_assist import SatAssist


class ProofLimit(Exception):pass

MAX_PROOF_BYTES=524288
MAX_PROOF_STEPS=4096


class _CaptureSink:
    """No native dependency is needed to reconstruct and check a CNF."""
    def add_clause(self,clause):pass
    def delete(self):pass


def _hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def _clause(row):
    if not isinstance(row,(list,tuple)) or any(type(x)is not int or x==0 for x in row):raise ValueError('Malformed proof clause.')
    return tuple(dict.fromkeys(row))


class RUPChecker:
    def __init__(self,clauses,*,max_steps=MAX_PROOF_STEPS,max_visits=2000000):
        self.clauses=tuple(_clause(row) for row in clauses);self.max_steps=max_steps;self.max_visits=max_visits
        self.stats={'proof_steps':0,'proof_clause_visits':0,'proof_propagations':0}

    def _rup(self,clauses,candidate,check):
        assigned={}
        for lit in candidate:
            var=abs(lit);value=lit<0
            if var in assigned and assigned[var]!=value:return True
            assigned[var]=value
        changed=True
        while changed:
            changed=False
            for row in clauses:
                self.stats['proof_clause_visits']+=1
                if self.stats['proof_clause_visits']>self.max_visits:raise ProofLimit('Independent RUP work cap reached.')
                if self.stats['proof_clause_visits']%128==0:check()
                missing=[];satisfied=False
                for lit in row:
                    value=assigned.get(abs(lit))
                    if value is None:missing.append(lit)
                    elif value==(lit>0):satisfied=True;break
                if satisfied:continue
                if not missing:return True
                if len(missing)==1:
                    lit=missing[0];assigned[abs(lit)]=lit>0;changed=True;self.stats['proof_propagations']+=1
        return False

    def verify(self,proof,check=lambda:None):
        if not isinstance(proof,(list,tuple)) or len(proof)>self.max_steps:raise ValueError('Proof step count exceeds bound.')
        if sum(len(line) for line in proof if isinstance(line,str))>MAX_PROOF_BYTES:raise ValueError('Proof byte bound exceeded.')
        active=list(self.clauses);empty=False
        for raw in proof:
            check()
            if not isinstance(raw,str) or len(raw)>100000:raise ValueError('Malformed proof line.')
            tokens=raw.split()
            if not tokens:continue
            deletion=tokens[0]=='d'
            if deletion:tokens=tokens[1:]
            try:row=[int(x) for x in tokens]
            except ValueError:raise ValueError('Proof contains a noninteger literal.') from None
            if not row or row[-1]!=0 or 0 in row[:-1]:raise ValueError('Proof clause lacks its terminator.')
            clause=_clause(row[:-1])
            # Keeping proved clauses is sound for RUP and simplifies deletion.
            if deletion:continue
            if not self._rup(active,clause,check):return False
            active.append(clause);self.stats['proof_steps']+=1
            if not clause:empty=True
        return empty


class BoundedDPLL:
    """Emit RUP branch clauses after complete unit-conflict derivations.

    A child failure learns the negation of its decisions. Two opposite failed
    children make their parent decision clause RUP. A partial traversal is not
    a certificate. This avoids unsafe native proof-log destruction observed in
    the local Windows PySAT build; it is intentionally a bounded producer.
    """
    def __init__(self,clauses,*,max_decisions=512,max_visits=500000,max_depth=128):
        self.clauses=[_clause(row) for row in clauses]
        self.max_decisions=max_decisions;self.max_visits=max_visits;self.max_depth=max_depth
        self.proof=[];self.known=set(self.clauses);self.bytes=0
        self.stats={'proof_dpll_decisions':0,'proof_dpll_visits':0,'proof_emitted_clauses':0}

    def _units(self,decisions,check):
        assigned={abs(lit):lit>0 for lit in decisions};changed=True
        while changed:
            changed=False;choice=None
            for clause in self.clauses:
                self.stats['proof_dpll_visits']+=1
                if self.stats['proof_dpll_visits']>self.max_visits:raise ProofLimit('DPLL clause-work cap reached.')
                if self.stats['proof_dpll_visits']%64==0:check()
                unknown=[];satisfied=False
                for lit in clause:
                    value=assigned.get(abs(lit))
                    if value is None:unknown.append(lit)
                    elif value==(lit>0):satisfied=True;break
                if satisfied:continue
                if not unknown:return False,None
                if len(unknown)==1:
                    lit=unknown[0];assigned[abs(lit)]=lit>0;changed=True
                elif choice is None or len(unknown)<len(choice):choice=unknown
            if not changed:return True,None if choice is None else choice[0]
        raise AssertionError('Unreachable propagation state.')

    def _emit(self,decisions):
        clause=tuple(-lit for lit in decisions)
        if clause in self.known:return
        text=' '.join(map(str,(*clause,0)));self.bytes+=len(text)
        if len(self.proof)>=MAX_PROOF_STEPS or self.bytes>MAX_PROOF_BYTES:raise ProofLimit('DPLL certificate size cap reached.')
        self.known.add(clause);self.clauses.append(clause);self.proof.append(text)
        self.stats['proof_emitted_clauses']+=1

    def solve(self,check=lambda:None):
        def visit(decisions):
            check()
            if len(decisions)>self.max_depth:raise ProofLimit('DPLL decision-depth cap reached.')
            possible,choice=self._units(decisions,check)
            if not possible:self._emit(decisions);return False
            if choice is None:return True
            if self.stats['proof_dpll_decisions']>=self.max_decisions:raise ProofLimit('DPLL decision cap reached.')
            self.stats['proof_dpll_decisions']+=1
            if visit((*decisions,choice)):return True
            if visit((*decisions,-choice)):return True
            self._emit(decisions);return False
        return visit(())


class MaskProofContext:
    def __init__(self,problem,initial,target='gold'):
        if target not in ('gold','edges'):raise ValueError('Unknown proof objective.')
        self.problem,self.initial,self.target=problem,tuple(initial),target
        self.original=tuple(sum(1<<code for code in sorted(set(row))) for row in problem.domains)
        if len(self.initial)!=problem.n or any(type(mask)is not int or mask<0 or mask&~root for mask,root in zip(self.initial,self.original)):
            raise ValueError('Proof root escapes original domains.')
        self._binding=self._current_binding()
        self.encoder=SatAssist(problem,self.initial,'edges',capture_clauses=True,factory=_CaptureSink)
        self.base=None;self.variables=None;self._verification_cursors=OrderedDict();self.last_verification_status=None
        self._verification_base=None
        self.stats={'proof_native_calls':0,'proof_checks':0,'proof_imports':0,'proof_unknown':0,
                    'proof_dpll_decisions':0,'proof_dpll_visits':0,'proof_emitted_clauses':0,
                    'proof_cursor_visits':0,'proof_cursor_resumptions':0}

    def _current_binding(self):
        return (self.problem.fingerprint,
                tuple(tuple(sorted(set(row))) for row in self.problem.domains),
                tuple(tuple(map(int,row)) for row in self.problem.masks),tuple(self.problem.edges))

    def _check(self):
        if self._current_binding()!=self._binding:raise ValueError('Proof puzzle binding changed.')

    def prepare(self,check=lambda:None,seconds=.1):
        self._check();check()
        if self.base is not None:return True
        if not self.encoder.prepare(check,seconds=seconds):return False
        snapshot=self.encoder.proof_base();self.base=tuple(snapshot['clauses']);self.variables=tuple(dict(row) for row in snapshot['variables'])
        return True

    def normalize(self,record):
        if not isinstance(record,(list,tuple)) or len(record)>self.problem.n:raise ValueError('Malformed proof assumptions.')
        result=[];seen=set()
        for item in record:
            if not isinstance(item,(list,tuple)) or len(item)!=2:raise ValueError('Malformed proof literal.')
            cell,mask=item
            if type(cell)is not int or not 0<=cell<self.problem.n or cell in seen or type(mask)is not int or mask<=0 or mask&~self.initial[cell]:
                raise ValueError('Proof assumptions escape their root.')
            seen.add(cell)
            if mask!=self.initial[cell]:result.append((cell,mask))
        return tuple(sorted(result))

    def binding(self,record):
        self._check()
        return {'fingerprint':self.problem.fingerprint,'target':self.target,'initial':[hex(x) for x in self.initial],
                'record':[[c,hex(m)] for c,m in record],'base_sha256':_hash(self.base)}

    def clauses(self,record):
        return self.base+tuple(tuple(lit for code,lit in self.variables[cell].items() if mask>>code&1) for cell,mask in record)

    def prove(self,record,check=lambda:None,*,seconds=.1,conflicts=10000):
        deadline=time.monotonic()+seconds
        def bounded():
            check()
            if time.monotonic()>=deadline:raise ProofLimit('Proof time budget reached.')
        producer=None
        try:
            if not self.prepare(check,seconds=max(0,deadline-time.monotonic())):return None
            record=self.normalize(record);bounded()
            producer=BoundedDPLL(self.clauses(record),max_decisions=min(512,max(1,conflicts)))
            if producer.solve(bounded):return None
            proof=list(producer.proof)
            if not any(line.strip()=='0' for line in proof):proof.append('0')
            result={'proof_format':'diamond-rup-mask-v1',**self.binding(record),'proof':proof}
            if not self.verify(result,bounded):raise RuntimeError('DPLL proof failed independent RUP checking.')
            return result
        except ProofLimit:self.stats['proof_unknown']+=1;return None
        finally:
            if producer is not None:
                for key,value in producer.stats.items():self.stats[key]+=value

    def verify(self,certificate,check=lambda:None):
        self._check()
        if not isinstance(certificate,dict) or set(certificate)!={'proof_format','fingerprint','target','initial','record','base_sha256','proof'}:return False
        if certificate.get('proof_format')!='diamond-rup-mask-v1':return False
        proof=certificate.get('proof')
        if (not isinstance(proof,(list,tuple)) or len(proof)>MAX_PROOF_STEPS
                or any(not isinstance(line,str) or len(line)>16384 for line in proof)
                or sum(map(len,proof))>MAX_PROOF_BYTES):return False
        try:
            if len(json.dumps(certificate))>MAX_PROOF_BYTES:return False
            record=self.normalize([(c,int(m,16)) for c,m in certificate['record']])
        except (KeyError,TypeError,ValueError):return False
        if not self.prepare(check,seconds=1):return False
        if any(certificate.get(k)!=v for k,v in self.binding(record).items()):return False
        checker=RUPChecker(self.clauses(record));self.stats['proof_checks']+=1
        try:valid=checker.verify(proof,check)
        except ProofLimit:self.stats['proof_unknown']+=1;return False
        except ValueError:return False
        if valid:self.stats['proof_imports']+=1
        return valid

    def verify_incremental(self,certificate,check=lambda:None,*,ticks=256):
        """Resume only a locally held exact certificate; imported cursors do not exist."""
        from constraint_proof_cursor import ResumableRUPChecker,_ValidatedRUPBase
        self._check();self.last_verification_status='invalid'
        if not isinstance(certificate,dict) or set(certificate)!={'proof_format','fingerprint','target','initial','record','base_sha256','proof'}:return False
        if certificate.get('proof_format')!='diamond-rup-mask-v1':return False
        proof=certificate.get('proof')
        if (not isinstance(proof,(list,tuple)) or len(proof)>MAX_PROOF_STEPS
                or any(not isinstance(line,str) or len(line)>16384 for line in proof)
                or sum(map(len,proof))>MAX_PROOF_BYTES):return False
        try:
            if len(json.dumps(certificate))>MAX_PROOF_BYTES:return False
            record=self.normalize([(c,int(m,16)) for c,m in certificate['record']])
        except (KeyError,TypeError,ValueError):return False
        if not self.prepare(check,seconds=1):
            self.last_verification_status='pending';return None
        if any(certificate.get(k)!=v for k,v in self.binding(record).items()):return False
        # The exact tuple key prevents mutated/replaced proof bytes from borrowing
        # a checked prefix. All root/base bindings above are checked on each call.
        key=(record,tuple(proof));cursor=self._verification_cursors.get(key)
        if cursor is None:
            if self._verification_base is None:self._verification_base=_ValidatedRUPBase(self.base)
            assumptions=tuple(tuple(lit for code,lit in self.variables[cell].items() if mask>>code&1) for cell,mask in record)
            cursor=ResumableRUPChecker(self._verification_base,proof,extra_clauses=assumptions)
            self._verification_cursors[key]=cursor;self.stats['proof_checks']+=1
            while len(self._verification_cursors)>8:self._verification_cursors.popitem(last=False)
        self._verification_cursors.move_to_end(key)
        before=cursor.stats.copy();was_finished=cursor.finished;self.last_verification_status='pending'
        try:valid=cursor.advance(check,ticks=ticks)
        finally:
            self.stats['proof_cursor_visits']+=cursor.stats['proof_clause_visits']-before['proof_clause_visits']
            self.stats['proof_cursor_resumptions']+=cursor.stats['proof_resumptions']-before['proof_resumptions']
        if cursor.limited:
            self.last_verification_status='limited'
            if not was_finished:self.stats['proof_unknown']+=1
        elif cursor.finished:
            self.last_verification_status='valid' if valid else 'invalid'
            if valid and not was_finished:self.stats['proof_imports']+=1
        return valid

    def close(self):
        self._verification_cursors.clear();self._verification_base=None;self.encoder.close()
