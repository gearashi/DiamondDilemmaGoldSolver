"""Trusted in-memory resumable RUP verification, with no imported cursor state.

Each yielded work tick commits either one parsed proof line or one clause visit.
Both the proved prefix and an unfinished unit propagation remain in the generator.
Only the final checked empty-clause trace yields True; partial work yields None.
"""
from dataclasses import dataclass
from constraint_proof import _clause, MAX_PROOF_BYTES, MAX_PROOF_STEPS, ProofLimit


@dataclass(frozen=True)
class _ValidatedRUPBase:
    """Immutable shared CNF; construction always validates every input clause.

    This private wrapper is created locally, never deserialized from a proof.
    There is deliberately no ``trusted=True`` constructor bypass.
    """
    clauses:tuple

    def __post_init__(self):
        source=tuple(self.clauses)
        normalized=tuple(_clause(row) for row in source)
        # A prepared encoder already owns canonical immutable clause tuples.
        # Preserve those objects rather than retaining a clone per proof cursor.
        rows=source if all(type(row)is tuple and row==checked
                           for row,checked in zip(source,normalized)) else normalized
        object.__setattr__(self,'clauses',rows)


class ResumableRUPChecker:
    def __init__(self,clauses,proof,*,max_steps=MAX_PROOF_STEPS,max_visits=2000000,extra_clauses=()):
        if (not isinstance(proof,(list,tuple)) or len(proof)>max_steps
                or any(not isinstance(line,str) or len(line)>16384 for line in proof)
                or sum(map(len,proof))>MAX_PROOF_BYTES):raise ValueError('Malformed or oversized proof.')
        base=clauses if type(clauses)is _ValidatedRUPBase else _ValidatedRUPBase(clauses)
        self.base=base.clauses;self.extra_clauses=tuple(_clause(row) for row in extra_clauses)
        self.proof=tuple(proof)
        self.max_visits=max_visits;self.finished=False;self.result=None;self.limited=False
        self.stats={'proof_steps':0,'proof_clause_visits':0,'proof_propagations':0,'proof_resumptions':0}
        self._iterator=self._verify()

    def _rup(self,active,candidate):
        assigned={}
        for lit in candidate:
            var=abs(lit);value=lit<0
            if var in assigned and assigned[var]!=value:return True
            assigned[var]=value
        changed=True
        while changed:
            changed=False
            for clause in active:
                if self.stats['proof_clause_visits']>=self.max_visits:raise ProofLimit('Resumable RUP total work cap reached.')
                self.stats['proof_clause_visits']+=1
                missing=[];satisfied=False
                for lit in clause:
                    value=assigned.get(abs(lit))
                    if value is None:missing.append(lit)
                    elif value==(lit>0):satisfied=True;break
                conflict=False
                if not satisfied:
                    if not missing:conflict=True
                    elif len(missing)==1:
                        lit=missing[0];assigned[abs(lit)]=lit>0;changed=True
                        self.stats['proof_propagations']+=1
                yield
                if conflict:return True
        return False

    def _verify(self):
        active=list(self.base);active.extend(self.extra_clauses);empty=False
        for raw in self.proof:
            tokens=raw.split()
            if not tokens:
                yield
                continue
            deletion=tokens[0]=='d'
            if deletion:tokens=tokens[1:]
            try:row=[int(x) for x in tokens]
            except ValueError:raise ValueError('Proof contains a noninteger literal.') from None
            if not row or row[-1]!=0 or 0 in row[:-1]:raise ValueError('Proof clause lacks its terminator.')
            clause=_clause(row[:-1])
            if not deletion:
                if not (yield from self._rup(active,clause)):return False
                active.append(clause);self.stats['proof_steps']+=1
                if not clause:empty=True
            yield
        return empty

    def advance(self,check=lambda:None,*,ticks=256):
        if type(ticks)is not int or ticks<0:raise ValueError('Invalid RUP slice budget.')
        if self.finished:return self.result
        self.stats['proof_resumptions']+=1
        for _ in range(ticks):
            # Check before advancing: cancellation cannot discard an uncommitted
            # native stack operation or cause a clause to be visited twice.
            check()
            try:next(self._iterator)
            except StopIteration as done:
                self.finished=True;self.result=bool(done.value);return self.result
            except ProofLimit:
                self.finished=True;self.limited=True;return None
            except ValueError:
                self.finished=True;self.result=False;return False
        return None
