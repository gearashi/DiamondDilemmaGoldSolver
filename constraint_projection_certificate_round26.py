"""Independently checked completeness for projected placement relations.

A table covers the projection of the necessary edge CNF plus explicitly
revalidated proper-loop clauses. It remains an overapproximation of Gold.
Raw edge witness boards justify observed rows; rows killed by later Gold cuts
may remain harmless extra allowed rows. A RUP refutation after blocking all
rows proves none of the still-possible rows are missing. Partial native
model enumeration is never imported as a table.
"""
from pathlib import Path
import hashlib,json,time,math
from constraint_proof import MaskProofContext,BoundedDPLL,ProofLimit
from constraint_proof_indexed_round26 import IndexedRUPChecker
from constraint_knowledge import revalidate_witness
from constraint_reasoning_store_round26 import LocalReasoningStore,encoded

FORMAT='diamond-projection-certificate-v1'
MAX_BYTES=8*1024*1024
STAT_KEYS=('relation_certificate_attempts','relation_certificate_unknown','relation_certificate_checks',
 'relation_certificate_rows','relation_certificate_dpll_decisions','relation_certificate_dpll_visits',
 'relation_certificate_checker_visits','relation_certificate_index_literals','relation_certificate_imports',
 'relation_certificate_saved_queries','relation_observed_imports','relation_observed_rows')
SOURCES=('constraint_projection_certificate_round26.py','constraint_higher_relations.py','constraint_sat_assist.py',
 'constraint_proof.py','constraint_proof_indexed_round26.py','constraint_proof_cursor.py','constraint_knowledge.py','search_problem.py','validator.py')

def _digest(value):return hashlib.sha256(encoded(value)).hexdigest()

class ProjectionCertificates:
    def __init__(self,problem,initial,target='gold'):
        self.problem=problem;self.initial=tuple(initial);self.target=target
        self.context=MaskProofContext(problem,initial,target,indexed=True)
        self.binding=self._binding();self.stats=dict.fromkeys(STAT_KEYS,0);self.store=None;self.catalog=[]
    def _binding(self):
        p=self.problem
        return _digest({'fingerprint':p.fingerprint,'domains':p.domains,'tiles':p.tiles,'edges':p.edges,
            'masks':p.masks.tolist(),'neighbors':p.neighbors.tolist(),'sides':p.sides.tolist(),'initial':self.initial,'target':self.target})
    def _check(self):
        if self._binding()!=self.binding:raise ValueError('Projection certificate input changed.')
    def configure_cache_dir(self,directory):
        self.store=None if directory is None else LocalReasoningStore(directory,{'projection':self.binding},max_bytes=32*1024*1024,max_entries=65)
    @staticmethod
    def sources():return [Path(__file__).with_name(name) for name in SOURCES]
    def _structure(self,record,check):
        self._check();check()
        if type(record)is not dict or set(record)!={'format','binding','scope','cells','rows','witnesses','gold','proof','digest'} or record['format']!=FORMAT or record['binding']!=self.binding or record['digest']!=_digest({k:v for k,v in record.items() if k!='digest'}) or len(encoded(record))>MAX_BYTES:raise ValueError('Projection certificate shape/binding differs.')
        scope,cells,rows,witnesses=record['scope'],record['cells'],record['rows'],record['witnesses'];n=self.problem.n
        if not isinstance(scope,(tuple,list)) or len(scope)!=n or any(type(m)is not int or m<0 or m&~r for m,r in zip(scope,self.initial)):raise ValueError('Projection scope escapes root.')
        if not isinstance(cells,(tuple,list)) or not 1<=len(cells)<=8 or any(type(c)is not int or not 0<=c<n for c in cells) or len(set(cells))!=len(cells):raise ValueError('Invalid projection cells.')
        if not isinstance(rows,(tuple,list)) or not isinstance(witnesses,(tuple,list)) or len(rows)!=len(witnesses) or len(rows)>4096:raise ValueError('Invalid table rows/witnesses.')
        normalized=[]
        for row,witness in zip(rows,witnesses):
            check()
            if not isinstance(row,(tuple,list)) or len(row)!=len(cells) or any(type(c)is not int or c<0 or not scope[cell]>>c&1 for cell,c in zip(cells,row)):raise ValueError('Malformed projected row.')
            if not isinstance(witness,(tuple,list)) or len(witness)!=n or any(type(c)is not int or c<0 or not mask>>c&1 for mask,c in zip(scope,witness)) or tuple(witness[c] for c in cells)!=tuple(row):raise ValueError('Projection witness differs.')
            report=self.problem.validate(witness)
            if not report['unique_tiles'] or report['matched_edges']!=report['total_edges']:raise ValueError('Projection witness is not an edge completion.')
            normalized.append(tuple(row))
        if tuple(sorted(set(normalized)))!=tuple(normalized):raise ValueError('Noncanonical projection rows.')
        if not isinstance(record['gold'],(tuple,list)) or len(record['gold'])>2048 or self.target=='edges' and record['gold']:raise ValueError('Invalid Gold theory.')
        gold=[]
        for raw in record['gold']:
            check();gold.append(revalidate_witness(self.problem,raw,target=self.target))
        if tuple(sorted(set(gold)))!=tuple(tuple(tuple(p) for p in row) for row in record['gold']):raise ValueError('Noncanonical Gold clauses.')
        self.stats['relation_certificate_rows']+=len(rows)
        return tuple(scope),tuple(cells),tuple(normalized),tuple(tuple(w) for w in witnesses),tuple(gold)
    def _formula(self,scope,cells,rows,gold,check):
        if not self.context.prepare(check,seconds=2):raise ProofLimit('Projection CNF construction incomplete.')
        variables=self.context.variables;clauses=list(self.context.base)
        for cell,mask in enumerate(scope):clauses.append(tuple(lit for code,lit in variables[cell].items() if mask>>code&1))
        for cut in gold:clauses.append(tuple(-variables[c][v] for c,v in cut if v in variables[c])) if all(v in variables[c] for c,v in cut) else None
        clauses.extend(tuple(-variables[c][v] for c,v in zip(cells,row)) for row in rows)
        return clauses
    def produce(self,scope,cells,rows,witnesses,gold=(),check=lambda:None,*,seconds=.05,max_decisions=512,max_visits=200000):
        if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or not 0<=seconds<=60:raise ValueError('Invalid projection proof budget.')
        self.stats['relation_certificate_attempts']+=1;deadline=time.monotonic()+seconds
        record={'format':FORMAT,'binding':self.binding,'scope':list(scope),'cells':list(cells),'rows':[list(r) for r in rows],
                'witnesses':[list(w) for w in witnesses],'gold':[list(map(list,c)) for c in sorted(gold)],'proof':[]}
        record['digest']=_digest(record);scope,cells,rows,_,gold=self._structure(record,check)
        callback=[None]
        def poll():
            try:check()
            except BaseException as exc:callback[0]=exc;raise
            if time.monotonic()>=deadline:raise ProofLimit('Projection proof slice ended.')
        producer=None
        try:
            clauses=self._formula(scope,cells,rows,gold,poll);producer=BoundedDPLL(clauses,max_decisions=max_decisions,max_visits=max_visits)
            if producer.solve(poll) is not False:
                self.stats['relation_certificate_unknown']+=1;return None
            record['proof']=producer.proof;record['digest']=_digest({k:v for k,v in record.items() if k!='digest'})
            self.verify(record,poll);return record
        except ProofLimit:
            if callback[0] is not None:raise
            self.stats['relation_certificate_unknown']+=1;return None
        finally:
            if producer is not None:
                self.stats['relation_certificate_dpll_decisions']+=producer.stats['proof_dpll_decisions'];self.stats['relation_certificate_dpll_visits']+=producer.stats['proof_dpll_visits']
    def verify(self,record,check=lambda:None):
        scope,cells,rows,witnesses,gold=self._structure(record,check);clauses=self._formula(scope,cells,rows,gold,check)
        checker=IndexedRUPChecker(clauses,record['proof'],context=(self.binding,scope,cells,rows))
        self.stats['relation_certificate_checks']+=1
        try:
            while not checker.finished:checker.advance(check,ticks=4096)
            if checker.result is not True:raise ValueError('Projection completeness proof failed.')
        finally:
            self.stats['relation_certificate_checker_visits']+=checker.stats['proof_clause_visits'];self.stats['relation_certificate_index_literals']+=checker.stats['proof_index_build_literals']
        self._check();return (scope,cells),rows,witnesses
    def save_observed(self,scope,cells,witnesses,check=lambda:None):
        if self.store is None:return False
        rows=tuple(sorted(witnesses));record={'format':FORMAT,'binding':self.binding,'scope':list(scope),'cells':list(cells),
            'rows':[list(row) for row in rows],'witnesses':[list(witnesses[row]) for row in rows],'gold':[],'proof':[]}
        record['digest']=_digest(record);self._structure(record,check)
        return self.store.save('observed',(scope,cells),record,sources=self.sources(),check=check)
    def load_observed(self,scope,cells,check=lambda:None):
        if self.store is None:return {}
        record=self.store.load('observed',(scope,cells),sources=self.sources(),check=check)
        if record is None:return {}
        callback=[None]
        def poll():
            try:check()
            except BaseException as exc:callback[0]=exc;raise
        try:
            got_scope,got_cells,rows,witnesses,_=self._structure(record,poll)
            if got_scope!=tuple(scope) or got_cells!=tuple(cells):return {}
        except (ValueError,TypeError,KeyError,IndexError):
            if callback[0] is not None:raise
            self.store.stats['reasoning_cache_invalid']+=1;return {}
        self.stats['relation_observed_imports']+=1;self.stats['relation_observed_rows']+=len(rows)
        return dict(zip(rows,witnesses))
    def save(self,record,check=lambda:None):
        # Only complete checked evidence is written. Disk integrity is never a
        # substitute for checking its proof again on process-cold import.
        key,_,_=self.verify(record,check)
        if self.store is None:return False
        if not self.store.save('table',key,record,sources=self.sources(),check=check):return False
        if key not in self.catalog:self.catalog.append(key)
        self.catalog=self.catalog[-32:]
        return self.store.save('catalog',(),self.catalog,sources=self.sources(),check=check)
    def load(self,check=lambda:None):
        if self.store is None:return []
        catalog=self.store.load('catalog',(),sources=self.sources(),check=check)
        if catalog is None:return []
        if type(catalog)is not list or len(catalog)>32:return []
        result=[];checked=[]
        for raw in catalog:
            check()
            if not isinstance(raw,(list,tuple)) or len(raw)!=2:continue
            if any(not isinstance(v,(list,tuple)) or any(type(x)is not int for x in v) for v in raw):continue
            key=tuple(tuple(v) for v in raw)
            record=self.store.load('table',key,sources=self.sources(),check=check)
            if record is None:continue
            callback=[None]
            def poll():
                try:check()
                except BaseException as exc:callback[0]=exc;raise
            try:verified=self.verify(record,poll)
            except (ValueError,TypeError,KeyError,IndexError,ProofLimit):
                if callback[0] is not None:raise
                self.store.stats['reasoning_cache_invalid']+=1;continue
            if verified[0]!=key:continue
            checked.append(key);result.append(verified);self.stats['relation_certificate_imports']+=1
            self.stats['relation_certificate_saved_queries']+=len(verified[1])+1
        self.catalog=checked;return result
    def close(self):self.context.close()
