"""Reusable structural seam CNF and unbound cardinality counters in a local SAT checkpoint."""
from array import array
import hashlib
import json
from pathlib import Path
import sys
import zlib

from constraint_work import digest


class Capture:
    """Forward clauses to the live SAT solver while collecting a compact terminated literal stream."""
    def __init__(self, solver):
        self.solver=solver
        self.literals=array('i')
        self.count=0

    def add_clause(self,clause):
        self.solver.add_clause(clause)
        self.literals.extend(clause)
        self.literals.append(0)
        self.count+=1

    def append_formula(self,clauses):
        for clause in clauses:self.add_clause(clause)


def encoded_scope(problem,rows,split):
    import pysat
    sources=[hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
             for name in ('r27_reason_maxsat.py','r31_defect_bands.py','r34_seam_cnf.py','r34_band_cores.py')]
    return digest([1,pysat.__version__,sources,problem.fingerprint,rows,split])


def schema(db):
    db.execute("""CREATE TABLE IF NOT EXISTS r34_band_cnf(
        scope TEXT,capacity INTEGER,meta TEXT NOT NULL,clauses BLOB NOT NULL,checksum TEXT NOT NULL,
        PRIMARY KEY(scope,capacity))""")


def save(db,scope,capacity,capture,meta,guard):
    if sys.byteorder!='little':capture.literals.byteswap()
    raw=zlib.compress(capture.literals.tobytes(),1)
    meta=json.dumps(meta,separators=(',',':'))
    sha=hashlib.sha256(digest([scope,capacity,meta]).encode()+raw).hexdigest()
    with db:
        db.execute('BEGIN IMMEDIATE')
        guard()
        db.execute('INSERT OR IGNORE INTO r34_band_cnf VALUES(?,?,?,?,?)',(scope,capacity,meta,raw,sha))


def load(db,scope,capacity,solver,check):
    row=db.execute("""SELECT capacity,meta,clauses,checksum FROM r34_band_cnf
                      WHERE scope=? AND capacity>=? ORDER BY capacity LIMIT 1""",(scope,capacity)).fetchone()
    if row is None:return None
    cap,meta,raw,checksum=row
    if hashlib.sha256(digest([scope,cap,meta]).encode()+raw).hexdigest()!=checksum:
        raise ValueError('Stored seam CNF checksum differs.')
    literals=array('i')
    literals.frombytes(zlib.decompress(raw))
    if sys.byteorder!='little':literals.byteswap()
    clause=[]
    for index,lit in enumerate(literals):
        if index%32768==0:check()
        if lit:
            clause.append(lit)
        else:
            solver.add_clause(clause)
            clause=[]
    if clause:raise ValueError('Unterminated structural clause in checkpoint.')
    return json.loads(meta)


def cached_encoding(db,database,scope,capacity,sat,check,deadline,stats,part,compile_rows):
    from r34_query_ownership import query_lease
    from r27_reason_maxsat import Budget
    meta=load(db,scope,capacity,sat,check)
    if meta is None:
        # One builder owns the immutable clause stream while other requests keep
        # their own independent native solver and branch progress.
        with query_lease(db,database,'seam-encoding:'+scope,deadline) as guard:
            meta=load(db,scope,capacity,sat,check)
            if meta is None:
                if guard is None:
                    raise Budget
                capture=Capture(sat)
                meta=compile_rows(capture)
                save(db,scope,capacity,capture,meta,guard)
                stats[part+'_builds']=1
                return meta
    stats[part+'_replays']=1
    return meta


def prepare(db,database,problem,rows,split,groups,budget,sat,check,deadline,stats,formula_type,threshold_builder,structural):
    from pysat.card import ITotalizer
    from r27_reason_maxsat import SeamFormula
    schema(db)
    def structure(capture):
        formula=formula_type(problem,rows,capture,check,exact_defects=split)
        literals=[[mapping[code] for code in row] for mapping,row in zip(formula.literals,rows)]
        return dict(top=formula.pool.top,literals=literals,defects=formula.defects,stats=dict(formula.stats))
    meta=cached_encoding(db,database,structural,0,sat,check,deadline,stats,'structural',structure)
    formula=SeamFormula.from_structure(rows,meta)
    def counters(capture):
        global_rhs=[]
        if formula.defects:
            with ITotalizer(lits=formula.defects,ubound=min(len(formula.defects)-1,budget),top_id=formula.pool.top) as total:
                capture.append_formula(total.cnf.clauses)
                global_rhs=list(total.rhs)
                formula.pool.occupy(formula.pool.top+1,total.top_id)
        thresholds=[threshold_builder([formula.defects[i] for i in group],min(len(group),budget+1),
                                      formula,capture,check) for group in groups] if split else []
        return dict(global_rhs=global_rhs,thresholds=thresholds)
    counter_scope=digest([structural,groups,split,'counters-v1'])
    meta=cached_encoding(db,database,counter_scope,budget,sat,check,deadline,stats,'counter',counters)
    if budget<len(formula.defects):
        sat.add_clause([-meta['global_rhs'][budget]])
    return formula,meta['thresholds']
