"""Budgeted Gold searches in physical-bond space with independent phase and geographic relaxations."""
import hashlib
import json
import math
import sqlite3
import time
from contextlib import closing
from itertools import chain
from pathlib import Path

from r27_structure_filters import domain_bits
from r28_geo_color import signed_phase_model
from r28_geo_sewing import PosePacking
from r29_gold_phase import EndpointCuts, SignedParity
from r30_gold import GoldMatching
from r31_geographic import physical_bonds
from r32_gold_bonds import endpoints, inventory, loop_groups


def pairset(rows):
    result=set()
    for pair in rows:
        if (not isinstance(pair,(list,tuple)) or len(pair)!=2 or
                any(type(v) is not int or v<0 for v in pair)):
            raise ValueError("Expected a pair of nonnegative integer side IDs")
        result.add(tuple(sorted(pair)))
    return frozenset(result)


def checked_core_pairs(rows, side_count):
    pairs = pairset(rows)
    if any(not 0 <= a < b < side_count or a//3 == b//3 for a,b in pairs):
        raise ValueError('Expected two different physical tile sides')
    return pairs


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


class Certificates:
    """Physical conflicts can be transported between any boards whose assumptions replay."""
    def __init__(self, problem):
        self.problem = problem
        self.inventory = inventory(problem)
        self.xor = SignedParity(problem)
        self.mark = len(self.xor.base.trail)
        self.arcs = endpoints(problem)
        self.sides = [tuple(k for k in range(1,12) if (s,k) in self.arcs)
                      for s in range(3*problem.n)]
        self.rows = set()

    def xor_conflict(self, pairs):
        self.xor.base.rollback(self.mark)
        for a,b in sorted(pairs):
            if not self.xor.bond(a,b):
                return frozenset((u,v) for typ,u,v in self.xor.base.conflict if typ=='bond')
        return None

    def conflicts(self, rows):
        pairs = checked_core_pairs(rows, len(self.sides))
        bad = pairs - self.inventory.possible
        if bad:
            return {frozenset([p]) for p in bad}
        # Incompatible incident bonds are already excluded by a perfect sewing.
        if len(set(chain.from_iterable(pairs))) != 2*len(pairs):
            return {frozenset((p,q)) for p in pairs for q in pairs if p<q and set(p)&set(q)}
        result = loop_groups(self.arcs, self.sides, pairs, self.problem.total_segments)
        core = self.xor_conflict(pairs)
        if core is not None:
            result.add(core)
        return result

    def add(self, candidates):
        """Only a replayed contradiction under unrestricted physical phases may enter."""
        before = len(self.rows)
        for candidate in candidates:
            for c in self.conflicts(candidate):
                if any(r<=c for r in self.rows):
                    continue
                self.rows = {r for r in self.rows if not c<r}
                self.rows.add(c)
        return len(self.rows) - before

    def clauses(self):
        return [sorted(c) for c in sorted(self.rows,key=lambda c:(len(c),sorted(c)))]


def reference_groups(problem, codes, extra=()):
    reference = pairset(physical_bonds(problem,codes))
    cert = Certificates(problem)
    scope = read_json(Path(__file__).parent/'docs/R31_GOLD_RESULTS.json')
    if problem.fingerprint == scope['binding']:
        group = scope['departures']['center']['witnesses']
        cert.add(group)
    cert.add(extra)
    # Check all sealed paths before filtering individually impossible bonds.
    closed = loop_groups(cert.arcs, cert.sides, reference, problem.total_segments)
    cert.add(closed)
    within = [c for c in cert.rows if c<=reference]
    return reference, cert, sorted(within,key=lambda g:(len(g), sorted(g)))


def loss_partition(problem, reference, groups, outside_losses, bounds):
    """Build disjoint ownership constraints on real mating variables."""
    outside = set(reference)-set().union(*groups)
    lower,upper = bounds
    if type(lower) is not int or type(upper) is not int or not 0<=lower<=upper<=len(reference):
        raise ValueError("Invalid reference loss interval")
    if outside_losses is not None and (type(outside_losses) is not int or not 0<=outside_losses<=len(outside)):
        raise ValueError("Invalid outside loss count")
    return dict(outside=sorted(outside),lower=lower,upper=upper,outside_losses=outside_losses,
                groups=[sorted(g) for g in groups])


def add_loss_budget(check, reference, spec):
    def absent(p):
        return 1 - check.mate.get(p,0)
    missed=sum(absent(p) for p in reference)
    check.m.add(missed>=spec['lower'])
    check.m.add(missed<=spec['upper'])
    if spec['outside_losses'] is not None:
        check.m.add(sum(absent(p) for p in spec['outside'])==spec['outside_losses'])
    for g in spec['groups']:
        check.m.add(sum(absent(tuple(p)) for p in g)>=1)


def first_loss_partition(groups, required=(), forbidden=(), path=()):
    """Expand only the selected first-missing branches. Children are disjoint."""
    keep, drop = set(required), set(forbidden)
    if keep & drop:
        return None
    for depth in range(len(path)+1):
        pending = [g-keep for g in groups if not drop.intersection(g)]
        if any(not g for g in pending):
            return None
        if not pending:
            if depth < len(path):
                raise ValueError('Branch path continues past a leaf')
            return sorted(keep), sorted(drop), []
        options = sorted(min(pending,key=lambda g:(len(g),sorted(g))))
        if len(options)==1:
            # Unary conflicts are discarded automatically before consuming an index.
            return first_loss_partition(groups,keep,drop|{options[0]},path[depth:])
        if depth==len(path):
            return sorted(keep),sorted(drop),options
        i = path[depth]
        if type(i) is not int or not 0<=i<len(options):
            raise ValueError('Branch index exceeds the disjoint partition')
        keep.update(options[:i])
        drop.add(options[i])


def dropout(problem, reference, cap, critical, seconds, stage='integer', required=(), excluded=(), domains=None):
    """Globally minimum loss in the chosen relaxation; a feasible objective is an upper bound there only."""
    from ortools.sat.python import cp_model
    if stage not in ('xor','integer','euler','mating') or seconds<=0:
        raise ValueError('Invalid stage or time allowance')
    data = EndpointCuts(problem)
    if stage=='mating':
        check=GoldMatching(problem,domains or domain_bits(problem),{},required,forbidden=excluded)
        m=check.m
        lost={p:1-check.mate.get(p,0) for p in reference}
    else:
        if stage=='xor':
            m=cp_model.CpModel()
            phases=[m.new_bool_var('tile_phase') for _ in range(problem.n)]
            base=SignedParity(problem,data.sides)
            if base.base.conflict is not None:
                m.add_bool_or([])
            for mask,rhs,_ in base.base.pivots.values():
                row=[x for i,x in enumerate(phases) if mask>>i&1]
                if rhs: m.add_bool_xor(row)
                else: m.add_bool_xor([*row,True])
        else:
            m,phases=signed_phase_model(problem,euler=stage=='euler',pins={})
        def signed(p):
            a,b=p
            ma,ca=data.sides[a]
            return phases[a//3], phases[b//3], ca^data.sides[b][1]^(ma.bit_count()&1)
        # A mismatch forces loss. A compatible phase can still lose the physical bond.
        # Keeping a phase-compatible reference bond has unit value in this relaxation.
        lost={}
        for p in reference:
            u,v,flip=signed(p)
            d=m.new_bool_var('lost_physical_bond')
            m.add(u!=v if flip else u==v).only_enforce_if(~d)
            lost[p]=d
        for p in required:
            u,v,flip=signed(p)
            m.add(u!=v if flip else u==v)
            if p in lost:m.add(lost[p]==0)
        for p in excluded:
            if p in lost:m.add(lost[p]==1)
    for core in critical:
        if core<=reference:m.add(sum(lost[p] for p in core)>=1)
    cost=sum(lost.values())
    if cap is not None:
        m.add(cost<=cap)
    m.minimize(cost)
    solver=cp_model.CpSolver()
    solver.parameters.num_search_workers=1
    solver.parameters.max_time_in_seconds=seconds
    status=solver.solve(m)
    out=dict(stage=stage,status=solver.status_name(status),branches=solver.num_branches,
             seconds=solver.wall_time,cap=cap)
    if status==cp_model.MODEL_INVALID:
        raise ValueError(solver.solution_info())
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL,cp_model.UNKNOWN):
        out['lower']=max(0,math.ceil(solver.best_objective_bound))
    if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
        out['relaxation_cost']=round(solver.objective_value)
        out['losses']=[p for p,d in lost.items() if solver.value(d)]
        if stage!='mating':out['phases']=[solver.value(x) for x in phases]
    if status==cp_model.INFEASIBLE and cap is not None:out['lower']=cap+1
    return out


def pose_lower(problem, reference, groups, codes, seconds, cells=None):
    """Minimum changed oriented tile poses necessary to hit all physical conflicts."""
    from ortools.sat.python import cp_model
    m=cp_model.CpModel()
    tile=[m.new_bool_var('change_tile_pose') for _ in range(problem.n)]
    if cells is not None:
        if (not isinstance(cells,(list,tuple)) or
                any(type(c) is not int or not 0<=c<problem.n for c in cells) or len(set(cells))!=len(cells)):
            raise ValueError('Free cells must be distinct valid cell IDs')
        free={codes[c]//3 for c in cells}
        for t,v in enumerate(tile):
            if t not in free:m.add(v==0)
    for g in groups:
        if g<=reference:
            m.add_bool_or([tile[t] for t in sorted({s//3 for p in g for s in p})])
    m.minimize(sum(tile))
    s=cp_model.CpSolver()
    s.parameters.max_time_in_seconds=seconds
    s.parameters.num_search_workers=1
    st=s.solve(m)
    return dict(status=s.status_name(st),lower=math.ceil(s.best_objective_bound),
                relaxed_poses=[i for i in range(problem.n) if s.value(tile[i])]
                if st in (cp_model.OPTIMAL,cp_model.FEASIBLE) else None)


def master(problem, reference, cert, budget, seconds, required=(), forbidden=(), domains=None):
    """Use phase, geographic packing and complete side matching as separation oracles."""
    from ortools.sat.python import cp_model
    m=cp_model.CpModel()
    loss={p:m.new_bool_var('break') for p in sorted(reference)}
    for p in required:
        if p in loss:m.add(loss[p]==0)
    for p in forbidden:
        if p in loss:m.add(loss[p]==1)
    m.add(sum(loss.values())<=budget)
    ds=tuple(domains or domain_bits(problem))
    packing=PosePacking(problem,ds)
    counts=dict(master=0,xor_or_loop=0,pose=0,mating=0)
    deadline=time.monotonic()+seconds
    seen=set()
    for row in cert.rows:
        if row<=reference:
            m.add_bool_or([loss[p] for p in row])
            seen.add(row)
    while time.monotonic()<deadline:
        s=cp_model.CpSolver()
        s.parameters.max_time_in_seconds=max(1e-6,deadline-time.monotonic())
        s.parameters.num_search_workers=1
        status=s.solve(m)
        counts['master']+=1
        if status not in (cp_model.FEASIBLE,cp_model.OPTIMAL):
            return dict(status=s.status_name(status),calls=counts,clauses=cert.clauses())
        kept={p for p,d in loss.items() if not s.value(d)}
        chosen=frozenset(kept|set(required))
        found=cert.conflicts(chosen)
        if found:
            cert.add(found)
            counts['xor_or_loop']+=1
        elif packing.check(chosen,max_steps=512)['feasible'] is False:
            # A complete negative under these geographic domains is local to this master.
            found={chosen}
            counts['pose']+=1
        else:
            left=deadline-time.monotonic()
            if left<=0:break
            check=GoldMatching(problem,ds,{},chosen,forbidden=forbidden)
            oracle=cp_model.CpSolver()
            oracle.parameters.max_time_in_seconds=max(1e-6,deadline-time.monotonic())
            oracle.parameters.num_search_workers=1
            counts['mating']+=1
            ans=oracle.solve(check.m)
            if ans==cp_model.INFEASIBLE:found={chosen}
            elif ans in (cp_model.FEASIBLE,cp_model.OPTIMAL):
                return dict(status='FEASIBLE_SEWING_RELAXATION',losses=sorted(reference-kept),
                            calls=counts,clauses=cert.clauses())
            else:break
        added=False
        for c in found:
            # Required constants may be removed only in this locally guarded master.
            row=frozenset(c-set(required))
            if not row<=kept:raise ArithmeticError('Separation failed to exclude master')
            if row not in seen:
                seen.add(row)
                m.add_bool_or([loss[p] for p in row])
                added=True
        if not added:raise ArithmeticError('Separation repeated a satisfied cut')
    return dict(status='UNKNOWN',calls=counts,clauses=cert.clauses())


def save_query(database,key,query,*,deadline=float('inf'),setup=None,related=None,publish=None,check=None):
    """Fence duplicate native calls and atomically publish final verified evidence."""
    from r34_query_ownership import LostQueryLease,query_lease
    if database is None:return query()
    with closing(sqlite3.connect(database,timeout=60)) as db:
        with db:
            db.execute('CREATE TABLE IF NOT EXISTS r32_gold(key TEXT PRIMARY KEY,result TEXT NOT NULL)')
            if setup:setup(db)
        def lookup():
            row=db.execute('SELECT result FROM r32_gold WHERE key=?',(key,)).fetchone()
            if row:
                answer=json.loads(row[0])
                if check:answer=check(answer)
            else:
                answer=related(db) if related else None
            return dict(answer,reused=True,cp_calls_now=0) if answer is not None else None
        cached=lookup()
        if cached is not None:return cached
        try:
            with query_lease(db,database,'geographic:'+key,deadline) as guard:
                cached=lookup()
                if cached is not None:return cached
                if guard is not None:
                    answer=query()
                    if answer['status'] in ('GOLD','INFEASIBLE'):
                        if check:answer=check(answer)
                        with db:
                            db.execute('BEGIN IMMEDIATE')
                            guard()
                            db.execute('INSERT OR IGNORE INTO r32_gold VALUES(?,?)',(key,json.dumps(answer)))
                            if publish:publish(db,answer)
                    return answer
        except LostQueryLease:
            pass
        return lookup() or dict(status='UNKNOWN',cp_calls_now=0)


def geographic(problem, reference, groups, checked, path, lower, upper, outside, seconds, database, domains=None, pins=None):
    from r30_gold import search
    if seconds<=0:raise ValueError('Invalid Gold solve budget.')
    from r29_gold_geography import checked_domains
    ds=tuple(checked_domains(problem,domains))
    spec=loss_partition(problem,reference,groups,outside,(lower,upper))
    # Unproved well-formed hints are harmless. Only a native model consuming
    # them needs to replay the physical proof; all paths check the side IDs.
    checked=tuple(checked_core_pairs(row,3*problem.n) for row in checked)
    parts=first_loss_partition(groups,path=path)
    if parts is None:return dict(status='INFEASIBLE',cp_calls=0)
    require,forbid,options=parts
    for a,b in [*require,*forbid]:
        if type(a) is not int or type(b) is not int or not (0 <= a < 3*problem.n and 0 <= b < 3*problem.n):
            raise ValueError("Invalid physical side index.")
    if database is not None:
        from r35_claim_catalogue import published_scope
        if (problem.fingerprint == 'e96470e2d302f4d3c350479f49a21bf05f7124f1e62cecfa34b879140403f853'
                and not published_scope(problem)):
            raise ValueError('Stored geographic queries require the authenticated published puzzle.')
    from r34_geographic_reuse import related,schema,verify_board
    from r29_gold_phase import checked_pins
    pins_for_board=checked_pins(problem.n,{0:0} if pins is None else pins)
    physical=[sorted(reference),spec,require,forbid,sorted((pins or {}).items())]
    context=digest([problem.fingerprint,*physical])
    def key_for_domains(rows):
        return digest([1,problem.fingerprint,list(rows),*physical])
    key=key_for_domains(ds)
    deadline=time.monotonic()+seconds
    def solve():
        from r35_geographic_loss import geographic_loss_cutoff
        cutoff=geographic_loss_cutoff(problem,reference,upper,require,forbid)
        if cutoff is not None:
            return cutoff
        remaining=deadline-time.monotonic()
        if remaining<=0:return dict(status='UNKNOWN',cp_calls_now=0)
        return search(problem,domains=ds,geography=True,channel=True,learned_geometry=True,
                      seconds=remaining,pins=pins,bonds=require,forbidden=forbid,
                      physical_loss=(reference,spec,checked))
    def positive(row):
        return verify_board(problem,row,ds,pins_for_board,require,forbid,reference,spec)
    def validate(row):
        if row['status']=='INFEASIBLE':
            return row
        result=positive(row) if row['status']=='GOLD' else None
        if result is None:
            raise ValueError('Geographic query failed independent validation.')
        return result
    def save(db,row):
        if row['status']=='GOLD':
            db.execute('INSERT OR IGNORE INTO r34_geo_boards VALUES(?,?)',(problem.fingerprint,key))
        else:
            db.execute('INSERT OR IGNORE INTO r34_geo_domains VALUES(?,?,?)',
                       (context,key,json.dumps([hex(v) for v in ds])))
    result=save_query(database,key,solve,deadline=deadline,setup=schema,
                      related=lambda db:related(db,problem,context,ds,positive,key_for_domains),publish=save,check=validate)
    return dict(result,branch=path,children=len(options))
