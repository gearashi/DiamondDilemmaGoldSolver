"""Independent cycle-distance CP-SAT verifier and checked conditional donor bounds."""
import argparse
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import domain_binding, read_json
from search_problem import SearchProblem


def cycle_type(donors):
    """Return nontrivial cycle lengths of a destination-to-source permutation."""
    if sorted(donors) != list(range(len(donors))):
        raise ValueError("Expected a permutation.")
    seen = set()
    lengths = []
    for start in range(len(donors)):
        if start in seen:
            continue
        i, length = start, 0
        while i not in seen:
            seen.add(i)
            length += 1
            i = donors[i]
        if length > 1:
            lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def _scope(model, board, pool, cycles):
    center = model.legal_board(board)
    if pool is None:
        pool = list(map(int, model.free))
    else:
        pool = list(pool)
    if any(type(i) is not int or not 0 <= i < model.n for i in pool) or len(set(pool)) != len(pool):
        raise ValueError("Invalid cell pool.")
    cycles = tuple(sorted(cycles, reverse=True))
    if any(type(k) is not int or k < 2 for k in cycles) or sum(cycles) > len(pool):
        raise ValueError("Cycle lengths do not fit the pool.")
    pool = sorted(pool)
    identity = [domain_binding(model), (center // 3).tolist(), pool, cycles]
    binding = hashlib.sha256(json.dumps(identity, separators=(",",":")).encode()).hexdigest()
    return center, pool, cycles, binding


class CycleModel:
    def __init__(self, model, board, pool, cycles):
        from ortools.sat.python import cp_model
        self.center, self.pool, self.cycles, self.binding = _scope(model,board,pool,cycles)
        self.model = model
        self.cp = cp_model.CpModel()
        n = model.n
        pool = self.pool
        cells = set(pool)
        donors = [self.cp.new_int_var(0,n-1,f"donor_{c}") if c in cells else self.cp.new_constant(c)
                  for c in range(n)]
        self.donors = donors
        actors = []
        self.actor_groups = []
        previous = {}
        for length in self.cycles:
            group = [self.cp.new_int_var_from_domain(cp_model.Domain.from_values(pool),"cycle_cell")
                     for _ in range(length)]
            for j,cell in enumerate(group):
                self.cp.add_element(cell,donors,group[(j+1)%length])
                if j:
                    self.cp.add(group[0] < cell)
            if length in previous:
                self.cp.add(previous[length] < group[0])
            previous[length] = group[0]
            self.actor_groups.append(group)
            actors.extend(group)
        if actors:
            self.cp.add_all_different(actors)
        moved = []
        for c in pool:
            flag = self.cp.new_bool_var("moved")
            self.cp.add(donors[c] != c).only_enforce_if(flag)
            self.cp.add(donors[c] == c).only_enforce_if(~flag)
            moved.append(flag)
        # The listed cycles already move every actor. Remaining donors are identity.
        self.cp.add(sum(moved)==len(actors))
        tiles = self.center // 3
        self.codes = []
        self.options = []
        for c in range(n):
            rows = [(j,int(3*tiles[j]+r)) for j in (pool if c in cells else (c,))
                    for r in range(3) if model.legal[c,3*tiles[j]+r]]
            vals = [v for _,v in rows]
            code = self.cp.new_int_var_from_domain(cp_model.Domain.from_values(vals),"oriented_tile")
            if c in cells:
                self.cp.add_allowed_assignments([donors[c],code],rows)
            self.codes.append(code)
            self.options.append(vals)
        self.seams = []
        for a,sa,b,sb in model.problem.edges:
            masks = [[int(model.masks[v,sa]) for v in self.options[a]],
                     [int(model.reverse[model.masks[v,sb]]) for v in self.options[b]]]
            if not set(masks[0]).intersection(masks[1]):
                continue
            if len(set(masks[0]+masks[1])) == 1:
                self.seams.append(1)
                continue
            ends = []
            for c,values in zip((a,b),masks):
                if len(set(values))==1:
                    ends.append(values[0])
                else:
                    end = self.cp.new_int_var_from_domain(cp_model.Domain.from_values(sorted(set(values))),"mask")
                    self.cp.add_allowed_assignments([self.codes[c],end],list(zip(self.options[c],values)))
                    ends.append(end)
            match = self.cp.new_bool_var("match")
            self.cp.add(ends[0]==ends[1]).only_enforce_if(match)
            self.seams.append(match)

    def _add_predecessors(self, m, target):
        """Undoing an isolated cycle leads into the checked lower-distance ball."""
        from r28_exchange import ExchangeBound
        from r30_cycles import Predecessors
        if self.cycles not in ((4,2),(3,3),(3,2,2),(2,2,2,2)):
            return 0
        checks=Predecessors(ExchangeBound(self.model,self.center),target)
        if not checks.certified:
            return 0
        # gain[c,j] bounds the local improvement against every fixed exterior
        # rotation. It adds over cells whose changed incident seams are disjoint.
        rows=[(a,b,int(checks.gain[a,b])) for a in self.pool for b in self.pool if a!=b]
        adjacency=[(a,b) for a in self.pool for b in self.pool if b in checks.near[a]]
        actors=[cell for group in self.actor_groups for cell in group]
        for group in self.actor_groups:
            terms=[]
            for ix,cell in enumerate(group):
                gain=m.new_int_var(-3,3,"isolated_donor_gain")
                m.add_allowed_assignments([cell,group[(ix+1)%len(group)],gain],rows)
                terms.append(gain)
            touches=[]
            group_indices={cell.index for cell in group}
            if adjacency:
                for a in group:
                    for b in actors:
                        if a.index==b.index or (b.index in group_indices and a.index>b.index):
                            continue
                        near=m.new_bool_var("piece_has_changed_neighbor")
                        m.add_allowed_assignments([a,b],adjacency).only_enforce_if(near)
                        touches.append(near)
            # A 208 endpoint must improve on its <=207 predecessor. Modeling
            # each touch only in the forward direction suffices for this disjunction.
            m.add(sum(terms)>=1).only_enforce_if([~v for v in touches])
        return len(self.actor_groups)

    def solve(self, *, target, pins=(), rules=(), seconds=5, optimize=False, core=False,
              move_cores=(), contact=False, predecessors=False, connected=False):
        from ortools.sat.python import cp_model
        if type(target) is not int or not 0<=target<=len(self.model.problem.edges) or seconds<=0:
            raise ValueError("Invalid target or solve budget.")
        if optimize and (rules or move_cores or contact or predecessors or connected):
            raise ValueError("Threshold-specific exclusions cannot be used to optimize smaller scores.")
        m = self.cp.clone()
        pins = list(pins)
        guards = {}
        n = self.model.n
        for c,j in pins:
            if type(c) is not int or type(j) is not int or not 0<=c<n or not 0<=j<n:
                raise ValueError("Physical pin must name a destination and a source cell.")
            if core:
                g = m.new_bool_var("held_donor")
                m.add(self.donors[c]==j).only_enforce_if(g)
                guards[g.index] = (c,j)
                m.add_assumption(g)
            else:
                m.add(self.donors[c]==j)
        undo_cuts=self._add_predecessors(m,target) if predecessors else 0
        applied = 0
        assignments = dict(pins)
        for rule in rules:
            if rule["binding"] != self.binding or rule["upper"] >= target:
                continue
            row = rule["pins"]
            if all(assignments.get(c)==j for c,j in row):
                return dict(status="INFEASIBLE",branches=0,applied=1,cp_calls=0,
                            upper=rule["upper"],reason="verified_donor_bound",predecessor_cuts=undo_cuts)
            if row:
                m.add_forbidden_assignments([self.donors[c] for c,j in row],[[j for c,j in row]])
            else:
                m.add_bool_or([])
            applied += 1
        for movement in move_cores:
            # External movement certificates must come from the domain-checked proof store.
            row=sorted(movement)
            if row:
                m.add_forbidden_assignments([self.donors[c] for c in row],[row])
            else:
                m.add_bool_or([])
        if contact or connected:
            # Each supported class requires an inter-cycle contact in the three
            # checked center scopes. Polarity and objective are part of this proof.
            record=read_json(Path(__file__).resolve().parent/'docs/R29_EXCHANGE_CENTERS.json')
            if (target<record['target'] or domain_binding(self.model)!=record['domain'] or
                    self.cycles not in ((4,2),(3,3),(3,2,2),(2,2,2,2)) or
                    not any((self.center//3).tolist()==r['pieces'] for r in record['centers'])):
                raise ValueError("Inter-cycle contact proof does not cover this center, family, or target.")
            near=[{i,*[int(b) for b in row if b>=0]} for i,row in enumerate(self.model.problem.neighbors)]
            pairs=[(a,b) for a in self.pool for b in self.pool if a!=b and near[a]&near[b]]
            contacts=[]
            cycle_edges=[]
            for i,first in enumerate(self.actor_groups):
                for j,second in enumerate(self.actor_groups[i+1:],i+1):
                    group_contacts=[]
                    for a in first:
                        for b in second:
                            e=m.new_bool_var("touches")
                            m.add_allowed_assignments([a,b],pairs).only_enforce_if(e)
                            contacts.append(e)
                            group_contacts.append(e)
                    cycle_edges.append((i,j,group_contacts))
            from r28_exchange import ExchangeBound
            from r31_route_geometry import proved_connected
            if connected and proved_connected(ExchangeBound(self.model,self.center),target,self.cycles):
                # Every proper bipartition must have at least one actual contact.
                for cut in range(1,1 << (len(self.actor_groups)-1)):
                    m.add_bool_or([e for i,j,lits in cycle_edges if ((cut>>i)^(cut>>j))&1 for e in lits])
            else:
                m.add_bool_or(contacts)
        if not optimize:
            m.add(sum(self.seams)>=target)
        else:
            m.maximize(sum(self.seams))
        solver = cp_model.CpSolver()
        solver.parameters.num_search_workers = 1
        solver.parameters.max_time_in_seconds = seconds
        status = solver.solve(m)
        if status == cp_model.MODEL_INVALID:
            raise RuntimeError(solver.solution_info())
        out = dict(status=solver.status_name(status),branches=solver.num_branches,applied=applied,cp_calls=1,
                   predecessor_cuts=undo_cuts)
        if status == cp_model.INFEASIBLE:
            if core:
                out["core"] = sorted(guards[i] for i in solver.sufficient_assumptions_for_infeasibility())
            return out
        if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            codes = [solver.value(x) for x in self.codes]
            ds = [solver.value(x) for x in self.donors]
            self.model.legal_board(codes)
            report = self.model.problem.validate(codes)
            if (cycle_type(ds)!=self.cycles or any(ds[c]!=c for c in set(range(n))-set(self.pool)) or
                    any(ds[c]!=j for c,j in pins) or
                    any(code//3!=self.center[j]//3 for code,j in zip(codes,ds)) or
                    (not optimize and report["matched_edges"]<target)):
                raise ArithmeticError("Independent route verification failed.")
            out.update(score=report["matched_edges"],codes=codes,gold=report["valid"])
            if optimize:
                out['upper'] = math.ceil(solver.best_objective_bound)
        return out


def rules_in(database, binding):
    if database is None or not Path(database).exists():
        return []
    with closing(sqlite3.connect(database)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='route_donor_rule'").fetchone():
            return []
        return [dict(binding=binding,pins=json.loads(pins),upper=upper)
                for pins,upper in db.execute("SELECT pins,upper FROM route_donor_rule WHERE binding=?",(binding,))]


def learn(query, pins, target, *, seconds=5, max_delete=30, database=None, contact=False, connected=False):
    """Learn only after a fresh unguarded threshold query confirms the smaller condition."""
    if max_delete<0:
        raise ValueError("Invalid deletion budget.")
    pins = sorted(set(map(tuple,pins)))
    first = query.solve(target=target,pins=pins,seconds=seconds,core=True,contact=contact,connected=connected)
    if first['status']!='INFEASIBLE':
        return dict(status=first['status'],queries=1,stored=False)
    keep = list(first['core'])
    calls = 1
    for pair in keep.copy()[:max_delete]:
        reduced = [x for x in keep if x!=pair]
        answer = query.solve(target=target,pins=reduced,seconds=seconds,contact=contact,connected=connected)
        calls += 1
        if answer['status']=='INFEASIBLE':
            keep = reduced
    numeric = query.solve(target=target,pins=keep,seconds=seconds,optimize=True)
    upper = min(target-1, numeric.get('upper',target-1))
    calls += 1
    # A fresh unguarded model checks both the learned literals and the numeric bound.
    fresh = CycleModel(query.model,query.center,query.pool,query.cycles)
    checked = fresh.solve(target=max(0,upper+1), pins=keep, seconds=seconds)
    calls += 1
    if checked['status']!='INFEASIBLE':
        return dict(status=checked['status'],queries=calls,stored=False)
    rule = dict(binding=query.binding,pins=keep,upper=upper)
    if database is not None:
        with closing(sqlite3.connect(database,timeout=30)) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS route_donor_rule(binding TEXT,pins TEXT,
                       upper INTEGER NOT NULL,PRIMARY KEY(binding,pins))""")
            db.execute("""INSERT INTO route_donor_rule VALUES (?,?,?) ON CONFLICT(binding,pins)
                       DO UPDATE SET upper=min(upper,excluded.upper)""",
                       (query.binding,json.dumps(keep),upper))
    return dict(status='CERTIFIED',queries=calls,stored=database is not None,rule=rule)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles",default="data/tiles.json")
    parser.add_argument("--incumbent",default="docs/R29_INCUMBENT.json")
    parser.add_argument("--cycles",type=int,nargs="+",required=True)
    parser.add_argument("--cells",type=int,nargs="+")
    parser.add_argument("--target",type=int,default=208)
    parser.add_argument("--seconds",type=float,default=5)
    parser.add_argument("--pins",type=Path,help="JSON list of [destination cell, donor cell].")
    parser.add_argument("--database",help="SQLite cache of checked conditional donor bounds.")
    parser.add_argument("--learn",action="store_true")
    parser.add_argument("--contact",action="store_true",help="Use the released-center 208 inter-cycle contact proof.")
    parser.add_argument("--connected",action="store_true",help="Require all actor cycles to connect under the release proof.")
    parser.add_argument("--predecessors",action="store_true",help="Use checked isolated-cycle predecessor gains where the released proof applies.")
    args=parser.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    model=BoardModel(problem)
    board=read_json(args.incumbent)["codes"]
    query=CycleModel(model,board,args.cells,args.cycles)
    pins=read_json(args.pins) if args.pins else []
    if args.learn:
        result=learn(query,pins,args.target,seconds=args.seconds,database=args.database,contact=args.contact,connected=args.connected)
    else:
        from r29_move_cores import load_cores
        cores=[]
        if args.database and Path(args.database).exists():
            with closing(sqlite3.connect(args.database)) as db:
                cores=load_cores(db,model,board,args.target)
        result=query.solve(target=args.target,pins=pins,seconds=args.seconds,
                           rules=rules_in(args.database,query.binding),move_cores=cores,contact=args.contact,
                           predecessors=args.predecessors,connected=args.connected)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
