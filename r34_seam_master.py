"""Checked hitting-set search for a restricted physical cycle family and geographic seam target."""
from collections import Counter
import time

import numpy as np

from r30_exact_routes import cycle_type


class SeamMaster:
    def __init__(self,query):
        self.query=query
        self.seams=[]
        terms=iter(query.seams)
        for a,s,b,t in query.model.problem.edges:
            left=set(map(int,query.model.masks[query.options[a],s]))
            right=set(map(int,query.model.reverse[query.model.masks[query.options[b],t]]))
            self.seams.append(next(terms) if left & right else 0)
        if next(terms,None) is not None:
            raise ValueError("Geographic seams differ from the route model.")
        self.cores=set()
        self.witnesses=[]

    def solve(self,target,seconds=0,pins=()):
        from ortools.sat.python import cp_model
        q=self.query
        seams=self.seams
        if not 0<=target<=len(seams) or seconds<0:
            raise ValueError("Invalid seam target or time limit.")
        pins=frozenset(pins)
        if any(c not in q.pool or j not in q.pool for c,j in pins):
            raise ValueError("A donor pin is outside the chosen pool.")
        deadline=time.monotonic()+seconds if seconds else None
        def solve(model):
            solver=cp_model.CpSolver()
            solver.parameters.num_search_workers=1
            if deadline is not None:
                available=deadline-time.monotonic()
                if available<=0:
                    return solver,cp_model.UNKNOWN
                solver.parameters.max_time_in_seconds=available
            status=solver.solve(model)
            if status==cp_model.MODEL_INVALID:
                raise RuntimeError(solver.solution_info())
            return solver,status

        def verify(codes):
            board=q.model.legal_board(codes)
            ds=np.argsort(q.center//3)[board//3]
            if cycle_type(ds.tolist())!=q.cycles or any(int(ds[c])!=c for c in range(len(ds)) if c not in q.pool):
                raise ArithmeticError("Target witness violates its cycle scope.")
            if any(int(ds[c])!=j for c,j in pins):
                return None
            report=q.model.problem.validate(list(map(int,board)))
            if report['matched_edges']>=target:
                return dict(codes=list(map(int,board)),score=report['matched_edges'],gold=report['valid'])
            return None

        work=Counter()
        for codes in self.witnesses:
            hit=verify(codes)
            if hit:
                return dict(status='FEASIBLE',work=dict(witness_reuse=1),**hit)
        master=cp_model.CpModel()
        fail={i:master.new_bool_var('failed') for i,term in enumerate(seams) if type(term) is not int}
        fixed=sum(type(term) is int and term==0 for term in seams)
        master.add(sum(fail.values())==min(len(fail),len(seams)-target-fixed))
        def add_cut(core):
            master.add_bool_or([fail[i] for i in core])
        for scope,core in self.cores:
            if scope<=pins:
                add_cut(core)
                work['inherited_cuts']+=1
        master.add_decision_strategy(list(fail.values()),cp_model.CHOOSE_FIRST,cp_model.SELECT_MAX_VALUE)
        # A candidate failure set covers every discovered conflict. Requiring
        # all other seams is a physical realizability query with free rotations.
        base=q.cp.clone()
        for c,j in pins:
            base.add(q.donors[c]==j)
        while True:
            ms,status=solve(master)
            work['master_calls']+=1
            if status==cp_model.INFEASIBLE:
                return dict(status='INFEASIBLE',work=dict(work))
            if status not in (cp_model.FEASIBLE,cp_model.OPTIMAL):
                return dict(status='UNKNOWN',work=dict(work))
            keep=[i for i,v in fail.items() if not ms.value(v)]
            sub=base.clone()
            ids={seams[i].index:i for i in keep}
            sub.add_assumptions([seams[i] for i in keep])
            solver,status=solve(sub)
            work['subsolver_calls']+=1
            work['subsolver_branches']+=solver.num_branches if status!=cp_model.UNKNOWN else 0
            if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
                codes=[solver.value(v) for v in q.codes]
                hit=verify(codes)
                if hit is None:
                    raise ArithmeticError("The physical witness misses the claimed seam threshold.")
                self.witnesses.append(hit['codes'])
                return dict(status='FEASIBLE',work=dict(work),**hit)
            if status!=cp_model.INFEASIBLE:
                return dict(status='UNKNOWN',work=dict(work))
            core=tuple(sorted(ids[k] for k in solver.sufficient_assumptions_for_infeasibility()))
            # Replay as hard seam equalities without assumption extraction.
            replay=base.clone()
            for i in core:
                replay.add(seams[i]==1)
            _,checked=solve(replay)
            work['replays']+=1
            if checked not in (cp_model.UNKNOWN,cp_model.INFEASIBLE):
                raise ArithmeticError("An infeasible physical seam core failed independent replay.")
            if checked==cp_model.UNKNOWN:
                return dict(status='UNKNOWN',work=dict(work))
            self.cores.add((pins,core))
            add_cut(core)
            work['new_cuts']+=1
