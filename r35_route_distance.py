"""Physical score target with a bounded or minimum number of unrestricted transpositions."""
import argparse
from math import ceil

from r27_incumbent import BoardModel
from r28_exchange import read_json
from r30_exact_routes import CycleModel,cycle_type
from search_problem import SearchProblem


class DistanceModel:
    def __init__(self,model,board,pool):
        self.base=CycleModel(model,board,pool,(),arbitrary=True)
        q=self.base
        rows=q.pool
        roots=[q.cp.new_int_var(0,i,'cycle_minimum') for i in range(len(rows))]
        depth=[q.cp.new_int_var(0,len(rows)-1,'steps_to_minimum') for _ in rows]
        starts=[]
        for i,c in enumerate(rows):
            idx=q.cp.new_int_var(0,len(rows)-1,'next_index')
            q.cp.add_element(idx,rows,q.donors[c])
            q.cp.add_element(idx,roots,roots[i])
            child_depth=q.cp.new_int_var(0,len(rows)-1,'next_depth')
            q.cp.add_element(idx,depth,child_depth)
            first=q.cp.new_bool_var('minimum_in_cycle')
            q.cp.add(roots[i]==i).only_enforce_if(first)
            q.cp.add(roots[i]!=i).only_enforce_if(~first)
            q.cp.add(depth[i]==0).only_enforce_if(first)
            q.cp.add(depth[i]==child_depth+1).only_enforce_if(~first)
            starts.append(first)
        self.distance=q.cp.new_int_var(0,max(0,len(rows)-1),'transpositions')
        q.cp.add(self.distance==len(rows)-sum(starts))

    def solve(self,target,*,minimum=0,maximum=None,optimize=False,seconds=0):
        from ortools.sat.python import cp_model
        q=self.base
        maximum=max(0,len(q.pool)-1) if maximum is None else maximum
        if not 0<=minimum<=maximum or seconds<0 or not 0<=target<=len(q.model.problem.edges):
            raise ValueError('Invalid physical threshold, distance interval or time limit.')
        if minimum==0:
            stationary=q.model.problem.validate(q.center.tolist())['matched_edges']
            if stationary>=target:
                return dict(status='OPTIMAL' if optimize else 'FEASIBLE',branches=0,conflicts=0,
                            codes=q.center.tolist(),distance=0,score=stationary,distance_lower_bound=0)
        m=q.cp.clone()
        m.add(sum(q.seams)>=target)
        m.add(self.distance>=minimum)
        m.add(self.distance<=maximum)
        if optimize:
            m.minimize(self.distance)
        solver=cp_model.CpSolver()
        solver.parameters.num_search_workers=1
        if seconds:
            solver.parameters.max_time_in_seconds=seconds
        status=solver.solve(m)
        if status==cp_model.MODEL_INVALID:
            raise ValueError(solver.solution_info())
        row=dict(status=solver.status_name(status),branches=solver.num_branches,conflicts=solver.num_conflicts)
        if status==cp_model.INFEASIBLE:
            return row
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            return row
        codes=[solver.value(v) for v in q.codes]
        q.model.legal_board(codes)
        inverse={int(code)//3:c for c,code in enumerate(q.center)}
        perm=[inverse[int(code)//3] for code in codes]
        if any(c!=d for c,d in enumerate(perm) if c not in q.pool):
            raise ArithmeticError('Witness moved a donor outside its permitted pool.')
        actual=sum(k-1 for k in cycle_type(perm))
        score=q.model.problem.validate(codes)['matched_edges']
        if not minimum<=actual<=maximum or actual!=solver.value(self.distance) or score<target:
            raise ArithmeticError('Distance threshold witness disagrees with independent physical evaluation.')
        row.update(codes=codes,distance=actual,score=score)
        if optimize:
            row['distance_lower_bound']=max(minimum,ceil(solver.best_objective_bound-1e-8))
        return row


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--incumbent',default='docs/R29_INCUMBENT.json')
    cli.add_argument('--cells',nargs='+',type=int,required=True)
    cli.add_argument('--target',type=int,default=208)
    cli.add_argument('--min-distance',type=int,default=0)
    cli.add_argument('--max-distance',type=int)
    cli.add_argument('--optimize',action='store_true')
    cli.add_argument('--seconds',type=float,default=0)
    a=cli.parse_args()
    import json
    model=BoardModel(SearchProblem(read_json(a.tiles)))
    print(json.dumps(DistanceModel(model,read_json(a.incumbent)['codes'],a.cells).solve(
        a.target,minimum=a.min_distance,maximum=a.max_distance,optimize=a.optimize,seconds=a.seconds)))
