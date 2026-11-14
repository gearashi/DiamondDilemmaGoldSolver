"""Independent-set assignment and certified fixed-permutation rotation repair.

All returned candidates are complete, independently auditable boards. Local
optima and bounds apply only to the stated frozen cells or tile permutation.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from validator import reverse_mask


def assignment(weights, legal):
    """Maximum-weight legal bijection for integer weights."""
    try:
        from ortools.graph.python.linear_sum_assignment import SimpleLinearSumAssignment
    except ImportError:
        # The Python 3.15 profile has no OR-Tools wheels. Reuse the checked
        # Hungarian implementation; a constant shift preserves every matching.
        from constraint_connectivity_round26 import assignment_upper_bound
        if not legal.any():
            raise ValueError('Assignment has no legal completion.')
        shift=int(weights[legal].min())
        matrix=[[int(w)-shift if ok else None for w,ok in zip(row,allow)] for row,allow in zip(weights,legal)]
        certificate=assignment_upper_bound(matrix)
        if certificate is None:
            raise ValueError('Assignment has no legal completion.')
        mates=np.array(certificate['assignment'])
        if len(mates)!=len(matrix) or not legal[np.arange(len(mates)),mates].all():
            raise ValueError('Assignment has no legal completion.')
        return mates
    rows, cols = np.nonzero(legal)
    solver = SimpleLinearSumAssignment()
    solver.add_arcs_with_cost(rows.astype(np.int32), cols.astype(np.int32),
                              -weights[rows, cols].astype(np.int64))
    if len(rows) == 0 or solver.solve() != solver.OPTIMAL or solver.num_nodes() != len(weights):
        raise ValueError('Assignment has no legal completion.')
    return np.array([solver.right_mate(i) for i in range(len(weights))])


class BoardModel:
    def __init__(self, problem):
        self.problem = problem
        self.n = problem.n
        self.a, self.sa, self.b, self.sb = np.array(problem.edges, dtype=int).reshape(-1, 4).T
        self.masks = problem.masks
        self.reverse = np.array([reverse_mask(v) for v in range(2048)], dtype=np.int16)
        self.legal = np.zeros((self.n, 3*self.n), dtype=bool)
        for cell, domain in enumerate(problem.domains):
            self.legal[cell, domain] = True
        self.free = np.array([i for i in range(self.n) if i not in problem.fixed], dtype=int)

    def matches(self, board):
        return self.masks[board[self.a], self.sa] == self.reverse[self.masks[board[self.b], self.sb]]

    def legal_board(self, values):
        board = np.asarray(values)
        if (board.shape != (self.n,) or board.dtype.kind not in 'iu' or np.any(board < 0)
                or np.any(board >= 3*self.n) or not np.array_equal(np.sort(board//3), np.arange(self.n))
                or not np.all(self.legal[np.arange(self.n), board])):
            raise ValueError('Expected a legal complete physical permutation.')
        return board.astype(int, copy=True)

    def independent_set(self, board, rng):
        bad = ~self.matches(board)
        defective = np.unique(np.r_[self.a[bad], self.b[bad]])
        order = np.r_[rng.permutation(np.intersect1d(self.free, defective)), rng.permutation(self.free)]
        blocked = set()
        cells = []
        for cell in order:
            if int(cell) not in blocked:
                cells.append(int(cell))
                blocked.add(int(cell))
                blocked.update(map(int, self.problem.neighbors[cell]))
        return np.array(cells, dtype=int)

    def block(self, values, cells, rng=None):
        board = self.legal_board(values)
        cells = np.asarray(cells, dtype=int)
        if not len(cells):
            return board
        if (len(set(cells)) != len(cells) or min(cells) < 0 or max(cells) >= self.n
                or any(c in self.problem.fixed for c in cells)
                or set(cells).intersection(self.problem.neighbors[cells].ravel())):
            raise ValueError('Assignment requires distinct nonadjacent movable cells.')
        choices = 3*(board[cells]//3)[:, None] + np.arange(3)
        legal = self.legal[cells[:, None, None], choices[None, :, :]]
        values = np.zeros(legal.shape, dtype=np.int32)
        for side in range(3):
            nb = self.problem.neighbors[cells, side]
            other = self.problem.sides[cells, side]
            target = self.reverse[self.masks[board[nb], other]]
            values += (nb >= 0)[:, None, None] & (self.masks[choices, side] == target[:, None, None])
        # A lower-order tie key cannot outweigh a single seam across the assignment.
        tie = np.zeros(legal.shape, dtype=np.int32) if rng is None else rng.integers(64, size=legal.shape)
        priced = np.where(legal, values*(64*len(cells)+1)+tie, -1)
        rotation = priced.argmax(axis=2)
        col = assignment(priced.max(axis=2), legal.any(axis=2))
        board[cells] = choices[col, rotation[np.arange(len(cells)), col]]
        return board

    def orientations(self, values, *, max_entries=59049, movable=None, rng=None):
        """Max-sum elimination with argmax reconstruction and safe factor bounds.

        If the next intermediate exceeds the limit, freeze one high-degree
        rotation to its supplied value. The resulting board is attainable. The
        original sum of separate seam maxima is still an admissible bound.
        """
        board = self.legal_board(values)
        if not isinstance(max_entries,int) or max_entries<3:
            raise ValueError('Rotation table limit must be at least three.')
        variable = set(range(self.n)) if movable is None else set(movable)
        if any(type(c) not in (int,np.int64,np.int32) or not 0<=c<self.n for c in variable):
            raise ValueError('Invalid rotation cell.')
        codes = 3*(board//3)[:, None] + np.arange(3)
        factors = []
        base = 0
        upper = 0
        fixed_for_limit = []
        rng = np.random.default_rng(0) if rng is None else rng
        for c in variable.copy():
            if self.legal[c,codes[c]].sum() == 1:
                variable.remove(c)
        for c in variable:
            factors.append(((int(c),),np.where(self.legal[c,codes[c]],0,-10000).astype(np.int32)))
        for a, sa, b, sb in self.problem.edges:
            left = codes[a] if a in variable else board[a:a+1]
            right = codes[b] if b in variable else board[b:b+1]
            grid = (self.masks[left,sa,None] == self.reverse[self.masks[right,sb]][None,:]).astype(np.int32)
            # Illegal rotations may loosen the bound but can never lower it.
            upper += int(grid.max())
            if np.all(grid == grid.flat[0]):
                base += int(grid.flat[0])
            else:
                axis = tuple(c for c in (a,b) if c in variable)
                arr = grid.reshape((3,)*len(axis))
                if len(axis)==2 and axis[0]>axis[1]:
                    axis=axis[::-1]
                    arr=arr.T
                factors.append((axis,arr))
        trace=[]
        table_entries=0
        while variable:
            neighbors={c:set() for c in variable}
            for scope,_ in factors:
                for c in scope:
                    neighbors[c].update(scope)
            c=min(variable,key=lambda v:(len(neighbors[v]),v))
            budget_exceeded = 3**len(neighbors[c])>max_entries
            if budget_exceeded:
                c=max(variable,key=lambda v:(len(neighbors[v]),-v))
                fixed_for_limit.append(c)
            incident=[(sc,tab) for sc,tab in factors if c in sc]
            factors=[(sc,tab) for sc,tab in factors if c not in sc]
            if budget_exceeded:
                for sc,tab in incident:
                    r=int(board[c]%3)
                    sub=np.take(tab,r,axis=sc.index(c))
                    sc=tuple(v for v in sc if v!=c)
                    if sc:
                        factors.append((sc,sub))
                    else:
                        base+=int(sub)
            else:
                scope=tuple(sorted(neighbors[c]))
                joined=np.zeros((3,)*len(scope),dtype=np.int32)
                for sc,tab in incident:
                    joined+=tab.reshape(tuple(3 if v in sc else 1 for v in scope))
                table_entries+=joined.size
                axis=scope.index(c)
                rest=tuple(v for v in scope if v!=c)
                permutation=rng.permutation(3)
                winner=permutation[np.take(joined,permutation,axis=axis).argmax(axis=axis)]
                trace.append((c,rest,winner))
                best=joined.max(axis=axis)
                if rest:
                    factors.append((rest,best))
                else:
                    base+=int(best)
            variable.remove(c)
        for c,rest,winner in reversed(trace):
            board[c]=codes[c,int(winner[tuple(board[v]%3 for v in rest)])]
        score=int(self.matches(board).sum())
        if score!=base or score>upper:
            raise AssertionError('Orientation reconstruction disagrees with factor score.')
        exact=not fixed_for_limit or score==upper
        return board,{'matched_edges':score,'upper_bound':score if exact else upper,
                      'exact':exact,'conditioned_cells':fixed_for_limit,'table_entries':table_entries}



def rotation_alternatives(model, values, *, k=4, seconds=5., seed=0, movable=None):
    """Enumerate the highest-scoring distinct orientations of this permutation.

    Between solves a no-good excludes only the preceding complete assignment.
    Each optimum/upper bound applies to the remaining fixed-permutation model.
    """
    from ortools.sat.python import cp_model
    import time
    if type(k) is not int or k<1 or not np.isfinite(seconds) or seconds<=0:
        raise ValueError('Supply positive solution and time limits.')
    board=model.legal_board(values)
    choices=3*(board//3)[:,None]+np.arange(3)
    movable=set(range(model.n)) if movable is None else set(movable)
    cp=cp_model.CpModel()
    terms=[]
    rotations=[]
    for c in range(model.n):
        allowed=[r for r in range(3) if model.legal[c,choices[c,r]] and (c in movable or r==board[c]%3)]
        rotations.append(cp.new_int_var_from_domain(cp_model.Domain.from_values(allowed),f'r{c}'))
        cp.add_hint(rotations[-1],int(board[c]%3))
    for edge,(a,sa,b,sb) in enumerate(model.problem.edges):
        matches=(model.masks[choices[a],sa,None]==model.reverse[model.masks[choices[b],sb]][None,:])
        if np.all(matches==matches.flat[0]):
            terms.append(int(matches.flat[0]))
        else:
            match=cp.new_bool_var(f'm{edge}')
            cp.add_allowed_assignments([rotations[a],rotations[b],match],
                [(x,y,int(matches[x,y])) for x in range(3) for y in range(3)])
            terms.append(match)
    cp.maximize(sum(terms))
    result=[]
    deadline=time.monotonic()+seconds
    for _ in range(k):
        remain=deadline-time.monotonic()
        if remain<=0:
            break
        solver=cp_model.CpSolver()
        solver.parameters.max_time_in_seconds=remain
        solver.parameters.num_search_workers=1
        solver.parameters.random_seed=seed
        status=solver.solve(cp)
        if status not in (cp_model.OPTIMAL,cp_model.FEASIBLE):
            break
        rotation=[solver.value(v) for v in rotations]
        candidate=choices[np.arange(model.n),rotation]
        score=int(model.matches(candidate).sum())
        if score!=round(solver.objective_value):
            raise RuntimeError('CP rotation model disagrees with raw seams.')
        result.append((candidate,dict(matched_edges=score,upper_bound=int(np.floor(solver.best_objective_bound+1e-6)),
            exact=status==cp_model.OPTIMAL,branches=int(solver.num_branches))))
        cp.add_forbidden_assignments(rotations,[rotation])
    return result


def polish_step(model,board,rng,*,max_entries=59049):
    cells=model.independent_set(board,rng)
    candidate=model.block(board,cells,rng)
    candidate,orientation=model.orientations(candidate,max_entries=max_entries,rng=rng)
    report=model.problem.validate(candidate.tolist())
    if report['matched_edges']!=orientation['matched_edges']:
        raise RuntimeError('Independent edge validation failed.')
    return candidate,report


def polish(problem, codes, *, seed=0, rounds=8, max_entries=59049, stop=lambda:False, progress=None):
    """Alternate exact set reassignment and rotation optimization, retain the best."""
    model=BoardModel(problem)
    board=model.legal_board(codes)
    rng=np.random.default_rng(seed)
    best=board.copy()
    best_report=problem.validate(best.tolist())
    rounds_done=0
    for _ in range(rounds):
        if stop():
            break
        board,report=polish_step(model,board,rng,max_entries=max_entries)
        if (report['matched_edges'],-report['line_components']) >= (best_report['matched_edges'],-best_report['line_components']):
            best,best_report=board.copy(),report
            if progress is not None:
                progress(best.tolist(),best_report)
        rounds_done+=1
        if report['valid']:
            break
    return {'codes':best.tolist(),'matched_edges':best_report['matched_edges'],'line_components':best_report['line_components'],
            'valid':best_report['valid'],'rounds':rounds_done}


def main():
    from search_problem import SearchProblem
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tiles',type=Path)
    parser.add_argument('incumbent',type=Path)
    parser.add_argument('--rounds',type=int,default=8)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--rotations-only',action='store_true',help='Use CP-SAT to certify orientations of this fixed permutation (requires OR-Tools).')
    parser.add_argument('--alternatives',type=int,default=2,help='Maximum distinct fixed-permutation orientations to certify.')
    parser.add_argument('--seconds',type=float,default=10.,help='Total fixed-permutation CP-SAT time budget.')
    args=parser.parse_args()
    read=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))
    hint=read(args.incumbent)
    problem=SearchProblem(read(args.tiles))
    codes=hint['codes'] if isinstance(hint,dict) else hint
    if args.rotations_only:
        try:
            alternatives=rotation_alternatives(BoardModel(problem),codes,k=args.alternatives,seconds=args.seconds,seed=args.seed)
        except ImportError:
            parser.error('Certified rotation enumeration requires the Python profile with OR-Tools.')
        result={'scope':'fixed physical permutation','alternatives':[]}
        for board,record in alternatives:
            report=problem.validate(board.tolist())
            if report['matched_edges']!=record['matched_edges']:
                raise RuntimeError('Independent orientation scoring failed.')
            result['alternatives'].append(dict(codes=board.tolist(),line_components=report['line_components'],valid=report['valid'],**record))
    else:
        result=polish(problem,codes,rounds=args.rounds,seed=args.seed)
    rendered=json.dumps(result,indent=2)
    if args.output:
        args.output.write_text(rendered+'\n',encoding='utf-8')
    print(rendered)


if __name__=='__main__':
    main()

