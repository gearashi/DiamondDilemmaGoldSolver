"""Bounded exact CNF comparisons for physical strand connectivity on residual boards."""
from collections import defaultdict
from itertools import combinations

from r27_structure_filters import Work, WorkLimit, bits, domain_bits
from validator import reverse_mask


class Formula:
    def __init__(self, work):
        self.work=work
        self.n=0
        self.rows=[]

    def var(self):
        self.work.tick('variables')
        self.n+=1
        return self.n

    def clause(self,values):
        row=list(values)
        self.work.tick('literals',1+len(row))
        self.rows.append(row)

    def one(self,row):
        row=list(row)
        self.clause(row)
        for a,b in combinations(row,2):self.clause((-a,-b))

    def either(self,row):
        row=tuple(set(row))
        if len(row)==1:return row[0]
        x=self.var()
        for a in row:self.clause((-a,x))
        self.clause([-x,*row])
        return x

    def both(self,a,b):
        x=self.var()
        self.clause((-x,a))
        self.clause((-x,b))
        self.clause((x,-a,-b))
        return x


def _base(problem,domains,cnf,physical):
    variables=[{x:cnf.var() for x in bits(d)} for d in domains]
    for row in variables:cnf.one(row.values())
    for t in range(problem.n):
        cnf.one(v for row in variables for x,v in row.items() if x//3==t)
    ends={}
    i=0
    for t,tile in enumerate(problem.tiles):
        for seg in tile['segments']:
            for j,(s,k) in enumerate(seg):ends[t,s,k]=2*i+j
            i+=1
    triggers=defaultdict(list)
    for a,s,b,t in problem.edges:
        for ac,av in variables[a].items():
            m=int(problem.masks[ac,s])
            for bc,bv in variables[b].items():
                cnf.work.tick('seam_candidates')
                if bc//3==ac//3:continue
                if reverse_mask(m)!=int(problem.masks[bc,t]):
                    cnf.clause((-av,-bv))
                    continue
                if physical and m:
                    joined=cnf.both(av,bv)
                    for j in bits(m):
                        left=ends[ac//3,(s-ac%3)%3,j+1]
                        right=ends[bc//3,(t-bc%3)%3,11-j]
                        triggers[tuple(sorted((left,right)))].append(joined)
    return variables,{k:cnf.either(v) for k,v in triggers.items()}


def _modular(cnf,edges,n,m):
    if type(m)is not int or m<=0 or n%m:raise ValueError('The modulus must divide the strand count.')
    width=max(1,(m-1).bit_length())
    position=[[cnf.var() for _ in range(width)] for _ in range(n)]
    successor=[]
    top=m-1
    for row in position:
        # Forbid the first most-significant excess bit above m-1.
        for k,x in enumerate(row):
            if not top&(1<<k):
                cnf.clause([-x,*(-row[j] if top&(1<<j) else row[j] for j in range(k+1,width))])
        wrap=-cnf.either([-x if top&(1<<k) else x for k,x in enumerate(row)])
        inc=[-row[0]]
        carry=row[0]
        for x in row[1:]:
            q=cnf.var()
            cnf.clause((-q,x,carry));cnf.clause((-q,-x,-carry))
            cnf.clause((q,-x,carry));cnf.clause((q,x,-carry))
            inc.append(q)
            carry=cnf.both(x,carry)
        successor.append([cnf.both(y,-wrap) for y in inc])
    for x in position[0]:cnf.clause([-x])
    incoming=[[] for _ in range(n)]
    outgoing=[[] for _ in range(n)]
    for (a,b),active in edges.items():
        i,j=a//2,b//2
        directions=(cnf.var(),cnf.var())
        x,y=directions
        cnf.clause((-active,x,y))
        cnf.clause((-x,-y))
        for a,b,v in ((i,j,x),(j,i,y)):
            cnf.clause((-v,active))
            incoming[b].append(v)
            outgoing[a].append(v)
            for x,y in zip(successor[a],position[b]):
                cnf.clause((-v,-x,y));cnf.clause((-v,x,-y))
    for row in incoming+outgoing:cnf.one(row)


class DenseElimination(Exception):
    """The least-degree active strand would exceed the solver's fill threshold."""


def _eliminate(cnf,edges,n,maximum_neighbors=None):
    adjacency=[{} for _ in range(n)]
    terms=defaultdict(list)
    for (a,b),x in edges.items():terms[tuple(sorted((a//2,b//2)))].append(x)
    for (a,b),items in terms.items():adjacency[a][b]=adjacency[b][a]=cnf.either(items)
    remaining=set(range(n))
    while len(remaining)>1:
        v=min(remaining,key=lambda k:(len(adjacency[k]),k))
        row=adjacency[v]
        if maximum_neighbors is not None and len(row)>maximum_neighbors:raise DenseElimination
        cnf.clause(row.values())
        for a,b in combinations(row,2):
            via=cnf.both(row[a],row[b])
            old=adjacency[a].get(b)
            edge=cnf.either([old,via]) if old else via
            adjacency[a][b]=adjacency[b][a]=edge
            cnf.work.counters['shortcut_pairs']+=1
        for a in row:del adjacency[a][v]
        remaining.remove(v)


def sat_boards(problem,domains=None,mode='eliminate',*,modulus=None,max_work=400000,
               conflicts=100000,retain=5,check=None):
    """Enumerate projected labelled boards. Conflicts/work limits give incomplete counts."""
    from pysat.solvers import Solver
    if mode not in ('lazy','eliminate','sparse','modular','reach'):raise ValueError('Unknown SAT comparison.')
    if type(conflicts)is not int or conflicts<1 or type(retain)is not int or retain<0:raise ValueError('Invalid SAT limits.')
    base=domain_bits(problem)
    ds=base if domains is None else tuple(domains)
    if len(ds)!=len(base) or any(type(v)is not int or v<0 or v&~b for v,b in zip(ds,base)):
        raise ValueError('Domains must restrict original choices.')
    w=Work(max_work,check)
    cnf=Formula(w)
    answers=[]
    count=0
    complete=False
    sat_stats={}
    try:
        variables,edges=_base(problem,ds,cnf,mode in ('eliminate','sparse','modular'))
        if mode=='modular':
            _modular(cnf,edges,problem.total_segments,modulus if modulus is not None else problem.total_segments)
        elif mode in ('eliminate','sparse'):
            try:
                _eliminate(cnf,edges,problem.total_segments,12 if mode=='sparse' else None)
            except DenseElimination:
                # Every prefix of the elimination encoding is necessary for
                # Gold. Keep its clauses and use checked lazy validation for
                # remaining connectivity, without replaying projected boards.
                w.counters['dense_elimination_fallback']+=1
        elif mode=='reach':
            # The released reachability encoding supplies the control on the same placement CNF.
            from constraint_sat_connectivity import add_reachability,SATConnectivityLimit
            try:
                add_reachability(problem,[list(row) for row in variables],variables,cnf.var,cnf.clause,
                                 check=(check or (lambda:None)))
            except SATConnectivityLimit:raise WorkLimit
        with Solver(name='g4',bootstrap_with=cnf.rows) as solver:
            while True:
                w.tick('sat_calls')
                if check:check()
                remaining=conflicts-solver.accum_stats()['conflicts']
                if remaining<=0:break
                solver.conf_budget(remaining)
                result=solver.solve_limited()
                sat_stats=solver.accum_stats()
                if result is None:break
                if result is False:
                    complete=True
                    break
                selected={x for x in solver.get_model() if x>0}
                codes=tuple(next(k for k,v in row.items() if v in selected) for row in variables)
                w.tick('projected_models')
                if problem.validate(list(codes))['valid']:
                    count+=1
                    if len(answers)<retain:answers.append(codes)
                solver.add_clause([-variables[c][v] for c,v in enumerate(codes)])
    except WorkLimit:pass
    return dict(complete=complete,solutions=count,witnesses=answers,work=dict(w.counters),
                sat=sat_stats,clauses=len(cnf.rows),units=w.used)

