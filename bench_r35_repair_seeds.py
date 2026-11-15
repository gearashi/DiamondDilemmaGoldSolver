"""Replay named-pose repair bounds with independent literal-level constraints and exact public repairs."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import argparse,json

from r28_exchange import read_json
from search_problem import SearchProblem
from validator import reverse_mask


def independently_allowed(p):
    from bench_r35_loss_proof import closes_proper_loop
    from pysat.solvers import Solver
    from r32_gold_bonds import inventory
    sides=[int(p.masks[3*(i//3),i%3]) for i in range(3*p.n)]
    raw={(a,b) for a,m in enumerate(sides) for b in range(a+1,len(sides))
         if a//3!=b//3 and reverse_mask(m)==sides[b]}
    # Independently replay each proposed individual mate removal. Unknown
    # edges stay available in this verification model.
    hint=inventory(p)
    sealed={e for e in hint.forbidden if closes_proper_loop(p.tiles,{e})}
    surviving=raw-sealed
    atoms={e:k+1 for k,e in enumerate(sorted(surviving))}
    incident=[[] for _ in sides]
    for (a,b),lit in atoms.items():incident[a].append(lit);incident[b].append(lit)
    impossible=set();required=set()
    with Solver(name='g42') as solver:
        for row in incident:
            solver.add_clause(row)
            for a,b in combinations(row,2):solver.add_clause([-a,-b])
        for e in hint.forbidden-sealed:
            if e in atoms and not solver.solve(assumptions=[atoms[e]]):impossible.add(e)
        for e in hint.required:
            if e in atoms and not solver.solve(assumptions=[-atoms[e]]):required.add(e)
    return surviving-impossible,dict(reciprocal=len(raw),sealed=len(sealed),perfect_matching_exclusions=len(impossible),
                                     verified_required=sorted(required))


class PoseOracle:
    """A fresh geographic SAT relaxation with explicit independently verified physical partners."""
    def __init__(self,problem,pairs):
        from pysat.solvers import Solver
        self.sat=Solver(name='g42')
        self.p=problem
        w=3*problem.n
        self.width=w
        self.top=problem.n*w
        self.clauses=0
        self.ds=[set(r) for r in problem.domains]
        table=[{} for _ in range(3)]
        for t in range(problem.n):
            sides=[3*t+i for i in range(3)]
            for rot in range(3):
                for j,dart in enumerate(sides):table[j][dart]=3*t+rot
                sides=[sides[-1],*sides[:-1]]
        allowed=defaultdict(set)
        for a,b in pairs:allowed[a].add(b);allowed[b].add(a)
        bytile=[[] for _ in range(problem.n)]
        for c,poses in enumerate(self.ds):
            self.exact([self.lit(c,v) for v in sorted(poses)])
            for v in sorted(poses):bytile[v//3].append(self.lit(c,v))
            for s,(b,j) in enumerate(zip(problem.neighbors[c],problem.sides[c])):
                b,j=int(b),int(j)
                if b<0:continue
                for dart,v in table[s].items():
                    if v in poses:
                        self.add([-self.lit(c,v),*(self.lit(b,table[j][m]) for m in sorted(allowed[dart])
                                                   if table[j][m] in self.ds[b])])
        for row in bytile:self.exact(row)
    def lit(self,c,v):return self.width*c+v+1
    def add(self,clause):
        self.sat.add_clause(clause)
        self.clauses+=1
    def exact(self,row):
        self.add(row)
        prev=None
        for idx,lit in enumerate(row):
            if prev:self.add([-prev,-lit])
            if idx+1<len(row):
                self.top+=1
                self.add([-lit,self.top])
                if prev:self.add([-prev,self.top])
                prev=self.top
    def replay(self,seed,row):
        if any(seed[c] not in self.ds[c] for c in row):return True
        return not self.sat.propagate(assumptions=[self.lit(c,seed[c]) for c in sorted(row)])[0]
    def close(self):self.sat.delete()


def minimum(problem,rows,seconds=30):
    from constraint_lns import _Control
    from constraint_repair_round24 import _minimum_hitting_set
    from ortools.sat.python import cp_model
    cells=[c for c in range(problem.n) if c not in problem.fixed]
    selected,work=_minimum_hitting_set(rows,cells,_Control(seconds,None),cp_model)
    if selected is not None and not all(selected&r for r in rows):
        raise ValueError('Hitting set misses a required change.')
    return dict(minimum=None if selected is None else len(selected),selection=None if selected is None else sorted(selected),branches=work)


def verify_rows(p,seed,rows,oracle):
    from constraint_knowledge import revalidate_witness
    units=[];closed=[];rejected=[]
    for row in rows:
        if oracle.replay(seed,row):units.append(row)
        else:
            try:
                proof=revalidate_witness(p,[(c,seed[c]) for c in sorted(row)])
            except ValueError:
                rejected.append(row)
            else:
                assert {c for c,_ in proof}<=row
                closed.append(row)
    checked=units+closed
    return dict(input_rows=len(rows),proved_by_units=len(units),proved_by_closed_circuit=len(closed),
                unresolved=len(rejected),unresolved_rows=[sorted(r) for r in rejected],checked_minimum=minimum(p,checked))

def write_cap88(n,rows,path):
    """Write the sorted change clauses and reproducible PySAT totalizer for at most 88 changes."""
    from pysat.card import CardEnc,EncType
    enc=CardEnc.atmost(lits=list(range(1,n+1)),bound=88,encoding=EncType.totalizer)
    rows=sorted(rows,key=lambda r:(len(r),sorted(r)))
    with Path(path).open('w',encoding='ascii',newline='\n') as out:
        out.write(f'p cnf {enc.nv} {len(rows)+len(enc.clauses)}\n')
        for r in rows:out.write(' '.join(str(v+1) for v in sorted(r))+' 0\n')
        for r in enc.clauses:out.write(' '.join(map(str,r))+' 0\n')



if __name__=='__main__':
    from r35_repair_seeds import seed_rows
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    cli.add_argument('--row-file',type=Path)
    cli.add_argument('--catalog',action='store_true',help='Independently replay every checked original placement row.')
    cli.add_argument('--write-cap88',type=Path,help='Write the original catalog cardinality refutation input.')
    cli.add_argument('--radius',type=int,default=2)
    cli.add_argument('--width',type=int,default=40)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    problem=SearchProblem(read_json(args.tiles));seed=read_json(Path(__file__).resolve().parent/'docs/R29_INCUMBENT.json')['codes']
    if args.catalog:
        from r35_repair_seeds import checked_catalog
        catalog=checked_catalog(problem)
        if catalog is None or catalog['codes']!=seed:raise ValueError('Unknown original repair catalog.')
        rows=[frozenset(row) for row in catalog['rows']]
    elif args.row_file:rows=[frozenset(r) for r in read_json(args.row_file)['rows']]
    else:rows,_=seed_rows(problem,seed,mode='compact',anchor_radius=args.radius,max_domain=args.width)
    physical,verified=independently_allowed(problem)
    verifier=PoseOracle(problem,physical)
    try:
        data=verify_rows(problem,seed,rows,verifier)
        data.update(physical=verified,variables=verifier.top,clauses=verifier.clauses,puzzle=problem.fingerprint)
        if args.write_cap88:
            if not args.catalog or data['unresolved']:raise ValueError('A fully verified published catalog is required.')
            write_cap88(problem.n,rows,args.write_cap88)
    finally:verifier.close()
    args.output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    print(data['input_rows'],data['proved_by_units'],data['checked_minimum']['minimum'],flush=True)
