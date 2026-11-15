"""Bench the variable-phase geographic board with signed physical poses or mesh-vertex colors."""
import argparse
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from time import monotonic

from pysat.card import CardEnc, EncType
from pysat.formula import IDPool
from pysat.solvers import Solver

from constraint_symmetry import canonical_root_domains
from io_utils import atomic_json
from r27_structure_filters import bits, domain_bits
from r28_exchange import read_json
from r29_gold_geography import checked_domains
from r29_gold_phase import checked_pins
from r35_loss_witness import raw_sides
from search_problem import SearchProblem

TARGET=int("d237aba7e91101e88041413e8c589902021e5126",16)


def encode(problem, fixed, domains, emit, mode='labels'):
    """fixed is a partial tile->bit map. All other physical signs remain free."""
    if mode not in ('vertices','labels'):
        raise ValueError('Unknown sign model.')
    fixed=checked_pins(problem.n,fixed)
    domains=checked_domains(problem,domains)
    side_profiles=raw_sides(problem.tiles)
    pool=IDPool(start_from=problem.n+1)
    by_tile=[[] for _ in problem.tiles]
    if mode=='vertices':
        # Only use identified vertex colors on the verified oriented mesh.
        from constraint_symmetry import proper_rotations
        proper_rotations(problem.board)
        if (tuple(map(tuple,problem.neighbors))!=problem.board.neighbor_cells or
                tuple(map(tuple,problem.sides))!=problem.board.neighbor_sides):
            raise ValueError('Canonical mesh adjacency required.')
        vertex_vars=[pool.id() for _ in problem.board.vertices]
    signs={t:t+1 for t in range(problem.n)}
    for tile,value in sorted(fixed.items()):
        emit([signs[tile] if value else -signs[tile]])

    def exactly_one(choices):
        emit(choices)
        if len(choices)>1:
            for clause in CardEnc.atmost(choices,1,vpool=pool,encoding=EncType.seqcounter).clauses:
                emit(clause)

    cells=[]
    for c,mask in enumerate(domains):
        row=[]
        for code in bits(mask):
            tile,turn=divmod(code,3)
            values=(fixed[tile],) if tile in fixed else ((0,1) if mode=='labels' else (None,))
            for value in values:
                pose=pool.id()
                row.append((code,value,pose))
                by_tile[tile].append(pose)
                if mode=='labels':
                    if tile not in fixed:
                        emit([-pose,signs[tile] if value else -signs[tile]])
                else:
                    for s,vertex in enumerate(problem.board.cells[c].vertices):
                        _,offset=side_profiles[3*tile+(s-turn)%3]
                        lit=vertex_vars[vertex] if offset==0 else -vertex_vars[vertex]
                        if tile in fixed:
                            emit([-pose,lit if fixed[tile] else -lit])
                        else:
                            emit([-pose,-signs[tile],lit])
                            emit([-pose,signs[tile],-lit])
        cells.append(row)
        exactly_one([v for _,_,v in row])
    for row in by_tile:
        exactly_one(row)
    for a,s,b,t in problem.edges:
        groups=[]
        for cell,side,reverse in ((a,s,False),(b,t,True)):
            labels=defaultdict(list)
            for code,phase,var in cells[cell]:
                tile,turn=divmod(code,3)
                slots,corner=side_profiles[3*tile+(side-turn)%3]
                if reverse:
                    slots=tuple(12-i for i in reversed(slots))
                    corner^=len(slots)%2
                label=(slots,corner^phase) if mode=='labels' else slots
                labels[label].append(var)
            groups.append(labels)
        left,right=groups
        for label in sorted(left.keys()|right.keys()):
            if label not in left or label not in right:
                for v in left.get(label,()) or right.get(label,()):
                    emit([-v])
                continue
            match=pool.id()
            for row in (left[label],right[label]):
                emit([-match,*row])
                for v in row:
                    emit([-v,match])
    return signs,cells,pool.top


def balanced_projections(p,fixed):
    from r35_loss_witness import reciprocal
    free=[t for t in range(p.n) if t not in fixed]
    if len(free)>16: raise ValueError('Enumerated signed-inventory projection limited to 16 free tiles.')
    raw=raw_sides(p.tiles)
    keys={}
    def factors(tile,value):
        return Counter((slots,base^value) for slots,base in raw[3*tile:3*tile+3])
    base=Counter()
    for t,v in fixed.items():base.update(factors(t,v))
    for t in free:
        for v in (0,1):keys[t,v]=factors(t,v)
    rows=[]
    for values in product((0,1),repeat=len(free)):
        tally=base.copy()
        for t,v in zip(free,values):tally.update(keys[t,v])
        if all((n%2==0 if reciprocal(k)==k else n==tally[reciprocal(k)]) for k,n in tally.items()):
            rows.append(values)
    return free,rows


def add_balanced_projection(p,fixed,emit):
    free,rows=balanced_projections(p,fixed)
    if not rows:
        emit([])
        return dict(eligible=0,clauses=1)
    clauses=0
    # Split only while the current prefix has at least one completion.
    def descend(prefix,sub):
        nonlocal clauses
        k=len(prefix)
        if not sub:
            emit([-(t+1) if v else t+1 for t,v in zip(free,prefix)])
            clauses+=1
            return
        if k==len(free):return
        for v in (0,1):descend((*prefix,v),[r for r in sub if r[k]==v])
    descend((),rows)
    return dict(eligible=len(rows),clauses=clauses,assignments=[dict(zip(free,row)) for row in rows])


def export_formula(p,fixed,domains,mode,path,*,inventory_cut=False):
    from tempfile import TemporaryFile
    from shutil import copyfileobj
    count=0
    with TemporaryFile() as stream:
        def emit(row):
            nonlocal count
            stream.write((' '.join(map(str,row))+' 0\n').encode('ascii'))
            count+=1
        _,_,top=encode(p,fixed,domains,emit,mode)
        projection=add_balanced_projection(p,fixed,emit) if inventory_cut else None
        stream.seek(0)
        with path.open('wb') as out:
            out.write(f'p cnf {top} {count}\n'.encode('ascii'))
            copyfileobj(stream,out)
    return dict(variables=top,clauses=count,inventory=projection)


def controls():
    from verify_r35_phase_board import profiles,certify,specimen,raw_boards
    work=Counter()
    for extras in (False,True):
        p,source=specimen(89,False,extras)
        source_phase,_=certify(p,profiles(p),source)
        word=sum(v<<i for i,v in enumerate(source_phase))
        full=domain_bits(p)
        free=(0,2,4,8)
        for reverse in (0,1):
            expected=word^(((1<<p.n)-1) if reverse else 0)
            fixed={t:(expected>>t)&1 for t in range(p.n) if t not in free}
            for ds in (full,canonical_root_domains(p,anchor_tile=6)[0]):
                expected_boards=set()
                for values in product((0,1),repeat=len(free)):
                    ph=sum(v<<t for t,v in [*fixed.items(),*zip(free,values)])
                    expected_boards.update((ph,b) for b in raw_boards(p,ph,ds))
                assert expected_boards
                for mode in ('vertices','labels'):
                    with Solver(name='cadical195') as sat:
                        signs,cells,_=encode(p,fixed,ds,sat.add_clause,mode)
                        actual=set()
                        while sat.solve():
                            model=set(sat.get_model())
                            phase=sum(1<<t for t,s in signs.items() if s in model)
                            selected=[next((code,v) for code,_,v in row if v in model) for row in cells]
                            board=tuple(code for code,_ in selected)
                            assert p.validate(board)['valid']==(not extras)
                            assert (phase,board) not in actual
                            actual.add((phase,board))
                            sat.add_clause([-v for _,v in selected]+[-signs[t] if phase>>t&1 else signs[t] for t in free])
                    assert expected_boards==actual
                    work['cases']+=1
                    work['checked_phase_boards']+=len(actual)
    return dict(work)


def run_case(p,args,fixed,queries,domains):
    scope=dict(fingerprint=p.fingerprint,hard_pins={str(k):v for k,v in fixed.items()},
               queries=[],mode=args.mode,solver=args.solver,anchor=args.anchor,controls=args.controls,started=True)
    def save(): atomic_json(args.output,scope)
    save()
    with Solver(name=args.solver) as sat:
        emitted=0
        def emit(clause):
            nonlocal emitted
            sat.add_clause(clause)
            emitted+=1
        start=monotonic()
        signs,cells,n=encode(p,fixed,domains,emit,args.mode)
        if args.inventory:scope['inventory']=add_balanced_projection(p,fixed,emit)
        scope.update(build_seconds=monotonic()-start,clauses=emitted,variables=n)
        save()
        for label,pins in queries:
            before=sat.accum_stats()
            sat.conf_budget(args.conflicts)
            assumptions=[signs[t] if v else -signs[t] for t,v in sorted(pins.items()) if t not in fixed]
            start=monotonic()
            result=sat.solve_limited(assumptions=assumptions)
            stats={k:v-before.get(k,0) for k,v in sat.accum_stats().items()}
            row=dict(label=label,pins=len(pins),status={None:'UNKNOWN',False:'UNSAT',True:'SAT'}[result],
                     seconds=monotonic()-start,budget=args.conflicts,work=stats)
            if result is False:
                core=sat.get_core() or []
                row['explicit_pins']={str(abs(lit)-1):int(lit>0) for lit in core}
                row['total_core_literals']=len(fixed)+len(core)
            elif result is True:
                model=set(sat.get_model())
                board=[next(code for code,_,v in group if v in model) for group in cells]
                from verify_r35_phase_board import profiles,certify
                phase=sum(1<<t for t,v in signs.items() if v in model)
                raw=certify(p,profiles(p),board)
                assert raw is not None and all((raw[0][t]^((phase>>t)&1))==(phase&1) for t in range(p.n))
                row.update(phase=hex(phase),validation=p.validate(board),board=board)
            scope['queries'].append(row)
            save()
            print(label,row['status'],row.get('total_core_literals'),stats['conflicts'],round(row['seconds'],2),flush=True)
    return scope


def main():
    c=argparse.ArgumentParser(description=__doc__)
    c.add_argument('--controls',action='store_true')
    c.add_argument('--tiles',type=Path)
    c.add_argument('--output',type=Path,required=True)
    c.add_argument('--mode',choices=('labels','vertices'),default='labels')
    c.add_argument('--anchor',type=int,default=25)
    c.add_argument('--inventory',action='store_true')
    c.add_argument('--formula',type=Path)
    c.add_argument('--phase',type=lambda s:int(s,0),default=TARGET)
    c.add_argument('--variable',help='comma-separated physical tiles, or all',default='all')
    c.add_argument('--release',action='append',default=[],help='comma-separated free signs in a query')
    c.add_argument('--skip-full',action='store_true')
    c.add_argument('--conflicts',type=int,default=15000)
    c.add_argument('--solver',default='cadical195')
    args=c.parse_args()
    if args.controls:
        result=controls()
        atomic_json(args.output,result)
        print(result)
        return
    if args.tiles is None:
        c.error('--tiles is required outside the planted controls.')
    if args.conflicts < 1:
        c.error('--conflicts must be positive.')
    p=SearchProblem(read_json(args.tiles))
    if not 0 <= args.phase < 1 << p.n:
        c.error('The reference phase is outside the tile range.')
    var=set(range(p.n)) if args.variable=='all' else {int(s) for s in args.variable.split(',') if s}
    if not var <= set(range(p.n)):
        c.error('A variable physical tile index is out of range.')
    full={t:(args.phase>>t)&1 for t in range(p.n)}
    fixed={t:v for t,v in full.items() if t not in var}
    queries=[] if args.skip_full else [('full',full)]
    for value in args.release:
        free={int(s) for s in value.split(',') if s}
        if not free<=var: raise ValueError('Cannot release a hard-coded sign.')
        queries.append(('free-'+value,{t:v for t,v in full.items() if t not in free}))
    ds,meta=canonical_root_domains(p,anchor_tile=args.anchor)
    assert meta['applied']
    if args.formula:
        report=export_formula(p,fixed,ds,args.mode,args.formula,inventory_cut=args.inventory)
        atomic_json(args.output,report)
    else:
        run_case(p,args,fixed,queries,ds)


if __name__=='__main__':
    main()
