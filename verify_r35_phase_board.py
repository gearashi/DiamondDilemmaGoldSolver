"""Independent finite oracle for phase-anchor projection on the oriented bipyramid."""
import argparse
import json
from collections import Counter
from hashlib import sha256
from pathlib import Path
from random import Random
from unittest.mock import patch

from geometry import build_board
from search_problem import SearchProblem
from r35_phase_anchors import narrow
from r30_gold import search


ROOT = Path(__file__).resolve().parent

def specimen(seed, mid=False, extras=False):
    rng=Random(seed)
    board=build_board(1)
    mesh=board.neighbors
    # Discover a Hamiltonian cycle without using solver support or phase code.
    def cycle(path):
        if len(path)==10:
            return path if any(v==path[0] for v,_ in mesh[path[-1]]) else None
        for v,_ in mesh[path[-1]]:
            if v not in path:
                result=cycle([*path,v])
                if result:
                    return result
    path=cycle([0])
    assert path
    label=list(range(10))
    rng.shuffle(label)
    turns=[rng.randrange(3) for _ in label]
    pose=[3*t+r for t,r in zip(label,turns)]
    endpoints=[[] for _ in label]
    segments=[[] for _ in label]
    sewn=set()
    def local(cell,side):
        return (side-turns[cell])%3
    for i,c in enumerate(path):
        v=path[(i+1)%len(path)]
        side=next(k for k,(dst,_) in enumerate(mesh[c]) if dst==v)
        opp=mesh[c][side][1]
        slot=6 if mid else rng.choice((1,2,3,4,5,7,8,9,10,11))
        endpoints[c].append((local(c,side),slot))
        endpoints[v].append((local(v,opp),12-slot))
        sewn.add(tuple(sorted((c,v))))
    for c,end in enumerate(endpoints):
        segments[label[c]].append(tuple(end))
    if extras:
        for c,side,v,opp in board.edges:
            if tuple(sorted((c,v))) in sewn:
                continue
            a,b=2,9
            segments[label[c]].append(((local(c,side),a),(local(c,side),b)))
            segments[label[v]].append(((local(v,opp),12-b),(local(v,opp),12-a)))
    return SearchProblem([dict(id=i,segments=s) for i,s in enumerate(segments)],board),pose

def profiles(p):
    # Physical corner 0 starts dark. Independently enumerate all 12 open
    # intervals and toggle on encountering an endpoint along the boundary.
    tabs=[]
    for tile in p.tiles:
        endpoints={e for arc in tile['segments'] for e in arc}
        intervals=[]
        state=0
        for side in range(3):
            row=[]
            for k in range(12):
                if (side,k) in endpoints:
                    state^=1
                row.append(state)
            intervals.append(tuple(row))
        assert state==0
        tabs.append(intervals)
    return tabs

def certify(p,tab,board):
    graph=[[] for _ in board]
    bonds=set()
    for c,s,d,t in p.edges:
        a,r=divmod(board[c],3)
        b,q=divmod(board[d],3)
        j,k=(s-r)%3,(t-q)%3
        x,y=tab[a][j],tab[b][k][::-1]
        diff=x[0]^y[0]
        if any((u^v)!=diff for u,v in zip(x,y)):
            return None
        bonds.add(tuple(sorted((3*a+j,3*b+k))))
        graph[a].append((b,diff))
        graph[b].append((a,diff))
    phase={0:0}
    todo=[0]
    while todo:
        a=todo.pop()
        for b,v in graph[a]:
            value=phase[a]^v
            if b not in phase:
                phase[b]=value
                todo.append(b)
            elif phase[b]!=value:
                return None
    assert len(phase)==len(board)
    return tuple(phase[t] for t in range(len(board))),bonds

def enumerate_domains(domains):
    order=sorted(range(len(domains)),key=lambda i:domains[i].bit_count())
    board=[None]*len(domains)
    def place(i,used):
        if i==len(order):
            yield tuple(board)
            return
        c=order[i]
        d=domains[c]
        while d:
            bit=d&-d
            d^=bit
            code=bit.bit_length()-1
            t=1<<(code//3)
            if not used&t:
                board[c]=code
                yield from place(i+1,used|t)
    yield from place(0,0)

def turned_board(p,board,*,swap=True,shift=0):
    # Independent vertex action on triangular macrofaces.
    idx={frozenset(p.board.vertices[v][0][0] for v in c.vertices):c for c in p.board.cells}
    def spin(v):
        if v in (0,1):
            return 1-v if swap else v
        return 2+(shift+((2-v) if swap else (v-2)))%5
    out=[None]*len(board)
    for cell,code in zip(p.board.cells,board):
        vertices=[p.board.vertices[v][0][0] for v in cell.vertices]
        image=[spin(v) for v in vertices]
        target=idx[frozenset(image)]
        verts=[p.board.vertices[v][0][0] for v in target.vertices]
        delta=verts.index(image[0])
        assert all(verts[(i+delta)%3]==x for i,x in enumerate(image))
        out[target.index]=3*(code//3)+(code%3+delta)%3
    return out

def check_anchors():
    answer=Counter()
    sym=Counter()
    status=Counter()
    reports=[]
    all_checks=0
    for seed,mid,extras in [(81,False,False),(94,True,False),(21,False,True),(37,True,True)]:
        p,board=specimen(seed,mid,extras)
        tab=profiles(p)
        ev=certify(p,tab,board)
        assert ev
        phase,bonds=ev
        known=sorted(bonds)
        assert p.validate(board)['valid']==(not extras)
        swapped=turned_board(p,board)
        assert certify(p,tab,swapped)==ev
        flip=tuple((1<<a)|(1<<b) for a,b in zip(board,swapped))
        free=(0,2,3,8)
        mask=sum(7<<(3*(board[i]//3)) for i in free)
        asym=tuple(mask if i in free else 1<<v for i,v in enumerate(board))
        corrupt=list(asym)
        corrupt[1]=1<<(3*(board[1]//3)+(board[1]%3+1)%3)
        comps=tuple(v^1 for v in phase)
        rules=[({},(),()),(dict(enumerate(phase)),(),()),(dict(enumerate(comps)),(),()),
               ({k:phase[k] for k in (0,6,8)},known[2:5],()),({2:phase[2],4:1^phase[4]},(),known[1:2]),
               ({},known[:1],known[:1]),({},(known[0],(known[0][0],known[4][1])),()),
               ({i:phase[i] for i in range(1,8)},(),known[3:4])]
        for family,ds in [('half-turn',flip),('four-free',asym),('rotated-fixed',tuple(corrupt))]:
            raw=list(enumerate_domains(ds))
            data=[(b,certify(p,tab,b)) for b in raw]
            eligible=[(b,ph,bs) for b,x in data if x for ph,bs in [x]]
            answer['whole_boards']+=len(raw)
            answer['signed_edge_perfect']+=len(eligible)
            answer['gold_layouts']+=sum(p.validate(b)['valid'] for b,_,_ in eligible)
            for pins,req,forb in rules:
                gold=[b for b,ph,bs in eligible if set(req)<=bs and not bs.intersection(forb)
                      and any(all(ph[k]^g==v for k,v in pins.items()) for g in (0,1))]
                answer['positive_scope_cases' if gold else 'empty_scope_cases']+=1
                for proj,unique in ((False,False),(True,True),(True,False),(False,True)):
                    opts=dict(required=req,forbidden=forb,anchors=[8,2,1,0,9],unique=unique,project=proj)
                    a=narrow(p,ds,pins,symmetry=False,**opts)
                    b=narrow(p,ds,pins,symmetry=True,**opts)
                    assert (a['status'],a['domains'])==(b['status'],b['domains']),(seed,family,pins,req,forb,proj,unique,a,b)
                    for result in (a,b):
                        if gold:
                            assert result['status']=='SURVIVING'
                        assert all(all(d&(1<<c) for c,d in zip(layout,result['domains'])) for layout in gold)
                    sym[b['symmetries']]+=1
                    status[b['status']]+=1
                    answer['verified_individual_witnesses']+=2*len(gold)
                    all_checks+=2
            reports.append(dict(seed=seed,odd_midpoints=mid,extra_closed_pairs=extras,domain=family,boards=len(raw),edge_and_phase=len(eligible)))
        # Stop after a measured number of root polls, including a partially
        # explored split. Every preexisting witness must still be retained.
        for target in (0,1,2,5,11,19):
            calls=[0]
            def virtual():
                tick=calls[0]
                calls[0]+=1
                return int(tick>=target)
            with patch('r35_phase_anchors.monotonic',virtual):
                res=narrow(p,asym,{},anchors=[board[0]//3,board[1]//3,board[2]//3],deadline=0.5)
            assert all(d&(1<<c) for c,d in zip(board,res['domains']))
            if res['status']=='LIMIT' and not res['stats'].get('anchor_splits',0):
                assert res['domains']==asym
            answer['interrupted_with_completed_earlier_split' if res['status']=='LIMIT' and res['stats'].get('anchor_splits',0) else 'other_deadline_cases']+=1
    return dict(results=dict(answer),filter_calls=all_checks,stabilizer_histogram=dict(sym),statuses=dict(status),work=reports)

def small_sphere(n):
    """Two oppositely oriented triangles, or the tetrahedron boundary."""
    from types import SimpleNamespace
    faces = [(0,1,2),(0,2,1)] if n==2 else [(0,2,1),(0,1,3),(0,3,2),(1,2,3)]
    incident = {}
    edges = []
    for cell,face in enumerate(faces):
        for side,start in enumerate(face):
            end=face[(side+1)%3]
            if (end,start) in incident:
                other,opposite=incident.pop((end,start))
                edges.append((other,opposite,cell,side))
            else:
                incident[start,end]=(cell,side)
    assert not incident
    shape=SimpleNamespace(cells=tuple(range(n)),edges=tuple(edges))
    if n==2:
        left,ls,right,rs=edges[0]
        _,lt,_,rt=edges[1]
        tiles=[dict(id=0,segments=[((ls,2),(lt,5))]),dict(id=1,segments=[((rs,10),(rt,7))])]
    else:
        # A labelled four-cycle gives one physically possible loop.
        ports=[[] for _ in range(4)]
        for slot,(i,j) in enumerate(((0,1),(1,3),(3,2),(2,0)),1):
            a,side,b,opp=next(e for e in edges if {e[0],e[2]}=={i,j})
            ports[a].append((side,slot))
            ports[b].append((opp,12-slot))
        tiles=[dict(id=i,segments=[x]) for i,x in enumerate(ports)]
    return SearchProblem(tiles,shape)


def raw_boards(p,phase,domains):
    """Join independently painted boundary intervals and enforce one copy per tile."""
    local=profiles(p)
    colors={}
    for code in range(3*p.n):
        tile,rotation=divmod(code,3)
        phase_bit=(phase>>tile)&1
        for side in range(3):
            colors[code,side]=tuple(v^phase_bit for v in local[tile][(side-rotation)%3])
    chosen=[-1]*p.n
    result=set()
    order=[]
    while len(order)<p.n:
        cell=min((i for i in range(p.n) if i not in order),key=lambda i:(
            -sum(int(v) in order for v in p.neighbors[i]),domains[i].bit_count(),i))
        order.append(cell)
    def extend(pos,used):
        if pos==p.n:
            result.add(tuple(chosen))
            return
        cell=order[pos]
        mask=domains[cell]
        while mask:
            bit=mask&-mask
            mask^=bit
            code=bit.bit_length()-1
            tile=1<<(code//3)
            if used&tile:
                continue
            for side,(other,opposite) in enumerate(zip(p.neighbors[cell],p.sides[cell])):
                if other>=0 and chosen[other]>=0 and colors[code,side]!=colors[chosen[other],int(opposite)][::-1]:
                    break
            else:
                chosen[cell]=code
                extend(pos+1,used|tile)
                chosen[cell]=-1
    extend(0,0)
    return result


def formula_models(p,phase,domains):
    """Reload exported DIMACS and enumerate distinct public placement projections."""
    from tempfile import TemporaryDirectory
    from pysat.formula import CNF
    from pysat.solvers import Solver
    from bench_r35_phase_board import encode_board,export_board
    emitted=[]
    n,poses=encode_board(p,phase,domains,lambda lits:emitted.append(list(lits)))
    with TemporaryDirectory() as folder:
        target=Path(folder)/'board.cnf'
        export_board(p,phase,domains,target)
        read=CNF(from_file=str(target))
        header=target.read_bytes().splitlines()[0].decode('ascii')
    assert header==f'p cnf {n} {len(emitted)}' and read.clauses==emitted
    assert len(poses)==p.n and len({v for row in poses for v in row.values()})==sum(map(len,poses))
    result=set()
    with Solver(name='g42',bootstrap_with=read.clauses) as solver:
        while solver.solve():
            values=set(solver.get_model())
            selected=[[(code,var) for code,var in row.items() if var in values] for row in poses]
            assert all(len(row)==1 for row in selected)
            board=tuple(row[0][0] for row in selected)
            assert board not in result
            result.add(board)
            solver.add_clause([-row[0][1] for row in selected])
    return result


def check_board_encoding():
    from constraint_symmetry import canonical_root_domains
    counts=Counter()
    def compare(p,phase,domains):
        expected=raw_boards(p,phase,domains)
        actual=formula_models(p,phase,domains)
        assert actual==expected,(p.n,phase,domains,len(actual),len(expected))
        counts['formulas']+=1
        counts['positive_formulas' if actual else 'empty_formulas']+=1
        counts['checked_projected_witnesses']+=len(actual)
        return actual
    cases=[]
    for n in (2,4):
        p=small_sphere(n)
        full=(1<<(3*n))-1
        identity=tuple(1<<(3*t) for t in range(n))
        domains=[(full,)*n,identity,(0,*([full]*(n-1))),tuple(full^7 for _ in range(n))]
        domains.append((1,1,*([full]*(n-2))))
        for domain in domains:
            seen={}
            for phase in range(1<<n):
                boards=compare(p,phase,tuple(domain))
                if domain==(full,)*n:
                    cases.append(dict(tiles=n,phase=phase,boards=len(boards)))
                opposite=((1<<n)-1)^phase
                if opposite in seen:
                    assert boards==seen[opposite]
                    counts['complement_equalities']+=1
                seen[phase]=boards
    orbit_checks=[]
    for seed,extras in ((81,False),(21,True)):
        p,planted=specimen(seed,False,extras)
        phase,_=certify(p,profiles(p),planted)
        word=sum(x<<i for i,x in enumerate(phase))
        full=((1<<(3*p.n))-1,)*p.n
        for signed in (word,word^1023):
            base=compare(p,signed,full)
            assert tuple(planted) in base
            assert all(p.validate(b)['valid']==(not extras) for b in base)
            for anchor_tile,face in ((0,0),(4,6)):
                domains,meta=canonical_root_domains(p,anchor_tile=anchor_tile,face=face)
                assert meta['applied']
                fixed=compare(p,signed,tuple(domains))
                lifted=set()
                for b in fixed:
                    orbit={tuple(turned_board(p,b,swap=swap,shift=shift)) for swap in (False,True) for shift in range(5)}
                    assert len(orbit)==10 and not (lifted&orbit)
                    assert sum(1 for v in orbit if v[face]//3==anchor_tile)==1
                    lifted.update(orbit)
                assert lifted==base
                orbit_checks.append(dict(seed=seed,multiple_loops=extras,phase=hex(signed),tile=anchor_tile,face=face,
                                         full_poses=len(base),quotient_poses=len(fixed)))
            # A complement pair and a nearby inconsistent phase test fixed domains separately.
            wrong=signed^1
            assert not compare(p,wrong,tuple(1<<v for v in planted))
    from bench_r35_phase_board import encode_board
    invalid=small_sphere(2)
    for domains in ((-1,63),(64,63),(63,),(63,63,63),(True,63)):
        try:
            encode_board(invalid,0,domains,lambda _:None)
        except ValueError:
            counts['invalid_domains_rejected']+=1
        else:
            raise AssertionError(f'Invalid domains accepted: {domains}')
    return dict(counts=dict(counts),small_unrestricted=cases,macroface_orbits=orbit_checks)


def public_search_checks():
    result=Counter()
    for mid,extra in ((False,False),(True,False),(False,True)):
        p,board=specimen(37,mid,extra)
        ds=[1<<c for c in board]
        phase,bonds=certify(p,profiles(p),board)
        for flip in (0,1):
            pins={i:v^flip for i,v in enumerate(phase)}
            for mode,physical in ((None,True),('shared',True),('roots',False)):
                actual=search(p,domains=ds,pins=pins,bonds=sorted(bonds)[:2],geography=True,
                              seconds=10,phase_anchors=mode,physical_support=physical)
                assert actual['status']==('INFEASIBLE' if extra else 'GOLD')
                result[actual['status']]+=1
    p,board=specimen(21,False,True)
    data=[dict(t,segments=list(t['segments'])) for t in p.tiles]
    i,j,k,l=sorted(x for seg in data[0]['segments'] for x in seg)
    data[0]['segments']=[(i,k),(j,l)]
    crossing=SearchProblem(data,p.board)
    for q,opt in ((p,{'neighbor_mode':'typo'}),(p,{'tour_mode':'absent'}),(p,{'corner_mode':'nonsense'}),(crossing,{})):
        errors=[]
        for mode in (None,'shared'):
            try:
                search(q,domains=[1<<v for v in board],bonds=[(0,3)],forbidden=[(0,3)],geography=True,phase_anchors=mode,**opt)
            except ValueError as exc:
                errors.append(str(exc))
        assert len(errors)==2 and errors[0]==errors[1]
        result['same_input_error']+=1
    return dict(result)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--checks',choices=('all','anchors','board','public'),default='all')
    cli.add_argument('--output',type=Path)
    args=cli.parse_args()
    if not __debug__:
        raise RuntimeError('Run this auditor without -O so every assertion is checked.')
    result={}
    for label,run in [('anchors',check_anchors),('board',check_board_encoding),('public',public_search_checks)]:
        if args.checks in ('all',label):
            result[label]=run()
    deps=('bench_r35_phase_board.py','r35_phase_anchors.py','r30_gold.py','r35_mixed_sewings.py','r35_loss_witness.py',
          'r35_signed_mate_support.py','constraint_symmetry.py','verify_r35_phase_board.py')
    result['source_sha256_lf']={f:sha256((ROOT/f).read_text(encoding='utf8').encode()).hexdigest() for f in deps}
    report=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.write_text(report,encoding='utf8')
    print(report)


if __name__=='__main__':
    main()
