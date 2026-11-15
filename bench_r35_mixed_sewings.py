"""Independently replay mixed-parent signed circuit counts and geographic-root exclusions."""
from collections import Counter,defaultdict
from itertools import combinations,product
from pathlib import Path
import argparse,json

from r28_exchange import read_json
from search_problem import SearchProblem
from bench_r35_near_reference_tour import raw_cycles
from r27_structure_filters import domain_bits
from r35_alternating_sewings import Family,common_phases,complete
from r35_mixed_sewings import mixed_solve


def labels(problem,edges,sides):
    adj=[[] for _ in range(problem.n)]
    for a,b in edges:
        m,x=sides[a]
        d=x^sides[b][1]^(m.bit_count()&1)
        adj[a//3].append((b//3,d))
        adj[b//3].append((a//3,d))
    phase={}
    for start in range(problem.n):
        if start in phase:continue
        phase[start]=0
        queue=[start]
        while queue:
            t=queue.pop()
            for u,d in adj[t]:
                need=phase[t]^d
                if u in phase:
                    if phase[u]!=need:return None
                else:
                    phase[u]=need
                    queue.append(u)
    return [phase[t] for t in range(problem.n)]


def local_geography(p,edges,domains,max_tiles=45,trial_limit=None,domain_first=False):
    """Enumerate assignments outward from each root, enforcing tile uniqueness and exact mates."""
    mate=defaultdict(set)
    for a,b in edges:mate[a].add(b);mate[b].add(a)
    trials=0
    capped=0
    root_start=0
    def visit(board,used,front):
        nonlocal trials,capped
        if trial_limit is not None and trials-root_start>=trial_limit:
            capped+=1
            return True
        trials+=1
        if len(board)>=max_tiles or not front:return True
        best=None
        for cell in sorted(front):
            candidates=None
            for s,(neighbor,j) in enumerate(p.board.neighbors[cell]):
                if neighbor not in board:continue
                tile,rot=divmod(board[neighbor],3)
                options={3*(m//3)+(s-m%3)%3 for m in mate[3*tile+(j-rot)%3]
                         if m//3 not in used}
                candidates=options if candidates is None else candidates&options
            options={v for v in candidates if domains[cell]&(1<<v)}
            if not options:return False
            priority=(len(options),distance[cell],cell) if domain_first else (distance[cell],len(options),cell)
            if best is None or priority<best[2]:best=cell,options,priority
        cell,options,_=best
        new_front=(front|{other for other,_ in p.board.neighbors[cell] if other>=0 and other not in board})-{cell}
        for code in options:
            board[cell]=code
            if visit(board,used|{code//3},new_front):return True
            del board[cell]
        return False
    survivors=[]
    for c,d in enumerate(domains):
        distance={c:0};ring=[c]
        while ring:
            nxt=[]
            for v in ring:
                for h,_ in p.board.neighbors[v]:
                    if h>=0 and h not in distance:distance[h]=distance[v]+1;nxt.append(h)
            ring=nxt
        for r in range(3):
            root_start=trials
            if d&(1<<r) and visit({c:r},{0},{b for b,_ in p.board.neighbors[c] if b>=0 and b!=c}):
                survivors.append((c,r))
    return dict(survivors=survivors,concrete_partial_assignments=trials,depth_limit=max_tiles,capped_roots=capped)


def union_sat(problem,edges,*,anchor=0):
    """Check each root in a geographic relaxation using SAT unit propagation on explicit poses."""
    from pysat.solvers import Solver
    p=problem
    width=3*p.n
    table=[{} for _ in range(3)]
    for tile in range(p.n):
        sides=[3*tile+j for j in range(3)]
        for rot in range(3):
            for side,dart in enumerate(sides):
                table[side][dart]=3*tile+rot
            sides=sides[-1:]+sides[:-1]
    external=defaultdict(set)
    for a,b in edges:external[a].add(b);external[b].add(a)
    def literal(c,v):return 1+c*width+v
    domains=[set(row) for row in p.domains]
    checked=rejected=supported=0
    with Solver(name='g42') as solver:
        clauses=0
        def add(clause):
            nonlocal clauses
            solver.add_clause(clause)
            clauses+=1
        for c,poses in enumerate(domains):
            add([literal(c,v) for v in poses])
            for root in range(3*anchor,3*anchor+3):
                if root in poses:
                    for other in poses:
                        if other!=root:add([-literal(c,root),-literal(c,other)])
            for s,(neighbor,j) in enumerate(p.board.neighbors[c]):
                if neighbor<0:continue
                for dart,code in table[s].items():
                    if code not in poses:continue
                    matches=[literal(neighbor,table[j][m]) for m in sorted(external[dart])
                             if table[j][m] in domains[neighbor]]
                    add([-literal(c,code),*matches])
        for c,poses in enumerate(domains):
            for root in range(3*anchor,3*anchor+3):
                if root not in poses:continue
                ok,_=solver.propagate(assumptions=[literal(c,root)])
                checked+=1
                if ok:supported+=1
                else:rejected+=1
    return dict(clauses=clauses,variables=width*p.n,checked_roots=checked,
                roots_rejected_by_unit_propagation=rejected,unresolved_roots=supported)


def allow_phase(f,phase):
    result={}
    for i,group in enumerate(f.groups+f.blanks):
        ok=[]
        for which,half in enumerate(group):
            if all(not (f.inputs.sides[a][1]^f.inputs.sides[b][1]^phase[a//3]^phase[b//3]^
                        (f.inputs.sides[a][0].bit_count()&1)) for a,b in half):ok.append(which)
        if not ok:return None
        if len(ok)==1:result[i]=ok[0]
    return result


def ring_donors(seed):
    """Two 10-triangle geometric Gold parents with one segment and two odd sides per tile."""
    from geometry import build_board
    from random import Random
    from r29_gold_phase import EndpointCuts
    from r31_geographic import physical_bonds
    from search_problem import SearchProblem
    board=build_board(1)
    rng=Random(seed)
    def hamilton(seq):
        if len(seq)==len(board.cells):
            return seq if any(other==seq[0] for other,_ in board.neighbors[seq[-1]]) else None
        steps=[other for other,_ in board.neighbors[seq[-1]] if other not in seq]
        rng.shuffle(steps)
        for other in steps:
            done=hamilton(seq+[other])
            if done:return done
        return None
    def absent(cycle):
        blank=[None]*len(cycle)
        for j,t in enumerate(cycle):
            adjacent={cycle[j-1],cycle[(j+1)%len(cycle)]}
            blank[t]=next(s for s,(other,_) in enumerate(board.neighbors[t]) if other not in adjacent)
        return blank
    home=absent(hamilton([0]))
    tiles=[{'id':t,'segments':[[[s,6] for s in range(3) if s!=b]]} for t,b in enumerate(home)]
    p=SearchProblem(tiles,board)
    away=absent(hamilton([0]))
    order=list(range(p.n))
    rng.shuffle(order)
    boards=[[3*t for t in range(p.n)],[3*t+(away[c]-home[t])%3 for c,t in enumerate(order)]]
    sides=EndpointCuts(p).sides
    parents=[]
    for codes in boards:
        assert p.validate(codes)['valid']
        pairs=physical_bonds(p,codes)
        phase=labels(p,pairs,sides)
        assert phase is not None
        parents.append(dict(pairs=pairs,phases=phase))
    return p,parents,boards


def original_sources():
    root=Path(__file__).resolve().parent/'docs'
    older=json.loads((root/'R35_CONNECTED_TOURS.json').read_bytes())
    near=json.loads((root/'R35_NEAR_TOUR_EXCHANGES.json').read_bytes())
    result={name:older[name] for name in ('raw','contract')}
    for i,item in enumerate(near['parents']):
        result['near'+str(i)]={**item,'phases':[int(near['phase'],0)>>t&1 for t in range(160)]}
    result['trade']=json.loads((root/'R35_EXCHANGE_PARENTS.json').read_bytes())[1]
    result['cp']=json.loads((root/'R35_ALTERNATING_SEWINGS.json').read_bytes())['fixed_phase_parent']
    return result


def finite(p,f,limit=8):
    all_groups=f.groups+f.blanks
    rows=[]
    for anchor in (0,1):
        flex=list(range(min(limit,len(all_groups))))
        pins={i:anchor for i in range(len(all_groups)) if i not in flex}
        signed=connected=0
        for values in product((0,1),repeat=len(flex)):
            values={**pins,**dict(zip(flex,values))}
            edges=set(f.fixed)
            for i,which in values.items():edges.update(all_groups[i][which])
            phase=labels(p,edges,f.inputs.sides)
            if phase is not None:
                signed+=1
                if raw_cycles(p,edges)==[p.total_segments]:connected+=1
        check=mixed_solve(f,fixed=pins)
        assert check['complete'] and check['connected_full']==connected
        rows.append(dict(anchor=anchor,total=1<<len(flex),signed=signed,connected=connected,nodes=check['work'].get('nodes',0)))
    return rows


def measure(tiles,limit=8):
    p=SearchProblem(read_json(tiles))
    donors=original_sources()
    result={}
    for (n1,a),(n2,b) in combinations(donors.items(),2):
        f=Family(p,a,b,None)
        old_fixed=[allow_phase(f,r['phases']) for r in (a,b)]
        edges=set(map(tuple,a['pairs']))|set(map(tuple,b['pairs']))
        g=mixed_solve(f,geography=True)
        independent=local_geography(p,edges,domain_bits(p))
        assert len(g['geographic_roots']['survivors'])==len(independent['survivors'])==0
        state=common_phases(p,a,b)[1]
        sample=finite(p,f,limit)
        if (n1,n2) in (('contract','cp'),('trade','cp')):
            # Geography for every child was already decided by both root tests.
            # Avoid treating an intentionally limited physical count as exact.
            limited=mixed_solve(f,max_nodes=100000)
            result[n1+'+'+n2]=dict(parent_union_phase=state,raw_children=1<<(len(f.groups)+len(f.blanks)),
                                   common_seams=len(f.fixed),geographic=g,independent_geographic=independent,
                                   independent_finite=sample,bounded_physical={k:limited[k] for k in ('connected_full','complete','work')})
            continue
        arms={}
        for name,memo,look,repeat in [('static_direct',False,0,False),('static_memo',True,0,False),
                                      *((f'window_{k}',True,k,False) for k in (1,2,4,8,16,48)),
                                      ('repeat_forcing',True,48,True)]:
            cap=400000 if not memo else 2000000
            x=mixed_solve(f,memo=memo,max_nodes=cap,lookahead=look,repeat_forcing=repeat)
            assert x['complete'] or not memo
            for example in x['examples']:
                mask=example['mask']
                mesh=f.pairs(mask&((1<<len(f.groups))-1),mask>>len(f.groups))
                complete(p,mesh,example['phases'],f.inputs)
                assert raw_cycles(p,mesh)==[p.total_segments] and labels(p,mesh,f.inputs.sides) is not None
            arms[name]={k:v for k,v in x.items() if k not in ('examples','geographic_roots')}
        assert len({r['connected_full'] for r in arms.values() if r['complete']})==1
        under=[]
        for pins in old_fixed:
            c=mixed_solve(f,fixed=pins) if pins is not None else {'complete':True,'connected_full':0}
            assert c['complete'];under.append(c['connected_full'])
        both=0
        if all(s is not None for s in old_fixed) and not any(i in old_fixed[1] and old_fixed[1][i]!=v for i,v in old_fixed[0].items()):
            match=mixed_solve(f,fixed=old_fixed[0]|old_fixed[1])
            assert match['complete'];both=match['connected_full']
        partition=[]
        for pins,excluded in ((old_fixed[0],()),(old_fixed[1],(a['phases'],)),({},(a['phases'],b['phases']))):
            if pins is None:
                partition.append({'connected_full':0,'work':{}})
                continue
            residual=mixed_solve(f,fixed=pins,exclude_phases=excluded)
            assert residual['complete']
            partition.append({k:residual[k] for k in ('connected_full','work')})
        assert sum(part['connected_full'] for part in partition)==arms['window_48']['connected_full']
        own=arms['window_48']['connected_full']-sum(under)+both
        assert own>=0
        short=n1+'+'+n2
        result[short]=dict(parent_union_phase=state,raw_children=1<<(len(f.groups)+len(f.blanks)),common_seams=len(f.fixed),
                    geographic=g,independent_geographic=independent,physical=arms,independent_finite=sample,
                    with_parent_phases=under,both_parent_phases=both,requiring_different_phase=own,disjoint_phase_scopes=partition)
        print(short,arms['window_48']['connected_full'],own,independent['concrete_partial_assignments'],flush=True)
    return dict(original_pairs=result,combined_unions=union_controls(p,donors))


def union_controls(p,donors):
    from r29_gold_phase import EndpointCuts
    from r35_mixed_sewings import donor_geography,geographic_roots
    from validator import reverse_mask
    contact=read_json(Path(__file__).resolve().parent/'docs/R35_MIXED_CONTACT_DONOR.json')
    assert contact['fingerprint']==p.fingerprint
    cphase=[int(contact['phase'],0)>>t&1 for t in range(p.n)]
    assert raw_cycles(p,contact['pairs'])==contact['circuits']
    switched=set(map(tuple,contact['pairs']))
    for removed,added in contact['connected_exchanges']:
        assert set(map(tuple,removed))<=switched
        switched.difference_update(map(tuple,removed))
        switched.update(map(tuple,added))
    assert raw_cycles(p,switched)==[p.total_segments]
    all_donors={**donors,'contact':dict(pairs=contact['pairs'],phases=cphase),
                'contact_switched':dict(pairs=sorted(switched),phases=cphase)}
    unions={}
    for label,keys in [('five_connected',[k for k in donors if k!='trade']),('all_six',list(donors)),('contact_extension',list(all_donors)[:-1]),('connected_contact_extension',list(all_donors))]:
        rows=[all_donors[k] for k in keys]
        edges=set().union(*(set(map(tuple,row['pairs'])) for row in rows))
        root=donor_geography(p,rows)
        independent=union_sat(p,edges)
        assert root['geographic_status']=='INFEASIBLE' and not independent['unresolved_roots']
        unions[label]=dict(donors=keys,physical_seams=len(edges),propagated=root['roots'],independent_cnf=independent)
    phases={tuple(rec['phases']) for rec in all_donors.values()}
    sides=EndpointCuts(p).sides
    unrestricted={}
    union=set()
    for phase in sorted(phases):
        edges={(a,b) for a,(m,col) in enumerate(sides) for b in range(a+1,len(sides))
               if a//3!=b//3 and reverse_mask(m)==sides[b][0] and
               phase[a//3]^phase[b//3]==col^sides[b][1]^(m.bit_count()&1)}
        root=geographic_roots(p,domain_bits(p),edges)
        key=hex(sum(bit<<i for i,bit in enumerate(phase)))
        unrestricted[key]=dict(physical_seams=len(edges),surviving_roots=len(root['survivors']))
        union.update(edges)
    root=geographic_roots(p,domain_bits(p),union)
    unrestricted['all_phase_mates']=dict(physical_seams=len(union),surviving_roots=len(root['survivors']))
    unions['outside_recorded_donors']=unrestricted
    return unions


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(measure(args.tiles),indent=2)+'\n',encoding='utf8')
