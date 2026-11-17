"""Repeat the Round 34 repair experiments on the installed puzzle, using isolated output."""
import argparse
from collections import Counter
import json
from pathlib import Path

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,read_json
from r29_exchange_route import physical_hash
from r31_fresh import Engine
from r32_route_pool import CoverCores,defect_cover
from search_problem import SearchProblem


def cover(problem, output):
    model=BoardModel(problem)
    centers=read_json('docs/R29_EXCHANGE_CENTERS.json')['centers']
    samples=read_json('docs/R32_ROUTE_FOLLOWUPS.json')['defects']
    cache=CoverCores()
    records=[]
    for center in centers:
        b=ExchangeBound(model,[3*x for x in center['pieces']])
        for row in (r for r in samples if r['center']==physical_hash(b.tiles)):
            for pool,movers,target in [(model.free,3,208),(tuple(range(80)),3,208),(model.free,2,209)]:
                args=b,row['prefix'],pool,movers,target
                control=defect_cover(*args,seconds=4)
                trial=defect_cover(*args,seconds=4,cores=cache)
                retried=False
                for seconds in (30,240):
                    if control['status']=='UNKNOWN':
                        control=defect_cover(*args,seconds=seconds)
                        retried=True
                    if trial['status']=='UNKNOWN':
                        trial=defect_cover(*args,seconds=seconds,cores=cache)
                        retried=True
                if 'UNKNOWN' not in (control['status'],trial['status']):
                    assert control['closed']==trial['closed']
                records.append(dict(center=center['sha256'],prefix=row['prefix'],eligible=len(pool),movers=movers,
                                    target=target,cold=control['status'],trial=trial['status'],retried=retried))
                atomic_json(output,dict(work=dict(cache.work),records=records))


def trajectories(problem,output,rounds):
    configurations={'baseline':{},'two':{'contacts':2},'unrestricted':{'contacts':-1},
                    'packing':{'exact_packing':True},'pairs':{'pair_cycles':3},
                    'forest':{'contacts':-1,'forest':True}}
    records=[]
    for seed in (340,341,342,343):
        for name,configuration in configurations.items():
            engine=Engine(problem,bonds=True,joint=True,cycle_budget=3,pool=10,**configuration)
            row=engine.run(seed,rounds=rounds)
            records.append(dict(name=name,seed=seed,rounds=rounds,initial=row['initial'],score=row['score'],
                                gold=row['gold'],work=row['work']))
            atomic_json(output,records)


def centers(problem,output,archives):
    from r34_centers import archive_boards,search
    boards=list(archive_boards(archives))
    records={}
    for policy in ('score','defects','novel'):
        db=output.with_name(output.stem+'-'+policy+'.sqlite')
        for target,limit in ((208,3),(198,1),(198,3)):
            result=search(problem,boards,db,family=(2,2,3),target=target,limit=limit,
                          policy=policy,seed=34,page=32,start=0,stop=64)
            records[f'{policy}-{target}-{limit}']=result
            atomic_json(output,records)


def predecessor_pages(problem,output,database):
    from contextlib import closing
    import random
    import sqlite3
    from r28_exchange import combination_batch,domain_binding
    from r30_cycles import canonical_cycles
    from r31_route_coverage import inherited_end
    from r32_route_store import colex,PageProofs
    from r34_cycle_orders import CycleOrders
    from r34_exchange_undo import CoveredUndo
    model=BoardModel(problem)
    bound=ExchangeBound(model,read_json('docs/R29_INCUMBENT.json')['codes'])
    owner=physical_hash(bound.tiles)
    scope=domain_binding(model)
    indices={int(c):i for i,c in enumerate(model.free)}
    rng=random.Random(1134)
    report={}
    with closing(sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True)) as db:
        proof=PageProofs(db,208)
        for family in [(2,2,3),(2,2,2,2)]:
            options=CycleOrders(family)
            def covered(g):
                f=tuple(sorted(map(len,g)))
                if f!=family:
                    return False
                rank=colex({c for cy in g for c in cy},indices)
                return inherited_end(db,'probe',scope,(f,),208,[owner],rank,rank+1,proofs=proof)>rank
            undo=CoveredUndo(bound,covered)
            intervals=db.execute("""SELECT p.lo,p.hi FROM r32_page_proof p JOIN r31_members m
                ON m.scope=p.scope AND m.owner=p.owner WHERE m.domain=? AND m.owner=? AND m.family=?
                ORDER BY p.lo""",(scope,owner,json.dumps(family))).fetchall()
            stats=Counter()
            for _ in range(256):
                lo,hi=rng.choice(intervals)
                rank=rng.randrange(lo,hi)
                cells=list(map(int,combination_batch(tuple(indices),sum(family),rank,1)[0]))
                group=canonical_cycles(cells,options[rng.randrange(len(options))])
                pair=tuple(rng.sample(sorted(set(indices)-set(cells)),2))
                stats['prior_proof']+=covered(group)
                stats['excluded_extensions']+=undo.rejects((*group,pair))
            report[str(family)]=dict(intervals=len(intervals),**stats)
    atomic_json(output,report)


def bands(problem,output):
    from r34_seam_master import SeamMaster
    from r30_exact_routes import CycleModel
    model=BoardModel(problem)
    board=read_json('docs/R29_INCUMBENT.json')['codes']
    records=[]
    for cells in ((0,1,2,3),(4,9,16,19,22,43),(2,6,9,17,43,66)):
        q=CycleModel(model,board,cells,(2,))
        master=SeamMaster(q)
        for target in (208,209):
            trial=master.solve(target,20)
            cold=q.solve(target=target,seconds=20)
            records.append(dict(cells=cells,target=target,trial=trial,baseline=cold))
    atomic_json(output,records)


def component_budgets(problem,output):
    import random
    from r30_bounds import pattern_distance_upper
    from r34_component_budget import ComponentBudget
    model=BoardModel(problem)
    b=ExchangeBound(model,read_json('docs/R29_INCUMBENT.json')['codes'])
    rng=random.Random(1534)
    records=[]
    for _ in range(36):
        cells=rng.sample(list(map(int,model.free)),8)
        c=ComponentBudget(b,cells)
        trial=c.upper(3,max_rows=100000)
        cold=pattern_distance_upper(b,cells,3)
        smaller=c.upper(2)
        repeat=c.upper(3)
        assert cold['upper']<=trial['upper']
        records.append(dict(cells=cells,trial=trial,baseline=cold,smaller=smaller,repeat=repeat,
                            four=c.upper(4),seven_or_more=c.upper(4,min_moved=7)))
        atomic_json(output,records)


def undo_pages(problem,output):
    from itertools import combinations
    import random
    from tempfile import TemporaryDirectory
    from unittest.mock import patch
    from contextlib import nullcontext
    from r30_multicenter import run
    from r30_cycles import canonical_cycles
    from r31_routes import rank
    from r34_cycle_orders import FIVE_FAMILIES,CycleOrders
    from r34_exchange_undo import CoveredUndo
    board=read_json('docs/R29_INCUMBENT.json')['codes']
    model=BoardModel(problem)
    b=ExchangeBound(model,board)
    rng=random.Random(6401)
    def covered(group):
        lengths=tuple(sorted(map(len,group)))
        return sum(lengths)<=5 or lengths in ((2,4),(3,3),(2,2,2))
    undo=CoveredUndo(b,covered)
    samples={}
    for family in FIVE_FAMILIES:
        opts=CycleOrders(family)
        hits=0
        for _ in range(250):
            cells=sorted(rng.sample(list(map(int,model.free)),sum(family)))
            group=canonical_cycles(cells,opts[rng.randrange(len(opts))])
            hits+=undo.rejects(group)
        samples[str(family)]=hits
    records={}
    with TemporaryDirectory(dir=output.parent) as temp:
        for family in FIVE_FAMILIES:
            if len(family)==1:
                continue
            size=sum(family)
            scopes=[]
            for anchor in ((30,62,105,104,107,106),(43,123,30,51,62,42)):
                near=sorted(set(int(j) for c in anchor for j in b.neighbors[c] if j!=b.n)-set(anchor))
                for _ in range(2):
                    cells=sorted([*anchor,*rng.sample(near,size-len(anchor))])
                    scopes.append(rank(cells,model.free))
            for mode in ('baseline','current'):
                stats=Counter()
                db=Path(temp)/(mode+'.sqlite')
                control=patch.object(CoveredUndo,'rejects',return_value=False) if mode=='baseline' else nullcontext()
                with control:
                    for lo in scopes:
                        arr=run(problem,[board],db,families=(family,),target=208,start=lo,stop=lo+2,page=2)
                        stats.update(arr['work'])
                        if arr['best'] is not None:
                            raise ValueError("Publish and verify new 208 witness.")
                records[str(family)+mode]=dict(stats)
                atomic_json(output,dict(samples=samples,records=records))


def interacting_undo(problem,output):
    import random
    from r34_exchange_undo import CoveredUndo
    board=read_json('docs/R29_INCUMBENT.json')['codes']
    model=BoardModel(problem)
    b=ExchangeBound(model,board)
    rng=random.Random(1134)
    records={}
    movable=set(map(int,model.free))
    for length in (4,5):
        other_size=7-length
        undo=CoveredUndo(b,lambda left:len(left)==1 and len(left[0])==other_size)
        counts=Counter()
        for i in range(80):
            cells=[rng.choice(sorted(movable))]
            while len(cells)<length:
                frontier=(set().union(*(undo.near[c] for c in cells))&movable)-set(cells)
                cells.append(rng.choice(sorted(frontier)))
            ports=(set().union(*(undo.near[c] for c in cells))&movable)-set(cells)
            rest=list(sorted(movable-ports-set(cells)))
            other=[rng.choice(sorted(ports)),*rng.sample(rest,other_size-1)] if i%2 else rng.sample(rest,other_size)
            changed=dict(zip(other,other[1:]+other[:1]))
            moves=dict(zip(cells,cells[1:]+cells[:1]))
            gain=undo._group_gain(moves,changed)
            if gain is None:
                counts['over_table_cap']+=1
            else:
                key='touches' if i%2 else 'unchanged_boundary'
                counts[key+'_priced']+=1
                counts[key+'_nonpositive']+=gain<=0
            cuts=undo.rejects((tuple(cells),tuple(other)))
            assert cuts==(gain is not None and gain<=0)
            counts['incremental_cuts']+=cuts
        records[length]=dict(counts)
        atomic_json(output,records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('cover','trajectories','centers','predecessors','bands','components','undo-pages','interacting-undo'))
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',action='append',default=[])
    p.add_argument('--database')
    p.add_argument('--rounds',type=int,default=24)
    args=p.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    if args.mode=='cover':
        cover(problem,args.output)
    elif args.mode=='trajectories':
        trajectories(problem,args.output,args.rounds)
    elif args.mode=='centers':
        centers(problem,args.output,args.archive)
    elif args.mode=='predecessors':
        predecessor_pages(problem,args.output,args.database)
    elif args.mode=='bands':
        bands(problem,args.output)
    elif args.mode=='components':
        component_budgets(problem,args.output)
    elif args.mode=='undo-pages':
        undo_pages(problem,args.output)
    else:
        interacting_undo(problem,args.output)


if __name__=='__main__':
    main()
