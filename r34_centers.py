"""Select and extend ranked physical neighborhoods from verified independent restarts."""
import argparse
from contextlib import closing
import json
import random
import sqlite3
from pathlib import Path


from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, domain_binding, read_json
from r29_exchange_route import legal_cell_rotations, owner_copies, physical_hash
from r30_bounds import budget_upper, defect_packings
from r30_cycles import FOUR_FAMILIES
from r34_cycle_orders import FIVE_FAMILIES
from r30_multicenter import orders,run,sparse_cycle_type
from search_problem import SearchProblem


def archive_boards(paths):
    for path in paths:
        path=Path(path)
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            tables={name for name, in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for table in ('r30_starts','r31_jobs'):
                if table in tables:
                    for data, in db.execute(f'SELECT result FROM {table} WHERE result IS NOT NULL ORDER BY scope,seed'):
                        yield json.loads(data)['codes']


def select(problem,boards,database,family,target,limit,policy='novel',seed=34,samples=64,*,distinct_orbits=True):
    if limit<1 or samples<1 or policy not in ('score','defects','novel'):
        raise ValueError("Invalid center policy.")
    family=tuple(sorted(family))
    if family not in FOUR_FAMILIES+FIVE_FAMILIES:
        raise ValueError("Expected a supported ranked cycle family.")
    size=sum(family)
    model=BoardModel(problem)
    if not 0<=target<=len(problem.edges) or size>len(model.free):
        raise ValueError("Invalid target or support size.")
    domain=domain_binding(model)
    from r35_center_orbits import CenterOrbits
    orbits=CenterOrbits(model)
    candidates={}
    mask_builds=mask_hits=0
    with closing(sqlite3.connect(database,timeout=90)) as feature_db,feature_db:
        feature_db.execute("""CREATE TABLE IF NOT EXISTS r35_center_masks(
            domain TEXT,posed BLOB,failed TEXT,PRIMARY KEY(domain,posed))""")
        for values in boards:
            board=model.legal_board(values)
            key=physical_hash(board//3)
            posed=board.astype('<i8',copy=False).tobytes()
            cached=feature_db.execute('SELECT failed FROM r35_center_masks WHERE domain=? AND posed=?',
                                      (domain,posed)).fetchone()
            if cached:
                mask=int(cached[0],16);mask_hits+=1
            else:
                mask=sum(1<<i for i,m in enumerate(model.matches(board)) if not m)
                feature_db.execute('INSERT OR IGNORE INTO r35_center_masks VALUES(?,?,?)',(domain,posed,hex(mask)))
                mask_builds+=1
            score=len(problem.edges)-mask.bit_count()
            if key not in candidates or score>candidates[key]['score']:
                candidates[key]=dict(board=board,score=score,mask=mask)
    submitted=len(candidates)
    if distinct_orbits:
        representatives={}
        for key,row in candidates.items():
            orbit,_=orbits.key(row['board']//3)
            previous=representatives.get(orbit)
            if previous is None or (row['score'],key)>(candidates[previous]['score'],previous):
                representatives[orbit]=key
        candidates={key:candidates[key] for key in representatives.values()}
    caps={}
    with closing(sqlite3.connect(database,timeout=90)) as db,db:
        db.execute("CREATE TABLE IF NOT EXISTS r30_budget(cap_key TEXT PRIMARY KEY,cap INTEGER NOT NULL)")
        for key,row in candidates.items():
            raw=f'{domain}:{size-len(family)}:{key}'
            orbit_key,_=orbits.key(row['board']//3)
            canonical=f'{domain}:{size-len(family)}:{orbit_key}'
            cap=db.execute('SELECT min(cap) FROM r30_budget WHERE cap_key IN (?,?)',(raw,canonical)).fetchone()[0]
            if cap is None:
                bound=ExchangeBound(model,row['board'])
                cap=orbits.cap(db,bound,domain,size-len(family),budget_upper)
            caps[key]=cap
    live={k for k in candidates if caps[k]>=target}
    transforms=None
    overlaps={}
    pair_counts={}
    from r35_shell_overlap import cycle_type,intersection,union_count
    from r34_cycle_orders import family_count
    from math import comb
    shell_size=comb(len(model.free),size)*family_count(family)
    def relatives(a,b):
        nonlocal transforms
        if transforms is None:
            transforms=legal_cell_rotations(model)
        if (a,b) not in overlaps:
            current=candidates[a]['board']//3
            other=candidates[b]['board']//3
            overlaps[a,b]=[]
            pair_counts[a,b]=[]
            for mapping,_ in owner_copies(model,[(other,size)],current,size,transforms=transforms):
                count=intersection(cycle_type(mapping),family,family,len(model.free))
                if count:
                    overlaps[a,b].append({c:d for c,d in enumerate(mapping) if c!=d})
                    pair_counts[a,b].append(count)
        return overlaps[a,b]
    test=[]
    def sampled_moves():
        if not test:
            options=orders(family)
            rng=random.Random(seed)
            free=tuple(map(int,model.free))
            for _ in range(samples):
                cells=sorted(rng.sample(free,size))
                order=options[rng.randrange(len(options))]
                test.append({c:cells[order[i]] for i,c in enumerate(cells)})
        return test
    novelty={}
    exact_novelty=sampled_novelty=0
    selected=[]
    groups=[]
    detail=[]
    while len(selected)<limit and live:
        def priority(key):
            row=candidates[key]
            if not selected or policy=='score':
                return row['score'],caps[key]
            diversity=min((row['mask']^candidates[k]['mask']).bit_count() for k in selected)
            if policy=='defects':
                return diversity,row['score'],caps[key]
            nonlocal exact_novelty,sampled_novelty
            lookup=key,tuple(selected)
            if lookup not in novelty:
                near=[rel for k in selected for rel in relatives(key,k)]
                if len(near)<=1:
                    covered=sum(sum(pair_counts[key,k]) for k in selected)
                else:
                    covered=union_count(family,near,len(model.free))
                if covered is None:
                    hits=sum(all(sparse_cycle_type(move,rel)!=family for rel in near) for move in sampled_moves())
                    novelty[lookup]=shell_size*hits//samples
                    sampled_novelty+=1
                else:
                    novelty[lookup]=shell_size-covered
                    exact_novelty+=1
            novel=novelty[lookup]
            return novel,caps[key],diversity,row['score']
        key=max(sorted(live),key=priority)
        detail.append(dict(center=key,score=candidates[key]['score'],upper=caps[key],priority=priority(key)))
        live.remove(key)
        selected.append(key)
        connected=[group for group in groups if any(relatives(key,k) for k in group)]
        groups=[group for group in groups if group not in connected]
        groups.append([k for group in connected for k in group]+[key])
    return candidates,groups,dict(selected=detail,cap_excluded={k:caps[k] for k in sorted(caps) if caps[k]<target},
                                  candidates=len(candidates),equivalent_centers=submitted-len(candidates),exact_novelty_counts=exact_novelty,sampled_novelty_counts=sampled_novelty,fresh_mask_prices=mask_builds,reused_mask_prices=mask_hits,groups=groups)


def search(problem,boards,database,*,family,target=208,limit=3,policy='novel',seed=34,samples=64,schedule='fair',group_pages=2,**settings):
    candidates,groups,report=select(problem,boards,database,family,target,limit,policy,seed,samples,
                                           distinct_orbits=settings.get('start',0)==0 and settings.get('stop') is None)
    # Separate disjoint shells retain their original singleton page scopes as
    # more centers are added. Possibly overlapping shells use exact joint ownership.
    if len(groups)<=1:
        reports=[run(problem,[candidates[k]['board'] for k in group],database,families=(family,),target=target,**settings)
                 for group in groups]
    else:
        from r35_center_schedule import interleave
        reports,report['schedule']=interleave(problem,candidates,groups,database,family,target,settings,run,
                                              quota=group_pages,policy=schedule)
    report['searches']=reports
    report['best']=max((r['best'] for r in reports if r['best'] is not None),
                       key=lambda row:row['score'],default=None)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--database',required=True)
    p.add_argument('--archive',action='append',required=True)
    p.add_argument('--family',choices=[','.join(map(str,f)) for f in FOUR_FAMILIES+FIVE_FAMILIES],default='2,2,3')
    p.add_argument('--target',type=int,default=208)
    p.add_argument('--limit',type=int,default=3)
    p.add_argument('--policy',choices=('score','defects','novel'),default='novel')
    p.add_argument('--seed',type=int,default=34)
    p.add_argument('--page',type=int,default=64)
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--group-pages',type=int,default=2,help='Ranked page quota for one disjoint center-group turn.')
    p.add_argument('--schedule',choices=('fair','yield'),default='fair')
    p.add_argument('--stop',type=int)
    p.add_argument('--seconds',type=float,default=0,help='Time limit per disjoint group (zero is unlimited).')
    args=p.parse_args()
    family=tuple(map(int,args.family.split(',')))
    report=search(SearchProblem(read_json(args.tiles)),archive_boards(args.archive),args.database,family=family,
                  target=args.target,limit=args.limit,policy=args.policy,seed=args.seed,page=args.page,
                  start=args.start,stop=args.stop,seconds=args.seconds,schedule=args.schedule,group_pages=args.group_pages)
    print(json.dumps(report))


if __name__=='__main__':
    main()
