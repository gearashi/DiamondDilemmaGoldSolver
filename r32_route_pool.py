"""Joint defect-cover bounds for forced donor prefixes with a small remaining mover budget."""
from collections import Counter
from itertools import combinations
import numpy as np


def defect_cover(bound, group, free, remaining, target, *, seconds=0.5):
    """A true seam failure can cover several obstructions; each equally changed cell can deactivate several."""
    from ortools.sat.python import cp_model
    from r31_route_geometry import cycle_mapping
    sources = cycle_mapping(group)
    ids = bound.tiles.copy()
    for cell,donor in sources.items():
        ids[cell] = bound.tiles[donor]
    available = set(free) - sources.keys()
    inventory = bound.model.problem
    neighbors = [[] for _ in range(bound.n)]
    requirements = []
    for edge,(a,s,b,t) in enumerate(inventory.edges):
        matches = (bound.masks[ids[a],:,s,None] == bound.model.reverse[bound.masks[ids[b],:,t]])
        if not matches.any():
            requirements.append(((edge,), {a,b}))
        else:
            neighbors[a].append((edge,b,matches.any(1)))
            neighbors[b].append((edge,a,matches.any(0)))
    for cell,adjacent in enumerate(neighbors):
        smaller = []
        for n in range(2,len(adjacent)+1):
            for row in combinations(adjacent,n):
                edges = frozenset(e for e,_,_ in row)
                if any(old <= edges for old in smaller):
                    continue
                if not np.logical_and.reduce([mask for _,_,mask in row]).any():
                    requirements.append((tuple(sorted(edges)),{cell, *(b for _,b,_ in row)}))
                    smaller.append(edges)
    model = cp_model.CpModel()
    failed, moved = {}, {}
    for edges,cells in requirements:
        clause = []
        for e in edges:
            if e not in failed:
                failed[e] = model.new_bool_var('failed')
            clause.append(failed[e])
        for c in available.intersection(cells):
            if c not in moved:
                moved[c] = model.new_bool_var('moved')
            clause.append(moved[c])
        model.add_bool_or(clause)
    model.add(sum(moved.values())<=remaining)
    model.add(sum(failed.values())<=len(inventory.edges)-target)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = seconds
    solver.parameters.num_search_workers = 1
    status = solver.solve(model)
    return dict(closed=status==cp_model.INFEASIBLE,status=solver.status_name(status),
                clauses=len(requirements),branches=solver.num_branches)


def load_prefix_certificates(bound, target, family):
    """Load promising full-pool prefixes. ConnectedJoin rechecks the proof on first use."""
    from pathlib import Path
    from r29_exchange_route import physical_hash
    from r28_exchange import read_json, domain_binding
    if target < 208 or tuple(sorted(family)) != (2,2,3):
        return frozenset()
    file = Path(__file__).parent/'docs/R32_ROUTE_FOLLOWUPS.json'
    try:
        record = read_json(file)
        if domain_binding(bound.model)!=record['domain']:
            return frozenset()
        owner = physical_hash(bound.tiles)
        pairs = {tuple(sorted(tuple(c) for c in row['prefix'])) for row in record['defects']
                 if row['center']==owner and row['result']['status']=='INFEASIBLE'}
        pairs = {g for g in pairs if len(g)==2 and all(len(c)==2 for c in g)
                 and len({i for c in g for i in c})==4 and all(type(i) is int and 0<=i<bound.n for c in g for i in c)}
    except (OSError,KeyError,TypeError,ValueError):
        return frozenset()
    return frozenset(pairs)


def evaluate(problem, boards, pool, family, target, *, database, paired=True, buckets=True, bridge=False,
             together=True, cover=False):
    """A finished pool is a complete connected scope per center; publish checked negatives to ranked search."""
    from contextlib import closing
    from itertools import combinations
    import sqlite3
    import hashlib
    import json
    from r27_incumbent import BoardModel
    from r28_exchange import ExchangeBound,domain_binding
    from r28_exchange_rotations import RotationOracle
    from r29_exchange_clusters import StarClusters
    from r29_exchange_route import physical_hash,rotation_optimum
    from r30_multicenter import orders
    from r31_route_coverage import register
    from r31_route_geometry import ConnectedJoin,proved_connected
    from r31_routes import rank
    from r32_route_join import traverse

    model = BoardModel(problem)
    pool = tuple(sorted(pool))
    family = tuple(sorted(family))
    bound = [ExchangeBound(model,b) for b in boards]
    if not bound or any(not proved_connected(b,target,family) for b in bound):
        raise ValueError("Pool requires a published connectivity proof for every supplied center.")
    joins = [ConnectedJoin(StarClusters(b),pool,family,target) for b in bound]
    scope = domain_binding(model)
    owners = [physical_hash(b.tiles) for b in bound]
    if len(set(owners))!=len(owners):
        raise ValueError("Duplicate physical center.")
    keys = [hashlib.sha256(json.dumps([1,scope,[tuple(map(int,b.tiles))],(family,),target]).encode()).hexdigest() for b in bound]
    supports = tuple(rank(s,model.free) for s in combinations(pool,sum(family)))
    with closing(sqlite3.connect(database,timeout=90)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS r30_pages(scope TEXT,lo INTEGER,hi INTEGER,work TEXT,best TEXT,PRIMARY KEY(scope,lo))")
        db.execute("CREATE TABLE IF NOT EXISTS optima(key TEXT PRIMARY KEY,result TEXT NOT NULL)")
        missing = []
        saved_best = []
        def covered(key,r):
            row = db.execute('SELECT hi,best FROM r30_pages WHERE scope=? AND lo<=? ORDER BY lo DESC LIMIT 1',(key,r)).fetchone()
            return row if row and row[0]>r else None
        for key,o in zip(keys,owners):
            register(db,key,scope,(family,),target,(o,))
            left = set()
            for r in supports:
                previous = covered(key,r)
                if previous is None:
                    left.add(r)
                elif previous[1]!='null':
                    saved_best.append(json.loads(previous[1]))
            missing.append(left)
        indices = [i for i,m in enumerate(missing) if m]
        if not indices:
            return dict(best=max(saved_best,key=lambda x:x['score'],default=None),complete=True,
                        work={'reused_center_supports':len(keys)*len(supports)})
        oracle = RotationOracle(model)
        best = {}
        rows,work = traverse([joins[i] for i in indices],pair=paired,buckets=buckets,bridge=bridge,together=together,
                             cover=cover)
        for sub,cells,order in rows:
            ix = indices[sub]
            r = rank(cells,model.free)
            if r not in missing[ix]:
                continue
            b = bound[ix]
            seed = b.witness_seed(cells,order)
            if seed is None:
                continue
            pkey=scope+':'+','.join(map(str,seed//3))
            old=db.execute('SELECT result FROM optima WHERE key=?',(pkey,)).fetchone()
            if old:
                result=json.loads(old[0])
                work['exact_reused']+=1
            else:
                result=rotation_optimum(b,oracle,seed)
                with db:
                    db.execute('INSERT OR IGNORE INTO optima VALUES(?,?)',(pkey,json.dumps(result)))
                work['exact_solved']+=1
            if result['score']>=target and result['score']>best.get((ix,r),{}).get('score',-1):
                best[ix,r]=result
        # Publish only after the complete finite traversal. Another writer may have completed the rank.
        with db:
            db.execute('BEGIN IMMEDIATE')
            for ix in indices:
                for r in sorted(missing[ix]):
                    if covered(keys[ix],r):
                        continue
                    n=len(orders(family))
                    db.execute('INSERT INTO r30_pages VALUES(?,?,?,?,?)',
                               (keys[ix],r,r+1,json.dumps({'joint_pool':1,'raw_center_endpoints':n}),json.dumps(best.get((ix,r)))))
                    work['published_center_supports']+=1
    return dict(best=max([*best.values(),*saved_best],key=lambda a:a['score'],default=None),complete=True,work=dict(work))



def stream_prefixes(problem, boards, pool, family, target, database, *, start=0, stop=None, seconds=0,
                    bridge=False, cover=False, max_pages=None):
    """Checkpoint each shared work item, splitting four swaps into first and second canonical cycles.

    The fixed root and a distinct second cycle identify one item even when no
    edge can connect them. Within a 3+2+2 item the root is the single triple,
    so every completion has exactly one owner. Within a 2+2+2+2 item choose
    the least two swaps in sorted order. This ordering admits any intermediate score.
    """
    from contextlib import closing
    import sqlite3, hashlib, json
    from itertools import combinations
    from time import monotonic
    from math import comb
    from r27_incumbent import BoardModel
    from r28_exchange import ExchangeBound,domain_binding
    from r28_exchange_cuda import unrank
    from r29_exchange_clusters import StarClusters
    from r29_exchange_route import rotation_optimum
    from r28_exchange_rotations import RotationOracle
    from r31_route_geometry import ConnectedJoin, connected, cycle_mapping, proved_connected
    from r32_route_join import PhysicalPrices,bridge_upper
    model = BoardModel(problem)
    bounds = [ExchangeBound(model,b) for b in boards]
    pool = tuple(sorted(pool))
    family = tuple(sorted(family))
    if not bounds or any(not proved_connected(b,target,family) for b in bounds):
        raise ValueError("Connected work items require validated connectivity for all centers.")
    if not 0 <= seconds or (max_pages is not None and max_pages<0):
        raise ValueError("Invalid time or work budget.")
    joins = [ConnectedJoin(StarClusters(b),pool,family,target) for b in bounds]
    prepared = [PhysicalPrices(j) for j in joins]
    permutations = [tuple(map(int,b.tiles)) for b in bounds]
    if len(set(permutations))!=len(bounds):
        raise ValueError("Duplicate physical center.")
    binding = domain_binding(model)
    scope = hashlib.sha256(json.dumps([1,binding,permutations,pool,family,target]).encode()).hexdigest()
    n = len(pool)
    # Split into mixed-radix work: prefix cycles, then the smaller of the two
    # remaining swaps. Only the final partner (at most C(155,2) rows) is open.
    root_size = 3 if family==(2,2,3) else 4
    prefixes = comb(n,3)*2 if root_size==3 else comb(n,4)*3
    stride = comb(n-root_size,2)
    total = prefixes*stride
    stop = total if stop is None else stop
    if not 0 <= start <= stop <= total:
        raise ValueError("Invalid joint work interval.")
    def prefix_at(i):
        r,orient = divmod(i,2 if root_size==3 else 3)
        positions = tuple(pool[t] for t in unrank(r,n,root_size))
        if root_size==3:
            a,b,c = positions
            return ((a,b,c),) if orient==0 else ((a,c,b),)
        a = positions[0]
        b = positions[1+orient]
        return ((a,b),tuple(i for i in positions if i not in (a,b)))
    work = Counter()
    oracle = RotationOracle(model)
    current = start
    pages = 0
    clock = monotonic()
    def expired():
        return seconds and monotonic()-clock>=seconds
    prev_id = None
    with closing(sqlite3.connect(database,timeout=90)) as db:
        db.execute('CREATE TABLE IF NOT EXISTS r32_joint_pages(scope TEXT,lo INTEGER,hi INTEGER,best TEXT,PRIMARY KEY(scope,lo))')
        db.execute('CREATE TABLE IF NOT EXISTS optima(key TEXT PRIMARY KEY,result TEXT NOT NULL)')
        while current<stop and not expired() and (max_pages is None or pages<max_pages):
            offs,pos = divmod(current,stride)
            if offs!=prev_id:
                # Reads precede expensive catalogues, so a fully completed interval reopens without pricing.
                prefix=prefix_at(offs)
                available=tuple(c for c in pool if all(c not in cyc for cyc in prefix))
                prev_id = offs
                prices = None
            w = Counter()
            with db:
                db.execute('BEGIN IMMEDIATE')
                previous = db.execute('SELECT hi FROM r32_joint_pages WHERE scope=? AND lo<=? ORDER BY lo DESC LIMIT 1',
                                      (scope,current)).fetchone()
                if previous and previous[0]>current:
                    current = previous[0]
                    continue
                nextpage = db.execute('SELECT min(lo) FROM r32_joint_pages WHERE scope=? AND lo>?',(scope,current)).fetchone()[0]
                end = min(stop, (offs+1)*stride, nextpage if nextpage is not None else stop)
                if prices is None:
                    active = []
                    for i,j in enumerate(joins):
                        if cover and defect_cover(j.bound,prefix,pool,4,target)['closed']:
                            w['overlap_closed']+=1
                        else:
                            active.append(i)
                    pairs = list(combinations(available,2))
                    prices = [(i,prepared[i].pair_catalog(prefix,pairs)) for i in active]
                    w['individual_cycle_prices'] += len(prices)*len(pairs)
                best = None
                if prices:
                    first = tuple(available[t] for t in unrank(pos,len(available),2))
                    end = current+1
                    if root_size==4 and first<=prefix[-1]:
                        # The first two swaps of this four-swap representation own the endpoint.
                        w['wrong_owner_work_items']+=1
                    else:
                        proposals = {}
                        for i,p in prices:
                            for pair in prepared[i].partners(p, first, w):
                                if first>=pair:
                                    continue
                                proposals.setdefault(pair,[]).append(i)
                        for pair,active in proposals.items():
                            group=tuple(sorted((*prefix,first,pair)))
                            w['shared_geometry_queries']+=1
                            if not connected(group,joins[0].close):
                                continue
                            mapping=cycle_mapping(group)
                            cells=tuple(sorted(mapping))
                            order=tuple(cells.index(mapping[c]) for c in cells)
                            for i in active:
                                j=joins[i]
                                ub=bridge_upper(j.stars,group) if bridge else None
                                if ub is None:
                                    ub=j.stars.price(cells,order)
                                w['stars']+=1
                                if ub<2*target:
                                    continue
                                seed=j.bound.witness_seed(cells,order)
                                if seed is None:
                                    continue
                                key=binding+':'+','.join(map(str,seed//3))
                                old=db.execute('SELECT result FROM optima WHERE key=?',(key,)).fetchone()
                                if old:
                                    row=json.loads(old[0])
                                    w['exact_reused']+=1
                                else:
                                    row=rotation_optimum(j.bound,oracle,seed)
                                    db.execute('INSERT INTO optima VALUES(?,?)',(key,json.dumps(row)))
                                    w['exact_solved']+=1
                                if row['score']>=target and (best is None or row['score']>best['score']):
                                    best=row
                db.execute('INSERT INTO r32_joint_pages VALUES(?,?,?,?)',(scope,current,end,json.dumps(best)))
                current=end
                pages+=1
                work.update(w)
        best=max((json.loads(row) for row, in db.execute("SELECT best FROM r32_joint_pages WHERE scope=? AND best!='null'",(scope,))),
                 key=lambda r:r['score'],default=None)
        done=db.execute('SELECT coalesce(sum(hi-lo),0) FROM r32_joint_pages WHERE scope=?',(scope,)).fetchone()[0]
    return dict(scope=scope,completed_items=done,total_items=total,range_finished=current>=stop,
                next_item=current,complete=done==total,best=best,new_pages=pages,work=dict(work))


def main():
    import argparse,json
    from search_problem import SearchProblem
    from r28_exchange import read_json
    from pathlib import Path
    p=argparse.ArgumentParser(description='Search a finite connected four-swap pool at all supplied centers.')
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--family',choices=['322','2222'],required=True)
    p.add_argument('--cells',nargs='+',type=int)
    p.add_argument('--stream',action='store_true',help='Resume disjoint prefix/swap intervals. Omitting cells uses all 160.')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--stop',type=int)
    p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--incumbent',action='append')
    p.add_argument('--database',required=True)
    p.add_argument('--target',type=int,default=208)
    p.add_argument('--bridge',action='store_true')
    p.add_argument('--cover',action='store_true')
    a=p.parse_args()
    if a.incumbent:
        boards=[read_json(p)['codes'] for p in a.incumbent]
    else:
        boards=[[3*t for t in c['pieces']] for c in read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')['centers']]
    problem=SearchProblem(read_json(a.tiles))
    if a.stream:
        result=stream_prefixes(problem,boards,range(problem.n) if a.cells is None else a.cells,
                               tuple(map(int,a.family)),a.target,a.database,start=a.start,stop=a.stop,
                               seconds=a.seconds,bridge=a.bridge,cover=a.cover)
    else:
        if not a.cells or a.start or a.stop is not None or a.seconds:
            p.error('Supply --cells for a finite pool, or use --stream.')
        result=evaluate(problem,boards,a.cells,tuple(map(int,a.family)),a.target,database=a.database,
                        bridge=a.bridge,cover=a.cover)
    print(json.dumps(result))


if __name__=='__main__':
    main()
