"""Solve and compare connected four-swap pools with every legal rotation free."""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
from math import comb, factorial
from numbers import Integral
from time import monotonic
import sqlite3

from search_problem import SearchProblem
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound,domain_binding,read_json
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r29_exchange_route import physical_hash,rotation_optimum
from r30_cycles import canonical_cycles
from r30_multicenter import released_six
from r31_route_geometry import ConnectedJoin,proved_connected
from r31_route_price import DefectPrices,Exterior,comparison_relations


def rank(cells,free):
    index = {c:i for i,c in enumerate(sorted(free))}
    return sum(comb(index[c],i) for i,c in enumerate(sorted(cells),1))


def search(problem,board,cells,family,target,*,database=':memory:',price=True,
           compare=False,defects=False,exterior=False):
    model = BoardModel(problem)
    b = ExchangeBound(model,board)
    cells = sorted(cells)
    if len(set(cells))!=len(cells):
        raise ValueError("Duplicate pool cell.")
    family = tuple(sorted(family))
    if not proved_connected(b,target,family):
        raise ValueError("This puzzle, center, family and threshold lack the required connectivity certificate; use r30_multicenter.")
    join = ConnectedJoin(StarClusters(b),cells,family,target)
    binding = domain_binding(model)
    tiles = tuple(map(int,b.tiles))
    key = hashlib.sha256(json.dumps([1,binding,[tiles],(family,),target]).encode()).hexdigest()
    if compare and not released_six(binding,[tiles],[(2,4),(3,3)],target):
        raise ValueError("This scope lacks a verified six-cell comparison bound.")
    oracle = RotationOracle(model)
    exterior_cell = None
    envelope = None
    best = None
    count = Counter()
    pricing = DefectPrices(b) if defects else None
    with closing(sqlite3.connect(database,timeout=90)) as db:
        db.execute("CREATE TABLE IF NOT EXISTS optima(key TEXT PRIMARY KEY,result TEXT NOT NULL)")
        db.execute("""CREATE TABLE IF NOT EXISTS r30_pages(scope TEXT,lo INTEGER,hi INTEGER,work TEXT,best TEXT,
                    PRIMARY KEY(scope,lo))""")
        db.commit()
        from itertools import combinations
        free = set(model.free)
        def prior_negative(r):
            page = db.execute("SELECT hi,best FROM r30_pages WHERE scope=? AND lo<=? ORDER BY lo DESC LIMIT 1",(key,r)).fetchone()
            return page and page[0]>r and page[1]=='null'
        # Scan a small pool up front so a completed revisit costs no cycle or
        # pricing work. The ranked interface handles wide pools by intervals.
        if comb(len(cells),sum(family))<=4096 and all(
                prior_negative(rank(s,free)) for s in combinations(cells,sum(family))):
            return dict(family=family,cells=cells,target=target,complete=True,best=None,
                        work={'whole_pool_reused':comb(len(cells),sum(family))})
        positives = []
        for support,order in join.survivors(price=price):
            r = rank(support,model.free)
            prior = db.execute("SELECT hi,best FROM r30_pages WHERE scope=? AND lo<=? ORDER BY lo DESC LIMIT 1",(key,r)).fetchone()
            if prior and prior[0]>r and prior[1]=='null':
                count['skipped_completed'] += 1
                continue
            if pricing and pricing.endpoint_upper(support,order) < target:
                count['donor_defects'] += 1
                continue
            if compare:
                ok,work = comparison_relations(b,canonical_cycles(support,order))
                count.update(work)
                if not ok:
                    count['six_comparison'] += 1
                    continue
            if exterior:
                if support!=exterior_cell:
                    envelope = Exterior(b,support)
                    exterior_cell = support
                cap = envelope.score(order)
                if cap is not None and cap<target:
                    count['exterior'] += 1
                    continue
            seed = b.witness_seed(support,order)
            if seed is None:
                continue
            perm = binding+':'+','.join(map(str,seed//3))
            prior = db.execute('SELECT result FROM optima WHERE key=?',(perm,)).fetchone()
            if prior:
                row = json.loads(prior[0])
                count['exact_reused'] += 1
            else:
                row = rotation_optimum(b,oracle,seed)
                with db:
                    db.execute('INSERT OR IGNORE INTO optima VALUES(?,?)',(perm,json.dumps(row)))
                count['exact_solved'] += 1
            if row['score']>=target:
                positives.append((r,row))
                if best is None or row['score']>best['score']:
                    best = row
        # A pool whose size equals the family size is a complete ranked support.
        # A completed larger pool may cover many nonconsecutive colex intervals.
        from itertools import combinations
        from r30_multicenter import orders
        from r31_route_coverage import register
        register(db,key,binding,(family,),target,(physical_hash(tiles),))
        found = {}
        for r,row in positives:
            if r not in found or row['score']>found[r]['score']:
                found[r] = row
        with db:
            db.execute("BEGIN IMMEDIATE")
            for support in combinations(cells,sum(family)):
                r = rank(support,model.free)
                previous = db.execute("SELECT hi FROM r30_pages WHERE scope=? AND lo<=? ORDER BY lo DESC LIMIT 1",(key,r)).fetchone()
                if not previous or previous[0]<=r:
                    work = {'raw_center_endpoints':len(orders(family)),'connected_pool':1}
                    db.execute("INSERT INTO r30_pages VALUES(?,?,?,?,?)",(key,r,r+1,json.dumps(work),json.dumps(found.get(r))))
                    count['published_supports'] += 1
    return dict(family=family,cells=cells,target=target,complete=True,best=best,work=dict(join.work+count))


def search_roots(problem,board,family,target,database,*,pool=None,start=0,stop=None,seconds=0,quantum=256,max_pages=None):
    """Durable, disjoint root intervals on the complete board or a restricted pool.

    Short cursor pages are atomic. A crash can replay its current page;
    complete subtrees, partial roots and exact physical scores survive.
    """
    from r28_exchange_cuda import unrank
    from r31_route_cursor import advance,restore_frames
    from itertools import permutations
    model = BoardModel(problem)
    bound = ExchangeBound(model,board)
    family = tuple(sorted(family))
    if not proved_connected(bound,target,family):
        raise ValueError("Root traversal requires a matching whole-family connected-cycle proof.")
    pool = tuple(sorted(model.free if pool is None else pool))
    if len(set(pool))!=len(pool) or any(isinstance(c,bool) or not isinstance(c,Integral) or c not in model.free for c in pool):
        raise ValueError("Expected distinct movable integer cells.")
    pool = tuple(map(int,pool))
    if type(quantum) is not int or quantum<1 or (max_pages is not None and (type(max_pages) is not int or max_pages<0)):
        raise ValueError("Invalid root cursor page limits.")
    groups = [(size,comb(len(pool),size)*factorial(size-1)) for size in sorted(set(family))]
    total = sum(count for _,count in groups)
    stop = total if stop is None else stop
    if not 0<=start<=stop<=total or seconds<0:
        raise ValueError("Invalid root range or time allowance.")
    binding = domain_binding(model)
    tiles = tuple(map(int,bound.tiles))
    owner = physical_hash(tiles)
    key = hashlib.sha256(json.dumps([1,binding,owner,pool,family,target]).encode()).hexdigest()
    ranked_key = hashlib.sha256(json.dumps([1,binding,[tiles],(family,),target]).encode()).hexdigest()
    join = ConnectedJoin(StarClusters(bound),pool,family,target)
    oracle = RotationOracle(model)
    work = Counter()
    clock = monotonic()
    current = start
    with closing(sqlite3.connect(database,timeout=90)) as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS r31_roots(scope TEXT,idx INTEGER,best TEXT,
                PRIMARY KEY(scope,idx));
            CREATE TABLE IF NOT EXISTS r30_pages(scope TEXT,lo INTEGER,hi INTEGER,work TEXT,best TEXT,
                PRIMARY KEY(scope,lo));
            CREATE TABLE IF NOT EXISTS optima(key TEXT PRIMARY KEY,result TEXT NOT NULL);
        """)
        db.execute("""CREATE TABLE IF NOT EXISTS r31_root_cursor(
                   scope TEXT, idx INTEGER, stack TEXT NOT NULL, best TEXT,
                   PRIMARY KEY(scope,idx))""")
        db.commit()
        frames = {}
        def expired():
            return bool(seconds and monotonic()-clock>=seconds)
        while current<stop and not expired() and (max_pages is None or work['cursor_pages']<max_pages):
            # Serialize only one bounded page. Another writer may commit this
            # root during our wait, so read its current cursor after acquiring.
            if seconds:
                db.execute("PRAGMA busy_timeout="+str(max(1,int(1000*(seconds-(monotonic()-clock))))))
            try:
                db.execute('BEGIN IMMEDIATE')
            except sqlite3.OperationalError as e:
                if seconds and 'locked' in str(e).lower() and expired():
                    break
                raise
            if db.execute('SELECT 1 FROM r31_roots WHERE scope=? AND idx=?',(key,current)).fetchone():
                db.commit()
                work['roots_reused'] += 1
                current += 1
                frames.clear()
                continue
            saved = db.execute('SELECT stack,best FROM r31_root_cursor WHERE scope=? AND idx=?',(key,current)).fetchone()
            if saved:
                stack = restore_frames(json.loads(saved[0]))
                best = json.loads(saved[1])
            else:
                offset = current
                for size,count in groups:
                    if offset<count:
                        break
                    offset -= count
                support_rank,orient = divmod(offset,factorial(size-1))
                cells = [pool[j] for j in unrank(support_rank,len(pool),size)]
                cycle = (cells[0],*list(permutations(cells[1:]))[orient])
                stack = [((cycle,),(0,0,0))]
                best = None
            def consume(support,order):
                nonlocal best
                r = rank(support,model.free)
                prev = db.execute('SELECT hi,best FROM r30_pages WHERE scope=? AND lo<=? ORDER BY lo DESC LIMIT 1',(ranked_key,r)).fetchone()
                if prev and prev[0]>r and prev[1]=='null':
                    work['ranked_negatives_reused'] += 1
                    return True
                seed = bound.witness_seed(support,order)
                if seed is None:
                    return True
                perm = binding+':'+','.join(map(str,seed//3))
                previous = db.execute('SELECT result FROM optima WHERE key=?',(perm,)).fetchone()
                if previous:
                    row = json.loads(previous[0])
                    work['exact_reused'] += 1
                else:
                    if seconds:
                        from r27_incumbent import rotation_alternatives
                        remaining = seconds-(monotonic()-clock)
                        if remaining<=0:
                            return False
                        trial = rotation_alternatives(model,seed,k=1,seconds=remaining)
                        if not trial or not trial[0][1]['exact']:
                            return False
                        codes,info = trial[0]
                        model.legal_board(codes)
                        codes = list(map(int,codes))
                        valid = problem.validate(codes)
                        if valid['matched_edges']!=info['matched_edges']:
                            raise RuntimeError("Physical score verification failed.")
                        row = dict(score=valid['matched_edges'],codes=codes,
                                   gold=valid['valid'],components=valid['line_components'])
                    else:
                        row = rotation_optimum(bound,oracle,seed)
                    db.execute('INSERT INTO optima VALUES(?,?)',(perm,json.dumps(row)))
                    work['exact_solved'] += 1
                if row['score']>=target and (best is None or row['score']>best['score']):
                    best = row
                return True
            steps = advance(join,stack,frames,work,consume,expired,quantum)
            if stack:
                db.execute('INSERT OR REPLACE INTO r31_root_cursor VALUES(?,?,?,?)',
                           (key,current,json.dumps(stack),json.dumps(best)))
            else:
                db.execute('INSERT INTO r31_roots VALUES(?,?,?)',(key,current,json.dumps(best)))
                db.execute('DELETE FROM r31_root_cursor WHERE scope=? AND idx=?',(key,current))
                current += 1
                frames.clear()
                work['roots_completed'] += 1
            db.commit()
            work['cursor_pages'] += 1
            if steps==0:
                break
        count, = db.execute('SELECT count(*) FROM r31_roots WHERE scope=?',(key,)).fetchone()
        best = max((json.loads(raw) for raw, in db.execute("SELECT best FROM r31_roots WHERE scope=? AND best!='null'",(key,))),key=lambda r:r['score'],default=None)
        for raw, in db.execute("SELECT best FROM r31_root_cursor WHERE scope=? AND best!='null'",(key,)):
            pending = json.loads(raw)
            if best is None or pending['score']>best['score']:
                best = pending
        # The entire full-board traversal may publish a negative range without
        # expanding a Python row per support. Existing positives stay intact.
        if count==total and best is None and set(pool)==set(model.free):
            from r31_route_coverage import register
            register(db,ranked_key,binding,(family,),target,(owner,))
            limit = comb(len(pool),sum(family))
            with db:
                db.execute('BEGIN IMMEDIATE')
                lo = 0
                gaps = []
                for a,b in db.execute('SELECT lo,hi FROM r30_pages WHERE scope=? ORDER BY lo',(ranked_key,)):
                    if a>lo:
                        gaps.append((lo,a))
                    lo = b
                if lo<limit:
                    gaps.append((lo,limit))
                for a,b in gaps:
                    db.execute('INSERT INTO r30_pages VALUES(?,?,?,?,?)',(ranked_key,a,b,json.dumps({'connected_full':b-a}),'null'))
    return dict(next_root=current,total_roots=total,completed_roots=count,complete=count==total,
                range_finished=current>=stop,best=best,work=dict(work+join.work))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--incumbent',default='docs/R29_INCUMBENT.json')
    p.add_argument('--cells',type=int,nargs='+')
    p.add_argument('--roots',action='store_true',help='Durable disjoint root search. Omitting cells uses the full board.')
    p.add_argument('--start',type=int,default=0)
    p.add_argument('--stop',type=int)
    p.add_argument('--seconds',type=float,default=0)
    p.add_argument('--cursor-page',type=int,default=256)
    p.add_argument('--family',choices=('322','2222'),required=True)
    p.add_argument('--target',type=int,default=208)
    p.add_argument('--database',required=True)
    p.add_argument('--compare-six',action='store_true')
    p.add_argument('--defects',action='store_true')
    p.add_argument('--exterior',action='store_true')
    a = p.parse_args()
    if a.roots:
        if a.compare_six or a.defects or a.exterior:
            p.error("The diagnostic endpoint options apply to support-pool mode.")
        print(json.dumps(search_roots(SearchProblem(read_json(a.tiles)),read_json(a.incumbent)['codes'],
                        tuple(map(int,a.family)),a.target,a.database,pool=a.cells,start=a.start,stop=a.stop,seconds=a.seconds,quantum=a.cursor_page)))
        return
    if not a.cells or a.start or a.stop is not None or a.seconds:
        p.error("Supply --cells for pool mode, or --roots for ranked subtrees.")
    result = search(SearchProblem(read_json(a.tiles)),read_json(a.incumbent)['codes'],a.cells,
                    tuple(map(int,a.family)),a.target,database=a.database,compare=a.compare_six,defects=a.defects,exterior=a.exterior)
    print(json.dumps(result))

if __name__=='__main__':
    main()
