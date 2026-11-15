"""Compare direct pose counting and all-different pricing in geographic Gold."""
import argparse
import json
from heapq import heappop, heappush
from pathlib import Path
from random import Random
from time import perf_counter
from unittest.mock import patch
from contextlib import ExitStack

from constraint_symmetry import proper_rotations
from r30_gold import search
from r35_pose_departures import indexed_rows, propagate, reference_images
from r35_pose_matching import owner_bounds, clear_owner_cache, tight_support, owner_candidate
from r35_repair_seeds import active_catalog
from search_problem import SearchProblem


def exact_margins(cells, tiles, weights, proof, slack):
    """Surplus over the best assignment when a tile edge is chosen or left spare."""
    k = len(cells)
    owner = {j:i for i,j in enumerate(proof['assignment'])}
    spare = k
    reverse = [[] for _ in range(k+1)]
    reduced = {}
    for i, row in enumerate(weights):
        for j, weight in enumerate(row):
            if weight is None:
                continue
            price = proof['rows'][i] + proof['columns'][j] - weight
            target = owner.get(j, spare)
            reverse[target].append((i, price))
            reduced[i,j] = price
    if len(tiles) > k:
        # Dummy zero-weight rows exchange any unused tile. Each unused
        # column has zero dual price in the certified optimal injection.
        for j,i in owner.items():
            reverse[i].append((spare, proof['columns'][j]))
    def returns(root):
        dist = [slack+1]*(k+1)
        dist[root] = 0
        heap = [(0, root)]
        while heap:
            price, here = heappop(heap)
            if price != dist[here]:
                continue
            for other, edge in reverse[here]:
                new = price+edge
                if new < dist[other]:
                    dist[other] = new
                    heappush(heap,(new,other))
        return dist
    inside = {}
    for i,c in enumerate(cells):
        distance = returns(i)
        for j,t in enumerate(tiles):
            if (i,j) in reduced:
                inside[c,t] = reduced[i,j] + distance[owner.get(j,spare)]
    distance = returns(spare)
    outside = {t:proof['columns'][j]+distance[owner.get(j,spare)] for j,t in enumerate(tiles)}
    return inside,outside


def cases(problem, seed):
    n = len(seed)
    full = (1 << (3*n)) - 1
    out = {'open': [full]*n}
    for name, rotation, kind, sizes in (
        ('shared', 0, 'shared', (69,)),
        ('chain', 4, 'chain', (70,)),
        ('stars', 6, 'stars', (5,)*22),
        ('saturated', 3, 'stars', (5,)*21+(4,)),
        ('spare', 8, 'stars', (5,)*20),
        ('one-possible-away', 1, 'stars', (5,)),
    ):
        ref = proper_rotations(problem.board)[rotation].apply(seed)
        order = list(range(n))
        Random(8953+rotation).shuffle(order)
        ds = [full]*n
        start = 0
        for size in sizes:
            chosen = order[start:start+size]
            start += size
            for i, c in enumerate(chosen):
                if kind == 'chain' and i == size-1:
                    ds[c] = 7 << (3*(ref[c]//3))
                    continue
                if kind == 'shared':
                    alternatives = order[size:size+2]
                elif kind == 'chain':
                    alternatives = [chosen[i+1]]
                else:
                    alternatives = chosen[1:] if i == 0 else chosen[:1]
                ds[c] = (1 << ref[c]) | sum(7 << (3*(ref[x]//3)) for x in alternatives)
        out[name] = ds
    interactions = [
        ('tight-six', 1, ((3,), (3,4), (0,3), (5,), (5,), (1,2,3))),
        ('tight-seven', 9, ((4,), (0,3,6), (1,3,4,5), (4,), (6,), (3,), (0,2))),
        ('slack-one', 2, ((4,), (0,3,4,7), (4,7), (0,2,5,6,7), (1,), (3,6), (1,7), (1,5,6))),
    ]
    for name,rotation,links in interactions:
        ref=proper_rotations(problem.board)[rotation].apply(seed)
        order=list(range(n)); Random(9287+rotation).shuffle(order)
        ds=[full]*n
        # Twenty-one five-tile stars require 63 unchanged poses in total.
        for at in range(0,105,5):
            group=order[at:at+5]
            for i,c in enumerate(group):
                alternatives=group[1:] if i==0 else group[:1]
                ds[c]=1<<ref[c] | sum(7<<(3*(ref[x]//3)) for x in alternatives)
        tail=order[105:105+len(links)]
        for i,c in enumerate(tail):
            ds[c]=1<<ref[c] | sum(7<<(3*(ref[tail[j]]//3)) for j in links[i])
        out[name]=ds
    for rotation,bank in ((0,5),(7,5),(3,10),(8,11)):
        order=list(range(n)); Random(91873+rotation).shuffle(order)
        ref=proper_rotations(problem.board)[rotation].apply(seed)
        a, b, c, reserve = order[:45], order[45:85], order[85:145], order[145:]
        def bank_mask(owners):
            return sum(7 << (3*(ref[x]//3)) for x in owners)
        ds=[full]*n
        for cell in a:
            ds[cell]=1<<ref[cell] | bank_mask(b+c)
        for cell in b:
            ds[cell]=1<<ref[cell] | bank_mask(c)
        for cell in c:
            ds[cell]=bank_mask(c+reserve[:bank])
        out[f'chain-bank-{rotation}-{bank}']=ds
    out['tile-only-broad']=[sum(7<<(3*t) for t in range(n) if (t+c)%7) for c in range(n)]
    out['two-turn-broad']=[sum(3<<(3*t) for t in range(n) if (t+c)%7) for c in range(n)]
    for width in (90,100,110,120,130,140,150):
        for trial in (0,1):
            order=list(range(n));Random(890100+10*width+trial).shuffle(order)
            donors=(order[width:]+order[:width])[:width-66]
            bank=sum(7<<(3*(seed[c]//3)) for c in donors)
            ds=[full]*n
            for c in order[:width]:
                ds[c]=(bank & ~(7<<(3*(seed[c]//3)))) | (1<<seed[c])
            out[f'dense-{width}-{trial}']=ds
    # A held-out interaction that survived full compiled arc consistency,
    # unweighted tile matching support and every direct certificate clause.
    trial=25
    ref=proper_rotations(problem.board)[trial%10].apply(seed)
    order=list(range(n));Random(93835+trial).shuffle(order)
    a,b,c,reserve,tail=order[:44],order[44:83],order[83:133],order[133:143],order[143:149]
    pool=lambda cells:sum(7<<(3*(ref[x]//3)) for x in cells)
    ds=[full]*n
    for cell in a:ds[cell]=1<<ref[cell]|pool(b+c)
    for cell in b:ds[cell]=1<<ref[cell]|pool(c)
    for cell in c:ds[cell]=pool(c+reserve)
    for i,links in enumerate(interactions[0][2]):
        ds[tail[i]]=1<<ref[tail[i]]|pool([tail[j] for j in links])
    out['nested-tight-25']=ds
    return out


def existing_closure(problem, compiled, domains, codes, images, rows):
    from r35_pose_departures import crossover_seams
    from r35_pose_image_pairs import pair_bounds
    cur=domains
    passes=0
    while True:
        passes+=1
        before=tuple(cur)
        cur=compiled.narrow(cur)
        if cur is None:return None,passes,'compiled'
        cur=tight_support(cur,codes)
        if cur is None:return None,passes,'tile_matching'
        cur,stats=propagate(cur,images,rows,65)
        if stats['over']:return None,passes,'direct_poses'
        cur,stats=pair_bounds(cur,images,65)
        if stats['over']:return None,passes,'pair_counts'
        cur,stats=crossover_seams(problem,cur,images)
        if stats['over']:return None,passes,'width_ten'
        if tuple(cur)==before:return cur,passes,'live'


def run(tiles, output, seconds, labels=None, native=None, ac_only=False):
    problem = SearchProblem(json.loads(tiles.read_bytes()), physical_gold_support=True)
    record = active_catalog(problem)
    images = reference_images(tuple(record['codes']))
    rows = indexed_rows(tuple(map(tuple, record['rows'])))
    reports = []
    with ExitStack() as cleanup:
        compiled=None
        if native:
            from bench_r35_pose_frontiers import Compiled
            compiled=Compiled(problem,native)
            cleanup.callback(compiled.close)
        for name, ds in cases(problem, record['codes']).items():
            if labels and name not in labels:
                continue
            def direct(_, domains):
                return propagate(domains, images, rows, 65)
            def prices(_, domains):
                narrowed, stats = direct(None, domains)
                if stats['over']:
                    return narrowed, stats
                return owner_bounds(narrowed, images, 65)
            clear_owner_cache()
            start = perf_counter()
            narrowed, first = prices(None, ds)
            elapsed = perf_counter()-start
            _, repeat = prices(None, ds)
            report = {'case': name, 'changed_domains': sum(v!=(1<<480)-1 for v in ds),
                      'singleton_cells': sum(v.bit_count()==1 for v in ds),
                      'remaining_poses':sum(v.bit_count() for v in narrowed),
                      'isolated_first_seconds': elapsed, 'initial': first, 'repeat': repeat}
            report['candidate']=owner_candidate(ds,65)
            if name in ('saturated','spare','tight-six','tight-seven','slack-one'):
                structural=tight_support(ds,record['codes'])
                weighted=narrowed
                joint=tight_support(weighted,record['codes'])
                if joint is not None:
                    report.update(tile_only_removed=sum((a^b).bit_count() for a,b in zip(ds,structural)),
                                  dual_only_removed=first['removed'],
                                  beyond_separate=sum(((a&b)&~c).bit_count() for a,b,c in zip(structural,weighted,joint)))
            if compiled:
                ac=compiled.narrow(ds)
                report['compiled_live']=ac is not None
                if ac is not None:
                    fixed=tight_support(ac,record['codes'])
                    report['ordinary_tile_live']=fixed is not None
                    if fixed is not None:
                        fixed,simple=direct(None,fixed)
                        report['direct_live_after_compiled_and_tile_matching']=not simple['over']
                        if not simple['over']:
                            clear_owner_cache()
                            weighted,new=owner_bounds(fixed,images,65)
                            report['after_compiled_and_tile_matching']=new
                            if not new['over'] and new['lower']==65:
                                full=tight_support(weighted,record['codes'])
                                if full is not None:
                                    report['incremental_tight_poses']=sum((x&~y).bit_count() for x,y in zip(weighted,full))
            if compiled:
                prior,passes,reason=existing_closure(problem,compiled,ds,record['codes'],images,rows)
                report['prior_fixed_point']={'passes':passes,'status':reason}
                if prior is not None:
                    clear_owner_cache()
                    _,report['owner_after_fixed_point']=owner_bounds(prior,images,65)
            if not ac_only and name not in ('open','one-possible-away','spare','tile-only-broad','two-turn-broad'):
                for label, apply in (('direct',direct),('owner',prices)):
                    with patch('r35_pose_departures.original_pose_departures',side_effect=apply):
                        started=perf_counter()
                        result=search(problem,geography=True,domains=ds,reference_bound=True,seconds=seconds)
                        report[label] = {key:result.get(key) for key in ('status','branches','cp_calls')}
                        report[label]['wall_seconds']=perf_counter()-started
            reports.append(report)
            output.write_text(json.dumps(reports, indent=2)+'\n',encoding='utf8')
            print(name, first, report.get('direct'), report.get('owner'), flush=True)



if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--seconds',type=float,default=8)
    ap.add_argument('--labels',nargs='+')
    ap.add_argument('--native',type=Path,help='Optional compiled CPU propagation library directory.')
    ap.add_argument('--ac-only',action='store_true',help='Skip the time-bounded native embed branches.')
    run(**vars(ap.parse_args()))
