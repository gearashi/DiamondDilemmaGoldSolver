"""Reproduce round 30 move, cluster, rotation and bounded-swap measurements."""
import argparse
from itertools import combinations
import math
from pathlib import Path

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, read_json, domain_binding
from r29_exchange_clusters import StarClusters
from r30_bounds import (RotationRegions, RotationForest, ResponseGroups, budget_upper,
    can_hit_cores, cell_mask, defect_cores, defect_packings, family_upper,
    grouped_rotation_bound, pattern_distance_upper, report_family, family_case_upper)
from search_problem import SearchProblem


def excluded_supports(n, size, cores):
    return sum((-1)**(k+1)*math.comb(n-joined.bit_count(),size)
               for k in range(1,len(cores)+1) for group in combinations(cores,k)
               for joined in [union(group)])


def union(rows):
    result = 0
    for r in rows:
        result |= r
    return result


def benchmark(problem, *, pair_limit=710):
    record = read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')
    model = BoardModel(problem)
    if record['domain']!=domain_binding(model):
        raise ValueError("The benchmark needs the release puzzle domains.")
    result = {}
    for row in record['centers']:
        bound = ExchangeBound(model,[3*t for t in row['pieces']])
        clusters = StarClusters(bound)
        cores = defect_cores(bound,208)
        pairs = RotationRegions(bound,2)
        paths = RotationRegions(bound,3)
        rings = RotationRegions(bound,6,rings=True)
        tree = RotationForest(model)
        packed = defect_packings(bound)
        regions = [sorted(set(range(model.n))-set(i for i in range(model.n) if cores[0] & (1<<i))),
                   list(range(6)), [0,1,18,30,44,118,146]]
        potential = [dict(cells=len(c), unrestricted=budget_upper(bound,packed,4),
                          bounded=budget_upper(bound,packed,4,c)) for c in regions]
        potential.extend(pattern_distance_upper(bound,regions[-1],d,max_nodes=3500) for d in (1,2))
        groups = ResponseGroups(bound)
        class_counts = []
        for c in range(model.n):
            by_score = groups.groups(c,range(model.n))
            class_counts.append(len(by_score))
        low = []
        high = []
        rng = np.random.default_rng(51)
        upper_before = np.zeros(2,int)
        count = 0
        for a,b in combinations(range(model.n),2):
            if not clusters.close[a,b] or count>=pair_limit:
                continue
            count += 1
            price = family_upper(clusters,(a,b),extra=4)
            upper_before += (price<416,price<=414)
            if price>=416:
                high.append((a,b))
            else:
                low.append((a,b))
        one = report_family(clusters,high[0] if high else (0,1),208,4,nodes=16)
        families = []
        for active in [(0,1),(0,1,2),(0,1,2,3),(0,1,138,139)]:
            pool = list(range(model.n))
            price = family_upper(clusters,active,extra=max(0,6-len(active)),candidates=pool)
            families.append(dict(cluster=active,upper_twice=price))
        forest_stats = []
        for perm in (bound.tiles,np.roll(bound.tiles,1)):
            seed = ExchangeBound(model,[3*int(t) for t in perm])
            opts = [tree.upper(perm,r) for r in (1,4)]
            hits = tree.hits
            repeat = tree.upper(perm,4)
            assert repeat==opts[-1]
            forest_stats.append(dict(star=seed.total//2,forest=opts,warm_message_hits=tree.hits-hits))
        samples = []
        free_set = [c for c in range(model.n) if not cores[0] & (1<<c)]
        for cells in (list(rng.choice(model.n,6,replace=False)) for _ in range(384)):
            mask = cell_mask(cells)
            core_closed = any(not core & mask for core in cores)
            upper = clusters.support_upper_twice(cells)
            samples.append((core_closed,upper<416,upper-pairs.discount(mask)<416))
        result[row['sha256'][:16]]=dict(
            domain=record['domain'],target=208,extra=4,closed_anchors=' '.join(f'{a}-{b}' for a,b in low),
            base_twice=bound.total,core_sizes=[core.bit_count() for core in cores],
            excluded_supports=[excluded_supports(model.n,k,cores) for k in (6,7,8)],
            core_first=sum(x[0] for x in samples), support_closed=sum(x[1] for x in samples),
            with_pairs=sum(x[2] for x in samples),sample_size=len(samples),
            partial_zero_hit=not can_hit_cores(cores,0,cell_mask(free_set),6),
            savings_twice=[r.discount() for r in (pairs,paths,rings)],
            pair_families=count,closed_by_family=int(upper_before[0]),
            pair_families_207_or_lower=int(upper_before[1]),family_refinement=one,
            cluster_families=families,pattern_search=potential,forest=forest_stats,
            donor_groups=dict(min=min(class_counts),mean=sum(class_counts)/model.n,
                              max=max(class_counts),initial_evaluations=model.n**2,
                              grouped_evaluations=sum(class_counts)))
    return result


def followups(problem):
    from r29_plateau import walk
    from search_fixtures import planted_gold
    model = BoardModel(problem)
    center = read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')['centers'][1]['pieces']
    bound = ExchangeBound(model,3*np.asarray(center))
    forest = RotationForest(model)
    swaps = []
    for a,b in combinations([0,17,18,29,30,41,44,45,62,118,146],2):
        tiles = bound.tiles.copy()
        tiles[a],tiles[b]=tiles[b],tiles[a]
        other = ExchangeBound(model,3*tiles)
        base = other.total//2
        upper = forest.upper(tiles,1)
        swaps.append(dict(cells=[a,b],star=base,forest=upper))
    response = ResponseGroups(bound)
    response_work = []
    for c in range(model.n):
        scores,classes=grouped_rotation_bound(bound,c,range(model.n),response_cache=response)
        legal = model.legal[c].reshape(model.n,3)
        price = np.zeros((model.n,3),dtype=int)
        for side,nb in enumerate(bound.neighbors[c]):
            if nb!=model.n:
                price += bound.support[bound.masks[:model.n,:,side],bound.tiles[nb]]
        exact = np.where(legal,price,-1).max(1).tolist()
        if [scores[t] for t in range(model.n)]!=exact:
            raise RuntimeError("Grouped rotation response lost a donor.")
        response_work.append(classes)
    stored = len(response.evaluations)
    for c in range(model.n):
        grouped_rotation_bound(bound,c,range(model.n),response_cache=response)
    fixture,geo,codes=planted_gold(31,1,1)
    small=SearchProblem(fixture,geo,{i:codes[i] for i in range(6,len(codes))})
    start=codes.copy()
    for i in range(6):
        start[i] = codes[(i+1)%6]
    walk_records=[]
    for use in (False,True):
        work=walk(small,start,':memory:',0,len(small.edges),max_centers=200,tie_limit=1,max_swaps=1,potentials=use)
        walk_records.append({key:work[key] for key in ('potential_closed','closed','solved','reached','score')})
    cluster = StarClusters(bound)
    split = family_case_upper(cluster,(0,1),[2,3,18,44],merge_slots=6,max_parts=256)
    return dict(tree_swaps=swaps,trees_tighter=sum(r['forest']<r['star'] for r in swaps),
                actual_group_counts=response_work,physical_full=model.n**2,
                representative_evaluations=stored,warm_new_evaluations=len(response.evaluations)-stored,
                bounded_walk=walk_records,cluster_overlap=split)


def unrestricted_families(problem):
    """Prove the single-contact family with no bound on distant movers."""
    record = read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')
    model = BoardModel(problem)
    if domain_binding(model)!=record['domain']:
        raise ValueError('The reference center has different physical domains.')
    rows = {}
    for row in record['centers']:
        bound = ExchangeBound(model,[3*t for t in row['pieces']])
        clusters = StarClusters(bound)
        pairs = [(a,b) for a,b in combinations(range(model.n),2) if clusters.close[a,b]]
        closed = [f'{a}-{b}' for a,b in pairs if family_upper(clusters,(a,b))<416]
        rows[row['sha256'][:16]] = dict(target=208,domain=record['domain'],count=len(closed),
                                       pair_families=len(pairs),closed_anchors=' '.join(closed))
    return rows


def seven_pilot(tiles, baseline):
    """Run two actual source versions on identical cold colex prefixes."""
    import json
    import subprocess
    import sys
    script = """
import json
import sys
from r28_exchange import read_json
from r29_exchange_route import run
from search_problem import SearchProblem
tiles,center_path,batch,pages = sys.argv[1:]
center = read_json(center_path)['centers'][1]
problem = SearchProblem(read_json(tiles))
result = run(problem,[3*t for t in center['pieces']],208,':memory:',distance=4,size=7,
             backend='cpu',reuse_balls=False,batch=int(batch),max_pages=int(pages))
names = ('center','offset','core_closed','support_closed','considered','exact','hits','seconds')
print(json.dumps({n:result.get(n,0) for n in names}))
"""
    project = Path(__file__).resolve().parent
    center = project/'docs/R29_EXCHANGE_CENTERS.json'
    result = []
    for batch,pages in [(12,1),(128,8)]:
        row = dict(batch=batch,pages=pages,threshold=208,orders_per_support=210)
        for key,path in [('v1.10.0',baseline),('current',project)]:
            output = subprocess.run([sys.executable,'-B','-c',script,str(tiles.resolve()),str(center),
                                     str(batch),str(pages)],cwd=path,check=True,capture_output=True,text=True)
            row[key] = json.loads(output.stdout)
        old,new = row['v1.10.0'],row['current']
        if (old['center'],old['offset'],old['hits']) != (new['center'],new['offset'],new['hits']):
            raise RuntimeError('Seven-cell pilot scope or results differ.')
        result.append(row)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pair-limit',type=int,default=710)
    parser.add_argument('--followups',action='store_true',help='Benchmark rotation grouping, bounded walks and tree updates.')
    parser.add_argument('--unrestricted-families',action='store_true',help='Prove touching pairs plus arbitrarily many isolated movers.')
    parser.add_argument('--seven-baseline',type=Path,help='Compare exact seven-cell prefixes to a v1.10.0 checkout.')
    args = parser.parse_args()
    problem = SearchProblem(read_json(args.tiles))
    if args.unrestricted_families:
        result = unrestricted_families(problem)
    elif args.seven_baseline is not None:
        result = seven_pilot(Path(args.tiles),args.seven_baseline)
    else:
        result = followups(problem) if args.followups else benchmark(problem,pair_limit=args.pair_limit)
    atomic_json(args.output,result)
