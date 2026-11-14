"""Measure independent versus shared four-swap pricing, union ownership, and CUDA filters."""
import argparse
from collections import Counter
import itertools
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import time

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, read_json
from r30_bounds import defect_cores
from r30_donor_dd import diagram
from r30_gpu_filters import contact_counts, core_passes
from r30_multicenter import SharedSupports, cycles, orders, run
from search_problem import SearchProblem


def signature_counts(tiles,k):
    """Exact counts of local-center overlaps. Ordinary cells can be relabeled in this count."""
    n = len(tiles[0])
    special = tuple(i for i in range(n) if len({row[i] for row in tiles})!=1)
    if len(special)>6:
        raise ValueError("Center-difference enumeration is intended for nearby centers.")
    normal = [i for i in range(n) if i not in special]
    families = {6:((2,4),(3,3)),7:((2,2,3),),8:((2,2,2,2),)}[k]
    pairs = Counter()
    raw = prior = duplicates = 0
    for c,base in enumerate(tiles):
        inverse = [{tile:i for i,tile in enumerate(t)} for t in tiles]
        for q in range(min(k,len(special))+1):
            weight = math.comb(len(normal),k-q)
            if not weight:
                continue
            for fixed in itertools.combinations(special,q):
                support = (*fixed,*normal[:k-q])
                # Only these cells can differ between an endpoint and any reference.
                involved = sorted(set(special).union(support))
                idx = {v:i for i,v in enumerate(involved)}
                for f in families:
                    for order in orders(f):
                        current = {i:base[support[j]] for i,j in zip(support,order)}
                        types = []
                        for inv in inverse:
                            rel = tuple(idx[inv[current.get(i,base[i])]] for i in involved)
                            types.append(cycles(rel))
                        raw += weight
                        if any(sum(h)<=5 or sum(h)-len(h)<=3 for h in types):
                            prior += weight
                            continue
                        other = [i for i,h in enumerate(types) if sum(h)-len(h)==4
                                 and h in ((2,4),(3,3),(2,2,3),(2,2,2,2))]
                        if min(other)!=c:
                            duplicates += weight
                            pairs[str((c,min(other)))] += weight
    return dict(raw_center_endpoints=raw,in_prior_balls=prior,already_owned_eligible=duplicates,
                remaining_unique=raw-prior-duplicates,ordered_pairs=dict(pairs))


def benchmark(problem,boards,samples=100,device=None):
    model = BoardModel(problem)
    bs = [ExchangeBound(model,c) for c in boards]
    rng = np.random.default_rng(3020)
    families = ((2,4),(3,3),(2,2,3),(2,2,2,2))
    metrics = {}
    for k in (6,7,8):
        fm = tuple(f for f in families if sum(f)==k)
        graph = SharedSupports(model,boards,fm)
        from r30_multicenter import CountingTerms
        stats = Counter()
        measured_dd = []
        for i in range(samples):
            cells = tuple(sorted(map(int,rng.choice(problem.n,k,replace=False))))
            local = CountingTerms(stats)
            values = {}
            for j,b in enumerate(bs):
                cold = graph.clusters[j].support_upper_twice(cells)
                if graph.upper(j,cells,values,local,stats)!=cold:
                    raise RuntimeError("Cross-center or partial-term price differs from independent search.")
            if i<3:
                for f in fm:
                    row = diagram(bs[1],cells,f)
                    baseline = max(graph.clusters[1].price(cells,o) for o in orders(f))
                    if row['upper_twice']!=baseline:
                        raise RuntimeError("Diagram and distinct exhaustive donor orders disagree.")
                    unmerged = diagram(bs[1],cells,f,merge=False)
                    if unmerged['upper_twice']!=baseline or row['transitions']>unmerged['transitions']:
                        raise RuntimeError("Decision diagram changed the exhaustive price or expanded more states.")
                    measured_dd.append(dict(family=f,orders=len(orders(f)),
                                            unmerged_transitions=unmerged['transitions'],**row))
        metrics[str(k)] = dict(separate_assignments=len(bs)*samples,shared=dict(stats),
                               diagram=measured_dd,overlap=signature_counts([list(b.tiles) for b in bs],k))
    comparisons = []
    with TemporaryDirectory() as tmp:
        for mode,flt,sharing in [('v29-equivalent',('star',),False),
                                 ('core-first',('core','star','diagram'),True),
                                 ('support-first',('star','core','diagram'),True)]:
            result = run(problem,boards,Path(tmp)/(mode+'.sqlite'),start=2000000,stop=2000512,
                         filters=flt,share=sharing,reuse_release=False)
            warm = run(problem,boards,Path(tmp)/(mode+'.sqlite'),start=2000000,stop=2000512,
                       filters=flt,share=sharing,reuse_release=False)
            if warm['new_pages'] or warm['best']!=result['best']:
                raise RuntimeError("Identical range replayed work.")
            comparisons.append(dict(mode=mode,stats=result['work'],warm_new_pages=warm['new_pages'],
                                    best=result['best']))
    if any(row['best']!=comparisons[0]['best'] for row in comparisons):
        raise RuntimeError("Filter ordering changed best result.")
    result = dict(supports_per_size=samples,by_size=metrics,rank_range=[2000000,2000512],
                  comparison=comparisons)
    # Equal input sets for a marginal first-proof comparison. The last 24 are
    # sampled outside one movement certificate to expose overlap with the star rule.
    masks = [defect_cores(b,208) for b in bs]
    outside = [v for v in range(problem.n) if not (masks[1][0]>>v)&1]
    rows = [tuple(sorted(map(int,rng.choice(pool,7,replace=False)))) for pool in
            [range(problem.n)]*96+[outside]*24]
    passes = core_passes(rows,masks,problem.n)
    graph = SharedSupports(model,boards,((2,2,3),))
    counts = {name:Counter() for name in ('core_first','support_first')}
    for row_index,cells in enumerate(rows):
        for j,cluster in enumerate(graph.clusters):
            core = not passes[j,row_index]
            star = cluster.support_upper_twice(cells)<416
            for name,order in [('core_first',[('core',core),('star',star)]),
                               ('support_first',[('star',star),('core',core)])]:
                counts[name]['center_queries']+=1
                for rule,failed in order:
                    counts[name][rule+'_evaluations']+=1
                    if failed:
                        counts[name]['first_'+rule]+=1
                        break
                else:
                    counts[name]['passed']+=1
    result['matched_order_test'] = dict(random=96,core_exterior=24,by_order=counts)
    geo = contact_counts(rows,graph.clusters[0].close)
    emitted_without_contact = len(rows)*len(bs)
    emitted_with_contact = int((geo>=2).sum())*len(bs)
    result['matched_contact_test'] = dict(center_queries=emitted_without_contact,
        center_queries_after_two_contacts=emitted_with_contact,
        donor_orders_avoided_before_other_screens=(emitted_without_contact-emitted_with_contact)*210)
    result['larger_rank_ranges'] = []
    with TemporaryDirectory() as tmp:
        for family in ((2,2,3),(2,2,2,2)):
            for mode in (('core','star'),('contact','core','star')):
                db = Path(tmp)/(''.join(map(str,family))+'_'.join(mode)+'.sqlite')
                args = dict(families=(family,),start=200000000000,stop=200000001024,filters=mode,page=512)
                row = run(problem,boards,db,**args)
                warm = run(problem,boards,db,backend=device or 'cpu',**args)
                if row['best']!=warm['best'] or warm['new_pages']!=0:
                    raise RuntimeError("Backend resume replayed a completed four-swap page.")
                result['larger_rank_ranges'].append(dict(family=family,filters=mode,start=args['start'],
                    stop=args['stop'],work=row['work'],warm_new_pages=warm['new_pages']))
    if device:
        rows = np.array([rng.choice(problem.n,8,replace=False) for _ in range(32768)],np.int32)
        masks = [defect_cores(b,208) for b in bs]
        scores = {}
        times = {}
        for backend in ('cpu',device):
            core_passes(rows[:8],masks,problem.n,backend)  # warm library load
            start = time.perf_counter()
            scores[backend] = core_passes(rows,masks,problem.n,backend)
            times[backend] = time.perf_counter()-start
        geo = contact_counts(rows,graph.clusters[0].close)
        gpu_geo = contact_counts(rows,graph.clusters[0].close,device)
        if not np.array_equal(geo,gpu_geo):
            raise RuntimeError("GPU contact geometry differs from CPU.")
        if not np.array_equal(scores['cpu'],scores[device]):
            raise RuntimeError("GPU filter disagrees with independent CPU result.")
        result['device'] = dict(supports=len(rows),center_queries=scores['cpu'].size,
                                rejections=int((~scores['cpu']).sum()),supports_with_fewer_than_two_contacts=int((geo<2).sum()),
                                supports_with_fewer_than_three_contacts=int((geo<3).sum()),warm_seconds=times)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--samples',type=int,default=100)
    p.add_argument('--cuda',action='store_true')
    args=p.parse_args()
    record=read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')
    tiles=SearchProblem(read_json(args.tiles))
    from r28_exchange import domain_binding
    if domain_binding(BoardModel(tiles))!=record['domain']:
        p.error('Benchmark requires the release puzzle.')
    boards=[[3*t for t in c['pieces']] for c in record['centers']]
    atomic_json(args.output,benchmark(tiles,boards,args.samples,'cuda' if args.cuda else None))
