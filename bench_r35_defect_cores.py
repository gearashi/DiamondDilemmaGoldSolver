"""Replay the Round 34 donor-prefix corpus with exact local clauses and durable checked cores."""
from collections import Counter
from itertools import combinations
import numpy as np


def direct_rows(bound,group,free,remaining,target):
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
    failures = len(inventory.edges)-target
    rows = [(edges,sum(1<<int(c) for c in available.intersection(cells))) for edges,cells in requirements]
    return rows


def main():
    import argparse,json,sqlite3
    from pathlib import Path
    from contextlib import closing
    from tempfile import TemporaryDirectory
    from r27_incumbent import BoardModel
    from r28_exchange import ExchangeBound,domain_binding,read_json
    from r32_route_pool import CoverCores,defect_cover
    from r35_defect_inventory import for_bound
    from search_problem import SearchProblem
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--historical',default='../round34/audit/r34/repair-cover-replay.json')
    cli.add_argument('--output',type=Path);args=cli.parse_args()
    model=BoardModel(SearchProblem(read_json(args.tiles)))
    centers={r['sha256']:ExchangeBound(model,[3*t for t in r['pieces']]) for r in read_json('docs/R29_EXCHANGE_CENTERS.json')['centers']}
    historical=read_json(args.historical)
    items=historical['records']; report=dict(historical_work=historical['work'],runs=[])
    for i,r in enumerate(items):
        bound=centers[r['center']];pool=model.free if r['eligible']==len(model.free) else tuple(range(r['eligible']))
        calc=for_bound(bound).rows(r['prefix'],pool); expected=direct_rows(bound,r['prefix'],pool,r['movers'],r['target'])
        assert calc==expected,(i,calc,expected)
    work=Counter()
    for bound in centers.values():work.update(for_bound(bound).work)
    report['exact_clauses']=dict(direct_stars=len(items)*model.n,**work)
    scratch=Path('audit/r35');scratch.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=scratch) as tmp:
        for label,cases in [('first',items),('reopen_reversed',list(reversed(items)))]:
            with closing(sqlite3.connect(Path(tmp)/'certs.db',isolation_level=None)) as db:
                cache=CoverCores(db,domain_binding(model),verify_seconds=4)
                results=Counter(); retried=0; traces=[]
                for i,r in enumerate(cases):
                    bound=centers[r['center']];pool=model.free if r['eligible']==len(model.free) else tuple(range(r['eligible']))
                    args1=(bound,r['prefix'],pool,r['movers'],r['target'])
                    for seconds in (4,30,240):
                        out=defect_cover(*args1,seconds=seconds,cores=cache)
                        if out['status']!='UNKNOWN':break
                        retried+=1
                    assert out['status']!='UNKNOWN' and out['closed']==(r['cold']=='INFEASIBLE'),(i,r,out)
                    results[out['status']]+=1
                    traces.append([r['center'],r['prefix'],r['eligible'],r['movers'],r['target'],out['status']])
                    if i%40==0:print(label,i,dict(results),flush=True)
                report['runs'].append(dict(label=label,statuses=dict(results),work=dict(cache.work),
                        retried=retried,live_cores=len(cache.cores),checked_queries=len(traces)))
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(report))


if __name__=='__main__':main()
