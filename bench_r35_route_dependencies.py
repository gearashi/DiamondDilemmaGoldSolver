"""Verify released intervals on installed ledgers and trace transformed cycle families on a finite board."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path
import sqlite3

import numpy as np

from r27_incumbent import BoardModel
from r27_structure_fixtures import planted
from r30_multicenter import SharedSupports,orders
from r32_route_store import PageProofs,colex
from r30_multicenter import cycles


def transformed():
    p,_=planted(1,4,71);families=((2,4),(3,3))
    model=BoardModel(p);original=[3*t for t in range(p.n)]
    changed=original.copy()
    for a,b in ((0,1),(5,6)):changed[a],changed[b]=changed[b],changed[a]
    graph=SharedSupports(model,[original,changed],families)
    perms=[np.asarray(v)//3 for v in (original,changed)]
    counts=Counter();ranks=Counter()
    for cells in combinations(range(p.n),6):
        for f in families:
            for o in orders(f):
                hit=graph.earlier_location(1,cells,o)
                if hit is None:continue
                owner,rank,actual=hit
                endpoint=perms[1].copy()
                endpoint[list(cells)]=perms[1][[cells[t] for t in o]]
                ok=False
                for rotation in graph.transforms:
                    rel=np.argsort(perms[owner])[endpoint[list(rotation)]]
                    moved=np.flatnonzero(rel!=np.arange(p.n))
                    if cycles(rel)==actual and colex(moved,graph.indices)==rank:ok=True;break
                assert ok
                counts[str((f,actual))]+=1
                ranks[(colex(cells,graph.indices),f,rank,actual)]+=1
    return dict(families=dict(counts),distinct_links=len(ranks),endpoints=sum(counts.values()))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--ledger',type=Path,default=Path('../runtime/four-swap-union.sqlite'))
    cli.add_argument('--output',type=Path);args=cli.parse_args()
    db=sqlite3.connect(args.ledger.absolute().as_uri()+'?mode=ro',uri=True)
    pages=db.execute("""SELECT p.scope,p.owner,m.family,m.target,p.lo,p.hi,p.deps,r.best
        FROM r32_page_proof p JOIN r31_members m ON m.scope=p.scope AND m.owner=p.owner
        JOIN r30_pages r ON r.scope=p.scope AND r.lo=p.lo""").fetchall()
    proofs={};queries=0
    for scope,owner,raw,target,lo,hi,deps,best in sorted(pages):
        checker=proofs.setdefault(target,None)
        if checker is None:checker=proofs[target]=PageProofs(db,target)
        if deps!='[]' or best!='null':continue
        family=tuple(json.loads(raw))
        for rank in range(lo,hi):
            assert checker.proven_end(scope,owner,rank,family=family)>rank
            queries+=1
    work=Counter()
    for v in proofs.values():work.update(v.work)
    report=dict(installed=dict(receipts=len(pages),source_points=queries,point_baseline_reads=queries,
                               old_rows_with_dependencies=sum(r[-2]!='[]' for r in pages),**work),
                finite=transformed())
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(report))

if __name__=='__main__':main()
