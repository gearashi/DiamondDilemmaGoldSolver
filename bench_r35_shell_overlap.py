"""Rebuild the three-center published overlap and audit new separation decisions."""
from collections import Counter
from itertools import combinations
from math import comb
from pathlib import Path
import argparse,json
import numpy as np
from r27_incumbent import BoardModel
from r28_exchange import read_json
from r29_exchange_route import owner_copies
from r34_centers import archive_boards
from r34_cycle_orders import CycleOrders
from r35_shell_overlap import class_mask,cycle_type,distance,intersection,order_matrix
from search_problem import SearchProblem


FAMILIES=((2,4),(3,3),(2,2,3),(2,2,2,2))


def third_discount(tiles,k):
    """Independently price the other published center's inner-ball exclusions."""
    base=np.asarray(tiles[2]);centers=[np.asarray(t) for t in tiles]
    relations=[np.argsort(center)[base] for center in centers]
    specials=tuple(i for i in range(len(base)) if any(r[i]!=i for r in relations))
    normal=tuple(i for i in range(len(base)) if i not in specials)
    full=discount=0
    for inside in range(min(k,len(specials))+1):
        outside=k-inside;weight=comb(len(normal),outside)
        for subset in combinations(specials,inside):
            support=(*subset,*normal[:outside]);cells=tuple(sorted(set(specials)|set(support)))
            idx={c:j for j,c in enumerate(cells)};local=np.array([idx[c] for c in support])
            for f in (f for f in FAMILIES if sum(f)==k):
                orders=order_matrix(f)
                own=np.broadcast_to(np.arange(len(cells)),(len(orders),len(cells))).copy();own[:,local]=local[orders]
                def other(j):
                    return np.asarray([idx[relations[j][c]] for c in cells])[own]
                eligible=set()
                for g in FAMILIES:
                    if sum(g)<=len(cells):eligible.update(map(int,class_mask(other(0),g)))
                full+=weight*len(eligible)
                middle=other(1)
                for row in eligible:
                    obj=cycle_type(middle[row])
                    if sum(obj)<=5 or distance(obj)<=3:discount+=weight
    return full,discount


def original(model,tiles):
    rel=np.argsort(tiles[0])[tiles[2]];delta=cycle_type(rel)
    historical=read_json('docs/R30_SHARED_RESULTS.json')
    report=[]
    for k in (6,7,8):
        before=sum(intersection(delta,f,g,model.n) for f in FAMILIES if sum(f)==k for g in FAMILIES)
        check,discount=third_discount(tiles,k);assert check==before,(check,before)
        expected=historical['by_size'][str(k)]['overlap']['already_owned_eligible']
        assert before-discount==expected,(k,before,discount,expected)
        report.append(dict(support=k,before_earlier_balls=before,earlier_excluded=discount,eligible=expected))
    assert sum(r['eligible'] for r in report)==668394211290
    return report


def main():
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--runtime',type=Path,default=Path('../runtime'));cli.add_argument('--output',type=Path);args=cli.parse_args()
    model=BoardModel(SearchProblem(read_json(args.tiles)))
    centers=[c['pieces'] for c in read_json('docs/R29_EXCHANGE_CENTERS.json')['centers']]
    report={'published_overlap':original(model,centers)}
    from r29_exchange_route import legal_cell_rotations
    transforms=legal_cell_rotations(model)
    physical=centers+[list(np.asarray(b)//3) for b in archive_boards([args.runtime/'fresh-cpu.sqlite',args.runtime/'fresh-cuda.sqlite'])]
    corpus=Counter()
    for a,b in combinations(physical,2):
        for f in FAMILIES:
            corpus['center_family_pairs']+=1
            for mapping,_ in owner_copies(model,[(a,sum(f))],b,sum(f),transforms=transforms):
                d=cycle_type(mapping);corpus['hamming_eligible_copies']+=1
                if distance(d)%2:corpus['parity_rejected']+=1
                elif intersection(d,f,f,model.n)==0:corpus['new_class_rejections']+=1
                else:corpus['intersecting_copies']+=1
    report['installed_centers']=dict(corpus)
    # Each new fixture is an actual legal original board with a different
    # physical permutation. Only its center/family relation is controlled.
    from r30_multicenter import SharedSupports
    from r35_shell_overlap import can_meet
    from unittest.mock import patch
    from r30_multicenter import sparse_cycle_type
    def partitions(total,minimum=2):
        if not total:yield ()
        for size in range(minimum,total+1):
            for rest in partitions(total-size,size):yield (size,*rest)
    for_family=[]
    for family in FAMILIES:
        checks=Counter()
        for h in range(2,2*sum(family)+1):
            for delta in partitions(h):
                if not can_meet(delta,family,family) or intersection(delta,family,family,model.n):continue
                first=np.asarray(centers[0]);second=first.copy();pos=0
                for n in delta:
                    second[pos:pos+n]=np.roll(first[pos:pos+n],-1);pos+=n
                model.legal_board(3*second)
                g=SharedSupports(model,[3*first,3*second],(family,))
                assert not g.copies[1]
                options=tuple(CycleOrders(family));cells=tuple(range(sum(family)))
                old=[]
                for m,_,tr in owner_copies(model,[(first,sum(family))],second,sum(family),transforms=transforms,with_transform=True):
                    if distance(cycle_type(m))%2==0:
                        old.append(({i:j for i,j in enumerate(m) if i!=j},0,np.argsort(tr)))
                with patch('r30_multicenter.sparse_cycle_type',wraps=sparse_cycle_type) as observe:
                    assert all(g.earlier_location(1,cells,o) is None for o in options)
                    checks['new_relative_tests']+=observe.call_count
                    g.copies[1]=old
                    assert all(g.earlier_location(1,cells,o) is None for o in options)
                    checks['old_relative_tests']+=observe.call_count
                checks['real_board_pairs']+=1
        for_family.append(dict(family=family,**checks))
    report['constructed_original_separations']=for_family
    from tempfile import TemporaryDirectory
    from search_fixtures import planted_gold
    from r34_centers import select
    from random import Random
    tiles,geo,gold=planted_gold(3412,1,4)
    q=SearchProblem(tiles,geo,fixed={9:gold[9]});f=(2,2,3)
    near=[list(gold)];rng=Random(3412)
    for selected_cells in ((0,1,2),(1,2,3),(4,5,6)):
        trial=list(gold)
        for a,b in zip(selected_cells,(*selected_cells[1:],selected_cells[0])):trial[a]=gold[b]
        near.append(trial)
    for _ in range(3):
        trial=list(gold);trial[:9]=rng.sample(gold[:9],9);near.append(trial)
    scratch=Path('audit/r35');scratch.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=scratch) as tmp:
        choices,_,detail=select(q,near,Path(tmp)/'novel.db',f,0,len(near),policy='novel')
    known=set();values=[]
    for item in detail['selected']:
        t=list(choices[item['center']]['board']//3);current=set()
        for cells in combinations(range(9),sum(f)):
            for order in order_matrix(f):
                result=t.copy()
                for c,i in zip(cells,order):result[c]=t[cells[i]]
                current.add(tuple(result))
        advance=len(current-known)
        if known:assert item['priority'][0]==advance,(item,advance)
        known.update(current);values.append(advance)
    report['materialized_novelty']=dict(incremental=values,union=len(known),exact_queries=detail['exact_novelty_counts'],
                                         sampled_queries=detail['sampled_novelty_counts'])
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(report))


if __name__=='__main__':main()
