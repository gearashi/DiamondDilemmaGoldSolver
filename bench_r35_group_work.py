"""Deterministic row-insertion budgets for alternative high-rank joint phase searches."""
from collections import Counter
from math import inf
from pathlib import Path
from random import Random
from unittest.mock import patch
import argparse
import json

from r32_gold_syndrome import OutsideSyndromes
from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_partition import insert
from r35_group_drops import reuse_drops
from r35_group_circuits import GroupCircuits
from bench_r35_group_fallback import legacy_drops
from search_problem import SearchProblem


class Budget(Counter):
    def __init__(self, limit):
        super().__init__()
        self.limit = limit

    def __setitem__(self,k,v):
        if k == 'insertions' and v > self.limit:
            raise TimeoutError
        super().__setitem__(k,v)

    def insert(self,basis,row):
        self['insertions'] += 1
        return insert(basis,row)


def compare(tiles, limit, seed):
    p=SearchProblem(read_json(tiles))
    docs=Path(__file__).parent/'docs'
    ref,_,g=reference_groups(p,read_json(docs/'R29_INCUMBENT.json')['codes'])
    req=inventory(p).required
    full=OutsideSyndromes(p,ref,g,req)
    remove=list(full.outside)
    Random(seed).shuffle(remove)
    patterns=[[],*read_json(docs/'R34_INTEGER_CORES_FINAL.json')['arrivals']]
    record=[]
    for removed in (8,16,32,56,80):
        q=OutsideSyndromes(p,ref-set(remove[:removed]),full.groups,req)
        circuit=GroupCircuits(q)
        rows=[]
        for pattern in patterns:
            indices=[q.outside.index(full.outside[i]) for i in pattern if full.outside[i] in q.outside]
            errors=1 | sum(2<<i for i in indices)
            if any((row&errors).bit_count()&1 for row in q._joint_dependencies):
                continue
            for cap in (15,17):
                arms={}
                for name in ('drop','reused','circuit'):
                    work=Budget(limit)
                    try:
                        if name=='drop':
                            with patch('bench_r35_group_fallback.insert',work.insert):
                                answer=legacy_drops(q,indices,cap,inf)
                        elif name=='reused':
                            with patch('r35_group_drops.insert',work.insert):
                                answer=reuse_drops(q,indices,cap,inf,work)
                        else:
                            answer=circuit.solve(indices,cap,inf,work)
                    except TimeoutError:
                        answer=None
                    arms[name]={'answer':answer,'work':dict(work)}
                if len({a['answer'] for a in arms.values() if a['answer'] is not None})>1:
                    raise AssertionError((removed,indices,cap,arms))
                rows.append(arms)
        result={}
        for name in ('drop','reused','circuit'):
            completed=[row[name] for row in rows if row[name]['answer'] is not None]
            common=[row[name] for row in rows if all(a['answer'] is not None for a in row.values())]
            result[name]=dict(completed=len(completed),positive=sum(a['answer'] for a in completed),
                              all_completed=len(common),insertions_common=sum(a['work']['insertions'] for a in common))
        print(seed,removed,result,flush=True)
        record.append(dict(removed=removed, count=len(rows),summary=result))
    return dict(seed=seed,row_insertion_limit=limit,cases=record)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    parser.add_argument('--limit',type=int,default=65536)
    parser.add_argument('--seeds',type=int,nargs='+',default=[35,71])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=[compare(args.tiles,args.limit,seed) for seed in args.seeds]
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
