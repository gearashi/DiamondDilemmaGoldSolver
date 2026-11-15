"""Compare exact outside reachability and price every pair of loss groups on real candidates."""
from functools import lru_cache
from itertools import combinations
from math import inf
from pathlib import Path
import argparse
import json
import numpy as np

from r32_gold_bonds import inventory
from r32_gold_loss import read_json,reference_groups
from r32_gold_syndrome import OutsideSyndromes
from search_problem import SearchProblem


def all_pairs(q,trace,cap):
    group_masks=[]
    bit=1
    protected=0
    for group in q.groups:
        mask=0
        for bond in sorted(group):
            mask|=bit
            if bond in q.required:
                protected|=bit
            bit<<=1
        group_masks.append(mask)
        bit<<=1
    pairs=list(combinations(range(len(group_masks)),2))
    counts=[0]*len(pairs)
    complete=0
    any_pair=0
    train=[]
    for pattern in trace:
        image=q._joint_images
        key=image.symbols[0]
        for i in pattern:
            key^=image.symbols[i+1]
        costs=np.array([[max(1,((key^v)&m).bit_count()) for m in group_masks]
                        for v in image.images if not (key^v)&protected],dtype=np.int16)
        if not costs.size:
            continue
        independent=costs.min(axis=0)
        slack=cap-len(pattern)-int(independent.sum())
        assert slack>=0
        complete+=int(costs.sum(axis=1).min()>cap-len(pattern))
        gains=(costs[:,[a for a,b in pairs]]+costs[:,[b for a,b in pairs]]).min(axis=0)
        active=[]
        for j,((a,b),together) in enumerate(zip(pairs,gains)):
            gain=int(together-independent[a]-independent[b])
            if gain>slack:
                counts[j]+=1
            if gain:
                active.append(((1<<a)|(1<<b),gain))
        any_pair+=int(any(v>slack for _,v in active))
        if active:
            @lru_cache(None)
            def best(i,used):
                if i==len(active):return 0
                pair,gain=active[i]
                skip=best(i+1,used)
                return skip if used&pair else max(skip,gain+best(i+1,used|pair))
            train.append(int(best(0,0)>slack))
    return dict(total=len(trace),joint_negatives=complete,any_single=any_pair,
                best_disjoint_cover=sum(train),pair_cuts=[(list(p),v) for p,v in zip(pairs,counts) if v])


def compare(tiles,full):
    p=SearchProblem(read_json(tiles))
    docs=Path(__file__).parent/'docs'
    ref,_,g=reference_groups(p,read_json(docs/'R29_INCUMBENT.json')['codes'])
    req=inventory(p).required
    results=[]
    for cap,e,start,stop in ((14,4,0,6903),(16,6,7044800,7052992),(17,4,0,6903),
                              (17,5,0,65536 if full else 8192)):
        wanted=None
        arms={}
        for enabled in (False,True):
            q=OutsideSyndromes(p,ref,g,req,prune_prefixes=enabled)
            trace=[]
            def probe(indices,*args):
                trace.append(tuple(sorted(indices)))
                return False
            q.joint=probe
            out=q.indexed(cap,e,inf,start,stop)
            if wanted is None:
                wanted=trace
            if trace!=wanted or out['next_rank']!=stop:
                raise AssertionError('changed rank coverage')
            arms[str(enabled)]=dict(work=out['work'],reach_states=q._reachable.states_built if q._reachable else 0)
        results.append(dict(cap=cap,errors=e,start=start,stop=stop,arms=arms,pairs=all_pairs(q,wanted,cap)))
        print(cap,e,arms['True']['work'].get('unreachable_prefixes',0),len(wanted),results[-1]['pairs']['best_disjoint_cover'],flush=True)
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',required=True)
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare(args.tiles,args.full),indent=2)+'\n',encoding='utf8')
