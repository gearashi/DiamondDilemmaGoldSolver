"""Compare reusable mixed-turn polygons against named component root-pose enumeration."""
import argparse
from collections import Counter
from itertools import product
import json
from pathlib import Path
from random import Random

from r27_structure_filters import Work,domain_bits
from r27_structure_fixtures import planted
from r28_exchange import read_json
from r28_geo_sewing import PosePacking
from r30_gold import search
from r31_geographic import physical_bonds
from r32_gold_bonds import gold_matching_supports,inventory
from r35_corner_chains import filter_options
from r35_turn_words import TurnWords,canonical
from search_problem import SearchProblem


def census(p):
    tab=TurnWords(p)
    out=[]
    for size in range(3,11):
        words={canonical(tuple((t,0,turn) for t,turn in enumerate(word)))[0]
               for word in product((1,2),repeat=size) if len(set(word))>1}
        possible=sum(bool(tab.roots(word)) for word in words)
        out.append(dict(length=size,mixed_classes=len(words),feasible=possible,impossible=len(words)-possible))
    return out


def scopes(p):
    rules=inventory(p)
    choices=sorted(set(physical_bonds(p,read_json('docs/R29_INCUMBENT.json')['codes']))&rules.possible)
    words=TurnWords(p)
    poses=PosePacking(p)
    out=[]
    distinct=[]
    for n in (4,12,24,48,96,144):
        stats=Counter()
        for seed in range(10):
            pins=Random(seed).sample(choices,n)
            fixed=dict(pins)
            fixed.update({b:a for a,b in pins})
            rows=[set() for _ in range(3*p.n)]
            for a,b in rules.possible:
                if fixed.get(a,b)==b and fixed.get(b,a)==a:
                    rows[a].add(b);rows[b].add(a)
            supported=gold_matching_supports(p,rows,Work(2000000))
            if supported is None:
                stats['prior_matching_closed']+=1
                continue
            options={(a,b) for a,row in enumerate(supported) for b in row if a<b}
            required={(a,b) for a,b in options if len(supported[a])==1 or len(supported[b])==1}
            prior=filter_options(p,options,required)[0]
            removed=words.conditional(options,required)
            if removed==options:
                stats['all_closed']+=1
                stats['closed_beyond_corners']+=bool(prior)
                assert poses.check(required,max_steps=10000)['feasible'] is False
            else:
                stats['removed']+=len(removed)
                stats['removed_beyond_corners']+=len(removed&prior)
                for pair in sorted(removed&prior):
                    explain=TurnWords(p)
                    conflicts=explain.conflicts(required|{pair},maximum=10000)
                    core=next(edges for edges in conflicts if pair in edges)
                    distinct.append(dict(pins=n,seed=seed,mate=pair,cycle=sorted(core)))
                for pair in removed:
                    result=poses.check(required|{pair},max_steps=10000)
                    assert result['feasible'] is False,(pair,result)
            stats['tested_scopes']+=1
        out.append(dict(bonds=n,**stats))
    return dict(scopes=out,work=words.stats,distinct_geographic_cores=distinct)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--output',type=Path)
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles))
    out=dict(word_classes=census(p),original_residuals=scopes(p))
    original=set(map(tuple,read_json('docs/R35_NONBLANK_SEWING.json')['nonblank']))
    poses=PosePacking(p)
    traces=[]
    for length in (5,7,10):
        words=TurnWords(p,lengths=length,max_steps=1000000)
        conflicts=words.conflicts(original,maximum=10000)
        for edges in conflicts:
            owners=tuple(sorted({d//3 for pair in edges for d in pair}))
            assert not poses.poses(owners,tuple(sorted(edges)))
        traces.append(dict(length=length,work=words.stats,unique_proven=len(conflicts),
                           poses_without_word_cache=len(conflicts)*3*p.n))
    out['original_nonblank_sewing']=traces
    out['held_out']=[]
    for seed in (35,71,103,157):
        q,d=planted(1,3,seed)
        case={'seed':seed}
        for label,length,channel in [('prior',None,False),('words',10,False),('channel',None,True)]:
            result=search(q,domains=d,geography=True,turn_length=length,channel=channel,seconds=15,max_rounds=40)
            case[label]={k:v for k,v in result.items() if k in ('status','cp_calls','branches','turn_word_work')}
            if result['status']=='GOLD':assert q.validate(result['codes'])['valid']
        out['held_out'].append(case)
    text=json.dumps(out,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':
    main()
