"""Price lost-bond branches using separately rechecked weighted covers of the surviving terms."""
import argparse
from collections import Counter
from itertools import combinations
import json
from pathlib import Path

from bench_r35_contact_dual import dual
from bench_r35_loss_boundary344 import problem_scope


def make(reference, required, source, cap, limit=0):
    terms=[(r['kind'],frozenset(map(tuple,r['bonds'])),frozenset(map(tuple,r['bonds']))-required) for r in source]
    indices={orig:i for i,(_,orig,_) in enumerate(terms)}
    optional=sorted(reference-required)
    branches=[]
    for b in optional:
        residual=[row for row in terms if b not in row[2]]
        proof=dual(residual);degree=Counter();shared=Counter();weights=[]
        for r in proof['weighted_cores']:
            g=frozenset(map(tuple,r['bonds']));w=r['weight'];weights.append((indices[g],w))
            for x in g-required:degree[x]+=w
            for pair in combinations(sorted(g-required),2):shared[pair]+=w
        maximum=max(degree.values());total=sum(w for _,w in weights)
        slack=(cap-1)*maximum-total
        if slack<0:
            branches.append(dict(assumed=b,weights=weights,forces=1,pairs=0,triples=0))
            continue
        blocked=[x for x in optional if x!=b and maximum-degree[x]>slack]
        live=[x for x in optional if x!=b and x not in blocked]
        triples=sum(2*maximum-degree[x]-degree[y]+shared[x,y]>slack for x,y in combinations(live,2))
        if blocked or triples:branches.append(dict(assumed=b,weights=weights,forces=0,pairs=len(blocked),triples=triples))
    branches.sort(key=lambda r:(-r['forces'],-r['pairs'],-r['triples'],r['assumed']))
    return branches[:limit] if limit else branches


def price(rows, terms, required, optional, cap, count, *, triples):
    all_single=set();all_pairs=set();all_three=set()
    for row in rows[:count or None]:
        b=tuple(row['assumed']);degree=Counter();shared=Counter();weight_sum=0
        for i,weight in row['weights']:
            if type(weight) is not int or weight<=0:raise ValueError('Invalid integer conditional weight.')
            g=terms[i]-required
            if not g or b in g:raise ValueError('A lost premise cannot charge an already hit group.')
            weight_sum+=weight
            for x in g:degree[x]+=weight
            for e in combinations(sorted(g),2):shared[e]+=weight
        maximum=max(degree.values());slack=(cap-1)*maximum-weight_sum
        if slack<0:all_single.add(b);continue
        blocked={x for x in optional if x!=b and maximum-degree[x]>slack}
        all_pairs.update(tuple(sorted((x,b))) for x in blocked)
        if triples:
            live=sorted(set(optional)-blocked-{b})
            all_three.update(tuple(sorted((b,x,y))) for x,y in combinations(live,2)
                             if 2*maximum-degree[x]-degree[y]+shared[x,y]>slack)
    all_pairs={e for e in all_pairs if not all_single.intersection(e)}
    all_three={e for e in all_three if not all_single.intersection(e) and not any(pair in all_pairs for pair in combinations(e,2))}
    return all_single,all_pairs,all_three


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True);cli.add_argument('--core',type=Path,required=True)
    cli.add_argument('--cap',type=int,default=26);cli.add_argument('--limit',type=int,default=0)
    cli.add_argument('--output',required=True,type=Path)
    a=cli.parse_args();p,ref,gs,req=problem_scope(a.tiles);src=json.loads(a.core.read_bytes())
    rows=make(ref,req,src['weighted_cores'],a.cap,a.limit)
    out=dict(fingerprint=p.fingerprint,cap=a.cap,conditionals=rows)
    a.output.write_text(json.dumps(out,separators=(',',':'))+'\n',encoding='utf8')
    terms=[frozenset(map(tuple,r['bonds'])) for r in src['weighted_cores']]
    for n in (8,32,len(rows)):print(n,list(map(len,price(rows,terms,req,ref-req,a.cap,n,triples=True))),flush=True)
