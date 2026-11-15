"""Exhaust root single-mate strand matching tests and price sparse conditioned residuals."""
import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random

from r27_structure_filters import Work,physical_graph as domain_graph,separator_ok
from r28_exchange import read_json
from r31_geographic import physical_bonds
from r32_gold_bonds import gold_matching_supports,inventory
from bench_r35_strand_oracle import check_sewing,physical_graph,verify_obstruction
from search_problem import SearchProblem


def rows_for(p,options,selected):
    mate=dict(selected);mate.update({b:a for a,b in selected})
    rows=[set() for _ in range(3*p.n)]
    for a,b in options:
        if mate.get(a,b)==b and mate.get(b,a)==a:
            rows[a].add(b);rows[b].add(a)
    return rows


def examine(p,raw,selected,root=False):
    rows=gold_matching_supports(p,rows_for(p,raw,selected),Work(4000000))
    if rows is None:
        return dict(matching_already_closed=1),[]
    options={(a,b) for a,row in enumerate(rows) for b in row if a<b}
    required={(a,b) for a,b in options if len(rows[a])==1 or len(rows[b])==1}
    stats=Counter()
    failures=[]
    def probe(extra):
        fixed=required|({extra} if extra is not None else set())
        result=check_sewing(p,options,fixed,Work(5000000))
        key='scope' if extra is None else 'single'
        stats[key+'_'+result['result']]+=1
        if result['result']=='obstruction':
            cert=result['certificate']
            graph,_=physical_graph(p,options,fixed)
            assert verify_obstruction(graph,cert)
            baseline=gold_matching_supports(p,rows_for(p,options,fixed),Work(4000000))
            prior=baseline is not None and separator_ok(graph,Work(40000000))
            if prior:stats['survived_matching_and_two_separator']+=1
            failures.append(dict(extra=extra,fixed=sorted(fixed),certificate=cert,vertices=len(graph),
                                 survived_matching_and_separators=prior))
        return result['result']
    if probe(None) in ('obstruction','proper_circuit'):
        return dict(stats),failures
    for a,b in sorted(options-required):
        if int(p.masks[3*(a//3),a%3]) and (root or min(len(rows[a]),len(rows[b]))<=3):
            probe((a,b))
    return dict(stats),failures


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--output',type=Path)
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles));base=inventory(p)
    board=read_json('docs/R29_INCUMBENT.json')['codes']
    ref=sorted(set(physical_bonds(p,board))&base.possible)
    failures=[]
    def trial(name,scope,pairs,fixed,root=False):
        counts,rejected=examine(p,pairs,fixed,root)
        failures.extend(dict(family=name,scope=scope,**row) for row in rejected)
        return counts
    out=dict(root=trial('root',None,base.possible,(),True),pinned=[],geographic=[])
    for count in (4,12,24,48,96,144):
        totals=Counter()
        for seed in range(10):
            totals.update(trial('pins',[count,seed],base.possible,Random(seed).sample(ref,count)))
        out['pinned'].append(dict(pins=count,**totals))
    for count in (4,8,16,32,64,96,128):
        totals=Counter()
        for seed in (34,35,36):
            free=set(Random(seed).sample(range(p.n),count))
            pool=sum(7<<(3*(board[c]//3)) for c in free)
            ds=tuple(pool if c in free else 1<<board[c] for c in range(p.n))
            graph=domain_graph(p,ds,Work(20000000))
            possible={(a,b) for a,b in base.possible if b in graph[a]}
            totals.update(trial('cells',[count,seed],possible,()))
        out['geographic'].append(dict(free=count,**totals))
    out['obstructions']=failures
    text=json.dumps(out,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':
    main()
