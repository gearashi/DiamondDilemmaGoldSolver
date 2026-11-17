"""Measure joint physical mate consequences beyond checks on single choices."""
import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random

from r27_structure_filters import Work
from r28_exchange import read_json
from r31_geographic import physical_bonds
from r32_gold_bonds import gold_matching_supports,inventory
from r35_corner_chains import filter_options
from r35_mate_correlations import price
from r35_turn_words import TurnWords
from search_problem import SearchProblem


def independent_pair(p,options,req,pins,head,implied):
    """Use independent native exact-one and raw-slot XOR models to check each premise."""
    from collections import defaultdict
    from ortools.sat.python import cp_model
    from r35_loss_witness import raw_sides,reciprocal
    graph=defaultdict(set)
    for a,b in options:graph[a].add(b);graph[b].add(a)
    vertices={head[0]};tasks=list(vertices)
    while tasks:
        u=tasks.pop()
        for v in graph[u]-vertices:vertices.add(v);tasks.append(v)
    def matching(missing):
        m=cp_model.CpModel();vars={e:m.new_bool_var('pair') for e in options if e[0] in vertices}
        for u in vertices:m.add_exactly_one(x for e,x in vars.items() if u in e)
        m.add(vars[head]==1)
        if missing:m.add(vars[missing]==0)
        return cp_model.CpSolver().solve(m)==cp_model.OPTIMAL
    assert matching(None) and all(not matching(f) for f in implied)
    sides=raw_sides(p.tiles)
    def phase(extra):
        m=cp_model.CpModel();x=[m.new_bool_var('phase') for t in range(p.n)]
        odd=defaultdict(set);constant=Counter()
        def toggle(row,t):
            if t in row:row.remove(t)
            else:row.add(t)
        for dart,(slots,zero) in enumerate(sides):
            labels=[min((slots,zero^z),reciprocal((slots,zero^z))) for z in (0,1)]
            constant[labels[0]]+=1
            for label in labels:toggle(odd[label],dart//3)
        for label,row in odd.items():
            terms=[x[t] for t in row]
            if constant[label]%2==0:terms.append(1)
            m.add_bool_xor(terms)
        for t,z in pins.items():m.add(x[t]==z)
        for a,b in req|set(extra):
            sa,ca=sides[a];sb,cb=sides[b];assert reciprocal((sa,0))[0]==sb
            m.add(x[a//3]!=x[b//3] if ca^cb^(len(sa)%2) else x[a//3]==x[b//3])
        return cp_model.CpSolver().solve(m)==cp_model.OPTIMAL
    assert phase([head]) and not phase([head,*implied])
    return dict(head=head,implied=implied,native_matching_premises=len(implied),raw_parity_head=True,raw_parity_bundle=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json');parser.add_argument('--output',type=Path)
    args=parser.parse_args();p=SearchProblem(read_json(args.tiles));avail=inventory(p).possible
    choices=sorted(set(physical_bonds(p,read_json('docs/R29_INCUMBENT.json')['codes']))&avail)
    results=[]
    for n in (0,4,12,24,48,96,144):
        for seed in range(10 if n else 1):
            fixed=dict(Random(seed).sample(choices,n))
            fixed.update({b:a for a,b in fixed.items()})
            graph=[set() for _ in range(3*p.n)]
            for a,b in avail:
                if fixed.get(a,b)==b and fixed.get(b,a)==a:
                    graph[a].add(b);graph[b].add(a)
            supported=gold_matching_supports(p,graph,Work(2000000))
            if supported is None:
                results.append(dict(n=n,seed=seed,prior_closed='matching'));continue
            options={(a,b) for a,row in enumerate(supported) for b in row if a<b}
            required={(a,b) for a,b in options if len(supported[a])==1 or len(supported[b])==1}
            options,_,_=filter_options(p,options,required)
            if not options:
                results.append(dict(n=n,seed=seed,prior_closed='corner'));continue
            options-=TurnWords(p).conditional(options,required)
            if not options:
                results.append(dict(n=n,seed=seed,prior_closed='word'));continue
            banned,stats=price(p,options,required,geography=True)
            results.append(dict(n=n,seed=seed,options=len(options),banned=sorted(banned),**stats))
        print(n,dict(Counter((r.get('prior_closed') or 'open') for r in results if r['n']==n)),flush=True)
    from r29_gold_phase import SignedParity
    from r28_geo_sewing import PosePacking
    xyz=SignedParity(p).sides
    for source,witness in read_json('docs/R35_CONNECTED_TOURS.json').items():
        for n in (40,80,120,160):
            rng=Random(355);order=rng.sample(range(p.n),n)
            pins={t:witness['phases'][t] for t in order}
            g=[set() for _ in xyz]
            for a,b in avail:
                m,ca=xyz[a];_,cb=xyz[b]
                if a//3 in pins and b//3 in pins and pins[a//3]^pins[b//3]!=ca^cb^(m.bit_count()&1):continue
                g[a].add(b);g[b].add(a)
            supported=gold_matching_supports(p,g,Work(2000000))
            if not supported:
                results.append(dict(phase_source=source,pins=n,prior_closed='signed matching'));continue
            options={(a,b) for a,row in enumerate(supported) for b in row if a<b}
            req={(a,b) for a,b in options if len(supported[a])==1 or len(supported[b])==1}
            options,_,_=filter_options(p,options,req)
            options-=TurnWords(p).conditional(options,req)
            if not options:
                results.append(dict(phase_source=source,pins=n,prior_closed='geometry'));continue
            banned,stats=price(p,options,req,pins,geography=True)
            native=[]
            for ex in stats.get('examples',[]):
                if ex['conflict']=='phase':native.append(independent_pair(p,options,req,pins,ex['head'],ex['implied']))
                if ex['conflict'] in ('corner','word'):
                    assert PosePacking(p).check(req|{ex['head'],*ex['implied']})['feasible'] is False
            results.append(dict(phase_source=source,pins=n,options=len(options),banned=sorted(banned),native_replay=native,**stats))
    if args.output:args.output.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(results))

if __name__=='__main__':main()
