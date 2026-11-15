"""Measure physical corner-chain and vertex-position filters at matched geography scopes."""
from collections import Counter
import argparse
import json
from pathlib import Path
from random import Random

from r27_structure_filters import Work, domain_bits
from r27_structure_fixtures import planted
from r28_exchange import read_json
from r28_geo_sewing import PosePacking
from r30_gold import GoldMatching,search
from r31_geographic import physical_bonds,ribbon_genus
from r32_gold_bonds import gold_matching_supports,inventory
from r35_corner_chains import degree_inventory,filter_options
from search_problem import SearchProblem


def residuals(p,rules,board):
    reference=sorted(set(physical_bonds(p,board)) & rules.possible)
    report=[]
    for n in (4,12,24,48,96,144):
        stats=Counter()
        for seed in range(10):
            pins=Random(seed).sample(reference,n)
            fixed=dict(pins)
            fixed.update({b:a for a,b in pins})
            rows=[set() for _ in range(3*p.n)]
            for a,b in rules.possible:
                if fixed.get(a,b)==b and fixed.get(b,a)==a:
                    rows[a].add(b)
                    rows[b].add(a)
            supported=gold_matching_supports(p,rows,Work(2000000))
            if supported is None:
                stats['published_baseline_closed']+=1
                continue
            pairs={(a,b) for a,row in enumerate(supported) for b in row if a<b}
            required={(a,b) for a,b in pairs if len(supported[a])==1 or len(supported[b])==1}
            pack=PosePacking(p)
            proofs=set()
            def independently_excluded(checked):
                key=frozenset(checked)
                if key in proofs:
                    return True
                darts=[d for pair in checked for d in pair]
                valid=(len(set(darts))!=len(darts) or ribbon_genus(p,checked)>0 or
                       pack.check(checked,max_steps=10000)['feasible'] is False)
                if valid:
                    proofs.add(key)
                return valid
            def verify(checked):
                assert independently_excluded(checked), sorted(checked)
            arms={}
            for mode,en in [('chains',False),('packing',True)]:
                live,state,work=filter_options(p,pairs,required,packing=en,check_exclusion=verify)
                if not live:
                    stats[mode+'_closed']+=1
                else:
                    stats[mode+'_removed_on_survivors']+=len(pairs-live)
                for reason in ('repeated_tile','overlong_chain','unavailable_degree','length_packing'):
                    stats[mode+'_'+reason]+=work.get(reason,0)
                arms[mode]=live
            assert arms['packing']<=arms['chains']
            stats['packing_only_on_chain_survivors']+=len(arms['chains']-arms['packing'])
            stats['independently_replayed_premises']+=len(proofs)
            if not arms['chains']:
                stats['closed_by_independent_geometry']+=independently_excluded(required)
            else:
                for pair in pairs-arms['chains']:
                    stats['exclusions_by_independent_geometry']+=independently_excluded(required|{pair})
        report.append(dict(pins=n,**stats))
    return report


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--output',type=Path)
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles))
    ds=domain_bits(p)
    rules=inventory(p)
    live,state,root=filter_options(p,rules.possible,rules.required)
    excluded=sorted(rules.possible-live)
    roots=[]
    for pair in excluded:
        preserved=set(rules.required)|{pair}
        for old in sorted(rules.required):
            if PosePacking(p).check(preserved-{old})['feasible'] is False:
                preserved.remove(old)
        assert PosePacking(p).check(preserved)['feasible'] is False
        roots.append(dict(removed=pair,required_core=sorted(preserved),genus=ribbon_genus(p,preserved)))
    cp=GoldMatching(p,ds,None,(),corner_mode='positions')
    assert not cp.m.validate()
    out=dict(root=dict(before=len(rules.possible),after=len(live),exclusions=roots,work=root,
                       degrees=dict(degree_inventory(p)),position_variables=len(cp.m.proto.variables),
                       position_constraints=len(cp.m.proto.constraints),positions=cp.corner_stats['positions']),
             original_residuals=residuals(p,rules,read_json('docs/R29_INCUMBENT.json')['codes']))
    fixed=read_json('docs/R35_NONBLANK_SEWING.json')
    options=dict(domains=ds,pins=dict(enumerate(fixed['phases'])),bonds=[tuple(pair) for pair in fixed['nonblank']],
                 geography=True,seconds=10)
    out['original_connected']={mode:{k:v for k,v in search(p,**options,corner_mode=mode).items()
                                     if k in ('status','cp_calls','branches','corner_preprocessing')}
                               for mode in ['chains','packing','positions']}
    out['planted']=[]
    for seed in (35,71,103,157):
        q,d=planted(1,3,seed)
        group=dict(seed=seed)
        for mode in (None,'chains','packing','positions','channel'):
            result=search(q,domains=d,geography=True,corner_mode=mode if mode!='channel' else None,
                          channel=mode=='channel',seconds=12,max_rounds=40)
            group[mode or 'prior']={k:v for k,v in result.items() if k in ('status','cp_calls','branches','corner_sewing_cuts')}
            if result['status']=='GOLD':
                assert q.validate(result['codes'])['valid']
        out['planted'].append(group)
    text=json.dumps(out,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':
    main()
