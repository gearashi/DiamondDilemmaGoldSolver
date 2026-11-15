"""Compare individual reference conflicts and certified cardinality in geographic Gold scopes."""
import argparse
import json
from pathlib import Path
from random import Random

from r27_structure_filters import domain_bits
from r28_exchange import read_json
from r30_gold import search
from r31_geographic import physical_bonds
from r32_gold_loss import reference_groups
from r32_gold_bonds import inventory
from r35_geographic_loss import certified_reference,retained_bound
from search_problem import SearchProblem


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json');cli.add_argument('--output',type=Path)
    args=cli.parse_args();p=SearchProblem(read_json(args.tiles))
    codes=read_json('docs/R29_INCUMBENT.json')['codes']; ref,limit=certified_reference(p)
    _,_,groups=reference_groups(p,codes); full=domain_bits(p)
    rows=[];seen=set()
    for seed in range(12):
        rng=Random(seed);drop=set()
        while any(not g&drop for g in groups):
            open_groups=[g for g in groups if not g&drop]
            options=sorted(set.union(set(),*open_groups))
            top=max(sum(e in g for g in open_groups) for e in options)
            drop.add(rng.choice([e for e in options if sum(e in g for g in open_groups)==top]))
        options=sorted(ref-drop);rng.shuffle(options)
        for keep in (186,187,188,189):
            chosen=frozenset(options[:keep])
            if len(chosen)!=keep or chosen in seen:continue
            seen.add(chosen)
            assert not any(g<=chosen for g in groups)
            out=retained_bound(p,list(full),ref,limit,chosen|inventory(p).required)
            row=dict(seed=seed,explicit_kept=keep,prior_conflicts=sum(g<=chosen for g in groups),
                     forced=out['forced'],discarded=out['over'],excluded_mates=len(out['forbidden']))
            if seed==0 and keep in (187,188):
                for mode in (False,True):
                    attempt=search(p,bonds=chosen,geography=True,max_rounds=1,seconds=4,reference_bound=mode)
                    row['bound' if mode else 'base']={k:attempt[k] for k in ('status','cp_calls','branches')}
            rows.append(row)
    geographic=[]
    order=list(range(p.n));Random(35).shuffle(order)
    for free in (0,2,5,8,12,20,32,48,80,160):
        movable=set(order[:free]);pool=sum(7<<(3*(codes[c]//3)) for c in movable)
        ds=[pool if c in movable else 1<<v for c,v in enumerate(codes)]
        out=retained_bound(p,ds,ref,limit,inventory(p).required)
        geographic.append(dict(free=free,forced=out['forced'],discarded=out['over'],
                               units=out['unit_poses'],excluded_mates=len(out['forbidden'])))
    tours={name:len(ref & set(map(tuple,doc['pairs']))) for name,doc in read_json('docs/R35_CONNECTED_TOURS.json').items()}
    data=dict(scope_bonds=len(ref),limit=limit,individual_conflicts=len(groups),
              bond_assumptions=rows,geographic=geographic,original_different_sewings=tours)
    if args.output:args.output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
    else:print(json.dumps(data))

if __name__=='__main__':main()
