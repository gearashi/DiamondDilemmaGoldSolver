"""Compare general odd combs against the existing physical LP cut separator."""
import argparse
from collections import Counter
import json
from pathlib import Path
from random import Random

from r27_structure_fixtures import planted
from r27_structure_filters import domain_bits
from r28_exchange import read_json
from r30_gold import GoldMatching
from r31_geographic import physical_bonds
from r31_gold_pricing import fractional_cuts
from r32_gold_bonds import inventory
from search_problem import SearchProblem


def prices(p,d,bonds,rounds):
    base=GoldMatching(p,d,None,bonds)
    before=len(base.m.proto.constraints)
    report=fractional_cuts(base,bonds=bonds,limit=rounds,extended_combs=True)
    kinds=Counter(report['cuts'])
    out=dict(status=report['status'],cuts=dict(kinds),extended=report.get('extended_work',{}),
             new_combs=report.get('extended_rows',[]),cut_budget=rounds,
             constraints_before=before,constraints_after=len(base.m.proto.constraints))
    return out


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='data/tiles.json')
    cli.add_argument('--output',type=Path)
    cli.add_argument('--rounds',type=int,default=256)
    cli.add_argument('--family',choices=('root','fixed','planted'),required=True)
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles))
    out={}
    if args.family=='planted':
        for seed in (35,71,103,157):
            for movers in (3,6,10):
                q,d=planted(1,movers,seed,uniform_ports=True)
                out[str(seed)+'_'+str(movers)]=prices(q,d,(),args.rounds)
    elif args.family=='root':
        out['unrestricted']=prices(p,domain_bits(p),(),args.rounds)
    else:
        prior=inventory(p)
        ref=sorted(set(physical_bonds(p,read_json('docs/R29_INCUMBENT.json')['codes']))&prior.possible)
        for size in (12,48,96):
            bonds=Random(35).sample(ref,size)
            out[str(size)]=prices(p,domain_bits(p),bonds,args.rounds)
    text=json.dumps(out,indent=2)+'\n'
    if args.output:args.output.write_text(text,encoding='utf-8')
    else:print(text)


if __name__=='__main__':
    main()
