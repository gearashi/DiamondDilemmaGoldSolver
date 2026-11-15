"""Check geographic exclusions conditional on full signed tile phases."""
import argparse
from pathlib import Path
import json

from bench_r35_mixed_sewings import union_sat
from io_utils import atomic_json
from r27_structure_filters import domain_bits
from r35_mixed_sewings import geographic_roots
from r35_signed_mate_support import raw_edges,reciprocal_classes
from search_problem import SearchProblem


def measure(problem, phases):
    sides, families=reciprocal_classes(problem.tiles)
    raw=[e for f in families for e in raw_edges(sides,f)]
    cases=[]
    for phase in phases:
        edges=[(a,b) for a,b,sign in raw if (((phase>>(a//3))^(phase>>(b//3)))&1)==sign]
        degrees=[0]*(3*problem.n)
        for a,b in edges:
            degrees[a]+=1
            degrees[b]+=1
        def rarity(tile):
            a,b,c=sorted(degrees[3*tile:3*tile+3])
            return a*b*c,b,c,tile
        probes=[]
        for tile in dict.fromkeys([0,*sorted(range(problem.n),key=rarity)]):
            result=geographic_roots(problem,domain_bits(problem),edges,anchor=tile)
            probes.append(dict(anchor=tile,root_count=result['checked_roots'],remaining=len(result['survivors']),
                               revisions=result['cell_revisions']))
            if not result['survivors']:
                break
        best=min(probes,key=lambda row:row['remaining'])
        independent=union_sat(problem,edges,anchor=best['anchor'])
        if independent['unresolved_roots']>best['remaining']:
            raise AssertionError('Independent full arc propagation missed a claimed exclusion.')
        cases.append(dict(phase=hex(phase),allowed_seams=len(edges),probes=probes,independent=independent))
    return cases


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    puzzle=SearchProblem(json.loads(args.tiles.read_bytes()))
    prior=Path(__file__).resolve().parent/'docs/R35_MIXED_SEWINGS.json'
    domains=json.loads(prior.read_bytes())['combined_unions']['outside_recorded_donors']
    phases=[int(key,16) for key in domains if key.startswith('0x')]
    atomic_json(args.output,dict(fingerprint=puzzle.fingerprint,cases=measure(puzzle,phases)))
