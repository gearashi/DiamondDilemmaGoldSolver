"""Screen branched induced geographic contact tables on original reference and donor bonds."""
import argparse
from collections import Counter
from pathlib import Path
from random import Random

from bench_r35_mixed_sewings import original_sources
from io_utils import atomic_json
from r28_exchange import read_json
from r31_geographic import physical_bonds
from r35_path_mates import PathMates
from search_problem import SearchProblem


def examples(edges, sizes, repeats, seed):
    rng=Random(seed)
    edges=sorted(edges)
    for size in sizes:
        for _ in range(repeats):
            root=rng.choice(edges)
            chosen={root}
            tiles={d//3 for d in root}
            while len(chosen)<size:
                candidates=[e for e in edges if e not in chosen and (e[0]//3 in tiles) != (e[1]//3 in tiles)]
                if not candidates:break
                e=rng.choice(candidates)
                chosen.add(e)
                tiles.update(d//3 for d in e)
            if len(chosen)==size:yield frozenset(chosen)


def measure(problem, sources, repeats):
    stats={}
    checker=PathMates(problem)
    for name,edges in sources.items():
        count=Counter()
        cases=[]
        for path in set(examples(edges,(4,6,9,12,16,22),repeats,35086)):
            result,geo,matched=checker.patterns(path)
            count["checked"]+=1
            if not result:continue
            count["compatible"]+=1
            certain=frozenset.intersection(*result)
            variables=sorted(frozenset.union(*result)-certain)
            if not variables:continue
            count["variable"]+=1
            if len(result)==1<<len(variables):continue
            count["correlated"]+=1
            count["joint_values_excluded"]+=(1<<len(variables))-len(result)
            if len(cases)<12:
                cases.append(dict(path=sorted(path),geometric=geo,compatible=matched,variables=variables,
                                  allowed=sorted(sorted(row-certain) for row in result)))
        stats[name]=dict(counts=dict(count),cases=cases)
    return stats


def public(problem, screen, seconds, rounds=32, profile="sparse"):
    from r30_gold import search
    for source in screen.values():
        for case in source["cases"]:
            case["runs"]={}
            for mode in ("units","table"):
                run=search(problem,geography=True,bonds=case["path"],path_mates=mode,
                           seconds=seconds,phase_anchors="off",max_rounds=rounds,
                           channel=profile=="channel",project_blanks=profile=="blank")
                case["runs"][mode]={key:run[key] for key in ("status","cp_calls","branches","path_mates")}
    return screen


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--repeats",type=int,default=32)
    cli.add_argument("--reuse",type=Path,help="Read an existing measured screen to compare the public search.")
    cli.add_argument("--seconds",type=float,default=0,help="Public seconds per correlated case and mode.")
    cli.add_argument("--rounds",type=int,default=32)
    cli.add_argument("--profile",choices=("sparse","blank","channel"),default="sparse")
    args=cli.parse_args()
    p=SearchProblem(read_json(args.tiles)); docs=Path(__file__).parent/"docs"
    sources={"reference":physical_bonds(p,read_json(docs/"R29_INCUMBENT.json")["codes"])}
    for name in ("raw","contract","near0","near1"):
        sources[name]=[tuple(sorted(e)) for e in original_sources()[name]["pairs"]]
    if args.reuse:
        previous=read_json(args.reuse)
        if previous["fingerprint"]!=p.fingerprint:raise ValueError("Different puzzle")
        result=previous["sources"]
    else:
        result=measure(p,sources,args.repeats)
    if args.seconds:public(p,result,args.seconds,args.rounds,args.profile)
    atomic_json(args.output,dict(fingerprint=p.fingerprint,sources=result))
    print({name:block["counts"] for name,block in result.items()})
