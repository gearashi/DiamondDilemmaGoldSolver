"""Screen physical-path implications on retained reference and independent donor graphs."""
import argparse
from collections import Counter,defaultdict,deque
from pathlib import Path

from bench_r35_mixed_sewings import original_sources
from io_utils import atomic_json
from r28_exchange import read_json
from r32_gold_loss import reference_groups
from r35_path_mates import PathMates, alternatives
from search_problem import SearchProblem


def candidates(edges, limit, maximum):
    graph = defaultdict(list)
    for a,b in sorted(edges):
        graph[a//3].append((b//3,(a,b)))
        graph[b//3].append((a//3,(a,b)))
    seen = set()
    for omitted in sorted(edges):
        start,goal = (a//3 for a in omitted)
        queue = deque([(start,(start,),())])
        emitted = 0
        while queue and emitted<limit:
            tile,path,bonds=queue.popleft()
            for neighbor,pair in graph[tile]:
                if pair==omitted or neighbor in path:
                    continue
                step=(*bonds,pair)
                if neighbor==goal:
                    premise=frozenset(step)
                    if premise not in seen:
                        seen.add(premise)
                        yield premise
                    emitted += 1
                elif len(step)<maximum:
                    queue.append((neighbor,(*path,neighbor),step))


def screen(problem, edges, limit, maximum, cache):
    checker=PathMates(problem)
    counts=Counter()
    examples=[]
    for premise in candidates(edges,limit,maximum):
        if premise in cache:
            patterns,geo,matched,clauses=cache[premise]
            counts["reused_paths"]+=1
        else:
            patterns,geo,matched=checker.patterns(premise)
            clauses=alternatives(patterns)
            cache[premise]=patterns,geo,matched,clauses
        counts["paths"]+=1
        counts["zero_geometry"]+=geo==0
        counts["zero_compatible"]+=matched==0
        if clauses:
            counts["new_rules"]+=len(clauses)
            counts["with_implication"]+=1
            for row in clauses:
                counts[f"width_{len(row)}"]+=1
            examples.append(dict(bonds=sorted(premise),alternatives=[sorted(row) for row in clauses],
                                 geometric=geo,compatible=matched,patterns=len(patterns)))
    return dict(summary=dict(counts),cases=examples)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--tiles",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--limit",type=int,default=1)
    cli.add_argument("--length",type=int,default=9)
    cli.add_argument("--sources",nargs="+",default=["reference"])
    args=cli.parse_args()
    problem=SearchProblem(read_json(args.tiles))
    docs=Path(__file__).parent/"docs"
    ref=reference_groups(problem,read_json(docs/"R29_INCUMBENT.json")["codes"])[0]
    donors={k:{tuple(sorted(b)) for b in row["pairs"]} for k,row in original_sources().items()}
    donors["reference"]=set(ref)
    donors["contact"]={tuple(sorted(b)) for b in read_json(docs/"R35_MIXED_CONTACT_DONOR.json")["pairs"]}
    result=dict(fingerprint=problem.fingerprint,limit=args.limit,length=args.length,sources={})
    cache={}
    for name in args.sources:
        result["sources"][name]=screen(problem,donors[name],args.limit,args.length,cache)
        atomic_json(args.output,result)
        print(name,result["sources"][name]["summary"],flush=True)


if __name__=="__main__":
    main()
