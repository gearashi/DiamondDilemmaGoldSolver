"""Explore exact physical swaps from a saved board and resume the shared graph."""
import argparse
import json
from pathlib import Path

from r28_exchange import read_json
from r29_plateau import walk
from search_problem import SearchProblem


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiles",default="data/tiles.json")
    p.add_argument("--incumbent",required=True)
    p.add_argument("--database",required=True)
    p.add_argument("--floor",type=int,default=207)
    p.add_argument("--target",type=int,default=208,help="Matched-edge threshold. At 240, inspect the Gold flag to verify one loop.")
    p.add_argument("--centers",type=int,default=20)
    p.add_argument("--seconds",type=float,default=0)
    p.add_argument("--output",type=Path)
    p.add_argument("--widest",action="store_true",help="Exhaust higher score floors first, then find a shortest route.")
    p.add_argument("--ties",type=int,default=8,help="Number of tied best rotations to inspect for the displayed board.")
    args = p.parse_args()
    result = walk(SearchProblem(read_json(args.tiles)),read_json(args.incumbent)["codes"],
                  args.database,args.floor,args.target,max_centers=args.centers,seconds=args.seconds,tie_limit=args.ties,widest=args.widest)
    if args.output:
        from io_utils import atomic_json
        atomic_json(args.output,result)
    print(json.dumps(result))


if __name__=="__main__":
    main()
