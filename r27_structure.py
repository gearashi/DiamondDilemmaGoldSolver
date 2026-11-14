"""Solve or count a restricted Gold puzzle using the tested structural alternatives."""
import argparse
import json
from pathlib import Path

from r27_structure_filters import domain_bits, filter_domains
from r27_structure_region import regional_count
from r27_structure_sat import sat_boards
from search_problem import SearchProblem


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('engine',choices=('sparse-sat','regional-count'))
    parser.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    parser.add_argument('--domains',type=Path,help='JSON array of allowed placement-code arrays, one per cell.')
    parser.add_argument('--region',help='Comma-separated cells on one side of a simple geographic cut.')
    parser.add_argument('--max-work',type=int,default=500000)
    parser.add_argument('--conflicts',type=int,default=100000)
    parser.add_argument('--subdivision',type=int,default=4,help='4 for the actual 160-cell board.')
    args=parser.parse_args()
    from geometry import build_board
    p=SearchProblem(json.loads(args.tiles.read_text(encoding='utf-8-sig')),build_board(args.subdivision))
    ds=domain_bits(p)
    if args.domains:
        data=json.loads(args.domains.read_text(encoding='utf-8-sig'))
        if (type(data)is not list or len(data)!=p.n or any(type(row)is not list or any(
                type(v)is not int or not 0<=v<3*p.n for v in row) for row in data)):
            parser.error('Domains must contain one list of valid integer placement codes per cell.')
        ds=tuple(sum(1<<v for v in set(row)) for row in data)
    if args.engine=='sparse-sat':
        out=sat_boards(p,ds,'sparse',max_work=args.max_work,conflicts=args.conflicts)
    else:
        region=(list(map(int,args.region.split(','))) if args.region else
                [c.index for c in p.board.cells if c.face==0])
        out=regional_count(p,region,ds,max_work=args.max_work)
    print(json.dumps(out,indent=2))
    return 0 if out['complete'] else 2


if __name__=='__main__':raise SystemExit(main())

