"""Explicit, isolated research entry points for Round 27 scoped reasoning."""
import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from search_problem import SearchProblem
from r27_reason_algebra import algebra
from r27_reason_cost import cost_bound
from r27_reason_maxsat import maximize_seams
from r27_reason_orbits import orbit_choices
from r27_reason_parallel import concurrent_sat
from r27_reason_search import scoped_search


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method',choices=('maxsat','cost','backdoor','dynamic','orbits','lp','gf2','strong','concurrent','mrv','backjump','orbit'))
    parser.add_argument('--data',type=Path,required=True,help='Source tiles.json')
    parser.add_argument('--domains',type=Path,help='JSON array of allowed orientation codes per cell')
    parser.add_argument('--seed-board',type=Path,help='Validated incumbent file with codes array')
    parser.add_argument('--threshold',type=int,help='Minimum accepted number of matching seams')
    parser.add_argument('--target',choices=('gold','edges'),default='gold')
    parser.add_argument('--seconds',type=float,default=30)
    parser.add_argument('--degree',type=int,choices=(1,2),default=1)
    parser.add_argument('--conflicts',type=int,help='MaxSAT native conflict budget')
    args = parser.parse_args(argv)
    def read(path):
        return json.loads(path.read_text(encoding='utf-8-sig'))
    problem = SearchProblem(read(args.data))
    domains = read(args.domains) if args.domains else None
    method = args.method
    if method == 'maxsat':
        seed = read(args.seed_board)['codes'] if args.seed_board else None
        result = maximize_seams(problem,domains=domains,seed_board=seed,threshold=args.threshold,
                                seconds=args.seconds,target=args.target,conflicts=args.conflicts)
    elif method == 'cost':
        result = cost_bound(problem,domains=domains)
    elif method == 'orbits':
        if args.target != 'gold':
            parser.error('Residual orbit reduction supports Gold existence.')
        result = orbit_choices(problem,domains=domains)
    elif method in ('lp','gf2'):
        result = algebra(problem,domains=domains,method=method,degree=args.degree,seconds=args.seconds)
    elif method == 'concurrent':
        result = concurrent_sat(problem,domains=domains,target=args.target,seconds=args.seconds)
    else:
        result = scoped_search(problem,domains=domains,target=args.target,mode=method,seconds=args.seconds)
    print(json.dumps(result,default=lambda obj:asdict(obj) if is_dataclass(obj) else str(obj)))
    return result


if __name__ == '__main__':
    main()

