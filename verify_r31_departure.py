"""Independently replay disjoint physical bond conflicts in the integer phase model."""
import argparse
import json
from pathlib import Path
from ortools.sat.python import cp_model
from r28_geo_color import signed_phase_model
from search_problem import SearchProblem
from validator import reverse_mask


def verify(problem, codes, groups):
    if sorted(v//3 for v in codes)!=list(range(problem.n)) or any(
            v not in domain for v,domain in zip(codes,problem.domains)):
        raise ValueError("Invalid reference placement")
    present=set()
    for a,sa,b,sb in problem.edges:
        x,y=codes[a],codes[b]
        if int(problem.masks[x,sa])==reverse_mask(int(problem.masks[y,sb])):
            present.add(tuple(sorted((3*(x//3)+(sa-x%3)%3,3*(y//3)+(sb-y%3)%3))))
    seen=set()
    sizes=[]
    for group in groups:
        bonds={tuple(sorted(pair)) for pair in group}
        if not bonds or not bonds<=present or seen & bonds:
            raise ValueError("Expected distinct, disjoint subsets of present bonds")
        model,_=signed_phase_model(problem,euler=False,pins={},bonds=sorted(bonds))
        cp=cp_model.CpSolver()
        cp.parameters.num_search_workers=1
        cp.parameters.max_time_in_seconds=30
        status=cp.solve(model)
        if status!=cp_model.INFEASIBLE:
            raise ValueError("Integer replay failed: "+cp.status_name(status))
        sizes.append(len(bonds))
        seen.update(bonds)
    return dict(fingerprint=problem.fingerprint,matched=len(present),sizes=sizes,
                minimum_required_bond_losses=len(sizes),physical_bonds_checked=len(seen))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',default='data/tiles.json')
    parser.add_argument('--center',default='docs/R29_INCUMBENT.json')
    parser.add_argument('--evidence',default='docs/R31_GOLD_RESULTS.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    def read(path):
        return json.loads(Path(path).read_text(encoding='utf-8-sig'))
    inp=read(args.evidence)
    problem=SearchProblem(read(args.tiles))
    if problem.fingerprint!=inp['binding']:
        raise ValueError('Wrong tile set')
    result=verify(problem,read(args.center)['codes'],inp['departures']['center']['witnesses'])
    message=json.dumps(result,indent=2)
    if args.output:
        args.output.write_text(message+'\n',encoding='utf-8')
    print(message)
