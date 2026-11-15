"""Rebuild the refined repair-distance formula or extract independently verified cuts for a candidate."""
import argparse
import json
from itertools import chain
from pathlib import Path

from bench_r35_repair_seeds import PoseOracle, independently_allowed
from r28_exchange import read_json
from r35_repair_propagation import RepairPropagation
from r35_repair_seeds import Repairs, checked_catalog
from search_problem import SearchProblem

ROOT=Path(__file__).resolve().parent


def local_cuts(problem,seed,selected,verifier,limit=32):
    """Delete fixed poses while preserving a geographic contradiction, then independently replay each cut."""
    if (len(selected)!=len(set(selected)) or any(type(c) is not int or not 0<=c<problem.n for c in selected)
            or type(limit) is not int or limit<1):
        raise ValueError('Invalid proposed changed cells or cut limit.')
    propagation=RepairPropagation(problem,Repairs(problem,seed,lambda:None).mates)
    fixed=sorted(set(range(problem.n))-set(selected))
    def contradiction(cells):
        return propagation.narrow(seed,cells,128,True) is None
    result=[]
    while len(result)<limit and contradiction(fixed):
        core=fixed[:]
        for width in (16,8,4,2,1):
            i=0
            while i<len(core):
                reduced=core[:i]+core[i+width:]
                if contradiction(reduced):
                    core=reduced
                else:
                    i+=width
        if not verifier.replay(seed,core):
            raise ValueError('Independent geographic formula did not confirm a proposed row.')
        result.append(core)
        if not core:break
        fixed.remove(core[(len(result)-1)%len(core)])
    return result


def write_formula(n,rows,cap,path):
    from pysat.card import CardEnc,EncType
    if type(cap) is not int or not 0<=cap<n:raise ValueError('Invalid geographic change cap.')
    encoding=CardEnc.atmost(lits=list(range(1,n+1)),bound=cap,encoding=EncType.totalizer)
    with Path(path).open('w',encoding='ascii',newline='\n') as out:
        out.write(f'p cnf {encoding.nv} {len(rows)+len(encoding.clauses)}\n')
        for row in rows:out.write(' '.join(str(c+1) for c in sorted(row))+' 0\n')
        for row in encoding.clauses:out.write(' '.join(map(str,row))+' 0\n')


def replay(problem,catalog,refined):
    seed=catalog['codes'];rows=refined['extra']
    if refined['puzzle']!=problem.fingerprint or refined['puzzle']!=catalog['puzzle']:
        raise ValueError('Repair refinement input identity differs.')
    n=problem.n
    if (refined['cap']+1!=len(refined['hit']) or len(set(refined['hit']))!=len(refined['hit'])
            or any(type(c) is not int or not 0<=c<n for c in refined['hit'])):
        raise ValueError('Invalid relaxed upper witness.')
    selected=set(refined['hit'])
    if not all(selected.intersection(row) for row in chain(catalog['rows'],rows)):
        raise ValueError('Published upper witness misses an obligation.')
    allowed,physical=independently_allowed(problem)
    verifier=PoseOracle(problem,allowed)
    try:
        for row in chain(rows,refined['additional']):
            if (not row or len(set(row))!=len(row) or any(type(c) is not int or not 0<=c<n for c in row)
                    or not verifier.replay(seed,row)):
                raise ValueError('An additional unchanged-pose conjunction did not replay.')
        return dict(puzzle=problem.fingerprint,base_rows=len(catalog['rows']),extra_rows=len(rows),additional_verified=len(refined['additional']),
                    cap=refined['cap'],upper=len(selected),variables=verifier.top,clauses=verifier.clauses,
                    physical=physical)
    finally:verifier.close()


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,default=ROOT/'data/tiles.json')
    cli.add_argument('--write-cnf',type=Path)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--selection',type=Path,help='A JSON file containing a candidate list of changed cell indices.')
    cli.add_argument('--cut-limit',type=int,default=32)
    args=cli.parse_args()
    problem=SearchProblem(read_json(args.tiles));catalog=checked_catalog(problem)
    if catalog is None:raise ValueError('Unrecognized original mesh or published catalog.')
    if args.selection:
        allowed,_=independently_allowed(problem);verifier=PoseOracle(problem,allowed)
        try:data={'extra':local_cuts(problem,catalog['codes'],read_json(args.selection),verifier,args.cut_limit)}
        finally:verifier.close()
    else:
        refined=read_json(ROOT/'docs/R35_REPAIR_REFINED.json')
        data=replay(problem,catalog,refined)
        if args.write_cnf:write_formula(problem.n,catalog['rows']+refined['extra'],refined['cap'],args.write_cnf)
    args.output.write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
