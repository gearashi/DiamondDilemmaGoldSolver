"""Solve a fixed signed phase directly on the board, with an independent literal per tile pose."""
import argparse
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryFile
from shutil import copyfileobj
from threading import Timer
from time import monotonic

from io_utils import atomic_json
from r28_exchange import read_json
from r35_loss_witness import raw_sides,reciprocal
from search_problem import SearchProblem


def encode_board(problem,phase,domains,emit):
    from pysat.card import CardEnc,EncType
    from pysat.formula import IDPool
    from r27_structure_filters import bits
    from r29_gold_geography import checked_domains

    if type(phase) is not int or not 0<=phase<1<<problem.n:
        raise ValueError('Expected one signed phase bit per tile.')
    domains=checked_domains(problem,domains)
    ids=IDPool();by_tile=[[] for _ in range(problem.n)]
    placement=[]
    raw=raw_sides(problem.tiles)
    signed=[(slots,base^((phase>>(dart//3))&1)) for dart,(slots,base) in enumerate(raw)]
    def exactly_one(values):
        values=list(values)
        emit(values)
        if len(values)>1:
            for clause in CardEnc.atmost(values,1,vpool=ids,encoding=EncType.seqcounter).clauses:
                emit(clause)
    for choices in domains:
        row={code:ids.id() for code in bits(choices)}
        placement.append(row);exactly_one(row.values())
        for code,var in row.items():by_tile[code//3].append(var)
    for locations in by_tile:exactly_one(locations)
    def labels(cell,side,reverse):
        groups=defaultdict(list)
        for code,v in placement[cell].items():
            tile,rot=divmod(code,3)
            label=signed[3*tile+(side-rot)%3]
            groups[reciprocal(label) if reverse else label].append(v)
        return groups
    for a,s,b,t in problem.edges:
        left,right=labels(a,s,False),labels(b,t,True)
        for label in sorted(left.keys()|right.keys()):
            if label not in left or label not in right:
                for v in left.get(label,()) or right.get(label,()):emit([-v])
                continue
            match=ids.id()
            for group in (left[label],right[label]):
                emit([-match,*group])
                for v in group:emit([-v,match])
    return ids.top,placement


def export_board(problem,phase,domains,path):
    count=0
    with TemporaryFile() as body:
        def emit(row):
            nonlocal count
            body.write((' '.join(map(str,row))+' 0\n').encode('ascii'))
            count+=1
        variables,_=encode_board(problem,phase,domains,emit)
        body.seek(0)
        with path.open('wb') as dest:
            dest.write(f'p cnf {variables} {count}\n'.encode('ascii'))
            copyfileobj(body,dest)


def solve_board(problem,phase,domains,seconds,*,candidate_path=None):
    from pysat.solvers import Solver
    if seconds<0:raise ValueError('Expected a nonnegative time budget.')
    start=monotonic()
    # Glucose supports asynchronous interruption in the Python bindings.
    with Solver(name='g42' if seconds else 'cadical153') as solver:
        variables,placement=encode_board(problem,phase,domains,solver.add_clause)
        built=monotonic()
        timer=Timer(seconds,solver.interrupt) if seconds else None
        if timer:timer.start()
        try:answer=solver.solve_limited(expect_interrupt=True)
        finally:
            if timer:timer.cancel();timer.join()
        result=dict(phase=hex(phase),status={True:'EDGE_PERFECT',False:'INFEASIBLE',None:'UNKNOWN'}[answer],
                    build_seconds=built-start,solve_seconds=monotonic()-built,variables=variables,
                    clauses=solver.nof_clauses(),**solver.accum_stats())
        if answer:
            model=set(solver.get_model())
            codes=[next(code for code,v in row.items() if v in model) for row in placement]
            checked=problem.validate(codes)
            if not checked['unique_tiles'] or checked['matched_edges']!=len(problem.edges):
                raise ArithmeticError('Physical board failed independent edge validation.')
            if checked['valid']:result['status']='GOLD'
            result.update(codes=codes,validation=checked)
            if candidate_path:atomic_json(candidate_path,dict(fingerprint=problem.fingerprint,**result))
        return result


if __name__=='__main__':
    from r27_structure_filters import domain_bits
    from constraint_symmetry import canonical_root_domains
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--phase',required=True,type=lambda s:int(s,0))
    cli.add_argument('--seconds',type=float,default=90,help='Native solve time in seconds after setup; zero runs to completion.')
    cli.add_argument('--anchor',type=int)
    cli.add_argument('--probe-tiles',type=int,nargs='+',help='Prune poses of these physical tiles before the native solve.')
    output=cli.add_mutually_exclusive_group(required=True)
    output.add_argument('--output',type=Path)
    output.add_argument('--formula',type=Path,help='Export deterministic DIMACS. With --probe-tiles it represents the domains after those deductions.')
    args=cli.parse_args()
    if args.seconds<0:cli.error('Expected a nonnegative solve budget.')
    p=SearchProblem(read_json(args.tiles))
    ds=domain_bits(p)
    if args.anchor is not None:
        ds,meta=canonical_root_domains(p,anchor_tile=args.anchor)
        if not meta['applied']:raise RuntimeError('Could not verify requested rotation symmetry.')
    probe=None
    if args.probe_tiles:
        from r35_phase_anchors import narrow
        pins={t:args.phase>>t&1 for t in range(p.n)}
        if not 0<=args.phase<1<<p.n:
            cli.error('Expected one phase bit per tile.')
        probe=narrow(p,ds,pins,anchors=args.probe_tiles)
        ds=probe.pop('domains')
        if probe['status']=='INFEASIBLE':
            if args.formula:
                ds=(0,*ds[1:])
            else:
                atomic_json(args.output,dict(phase=hex(args.phase),status='INFEASIBLE',decisions=0,probe=probe))
                raise SystemExit(0)
    if args.formula:
        export_board(p,args.phase,ds,args.formula)
        raise SystemExit(0)
    result=solve_board(p,args.phase,ds,args.seconds,candidate_path=args.output.with_suffix('.verified-board.json'))
    if probe:result['probe']=probe
    atomic_json(args.output,result)
    print({k:v for k,v in result.items() if k not in ('codes','validation')})
