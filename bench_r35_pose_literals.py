"""Measure native watched pose clauses in the actual public SAT and CP-SAT models."""
from pathlib import Path
from unittest.mock import patch
import argparse,json
from time import perf_counter
from constraint_models import _Run,_domains,_sat,_cpsat
from r35_pose_departures import reference_images
from r35_repair_seeds import active_catalog
from search_problem import SearchProblem


class Built(Exception):pass


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tiles',type=Path,default=Path('../data/tiles.json'))
    ap.add_argument('--engine',choices=('sat','cp-sat'),required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--conditional',action='store_true',help='Compare native watches after all available original host deductions.')
    ap.add_argument('--strategies',nargs='+',choices=('off','binary','all','image','touched','pair'),default=('off','binary','all'))
    ap.add_argument('--rotated',action='store_true',help='Test all nine other geometric orientations of the two binary gains and one negative control.')
    args=ap.parse_args()
    if any(k in ('image','touched','pair') for k in args.strategies) and not args.conditional:
        ap.error('Compact watches require the conditional SAT comparison.')
    if args.engine=='cp-sat' and (args.conditional or tuple(args.strategies)!=('off','binary','all')):
        ap.error('The CP-SAT screen compares off, binary and all on the original six scopes.')
    p=SearchProblem(json.loads(args.tiles.read_bytes()),physical_gold_support=True)
    record=active_catalog(p)
    seed=record['codes']
    groups=[*record['rows'],*record['post_master']]
    images=reference_images(tuple(seed))
    work=_Run(p,args.engine,300,0,'gold',lambda:False,lambda x:None)
    domains=_domains(work)
    ds=list(map(set,domains))
    candidates=[];image_id={}
    for n,pairs in enumerate(images):
        for row in groups:
            terms=tuple((pairs[c][0],pairs[c][1].bit_length()-1) for c in row)
            if all(v in ds[c] for c,v in terms):
                candidates.append(terms);image_id[terms]=n
    cases={'fresh':[],'guard8':groups[-4],'guard6':groups[-3],'guard5':groups[-1],
           'single':[5],'one_short':groups[-4][:-1]}
    if args.conditional:
        from r35_pose_frontiers import FrontierPoses
        receiver=FrontierPoses(p)
        cases={}
        for j,row in enumerate(record['post_master']):
            for k,c in enumerate(row):
                cases[f'guard{j}-omit{c}']=row[:k]+row[k+1:]
    cases={name:[(c,seed[c]) for c in row] for name,row in cases.items()}
    if args.rotated:
        if not args.conditional:raise ValueError('Rotated prospects require the conditional comparator.')
        cases={}
        base=record['post_master'][0]
        for j,image in enumerate(images[1:],1):
            for c in (134,142,153):
                pins=[(image[b][0],image[b][1].bit_length()-1) for b in base if b!=c]
                if all(v in ds[b] for b,v in pins):cases[f'rotation{j}-omit{c}']=pins
    result=dict(engine=args.engine,retained=[sum(len(t)==2 for t in candidates),len(candidates)],cases={})
    import r34_geographic_mates as mates
    if args.engine=='sat':
        from pysat.solvers import Glucose4
        base=[];bind=[]
        class Recording:
            def __enter__(self):return self
            def __exit__(self,*exc):return False
            def add_clause(self,row):base.append(list(row))
        old=mates.post_sat
        def end(problem,lits,add,check,**kw):
            stats=old(problem,lits,add,check,**kw)
            bind.extend(lits);raise Built
        try:
            with patch('pysat.solvers.Glucose4',Recording),patch.object(mates,'post_sat',end):
                _sat(work,domains,{})
        except Built:pass
        result['base_clauses']=len(base)
        print('captured',len(base),flush=True)
        for name,held in cases.items():
            for kind in args.strategies:
                extra=[]
                assumptions=[bind[c][v] for c,v in held]
                if args.conditional:
                    cube=[sum(1<<v for v in domain) for domain in domains]
                    for c,v in held:cube[c]=1<<v
                    reduced,receipt=receiver.prune(cube)
                    if receipt and receipt['over']:raise ValueError('Expected a host-supported one-short input')
                    for c,(old,new) in enumerate(zip(cube,reduced)):
                        dropped=old & ~new
                        while dropped:
                            bit=dropped & -dropped
                            assumptions.append(-bind[c][bit.bit_length()-1])
                            dropped-=bit
                if kind in ('image','touched','pair'):
                    active={j for j,image in enumerate(images) if any(reduced[c]==bit for c,bit in image)}
                    for terms in candidates:
                        if image_id[terms] not in active or any(not reduced[c]&(1<<v) for c,v in terms):continue
                        residual=[(c,v) for c,v in terms if reduced[c]!=(1<<v)]
                        if kind!='image' and (len(residual)==len(terms) or kind=='pair' and len(residual)!=2):continue
                        if len(residual)>=2:extra.append([-bind[c][v] for c,v in terms])
                else:
                    extra=[[-bind[c][v] for c,v in terms] for terms in candidates if kind=='all' or kind=='binary' and len(terms)==2]
                with Glucose4(bootstrap_with=base) as solver:
                    for row in extra:solver.add_clause(row)
                    # Unit facts mirror the independently narrowed case at a fresh root.
                    for lit in assumptions:solver.add_clause([lit])
                    from threading import Timer
                    solver.conf_budget(128)
                    solver.prop_budget(500000)
                    limit=1.5 if args.conditional else 8
                    timer=Timer(limit,solver.interrupt)
                    begin=perf_counter()
                    timer.start()
                    try:value=solver.solve_limited(expect_interrupt=True)
                    finally:timer.cancel();timer.join()
                    elapsed=perf_counter()-begin
                    counts=solver.accum_stats()
                    result['cases'].setdefault(name,{})[kind]=dict(answer=value,clauses=len(extra),seconds=elapsed,wall_stop=elapsed>=limit,**counts)
                    args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
                    print(name,kind,value,counts['decisions'],round(elapsed,3),flush=True)
            print(name,[(k,v['answer'],v['decisions']) for k,v in result['cases'][name].items()],flush=True)
            args.output.write_text(json.dumps(result,indent=2),encoding='utf8')
    else:
        from ortools.sat.python import cp_model
        original=[]
        old=mates.post_cpsat
        def end(problem,model,codes,rows,check,**kw):
            old(problem,model,codes,rows,check,**kw)
            original.append((model,codes))
            raise Built
        try:
            with patch.object(mates,'post_cpsat',end):_cpsat(work,domains,{})
        except Built:pass
        template,raw_codes=original[0]
        for name,held in cases.items():
            for kind in ('off','binary','all'):
                model=template.clone()
                codes=[model.get_int_var_from_proto_index(v.index) for v in raw_codes]
                for c,v in held:model.add(codes[c]==v)
                if kind!='off':
                    for terms in candidates:
                        if kind=='all' or len(terms)==2:
                            model.add_forbidden_assignments([codes[c] for c,v in terms],[[v for c,v in terms]])
                model.add_decision_strategy(codes,cp_model.CHOOSE_MIN_DOMAIN_SIZE,cp_model.SELECT_MIN_VALUE)
                solver=cp_model.CpSolver()
                solver.parameters.num_search_workers=1
                solver.parameters.random_seed=0
                solver.parameters.max_deterministic_time=.25
                begin=perf_counter()
                status=solver.solve(model)
                result['cases'].setdefault(name,{})[kind]=dict(answer=solver.status_name(status),seconds=perf_counter()-begin,
                    branches=solver.num_branches,conflicts=solver.num_conflicts,deterministic=solver.response_proto.deterministic_time,
                    binary_propagations=solver.response_proto.num_binary_propagations,integer_propagations=solver.response_proto.num_integer_propagations)
            print(name,[(k,v['answer'],v['branches']) for k,v in result['cases'][name].items()],flush=True)
            args.output.write_text(json.dumps(result,indent=2),encoding='utf8')

if __name__=='__main__':main()
