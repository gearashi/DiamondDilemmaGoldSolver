"""Serial supported-platform persistence diagnostic, each restart a fresh process.

Synthetic inventory encodings and a finite puzzle establish bounded mechanism
usefulness and preservation. They do not predict original-puzzle performance.
"""
import argparse,hashlib,itertools,json,os,platform,subprocess,sys,time
from pathlib import Path


def child(payload):
    completed=subprocess.run([sys.executable,'-B',__file__,'--child'],input=json.dumps(payload),
        capture_output=True,text=True,timeout=90)
    if completed.returncode:raise RuntimeError('Persistence child failed: '+completed.stderr[-5000:])
    lines=completed.stdout.splitlines()
    if not lines or lines[-1]!='cleanup_returned':raise RuntimeError('Native child did not complete cleanup')
    return json.loads(lines[-2])


def run_child(request):
    from constraint_sat_persistence import VerifiedNativeClauses
    from test_constraint_sat_persistence import finish
    from pysat.solvers import Glucose4
    store=VerifiedNativeClauses(request['base'],[dict(row) for row in request['mapping']],
        fingerprint=request['fingerprint'],target='edges',initial=request['initial'])
    start=time.monotonic()
    if request['mode']=='produce':
        solver=store.create_solver(request['base']);answers=[]
        try:
            for assumptions in request.get('queries',[[],[],[]]):
                solver.conf_budget(1);answers.append(solver.solve_limited(assumptions=assumptions,expect_interrupt=True))
            for _ in range(1024):
                store.harvest(solver);finish(store)
                if store.offset>=os.fstat(solver.prfile.fileno()).st_size:break
            output={'snapshot':store.snapshot(),'answers':answers,'counters':dict(store.stats),
                    'native':solver.accum_stats(),'placement_clauses':len(store.projected),'reusable_clauses':len(store.reusable)}
        finally:solver.delete()
    else:
        if request.get('snapshot'):
            store.queue_import(request['snapshot']);finish(store)
        solver=Glucose4(bootstrap_with=request['base'])
        try:
            for row in store.projected if request['mode']=='placement' else store.reusable:solver.add_clause(row)
            boards=[];answers=[]
            if request.get('enumerate'):
                while solver.solve():
                    positive={lit for lit in solver.get_model() if lit>0}
                    board=[next(code for code,lit in row if lit in positive) for row in request['mapping']]
                    boards.append(board)
                    solver.add_clause([-dict(row)[code] for row,code in zip(request['mapping'],board)])
                    if len(boards)>10000:raise RuntimeError('Finite enumeration exceeded its bound')
            else:answers=[solver.solve(assumptions=assumptions) for assumptions in request.get('queries',[[]])]
            output={'answers':answers,'boards':boards,'native':solver.accum_stats(),'counters':dict(store.stats),
                    'placement_clauses':len(store.projected),'reusable_clauses':len(store.reusable)}
        finally:solver.delete()
    output['elapsed_seconds']=time.monotonic()-start
    print(json.dumps(output),flush=True);print('cleanup_returned',flush=True)


def inventory(sequence=False):
    pigeons,holes=8,7;var=lambda p,h:1+p*holes+h
    base=[[var(p,h) for h in range(holes)] for p in range(pigeons)]
    top=pigeons*holes
    if sequence:
        from pysat.card import CardEnc,EncType
        for h in range(holes):
            cnf=CardEnc.atmost([var(p,h) for p in range(pigeons)],bound=1,top_id=top,encoding=EncType.seqcounter)
            base.extend(cnf.clauses);top=cnf.nv
    else:base.extend([-var(p,h),-var(q,h)] for h in range(holes) for p in range(pigeons) for q in range(p+1,pigeons))
    return {'base':base,'mapping':[[[h,var(p,h)] for h in range(holes)] for p in range(pigeons)],
            'initial':[127]*pigeons,'fingerprint':'inventory-8-7-'+('sequence' if sequence else 'pairwise')}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');parser.add_argument('--output',type=Path);args=parser.parse_args()
    from constraint_sat_persistence import require_capability
    require_capability()
    if args.child:return run_child(json.load(sys.stdin))
    if args.output is None:parser.error('--output is required')
    results=[]
    for sequence in (False,True):
        case=inventory(sequence);produced=child(dict(case,mode='produce'))
        runs={mode:child(dict(case,mode=mode,snapshot=produced['snapshot'] if mode!='baseline' else None)) for mode in ('baseline','placement','all_base')}
        if not any(x is None for x in produced['answers']):raise RuntimeError('Native production did not interrupt')
        if any(run['answers']!=[False] for run in runs.values()):raise RuntimeError('Inventory UNSAT result changed')
        results.append({'case':case['fingerprint'],'produced':produced,'restarts':runs})
    from test_constraint_sat_persistence import fixture,problem_store,brute
    from constraint_work import WorkScope
    problem,_=fixture();scope=WorkScope.for_problem(problem,'edges');store=problem_store(problem,scope)
    mapping=[list(row) for row in store.variables]
    # Mix complete roots with local duplicate-inventory UNSAT assumptions. None
    # of these assumption sets enter the independent root proof authority.
    queries=[[],*[ [dict(mapping[a])[code],dict(mapping[b])[code]] for a in range(4) for b in range(a+1,4) for code in range(12)]]
    case={'base':store.base.clauses,'mapping':mapping,'initial':scope.root,'fingerprint':problem.fingerprint,'queries':queries}
    produced=child(dict(case,mode='produce'))
    expected=sorted([list(row) for row in brute(problem,'edges')])
    runs={mode:child(dict(case,mode=mode,enumerate=True,snapshot=produced['snapshot'] if mode!='baseline' else None)) for mode in ('baseline','placement','all_base')}
    if len(expected)!=72 or any(sorted(run['boards'])!=expected for run in runs.values()):raise RuntimeError('Fresh-process finite solution set changed')
    results.append({'case':'finite-puzzle-72','produced':produced,'restarts':runs,'all_72_preserved':True})
    tests=subprocess.run([sys.executable,'-B','-m','unittest','-v','test_constraint_sat_persistence','test_constraint_sat_session'],
        env=dict(os.environ,DIAMOND_TEST_NATIVE_SAT_PERSISTENCE='1'),capture_output=True,text=True,timeout=120)
    import pysat
    report={'python':platform.python_version(),'platform':platform.platform(),'python_sat':pysat.__version__,
            'results':results,'tests':{'exit_code':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr},
            'source_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in Path(__file__).parent.glob('*.py')},
            'limits':['Verified native clause subset only; not full native solver state.',
                      'Exact source CNF and variable-map binding; no cross-engine clause import.',
                      '4 MiB monitored native log threshold may overshoot between polls; measured peak is reported.',
                      'Synthetic inventory discriminator and finite puzzle preservation, not original-puzzle gain.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    for result in results:print(json.dumps({'case':result['case'],'produced':result['produced']['counters'],
        'restarts':{mode:run['native'] for mode,run in result['restarts'].items()}}),flush=True)
    if tests.returncode:raise RuntimeError('Persistence unit tests failed; inspect report')
    if not any(row['restarts']['all_base']['native']['conflicts']<row['restarts']['baseline']['native']['conflicts'] for row in results[:2]):
        raise RuntimeError('No completed fresh-process saved-reasoning discriminator')


if __name__=='__main__':main()
