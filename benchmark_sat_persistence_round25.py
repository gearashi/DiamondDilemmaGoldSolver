"""Serial supported-platform persistence diagnostic, each restart a fresh process.

Synthetic inventory encodings and a finite puzzle establish bounded mechanism
usefulness and preservation. They do not predict original-puzzle performance.
"""
import argparse,hashlib,itertools,json,os,platform,subprocess,sys,time
from pathlib import Path
from benchmark_provenance import require_assertions,source_snapshot,runtime_snapshot,repository_snapshot

ROOT=Path(__file__).resolve().parent


def child(payload):
    require_assertions()
    completed=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--child'],input=json.dumps(payload),
        capture_output=True,text=True,timeout=90,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    if completed.returncode:raise RuntimeError('Persistence child failed: '+completed.stderr[-5000:])
    lines=completed.stdout.splitlines()
    if not lines or lines[-1]!='cleanup_returned':raise RuntimeError('Native child did not complete cleanup')
    return json.loads(lines[-2])


def run_child(request):
    require_assertions()
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
    require_assertions()
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


def require_capability():
    require_assertions()
    # Kept lazy so dirty/output/optimized-Python guards precede native probing.
    from constraint_sat_persistence import require_capability as check
    check()


def run_comparisons():
    require_assertions()
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
        env=dict(os.environ,DIAMOND_TEST_NATIVE_SAT_PERSISTENCE='1'),capture_output=True,text=True,timeout=120,cwd=ROOT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    return results,{'exit_code':tests.returncode,'stdout':tests.stdout,'stderr':tests.stderr}


def validate_comparisons(results,tests):
    if tests['exit_code']:
        raise RuntimeError('Persistence unit tests failed; no successful evidence report was written.\n'+tests.get('stderr','')[-5000:])
    if not any(row['restarts']['all_base']['native']['conflicts']<row['restarts']['baseline']['native']['conflicts'] for row in results[:2]):
        raise RuntimeError('No completed fresh-process saved-reasoning discriminator')


def main():
    require_assertions()
    parser=argparse.ArgumentParser();parser.add_argument('--child',action='store_true');parser.add_argument('--output',type=Path)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args()
    if args.child:
        # Parent owns the complete frozen envelope; a child never writes a
        # separately labelled frozen report or imports native code before guard.
        require_capability()
        return run_child(json.load(sys.stdin))
    if args.output is None:parser.error('--output is required')
    if args.output.suffix.lower()!='.json' or args.output.exists() or args.output.is_symlink():
        raise ValueError('SAT persistence output must be a fresh .json path.')
    before=source_snapshot(ROOT);runtime_before=runtime_snapshot();repository_before=repository_snapshot(ROOT)
    if args.evidence_stage=='frozen' and (not repository_before.get('commit') or repository_before.get('dirty') is not False):
        raise RuntimeError('Frozen SAT persistence evidence requires a clean, readable committed checkout.')
    require_capability()
    results,tests=run_comparisons()
    validate_comparisons(results,tests)
    after=source_snapshot(ROOT);runtime_after=runtime_snapshot();repository_after=repository_snapshot(ROOT)
    if before!=after or runtime_before!=runtime_after:
        raise RuntimeError('Sources or runtime changed during SAT persistence evaluation.')
    if repository_before.get('commit')!=repository_after.get('commit'):
        raise RuntimeError('Git HEAD changed during SAT persistence evaluation.')
    if args.evidence_stage=='frozen' and repository_before!=repository_after:
        raise RuntimeError('Git identity changed during frozen SAT persistence evaluation.')
    from importlib.metadata import version
    report={'python':platform.python_version(),'platform':platform.platform(),'python_sat':version('python-sat'),
            'results':results,'tests':tests,'evidence_stage':args.evidence_stage,'evidence_status':'passed',
            'source_sha256':{name:record['raw_sha256'] for name,record in before.items()},
            'source_before':before,'source_after':after,'runtime_before':runtime_before,'runtime_after':runtime_after,
            'repository_before':repository_before,'repository_after':repository_after,'source_unchanged':True,'runtime_unchanged':True,
            'child_evidence_scope':'Fresh-process producers/restarts and finite regression children share this parent source/runtime/Git envelope; children do not make independent frozen-evidence claims.',
            'limits':['Verified native clause subset only; not full native solver state.',
                      'Exact source CNF and variable-map binding; no cross-engine clause import.',
                      '4 MiB monitored native log threshold may overshoot between polls; measured peak is reported.',
                      'Synthetic inventory discriminator and finite puzzle preservation, not original-puzzle gain.']}
    encoded=json.dumps(report,indent=2,allow_nan=False)+'\n'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:stream.write(encoded)
    for result in results:print(json.dumps({'case':result['case'],'produced':result['produced']['counters'],
        'restarts':{mode:run['native'] for mode,run in result['restarts'].items()}}),flush=True)


if __name__=='__main__':main()
