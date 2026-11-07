"""Bounded Linux proof-prefix diagnostics, never an application search engine.

A checked RUP prefix is only a source of entailed clauses. It is not a snapshot
of the native solver. All assumptions remain outside the independent base.
"""
import argparse,hashlib,itertools,json,platform,subprocess,sys
from pathlib import Path
from constraint_proof import RUPChecker,ProofLimit,_CaptureSink

PRODUCE=r"""
import json,sys
from pysat.solvers import Solver
s=json.load(sys.stdin);solver=Solver(name=s['backend'],bootstrap_with=s['base'],with_proof=True)
try:
    rounds=[]
    for _ in range(3):
        solver.conf_budget(1)
        answer=solver.solve_limited(assumptions=s['assumptions'],expect_interrupt=True)
        proof=solver.get_proof()
        rounds.append({'answer':answer,'proof':proof,'stats':solver.accum_stats()})
        if answer is not None:break
    print(json.dumps({'status':'produced','rounds':rounds}),flush=True)
finally:solver.delete()
print(json.dumps({'status':'cleanup_returned'}),flush=True)
"""
RESTART=r"""
import itertools,json,sys
from pysat.solvers import Solver
s=json.load(sys.stdin);solver=Solver(name=s['backend'],bootstrap_with=s['base']+s['learned'])
try:
    results=[]
    for assumptions in s['queries']:
        solver.conf_budget(100)
        answer=solver.solve_limited(assumptions=assumptions,expect_interrupt=True)
        results.append(answer)
    boards=[]
    if s.get('variables'):
        while solver.solve():
            positive={x for x in solver.get_model() if x>0}
            board=[next(code for code,lit in row if lit in positive) for row in s['variables']]
            boards.append(board)
            solver.add_clause([-next(lit for code,lit in row if code==choice) for row,choice in zip(s['variables'],board)])
            if len(boards)>10000:raise RuntimeError('Enumeration bound exceeded')
    print(json.dumps({'status':'restarted','queries':results,'boards':boards}),flush=True)
finally:solver.delete()
print(json.dumps({'status':'cleanup_returned'}),flush=True)
"""

def child(source,payload):
    try:
        process=subprocess.run([sys.executable,'-B','-c',source],input=json.dumps(payload),capture_output=True,text=True,timeout=15)
    except subprocess.TimeoutExpired:return {'status':'timeout','clean_exit':False}
    records=[]
    for line in process.stdout.splitlines():
        try:records.append(json.loads(line))
        except ValueError:pass
    return {'exit_code':process.returncode,'clean_exit':process.returncode==0 and any(r.get('status')=='cleanup_returned' for r in records),
            'records':records,'stderr':process.stderr[-2000:]}


def checked_prefix(base,proof,projection):
    active=list(map(tuple,base));known=set(active);selected=[];accepted=0;rejected=None
    checker=RUPChecker(base,max_steps=20000,max_visits=4000000)
    if not isinstance(proof,list):return {'status':'unavailable','clauses':[]}
    try:
        for index,line in enumerate(proof):
            tokens=line.split()
            if not tokens:continue
            if tokens[0]=='d':continue
            row=tuple(map(int,tokens))
            if not row or row[-1]!=0 or 0 in row[:-1]:rejected={'index':index,'reason':'malformed'};break
            row=tuple(dict.fromkeys(row[:-1]))
            if not checker._rup(active,row,lambda:None):rejected={'index':index,'reason':'not RUP under original root','clause':list(row)};break
            accepted+=1
            if row not in known:
                active.append(row);known.add(row)
                if row and all(abs(lit) in projection for lit in row):selected.append(list(row))
        status='checked'
    except ProofLimit:status='bounded';rejected={'reason':'independent check work limit'}
    return {'status':status,'accepted_prefix_steps':accepted,'first_rejected':rejected,'clauses':selected,'checker':checker.stats}


def php_case():
    pigeons,holes=8,7
    variable=lambda p,h:1+p*holes+h
    gate=pigeons*holes+1
    base=[[variable(p,h) for h in range(holes)] for p in range(pigeons)]
    base += [[-variable(p,h),-variable(q,h)] for h in range(holes) for p in range(pigeons) for q in range(p+1,pigeons)]
    guarded=[[-gate,*row] for row in base]
    return {'name':'guarded_inventory_pigeonhole','base':guarded,'assumptions':[gate],'projection':list(range(1,gate)),
            'queries':[[],[-gate],[gate]],'expected_queries':[True,True,False],
            'non_claim':'Synthetic inventory analogue, not the original puzzle; selector-containing clauses are not importable placement clauses.'}


def finite_case():
    from types import SimpleNamespace
    from search_problem import SearchProblem
    from constraint_sat_assist import SatAssist
    pairings=[[[[1,10],[1,2]],[[2,3],[0,2]],[[2,9],[0,10]]],
              [[[0,10],[1,10]],[[2,9],[0,2]],[[1,2],[2,3]]],
              [[[0,10],[0,2]],[[2,9],[1,2]],[[2,3],[1,10]]],
              [[[0,2],[2,3]],[[1,10],[1,2]],[[0,10],[2,9]]]]
    edges=[(0,0,1,0),(0,1,2,0),(0,2,3,0),(1,1,2,1),(1,2,3,1),(2,2,3,2)]
    p=SearchProblem([{'id':i,'segments':row} for i,row in enumerate(pairings)],SimpleNamespace(cells=tuple(range(4)),edges=edges))
    initial=[sum(1<<v for v in row) for row in p.domains]
    encoder=SatAssist(p,initial,'edges',factory=_CaptureSink,capture_clauses=True)
    try:
        if not encoder.prepare(seconds=5):raise RuntimeError('Finite model construction did not finish')
        model=encoder.proof_base()
    finally:encoder.close()
    variables=[dict(row) for row in model['variables']]
    expected=[]
    for permutation in itertools.permutations(range(4)):
        for rotations in itertools.product(range(3),repeat=4):
            board=[3*t+r for t,r in zip(permutation,rotations)];report=p.validate(board)
            if report['unique_tiles'] and report['matched_edges']==len(edges):expected.append(board)
    # Duplicate physical tile assumptions are intentionally local UNSAT. An
    # empty proof clause for this call must never be admitted at the root.
    assumptions=[variables[0][0],variables[1][0]]
    return {'name':'finite_puzzle_scoped_conflict','base':[list(row) for row in model['clauses']],
            'assumptions':assumptions,'projection':[lit for row in variables for lit in row.values()],
            'queries':[[],assumptions,[variables[0][expected[0][0]],variables[1][expected[0][1]]]],
            'expected_queries':[True,False,True],'variables':[[list(pair) for pair in row.items()] for row in variables],
            'expected_boards':expected,'fingerprint':p.fingerprint}


def main():
    import pysat
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    cases=[php_case(),finite_case()];rows=[]
    for backend in ['g3','g4','g42','gc3','gc4','mcb']:
        for case in cases:
            request={'backend':backend,**case};production=child(PRODUCE,request)
            data=next((r for r in production.get('records',[]) if r.get('status')=='produced'),{})
            snapshots=data.get('rounds',[]);proof=snapshots[-1].get('proof') if snapshots else None
            checked=checked_prefix(case['base'],proof,set(case['projection'])) if production['clean_exit'] else {'status':'unclean producer','clauses':[]}
            learned=checked['clauses']
            restarted=child(RESTART,{'backend':backend,'base':case['base'],'learned':learned,'queries':case['queries'],'variables':case.get('variables')})
            result=next((r for r in restarted.get('records',[]) if r.get('status')=='restarted'),{})
            queries=result.get('queries',[])
            sound_queries=len(queries)==len(case['expected_queries']) and all(actual is None or actual is expected for actual,expected in zip(queries,case['expected_queries']))
            exact_finite=(not case.get('variables') or sorted(result.get('boards',[]))==sorted(case['expected_boards']))
            row={'backend':backend,'case':case['name'],'original_cnf_sha256':hashlib.sha256(json.dumps(case['base']).encode()).hexdigest(),
                 'producer':production,'checked_prefix':checked,'restart':restarted,'query_answers_sound':sound_queries,
                 'finite_solution_set_equal':exact_finite,'finite_solution_count':len(case.get('expected_boards',[])),
                 'observed_interrupted_call':any(r.get('answer') is None for r in snapshots),'projected_clauses_imported':len(learned)}
            rows.append(row)
            print(json.dumps({k:row[k] for k in ('backend','case','observed_interrupted_call','projected_clauses_imported','query_answers_sound','finite_solution_set_equal')}),flush=True)
    result={'python':platform.python_version(),'platform':platform.platform(),'python_sat':pysat.__version__,'results':rows,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'limits':['Bounded capability tests, not original-puzzle performance.','A proof prefix is not a full learned database or native search snapshot.','Only independently checked clauses using original placement variables are imported.','No native proof logger or importer is enabled in the app.']}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    if any(not r['query_answers_sound'] or not r['finite_solution_set_equal'] for r in rows):raise RuntimeError('Restart validation failed; inspect artifact.')
if __name__=='__main__':main()
