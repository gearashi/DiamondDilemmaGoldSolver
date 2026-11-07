"""Finite protected-model discriminator; no wall-time winner claims."""
import hashlib,itertools,json,random,sys
from pathlib import Path
from constraint_sat_session import PersistentSatSession
from constraint_work import WorkScope
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face,brute
from test_constraint_sat_domains import fixture
from test_constraint_separator import ring
from constraint_pruning import values


def compare():
    rows=[]
    for label,problem in [('double',SearchProblem(simple_tiles(),double_face())),('tetra',fixture()[0]),('ring4',ring(4)),('ring5',ring(5))]:
        root=tuple(sum(1<<c for c in row) for row in problem.domains);rng=random.Random(2605)
        queries=([(1<<a,1<<b) for a,b in itertools.product(range(6),repeat=2)] if label=='double' else
                 [tuple(sum(1<<c for c in values(mask) if rng.random()<.5) for mask in root) for _ in range(64)])
        solutions=brute(problem,'edges');expected=[any(all(mask>>v&1 for mask,v in zip(query,board)) for board in solutions) for query in queries]
        results=[]
        for enabled in (False,True):
            session=PersistentSatSession(problem,WorkScope.for_problem(problem,'edges'),simplification=enabled)
            try:
                session.prepare(lambda:None,2);answers=[]
                for query in queries:
                    assumptions=[-lit for cell,row in enumerate(session.variables) for code,lit in row.items() if not query[cell]>>code&1]
                    answer=session.solver.solve(assumptions=assumptions);answers.append(answer)
                    if answer and enabled:session._simplified.reconstruct(session.solver.get_model())
                if answers!=expected:raise RuntimeError('Independent protected-assignment oracle disagrees.')
                results.append({'enabled':enabled,'base_clauses':len(session._base_clauses),'search_clauses':len(session._search_formula()),
                    'native':dict(session.solver.accum_stats()),'sat_answers':sum(answers),'simplification_stats':{k:v for k,v in session.stats.items() if k.startswith('simplify_')}})
            finally:session.close()
        rows.append({'fixture':label,'seed':2605,'query_masks':queries,'independent_edge_solutions':len(solutions),'results':results})
    return rows

if __name__=='__main__':
    if not __debug__:raise RuntimeError('Assertions must remain enabled.')
    files=('constraint_sat_simplify_round26.py','constraint_sat_session.py','test_constraint_sat_simplify_round26.py',Path(__file__).name)
    before={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files};rows=compare()
    after={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files}
    if before!=after:raise RuntimeError('Sources changed during comparison.')
    report={'scope':'Finite exact placement projections, no original-puzzle speed claim','results':rows,'sources':before}
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(json.dumps(report,indent=2))
    print(json.dumps([{'fixture':r['fixture'],'results':[{k:v for k,v in a.items() if k!='simplification_stats'} for a in r['results']]} for r in rows]))
