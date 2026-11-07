"""Finite Gold pool/master comparisons, called by the frozen repair runner.

No performance claim: constructor/native-call counts and independent completed
outcomes are the evidence. Native UNKNOWN/exception attempts stay separate.
"""
from contextlib import contextmanager
from unittest.mock import patch
import itertools
import json
from collections import Counter


@contextmanager
def native_work(cp):
    original=cp.CpSolver.solve;report={'calls':0,'statuses':{},'branches':0,'conflicts':0,'exceptions':0}
    def solve(solver,*args,**kwargs):
        report['calls']+=1
        try:status=original(solver,*args,**kwargs)
        except BaseException:
            report['exceptions']+=1;raise
        name=solver.status_name(status);report['statuses'][name]=report['statuses'].get(name,0)+1
        report['branches']+=int(solver.num_branches);report['conflicts']+=int(solver.num_conflicts)
        return status
    with patch.object(cp.CpSolver,'solve',solve):yield report


def finite():
    from ortools.sat.python import cp_model as cp
    from constraint_lns import _Control
    from constraint_repair_round24 import GoldRepairOracle,_minimum_hitting_set
    from constraint_repair_resume_round25 import _fragment_master
    from constraint_repair_gold_round26 import GoldRepairPool
    from test_retained_gold_repair_round26 import hit_optimum,fragment_optimum
    from search_problem import SearchProblem
    from test_constraint_dfs import tetrahedron_tiles,brute
    problem=SearchProblem(*tetrahedron_tiles());solutions=brute(problem,'gold');seed=[0,4,6,9]
    queries=[()]+[((cell,code),) for cell in range(4) for code in problem.domains[cell]]
    def verify(assumptions,answer):
        possible=any(all(board[c]==v for c,v in assumptions) for board in solutions)
        assert answer['status']==('solved' if possible else 'infeasible')
        if possible:
            assert tuple(answer['codes']) in solutions
            assert all(answer['codes'][c]==v for c,v in assumptions)
        else:assert set(answer['core'])<=set(assumptions)
    cold_models=0;cold_variables=0;cold_constraints=0
    with native_work(cp) as cold_native:
        for assumptions in queries:
            oracle=GoldRepairOracle(problem,_Control(10,None));cold_models+=1
            cold_variables+=len(oracle.model.model.proto.variables);cold_constraints+=len(oracle.model.model.proto.constraints)
            verify(assumptions,oracle.query(assumptions))
    pool=GoldRepairPool(problem)
    with native_work(cp) as warm_native:
        for index,assumptions in enumerate(queries):
            verify(assumptions,pool.oracle(_Control(10,None)).query(assumptions))
            if index==len(queries)//2:pool=GoldRepairPool(problem,resume=json.loads(json.dumps(pool.snapshot())))
        first=pool.stats
        for assumptions in queries:verify(assumptions,pool.oracle(_Control(10,None)).query(assumptions))
    assert pool.stats['gold_pool_native_calls']==first['gold_pool_native_calls']
    assert pool.stats['gold_pool_model_builds']<cold_models
    gold={'contexts':len(queries),'independent_gold_boards':len(solutions),
          'cold':{'models':cold_models,'variables':cold_variables,'constraints':cold_constraints,'native':cold_native},
          'retained_with_one_restore':{'stats':pool.stats,'native':warm_native},
          'repeated_contexts_native_calls':pool.stats['gold_pool_native_calls']-first['gold_pool_native_calls']}
    core_rows=[[],[[0,1]],[[0,1],[1,2]],[[0,1],[1,2],[2,3]],[[0,1],[1,2],[2,3],[0,3]]]
    pool=GoldRepairPool(problem);cells=list(range(4))
    with native_work(cp) as cold_hit:
        for rows in core_rows:
            chosen,_=_minimum_hitting_set([frozenset(r) for r in rows],cells,_Control(10,None),cp)
            assert len(chosen)==hit_optimum(cells,rows)
    with native_work(cp) as warm_hit:
        for rows in core_rows:
            chosen,_=pool.minimum_hitting_set(rows,cells,_Control(10,None),cp,seed=seed)
            assert len(chosen)==hit_optimum(cells,rows)
        before=pool.stats;pool=GoldRepairPool(problem,resume=json.loads(json.dumps(pool.snapshot())))
        chosen,_=pool.minimum_hitting_set(core_rows[-1],cells,_Control(10,None),cp,seed=seed)
    assert pool.stats['gold_pool_master_calls']==before['gold_pool_master_calls']
    hit={'contexts':len(core_rows),'cold_models':len(core_rows),'cold_native':cold_hit,'retained_native':warm_hit,'retained_stats':pool.stats}
    fragments=[{'source':0,'placements':[(0,0)]},{'source':0,'placements':[(1,3)]},
               {'source':1,'placements':[(0,3)]},{'source':1,'placements':[(1,0)]}]
    conflicts=[[],[[0,1]],[[0,1],[2,3]],[[0,1],[2,3],[0]]]
    pool=GoldRepairPool(problem)
    with native_work(cp) as cold_fragment:
        for rows in conflicts:
            chosen,_=_fragment_master(fragments,rows,_Control(10,None),cp)
            assert sum(len(fragments[i]['placements']) for i in chosen)==fragment_optimum(fragments,rows)
    with native_work(cp) as warm_fragment:
        for rows in conflicts:
            chosen,_=pool.fragment_master(fragments,rows,_Control(10,None),cp)
            assert sum(len(fragments[i]['placements']) for i in chosen)==fragment_optimum(fragments,rows)
        pool=GoldRepairPool(problem,resume=json.loads(json.dumps(pool.snapshot())))
        chosen,_=pool.fragment_master(fragments,conflicts[-1],_Control(10,None),cp)
    fragment={'contexts':len(conflicts),'cold_models':len(conflicts),'cold_native':cold_fragment,
              'retained_native':warm_fragment,'retained_stats':pool.stats}
    from search_fixtures import planted_gold
    data,mesh,witness=planted_gold(261717,size=4)
    large=SearchProblem(data,mesh,fixed={i:v for i,v in enumerate(witness) if i>=3})
    completions=[]
    for pieces in itertools.permutations([witness[i]//3 for i in range(3)]):
        for rotations in itertools.product(range(3),repeat=3):
            board=list(witness)
            for i,(piece,rotation) in enumerate(zip(pieces,rotations)):board[i]=piece*3+rotation
            if large.validate(board)['valid']:completions.append(board)
    assert completions
    pool=GoldRepairPool(large);contexts=[(),((0,witness[0]),),((0,witness[0]//3*3+(witness[0]+1)%3),)]
    with native_work(cp) as native160:
        for assumptions in contexts:
            answer=pool.oracle(_Control(10,None)).query(assumptions)
            possible=[b for b in completions if all(b[c]==v for c,v in assumptions)]
            assert answer['status']==('solved' if possible else 'infeasible')
            if possible:assert answer['codes'] in completions
    controlled={'cells':160,'fixed_cells':157,'enumerated_assignments':162,'gold_boards':len(completions),
                'contexts':len(contexts),'native':native160,'stats':pool.stats}
    return {'gold_queries':gold,'hitting_master':hit,'fragment_master':fragment,'controlled160':controlled,
            'limits':['Controlled 160-cell fixture has only three free cells; it is not original-puzzle progress.',
                      'Native branches/conflicts are distinct backend units, not elapsed-time evidence.',
                      'Restored optimal bounds and failures are trusted-local records, not portable proofs.',
                      'Incomplete native searches retry; only completed queries and optima suppress work.',
                      'Bounded FIFO eviction may cause later repetition; counters expose evictions.']}
