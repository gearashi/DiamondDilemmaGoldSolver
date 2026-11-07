"""Reproducible bounded logical-work discriminators for R24 regional proposals.

Run: python -B benchmark_regions_round24.py --output PATH
Optional --original-tiles PATH adds one capped full-original factor query.
Results are finite scope; they do not establish an original Gold solution.
"""
import argparse
import itertools
import json
from math import comb
from pathlib import Path
from fractions import Fraction
from constraint_region_symbolic import (DirectRegionDiagram,DirectRegionSession,inventory_supports,
    subset_convolution,endpoint_matchings,representative_matchings,single_cycle)
from constraint_region_elimination import (Factor,EliminationPruner,eliminate,
    relaxed_diagram_possible,refined_diagram_possible)
from constraint_region_master import (LazyRegionMaster,priced_inventory_master,check_farkas)
from constraint_region_oracles import ResidualProofCache,LabelledRouteOracle
from constraint_region_network import ExactRegionDiagram
from constraint_regions import HierarchicalRegionPruner
from constraint_separator import _Budget
from test_constraint_separator import ring
from test_constraint_advanced import domains_of,bits


def baseline_inventory(groups):
    supported=[set() for _ in groups];memo={};work={'state_calls':0,'inventory_trials':0}
    def solve(index,used):
        work['state_calls']+=1;key=(index,used)
        if key in memo:return memo[key]
        if index==len(groups):return True
        possible=False
        for inventory in groups[index]:
            work['inventory_trials']+=1
            if inventory&used:continue
            if solve(index+1,used|inventory):supported[index].add(inventory);possible=True
        memo[key]=possible;return possible
    solve(0,0);return supported,work


def run(original_tiles=None):
    report={'schema':1,'scope':'Finite exact logical-work discriminators; no original Gold solution claim.',
            'candidates':{}}
    results=report['candidates']
    problem=ring(7);a=[bits([0,3])]*3+[1<<(3*c) for c in range(3,7)]
    b=a.copy();b[3],b[4]=b[4],b[3]
    assert not any(len(set(row))==3 for row in itertools.product((0,3),repeat=3))
    cache=ResidualProofCache(problem);cache.remember_completed(a,status='unsatisfiable')
    assert cache.contains(b)
    results['8']={'outcome':'verified_reuse','different_fixed_histories':a!=b,'groups':cache.group(((0,a),(1,b))),
                  'stats':cache.stats,'scope':'Same labels with exactly equivalent contracted fixed interiors; general graph isomorphism not implemented.'}
    problem=ring(7);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
    eager=ExactRegionDiagram(owner,range(7),domains,lambda:None,max_rows=6000)
    direct=DirectRegionDiagram(owner,range(7),domains,max_rows=512)
    assert set(eager.histories())==set(direct.histories())
    session=DirectRegionSession(owner,range(7),domains,max_rows=512);slices=0
    while not session.advance(7):slices+=1
    assert session.steps==direct.steps
    results['17']={'outcome':'verified_less_enumeration','histories':direct.rows,'eager_candidate_trials':eager.steps,
        'direct_candidate_trials':direct.steps,'direct_states':direct.states,'continuation_slices':slices,
        'continuation_repeated_trials':session.steps-direct.steps}
    groups=[{sum(1<<p for p in pair) for pair in itertools.combinations(range(12),2)}]*6
    expected,oldwork=baseline_inventory(groups);actual,work=inventory_supports(groups)
    assert actual==expected
    results['18']={'outcome':'verified_exact_symbolic_allocation','baseline':oldwork,'symbolic':work,
                   'supported_inventories_per_region':list(map(len,actual))}
    problem=ring(160);domains=[bits([3*c,3*((c+80)%160)]) for c in range(160)]
    helper=EliminationPruner(problem,max_steps=200000);actual=domains.copy();helper.prune(actual,full=True)
    assert actual==domains and helper.stats['factor_unknown']==0
    base=[3*c for c in range(160)];assert problem.validate(base)['valid']
    for cell in range(80):
        witness=base.copy();witness[cell],witness[cell+80]=witness[cell+80],witness[cell]
        assert problem.validate(witness)['valid']
    capped=False
    try:DirectRegionDiagram(HierarchicalRegionPruner(problem),range(160),domains,max_rows=4096)
    except _Budget:capped=True
    assert capped
    results['19']={'outcome':'verified_complete_160_cell_support_pass','stats':helper.stats,
        'explicit_global_witnesses':81,'baseline_full_inventory_state_cap':4096,'baseline_capped':capped,
        'scope':'Synthetic 160-cell ring, paired long-range physical resources; not original tile data.'}
    neq=frozenset((a,b) for a in range(3) for b in range(3) if a!=b)
    factors=[Factor((a,b),neq) for a,b in itertools.combinations(range(4),2)]
    relaxed,w3=eliminate(factors,range(4),width=3);exact,w4=eliminate(factors,range(4),width=4)
    assert not exact
    results['20']={'outcome':'verified_optimistic_relaxation','width3_possible':relaxed,'width4_possible':exact,
                   'width3_work':w3,'width4_work':w4,'scope':'Abstract K4 three-color finite relation.'}
    factors=[Factor(pair,frozenset({(0,1),(1,0)})) for pair in ((0,1),(1,2),(0,2))]
    choices={v:(0,1) for v in range(3)}
    coarse,cwork=relaxed_diagram_possible(factors,choices,width=1)
    refined,rwork=refined_diagram_possible(factors,choices)
    assert coarse and not refined
    results['21']={'outcome':'verified_refinement_deduction','coarse':cwork,'refined':rwork}
    matchings=tuple(endpoint_matchings(range(8)));selected,work=representative_matchings(matchings)
    for outside in matchings:assert any(single_cycle(row,outside) for row in matchings)==any(single_cycle(row,outside) for row in selected)
    assert len(selected)<len(matchings)
    results['22']={'outcome':'verified_existence_preservation','work':work,'outside_matchings_checked':len(matchings),
                   'scope':'Identical inventory and boundary class; no unconditional support/count preservation claim.'}
    n=10;counts=[int(mask.bit_count()==5) for mask in range(1<<n)]
    product,operations=subset_convolution(counts,counts,n)
    assert product[-1]==comb(10,5) and sum(product)==comb(10,5)
    pairs=comb(10,5)**2;assert operations<pairs
    results['23']={'outcome':'verified_fewer_join_operations','explicit_inventory_pair_tests':pairs,
                   'integer_transform_operations':operations,'joined_history_count':sum(product)}
    problem=ring(8);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
    master=LazyRegionMaster(owner,domains,((0,1,2,3),(4,5,6,7)));assert master.solve()
    assert problem.validate([master.solution[c] for c in range(8)])['valid']
    eager_rows=sum(DirectRegionDiagram(owner,part,domains).rows for part in master.regions)
    results['24']={'outcome':'verified_lazy_realization','eager_rows':eager_rows,'lazy':master.stats}
    problem=ring(6);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
    diagrams=[DirectRegionDiagram(owner,part,domains) for part in ((0,1),(2,3),(4,5))]
    pricing=priced_inventory_master(diagrams,6)
    if pricing['status']=='unknown':
        results['26']={'outcome':'inconclusive_backend_or_budget','pricing':pricing}
    else:
        assert pricing['status']=='possible'
        bad=[bits([0,3,6,9])]*6
        failure=priced_inventory_master([DirectRegionDiagram(owner,part,bad) for part in ((0,1),(2,3),(4,5))],6)
        assert failure['status']=='infeasible'
        assert check_farkas([{0:1,1:1},{0:1,2:1}],(1,1,1),(-1,1,1))
        results['26']={'outcome':'verified_exact_certificate_and_pricing','feasible':pricing,'infeasible':failure,
                       'all_eager_histories':sum(d.rows for d in diagrams)}
    graph={}
    for a,b,labels in ((0,1,((0,0),)),(1,3,()),(0,2,((0,1),)),(2,3,())):
        graph.setdefault(a,[]).append((b,labels));graph.setdefault(b,[]).append((a,labels))
    oracle=LabelledRouteOracle(graph);assert oracle.possible(0,3,count=1)
    single_steps=oracle.steps;assert oracle.possible(0,3,count=2) is False
    results['27']={'outcome':'verified_joint_route_obstruction','individual_path_steps':single_steps,'joint_route_steps':oracle.steps,
                   'scope':'Two individually available routes require inconsistent orientations of the same cell.'}
    if original_tiles:
        from search_problem import SearchProblem
        data=json.loads(Path(original_tiles).read_text(encoding='utf-8-sig'))
        problem=SearchProblem(data);domains=domains_of(problem);helper=EliminationPruner(problem,max_steps=50000)
        actual=domains.copy();helper.prune(actual,full=True)
        report['original_capped_factor_query']={'cells':problem.n,'stats':helper.stats,
            'removed':sum((a^b).bit_count() for a,b in zip(domains,actual)),
            'scope':'One 50,000-operation construction/reasoning cap; unknown is not infeasibility.'}
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--original-tiles')
    args=parser.parse_args();result=run(args.original_tiles)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'candidates':len(result['candidates']),'output':args.output}))
