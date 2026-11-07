"""Bounded completed-work evidence for F09-F13; no original-board speed claim."""
import argparse,json
from pathlib import Path
from benchmark_provenance import source_snapshot,runtime_snapshot,repository_snapshot,require_assertions


def f09():
    import itertools,random
    from search_problem import SearchProblem
    from test_constraint_dfs import board
    from test_constraint_advanced import domains_of
    from constraint_regions import HierarchicalRegionPruner
    from constraint_region_compiler_round26 import RegionalCompilerSession
    rng=random.Random(260905);positions=(1,3,6,9,11)
    selected=rng.sample(list(itertools.permutations(range(5))),7);sequence=(0,3,6,2,5,1,4)
    data=[{'id':i,'segments':[[[0,positions[j]],[1,positions[4-p[j]]]] for j in range(5)]} for i,p in enumerate(selected)]
    problem=SearchProblem(data,board(7,[(sequence[i],1,sequence[(i+1)%7],0) for i in range(7)]))
    owner=HierarchicalRegionPruner(problem);domains=domains_of(problem);results={}
    assert sum(sum(p[i]>p[j] for i in range(5) for j in range(i+1,5)) for p in selected)%2==1
    for name,policy in (('natural',{'order':tuple(range(7))}),('ordered',{}),('ranked',{'existence_only':True})):
        session=RegionalCompilerSession(owner,domains,**policy)
        while not session.complete and not session.blocked:session.advance(512)
        result=session.result();assert result is not None and not result['feasible']
        results[name]={'complete':True,'feasible':False,'order':session.order,'stats':session.stats}
    assert results['ranked']['stats']['regional_compiler_transitions']<results['ordered']['stats']['regional_compiler_transitions']<results['natural']['stats']['regional_compiler_transitions']
    return {'fixture':'7-cell five-strand scrambled ring, seed260905','oracle':'Odd product parity cannot be the even five-cycle required by Gold.',
      'cases':results,'limitations':['Rank is existence-only; counts and placement supports are deliberately unavailable.','Additional order evaluation, basis construction and XOR work is reported separately.','This is a completed bounded negative, not an original 160-cell solve.']}


def f10():
    from test_constraint_regions_round26 import SymbolicRegionalMessageTests
    from constraint_region_messages_round26 import SymbolicMessageJoin
    from constraint_region_interfaces_round26 import join_messages
    from constraint_region_tree_round26 import _pack,_unpack
    from test_constraint_dfs import brute
    problem,owner,domains,(left,right)=SymbolicRegionalMessageTests.fixture()
    baseline,_,pairs=join_messages(owner,(0,1,2),left,(3,4,5),right,domains)
    results={}
    for name,split in (('cold',False),('resumed',True)):
        session=SymbolicMessageJoin(owner,(0,1,2),left,(3,4,5),right,domains);steps=0;restores=0
        while not session.advance():
            steps+=1
            if split and steps%11==0:
                state=_unpack(json.loads(json.dumps(_pack(session.snapshot()))))
                session=SymbolicMessageJoin(owner,(0,1,2),left,(3,4,5),right,domains,state=state);restores+=1
        exact=lambda rows:{sig:(entry['count'],entry['supports']) for sig,entry in rows.items()}
        assert exact(session.result)==exact(baseline)
        rows=sum(entry['count'] for entry in session.result.values());assert rows==len(brute(problem,'gold'))
        results[name]={'rows':rows,'stats':session.stats,'controller_steps':steps+1,'restores':restores}
    assert results['cold']['stats']==results['resumed']['stats']
    assert results['cold']['stats']['recursive_region_symbolic_geometry_tests']<pairs
    return {'fixture':'Two three-cell messages on the six-cell ring','baseline_inventory_pair_geometry_checks':pairs,'cases':results,
      'limitations':['Both policies pay the same local relation compilation before this join comparison.','ZDD construction/operator overhead is separate from eliminated geometry checks.','Serialization/reconstruction is reported as restore count, not claimed free.','Shared inventories and exact Gold pairings remain correlated; this is not independent component multiplication.']}


def f11():
    from test_constraint_separator import ring
    from test_constraint_advanced import domains_of
    from constraint_regions import HierarchicalRegionPruner
    from constraint_region_proof_round26 import build_core,RegionalProofReuse,STAT_KEYS
    problem=ring(6);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem);domains[0]=domains[1]=1
    build_stats=dict.fromkeys(STAT_KEYS,0);record=build_core(owner,(0,1),domains,generalize=8,stats=build_stats);assert record is not None
    baseline=dict.fromkeys(STAT_KEYS,0);warm=RegionalProofReuse(owner)
    for cell in (2,3,4,5):
        child=domains.copy();child[cell]=1<<problem.domains[cell][-1]
        fresh=RegionalProofReuse(owner);assert fresh.import_certificate(record);assert fresh.infeasible(child)
        for key,value in fresh.stats.items():baseline[key]+=value
        assert warm.import_certificate(record);assert warm.infeasible(child)
    assert warm.stats['regional_core_verified_states']<baseline['regional_core_verified_states']
    assert warm.stats['regional_core_verified_import_hits']==3
    return {'fixture':'Duplicate physical piece in cells0/1, four increasingly fixed outside contexts','oracle':'The same physical piece cannot occupy two cells.',
      'shared_producer_cost':build_stats,'fresh_verification':baseline,'verified_assumption_index':warm.stats,
      'limitations':['Only byte-identical previously admitted proofs bypass repeat verification.','Novel external certificates always undergo independent transition-completeness checking.','Forced fixed outside contractions are supported; arbitrary active graph contractions are not certified.']}


def f12():
    from constraint_connectivity_round26 import ReversibleBlockCertificates,ObligationChainOracle,joint_demands,verify_joint_demands,assignment_upper_bound
    from constraint_region_multichain import JointChainOracle
    from constraint_advanced import cyclic_edge_blocks
    initial=((0,1),(1,2),(0,2),(3,4),(4,5),(3,5));changed=initial+((3,6),(4,6));contexts=(initial,changed,initial,changed)
    helper=ReversibleBlockCertificates();baseline_polls=0;dynamic_polls=0
    def base_check():
        nonlocal baseline_polls
        baseline_polls+=1
    def delta_check():
        nonlocal dynamic_polls
        dynamic_polls+=1
    for edges in contexts:
        expected=cyclic_edge_blocks(7,edges,base_check);actual=helper.update(edges,delta_check)
        for i,edge in enumerate(edges):assert (actual['blocks'][tuple(sorted(edge))] is None)==(expected[i]<0)
    graph={0:((3,((0,0),)),),1:((2,((0,0),)),),2:((1,((0,0),)),),3:((0,((0,0),)),),4:((5,((0,0),)),),5:((4,((0,0),)),)}
    chains=((0,1),(2,3));obligations={(0,0):((1,2),(0,3),(4,5))}
    old=JointChainOracle(graph,chains);old_answer=old.possible();new=ObligationChainOracle(graph,chains,obligations);new_answer=new.possible()
    assert old_answer is True and new_answer is False
    rows=[[(0,(2,0)),(0,(0,2))]];assert not joint_demands(rows,(2,2));verified,verify_steps=verify_joint_demands(rows,(2,2));assert verified
    bound=assignment_upper_bound([[3,1],[3,1]])
    assert bound['bound']==4
    return {'fixture':'Two unchanged/delta/restored cyclic components; omitted third strand; correlated two-demand orientation',
      'graph':{'fresh_traversal_callback_steps':baseline_polls,'incremental_callback_steps':dynamic_polls,'stats':helper.stats},
      'routing':{'old_possible':old_answer,'old_steps':old.steps,'all_strands_possible':new_answer,'new_steps':new.steps,'label_checks':new.label_tests,'refinements':new.refinements,'added_obligations':new.added},
      'joint_demand':{'independent_verified_failure':verified,'verification_steps':verify_steps},'assignment_bound':bound,
      'limitations':['Callbacks, edge counts and route states are distinct work units, not a timing comparison.','The stronger route can do more work while proving a contradiction the older relaxation misses.','Positive capacity/routing relaxations never establish Gold.']}


def f13():
    from unittest.mock import patch
    from contextlib import nullcontext
    from test_constraint_separator import ring
    from test_constraint_advanced import domains_of,bits
    from constraint_regions import HierarchicalRegionPruner
    from constraint_region_integer_round26 import IntegerRegionalSession
    problem=ring(6);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
    for cell in (0,2,4):domains[cell]=bits((0,3))
    results={}
    for name,enabled in (('integer_weighted_bound_fallback',False),('branch_and_price',True)):
        context=nullcontext() if enabled else patch('constraint_region_branch_price_round26.phase_one',return_value={'status':'unavailable'})
        session=IntegerRegionalSession(owner,domains,region_size=2)
        with context:
            while not session.complete and not session.blocked:session.advance(128)
        result=session.result();assert result is not None and not result['feasible']
        results[name]={'complete':True,'feasible':False,'stats':session.stats}
        for node in session.lp_nodes.values():
            if node.status=='infeasible':assert node._verify_negative(node.dual,lambda:None)
    lp_available=bool(results['branch_and_price']['stats']['integer_bp_native_calls'])
    if lp_available:assert results['branch_and_price']['stats']['integer_bp_certified_failures']>0
    return {'fixture':'Three constrained cells sharing only two physical pieces across three nonempty regions','oracle':'Pigeonhole principle on cells0/2/4 and physical pieces0/1.',
      'optional_native_lp_exercised':lp_available,'cases':results,
      'limitations':['LP proposals alone prove nothing; omitted paths and Farkas inequalities are replayed exactly.','The fallback already has exact weighted bounds, so this comparison includes a strong existing regional relaxation.','No universal branch-count or wall-time dominance is claimed.','Native solver objects are rebuilt; completed native results and exact pricing subqueries are retained.']}


RUNNERS={'F09':f09,'F10':f10,'F11':f11,'F12':f12,'F13':f13}

def main(argv=None):
    require_assertions()
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=tuple(RUNNERS));parser.add_argument('--output',required=True)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args(argv)
    output=Path(args.output).resolve();root=Path(__file__).resolve().parent
    if output.suffix.lower()!='.json' or output.exists() or not output.parent.is_dir():raise ValueError('A fresh JSON output in an existing directory is required.')
    before=source_snapshot(root);runtime_before=runtime_snapshot();repo_before=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repo_before.get('available') or repo_before.get('dirty') or not repo_before.get('commit')):raise RuntimeError('Frozen evidence requires a readable clean Git commit.')
    results={family:run() for family,run in RUNNERS.items() if args.family is None or args.family==family}
    after=source_snapshot(root);runtime_after=runtime_snapshot();repo_after=repository_snapshot(root)
    if before!=after or runtime_before!=runtime_after:raise RuntimeError('Source/runtime changed during regional finite evidence; discard the run.')
    if repo_before!=repo_after:raise RuntimeError('Repository identity changed during regional finite evidence; discard the run.')
    report={'schema':'diamond-regional-round26-evidence-v1','status':'passed','evidence_stage':args.evidence_stage,'families':results,
      'interpretation':'Completed finite logical-work evidence; no timing or original-board solution claim.',
      'provenance':{'sources_before':before,'sources_after':after,'runtime_before':runtime_before,'runtime_after':runtime_after,'repository_before':repo_before,'repository_after':repo_after}}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'passed','families':tuple(results),'evidence_stage':args.evidence_stage,'source_runtime_stable':True}))
    return report

if __name__=='__main__':main()
