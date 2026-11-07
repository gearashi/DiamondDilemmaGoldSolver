"""Finite logical-work reports for ordered learning proposals; not speed claims."""
import argparse,hashlib,json,platform,subprocess,importlib.metadata
from pathlib import Path
from benchmark_provenance import source_snapshot, runtime_snapshot, repository_snapshot, require_assertions


def core_lifting():
    from test_constraint_learning_round25 import run_core_fixture
    baseline=run_core_fixture(False);candidate=run_core_fixture(True)
    assert baseline['status']==candidate['status']=='infeasible'
    assert baseline['checkpoint']['frontier']['boxes']==candidate['checkpoint']['frontier']['boxes']==[]
    original=baseline['stats']['session_native_calls']
    improved=candidate['stats']['session_native_calls']+candidate['stats']['session_core_verifications']
    assert improved<original
    return {'proposal':1,'assertion_tested':'Twelve disjoint contradictory 4-cell cubes are exhausted with fewer total SAT queries, including independent verification.',
            'baseline':baseline['stats'],'candidate':candidate['stats'],'total_queries':[original,improved],
            'non_claims':['No original Gold-puzzle solution or time-to-solution improvement established.','This finite work count does not compare native conflict work or wall-clock speed.']}


def shared_assistant():
    from test_constraint_job_sat import compare_job_models
    report=compare_job_models()
    assert report['shared']['model_builds']<report['baseline']['model_builds']
    assert report['shared']['native_support_calls']<report['baseline']['native_support_calls']
    return {'proposal':2,'assertion_tested':'Twenty alternating related jobs share one exact SAT model and completed support work, preserving independently enumerated completions.',
            **report,'non_claims':['Same-process model and query reuse, not durable native SAT internals.','No original-puzzle solution or universal speedup established.']}


def native_explanations():
    from test_constraint_native_reasons import compare_native_explanations
    report=compare_native_explanations()
    assert report['recorded']['checker_work']<report['replayed']['checker_work']
    assert report['recorded']['solutions']==report['replayed']['solutions']
    return {'proposal':4,'assertion_tested':'Native reason trails preserve all independently enumerated completions while reducing repeated Python explanation-checking work.',
            **report,'non_claims':['Counters measure Python checking only; native trace recording overhead is additional.','No original-puzzle time-to-solution claim.']}


def resumable_proof():
    from test_constraint_learning_round25 import run_proof_cursor_fixture
    report=run_proof_cursor_fixture()
    assert report['resumed_completed'] and not report['baseline_completed']
    assert report['resumed_clause_visits']<report['baseline_clause_visits']
    return {'proposal':3,'assertion_tested':'Retained RUP verification completes the same finite proof across small leases without revisiting its checked prefix or partial propagation.',
            **report,'non_claims':['Trusted cursor survives warm leases only, not process restart.','No original-puzzle search reduction or universal speedup established.']}


def symbolic_coverage():
    from test_constraint_learning_round25 import run_symbolic_fixture,run_growing_symbolic_fixture
    report=run_symbolic_fixture()
    assert report['coverage_symbolic_builds']==1 and report['coverage_symbolic_reuses']==7
    assert report['coverage_symbolic_apply_steps']<report['baseline_subtraction_checks']
    return {'proposal':6,'assertion_tested':'Eight 160-cell coverage queries reuse one exact completed-region DAG, avoiding repeated flat cube subtraction and preserving the same remaining box.',
            **report,'growing_ledger':run_growing_symbolic_fixture(),'non_claims':['Native public search frontiers still export flat cubes under the existing cap.','Logical-work counters represent different operations, not identical CPU cost.','No original Gold solution or predicted solve time.']}



def mixed_proofs():
    from test_constraint_mixed_proof import mixed_comparison
    report=mixed_comparison();report.pop('certificates')
    assert report['shared_checked_nodes']<report['cold_checked_nodes']
    return {'proposal':7,'assertion_tested':'Two finite Gold refutations reuse independently checked Hall, proper-loop and RUP deductions while preserving all independent edge/Gold enumeration results.',
            **report,'non_claims':['The finite case has two cells; no original-puzzle reduction or time-to-solution claim.','Exact-integer and affine leaves have separate semantic tampering and solution-preservation regressions.']}


def higher_relations():
    from test_constraint_learning_round25 import run_higher_fixture
    report=run_higher_fixture()
    assert report['candidate_total_queries']<report['baseline']['sat_assist_domain_calls']
    assert report['candidate_native_decisions']<=report['baseline_native_decisions']
    return {'proposal':8,'assertion_tested':'Complete joint projection tables reduce total native queries and decisions, including compilation and subsequent ordinary assistance, across 135 contexts after existing propagation fixpoints.',
            **report,'non_claims':['This is bounded live relation reuse, not full arbitrary-arity closure or disk proof transport.','Both sides enable witness pooling and support-cover batches.','No original Gold solution or universal runtime improvement.']}



def incremental_affine():
    from test_constraint_affine_round25 import compare_affine_updates
    report=compare_affine_updates()
    assert report['incremental']['stats']['affine_equations']<report['rebuild']['stats']['affine_equations']
    assert report['incremental']['stats']['affine_xors']<report['rebuild']['stats']['affine_xors']
    return {'proposal':9,'assertion_tested':'Thirteen exact nested domain contexts reuse ancestor affine equations, preserving consistency and independently enumerated completions with fewer equation insertions and Gaussian row work steps.',
            **report,'non_claims':['The discriminator isolates basis construction; candidate checking is covered separately by 180 differential contexts.','Counters exclude context validation and dictionary copies, and do not assert a wall-clock speed ratio.','No original-puzzle solution or search-node reduction established.']}


def conflict_pb():
    from test_constraint_learning_round25 import run_conflict_pb_fixture
    from constraint_conflict_pb import guided_cut,guided_cut_parent_priority,STAT_KEYS
    from constraint_pb import primitive_rows
    from test_constraint_sat_domains import fixture,admitted
    from test_constraint_dfs import brute
    problem,domains=fixture();answers=admitted(brute(problem,'edges'),domains)
    unsupported=[(c,v) for c,m in enumerate(domains) for v in range(12) if m>>v&1 and not any(a[c]==v for a in answers)]
    base=primitive_rows(problem,tuple(enumerate(domains)));comparisons=[]
    for kernel,pairs,new in ((guided_cut_parent_priority,256,64),(guided_cut,512,32),(guided_cut,2048,64),(guided_cut,8192,128)):
        stats=dict.fromkeys(STAT_KEYS,0);hits=[]
        for cell,code in unsupported:
            trial=domains.copy();trial[cell]=1<<code
            if kernel(base,trial,max_pairs=pairs,max_new=new,stats=stats):hits.append([cell,code])
        comparisons.append({'kernel':kernel.__name__,'max_pairs':pairs,'max_new':new,'hits':hits,'stats':stats})
    report=run_conflict_pb_fixture()
    assert report['guided']['conflict_pb_prunes']==2 and comparisons[2]['hits']==comparisons[3]['hits']
    return {'proposal':10,'assertion_tested':'Conflict-guided exact integer cuts remove two candidates surviving ordinary plus expanded static-PB fixpoints, preserving every independently enumerated edge and Gold completion.',
            **report,'configuration_discriminator':comparisons,
            'non_claims':['This performs additional arithmetic to obtain stronger deductions; neither total-work reduction nor speedup is claimed.',
                          'Configuration microcomparisons use the twelve independently known unsupported values; retained integrated run accounts for all thirty-three actual trials.',
                          'Bounded positive-combination/gcd cuts, not general cutting-planes conflict analysis or complete PB closure.']}


def sat_provenance():
    import pysolvers
    path=Path(pysolvers.__file__).resolve()
    return {'python_sat':importlib.metadata.version('python-sat'),'native_module':path.name,
            'native_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}



def integer_lattice():
    from test_constraint_lattice_round25 import lattice_comparison
    report=lattice_comparison()
    return {'proposal':11,'assertion_tested':'A full signed seam and inventory integer-lattice relaxation removes three choices beyond ordinary, affine-GF2 and static-PB fixpoints, preserving six edge and four Gold completions.',
            **report,'non_claims':['The three puzzle exclusions also follow from rational-span reasoning; no divisibility-only puzzle advantage was found in the additional 3000-context corpus.','A separate exact determinant-3 arithmetic fixture distinguishes integer membership from rational and GF(2) membership.','Bounded active equations/placement columns; caps are unknown, not exclusions.','No original Gold solution or universal time-to-solution advantage.']}


def hall_rematch():
    from test_constraint_learning_round25 import run_hall_rematch_fixture
    report=run_hall_rematch_fixture()
    assert report['rematch']['delta']['hall_scc_vertices']<report['baseline']['delta']['hall_scc_vertices']
    return {'proposal':12,'assertion_tested':'After a forced perfect-matching repair, unaffected residual components reuse their exact SCC partition; independent all-matching enumeration confirms identical supports.',
            **report,'non_claims':['Only SCC traversal decreases; matching-edge tests are unchanged.','No search-node or original-puzzle runtime gain established.']}


def reversible_gold_graph():
    from test_constraint_learning_round25 import run_gold_graph_fixture
    report=run_gold_graph_fixture()
    assert report['candidate']['gold_graph_seam_tests']<report['baseline_seam_tests']
    return {'proposal':13,'assertion_tested':'Sixteen exact related domain contexts retain identical strand and cyclic-block deductions with fewer repeated seam-support tests, preserving the planted full Gold witness.',**report}


def main():
    require_assertions()
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path);parser.add_argument('--candidate',type=int,choices=[1,2,3,4,6,7,8,9,10,11,12,13],default=1);parser.add_argument('--evidence-stage',choices=['development','frozen'],default='development');args=parser.parse_args()
    root=Path(__file__).resolve().parent
    paths=['constraint_sat_session.py','constraint_work.py','constraint_sat_assist.py','constraint_knowledge.py','search_problem.py','validator.py','test_constraint_learning_round25.py','benchmark_learning_round25.py']
    hashes=lambda:{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths}
    if args.candidate==2:paths+=['constraint_job_sat.py','constraint_dfs.py','constraint_parallel.py','constraint_pruning.py','test_constraint_job_sat.py','test_constraint_sat_domains.py','search_options.py']
    if args.candidate==4:
        paths+=['constraint_native_search.py','constraint_native_search.c','constraint_native_learning.py','constraint_lcg.py','test_constraint_native_reasons.py','test_constraint_native_learning.py','test_constraint_native_search.py','.native/search-build.json']
        metadata=json.loads((root/'.native/search-build.json').read_text(encoding='utf-8'))
        paths.append('.native/'+metadata['library'])
    if args.candidate==3:paths+=['constraint_proof.py','constraint_proof_cursor.py','constraint_mask_conflicts.py']
    if args.candidate==6:paths+=['constraint_symbolic_work.py']
    if args.candidate==7:paths+=['constraint_mixed_proof.py','constraint_proof.py','constraint_proof_cursor.py','constraint_mask_conflicts.py','constraint_pb.py','constraint_algebra.py','test_constraint_mixed_proof.py']
    if args.candidate==8:paths+=['constraint_higher_relations.py','constraint_dfs.py','constraint_job_sat.py','constraint_pruning.py','test_constraint_sat_domains.py','search_options.py']
    if args.candidate==9:paths+=['constraint_algebra.py','constraint_round24.py','search_options.py','test_constraint_affine_round25.py','test_constraint_dfs.py','test_constraint_sat_domains.py']
    if args.candidate==10:paths+=['constraint_conflict_pb.py','constraint_pb.py','constraint_round24.py','search_options.py','constraint_pruning.py','test_constraint_sat_domains.py']
    if args.candidate==11:paths+=['constraint_lattice.py','constraint_algebra.py','constraint_pb.py','constraint_round24.py','search_options.py','test_constraint_lattice_round25.py','test_constraint_sat_domains.py','test_constraint_dfs.py']
    if args.candidate==12:paths+=['constraint_incremental_hall.py','constraint_pruning.py','constraint_dfs.py','search_options.py','test_constraint_incremental_hall.py']
    if args.candidate==13:paths+=['constraint_gold_graph.py','constraint_advanced.py','constraint_pruning.py','constraint_dfs.py','search_options.py','search_fixtures.py','geometry.py','test_constraint_advanced.py','test_constraint_sat_domains.py']
    frozen_before=source_snapshot(root);runtime_before=runtime_snapshot();repository=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repository.get('commit') or repository.get('dirty') is not False):
        raise RuntimeError('Frozen learning evidence requires a clean, readable committed checkout.')
    before=hashes();result={1:core_lifting,2:shared_assistant,3:resumable_proof,4:native_explanations,6:symbolic_coverage,7:mixed_proofs,8:higher_relations,9:incremental_affine,10:conflict_pb,11:integer_lattice,12:hall_rematch,13:reversible_gold_graph}[args.candidate]();after=hashes();assert before==after
    frozen_after=source_snapshot(root);runtime_after=runtime_snapshot();repository_after=repository_snapshot(root)
    if args.evidence_stage=='frozen' and repository_after!=repository:
        raise RuntimeError('The committed checkout changed during frozen learning evaluation.')
    if frozen_before!=frozen_after or runtime_before!=runtime_after:
        raise RuntimeError('Sources or runtime changed during learning evaluation.')
    result.update(evidence_stage=args.evidence_stage,repository_after=repository_after,source_before=frozen_before,source_after=frozen_after,runtime_before=runtime_before,runtime_after=runtime_after,source_unchanged=frozen_before==frozen_after,runtime_unchanged=runtime_before==runtime_after,python=platform.python_version(),commit=repository.get('commit'),repository=repository,source_sha256=before,inputs_unchanged=before==after)
    if args.candidate in (1,2,8):result['sat_backend']=sat_provenance()
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'proposal':args.candidate,'inputs_unchanged':True,**{k:result[k] for k in ('total_queries','baseline_clause_visits','resumed_clause_visits','slices','baseline_subtraction_checks','coverage_symbolic_apply_steps','coverage_symbolic_nodes') if k in result}}))
if __name__=='__main__':main()
