"""Source-bound completed-work comparisons for CPU learning families F02-F07.

Each report records logical work, additional checking/setup, and explicit bounds.
It does not rank capped searches by their smaller node counts or claim an
original-puzzle solution. Run families serially from a frozen checkout.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
from benchmark_provenance import source_snapshot,runtime_snapshot,repository_snapshot,require_assertions


def f02():
    import random,tempfile
    from constraint_propagation_cursor_round26 import BasicCursor
    from constraint_cpu_archive_round26 import CPUReasoningArchive
    from constraint_lookahead import CompletedProbeCache
    from constraint_dfs import _build_support_tables,_ArcFilter
    from test_constraint_sat_domains import fixture,admitted
    from test_constraint_dfs import brute
    problem,_=fixture();root=[sum(1<<c for c in row) for row in problem.domains]
    answers=brute(problem,'edges');tables=_build_support_tables(problem,root,'edges',lambda:None)
    rng=random.Random(260202);contexts=[];cases=[]
    for index in range(12):
        witness=rng.choice(answers);initial=[(1<<witness[c])|rng.randrange(4096) for c in range(4)];contexts.append(initial)
        uninterrupted=BasicCursor(initial,tables.arcs,tables.incoming,tables.tile_bits);expected_work={}
        uninterrupted.advance(_ArcFilter(tables),stats=expected_work)
        prefix=BasicCursor(initial,tables.arcs,tables.incoming,tables.tile_bits);prefix_work={};calls=0
        def stop():
            nonlocal calls
            calls+=1
            if calls==11:raise InterruptedError('finite checkpoint')
        try:prefix.advance(_ArcFilter(tables),stop,stats=prefix_work)
        except InterruptedError:pass
        cold=BasicCursor(initial,tables.arcs,tables.incoming,tables.tile_bits);cold_work=dict(prefix_work)
        cold.advance(_ArcFilter(tables),stats=cold_work)
        payload=json.loads(json.dumps(prefix.snapshot()));resumed=BasicCursor.restore(payload,tables.arcs,tables.incoming,tables.tile_bits)
        resumed_work=dict(prefix_work);resumed.advance(_ArcFilter(tables),stats=resumed_work)
        assert cold.domains==resumed.domains==uninterrupted.domains
        assert resumed_work==expected_work
        assert admitted(answers,initial)==admitted(answers,resumed.domains)
        cases.append({'input':initial,'output':resumed.domains,'cold_restart_work_including_prefix':cold_work,
                      'durable_work_including_prefix':resumed_work,'snapshot_bytes':len(json.dumps(payload).encode())})
    # Actual archived completed singleton work survives a new helper owner.
    probe_calls=0;cache=CompletedProbeCache();context=(problem.fingerprint,'edges',tuple(root))
    trial=[1,4095,4095,4095]
    def oracle(values,cell):
        nonlocal probe_calls
        probe_calls+=1;rows=admitted(answers,values)
        if not rows:return False
        values[:]=[sum({1<<row[c] for row in rows}) for c in range(4)];return True
    a=trial.copy();cache.run(context,a,0,oracle,lambda:None,{})
    with tempfile.TemporaryDirectory() as directory:
        archive=CPUReasoningArchive(directory,problem.fingerprint,'edges',tuple(root));archive.save_probes(cache)
        restored=archive.load_probes();assert restored is not None;b=trial.copy();restored.run(context,b,0,oracle,lambda:None,{})
        assert a==b and probe_calls==1
        disk={'independent_oracle_calls':probe_calls,'restores':1,'archive_stats':dict(archive.local.stats)}
    return {'fixture':'12 seeded four-cell valid-completion contexts and one archived exact singleton query','seed':260202,
      'cases':cases,'completed_probe':disk,
      'limitations':['The native basic-propagation call is an opaque retryable unit; its internal instruction cursor is not serialized.',
        'Every restored cursor is trusted local continuation bound to source/root/options, not imported proof authority.',
        'Snapshot bytes and setup/restoration costs are separate from propagation work.']}


def f03():
    import multiprocessing
    from constraint_sat_service_round26 import SATServiceOwner
    from constraint_sat_assist import SatAssist
    from test_constraint_sat_service_round26 import query_child
    from search_problem import SearchProblem
    from test_constraint_dfs import simple_tiles,double_face,brute
    problem=SearchProblem(simple_tiles(),double_face());root=(63,63);answers=brute(problem,'edges')
    assert not any(row[0]==row[1]==0 for row in answers)
    baseline={};builds=0
    for _ in range(16):
        helper=SatAssist(problem,root,'edges')
        try:
            assert helper.prepare(seconds=2);builds+=1
            assert helper.query([(0,0),(1,0)],seconds=2)==((0,0),(1,0))
            for k,v in helper.stats.items():baseline[k]=baseline.get(k,0)+v
        finally:helper.close()
    context=multiprocessing.get_context('spawn')
    with SATServiceOwner(problem,'edges') as owner:
        first=owner.client();assert first.prepare(seconds=2);first.close()
        queue=context.Queue();workers=[context.Process(target=query_child,args=(owner.client(),queue)) for _ in range(2)]
        try:
            for process in workers:process.start()
            results=[queue.get(timeout=30) for _ in workers]
            assert all(status=='unsat' and rows==[((0,0),(1,0))]*8 for rows,status in results),results
            for process in workers:process.join(timeout=20)
            assert all(not process.is_alive() and process.exitcode==0 for process in workers)
            counts=owner.counters()[0]
            assert counts['service']['sat_service_models']==1 and counts['service']['sat_service_query_calls']==1
            assert counts['service']['sat_service_cache_hits']==15
        finally:
            for process in workers:
                if process.pid is not None:
                    if process.is_alive():process.terminate()
                    process.join(timeout=10)
            queue.close();queue.join_thread()
    return {'fixture':'16 identical independently impossible original-root placement queries from two spawned worker processes',
      'cold_models':builds,'cold_sat_work':baseline,'shared_service':counts,
      'limitations':['This repeated-query fixture measures cross-worker deduplication, not unrelated-query throughput.',
        'Native solver stacks are not serialized; restart reconstructs the model and independently reloads semantic support boards.',
        'One to eight affinity owners are supported; this finite discriminator uses one.']}


def f04():
    from test_constraint_proof_indexed_round26 import compare_indexed_checking,complete
    from constraint_proof_indexed_round26 import IndexedRUPChecker
    from constraint_proof_reuse_round26 import clear,compact
    from constraint_proof import RUPChecker
    comparison=compare_indexed_checking();base=[(1,),(-1,)];proof=['2 -2 0']*64+['0'];clear()
    first=IndexedRUPChecker(base,proof,reuse=True);assert complete(first) is True
    second=IndexedRUPChecker(base,proof,reuse=True);assert complete(second) is True
    assert second.stats['proof_shared_steps']==65 and second.stats['proof_clause_visits']==0
    base2=[(1,2),(1,-2),(-1,2),(-1,-2)];long=['3 -3 0']*20+['1 0','-1 0','0']+['4 -4 0']*10
    stats={};small=compact(base2,long,stats=stats);assert small is not None and RUPChecker(base2).verify(small)
    clear()
    return {'indexed_checking':comparison,'shared_prefix':{'first':first.stats,'second':second.stats},
      'compaction':{'input_lines':len(long),'output_lines':len(small),'work':stats},
      'limitations':['Prefix reuse always rebuilds its clause index and is bound to exact assumptions/source/proof bytes.',
        'Compaction performs two independent checks up front; fewer exported lines is not a one-off runtime saving.',
        'Persistent theory and isolated native production have separate benchmark_theory_round26.py and benchmark_isolated_proof_round26.py reports.',
        'Regional and GPU deduction transport are checked separately; unsupported or capped replay never proves a conflict.']}


def f05():
    from test_constraint_sat_guards_round26 import compare_guard_reuse
    from benchmark_sat_simplify_round26 import compare
    guards=compare_guard_reuse()
    assert all(row['status']=='infeasible' and row['remaining_boxes']==0 for row in guards.values()),'Incomplete guard comparison.'
    return {'guards':guards,'simplification':compare(),
      'limitations':['All placement answers are compared against independent finite enumeration.',
        'Simplification is not uniformly better: native decisions can increase even when propagation decreases.',
        'Windows native proof logging stays disabled; the isolated Linux/WSL producer is tested separately.']}


def f06():
    import tempfile
    from test_constraint_relational_round26 import native_semijoin_fixture
    from test_constraint_separator import ring
    from test_constraint_dfs import brute
    from constraint_higher_relations import ProjectedRelations
    from constraint_sat_assist import SatAssist
    result=native_semijoin_fixture();problem=ring(4);root=tuple(sum(1<<c for c in row) for row in problem.domains)
    solutions=brute(problem,'edges');objects=[]
    try:
        with tempfile.TemporaryDirectory() as directory:
            helper=SatAssist(problem,root,'edges');objects.append(helper)
            first=ProjectedRelations(problem,root,'edges',certificates=True,guided=True);objects.append(first);first.configure_cache_dir(directory)
            current=list(root);first.prune(current,helper,seconds=2,max_calls=3,projection_cells=(0,1,2),max_models=64)
            assert first.stats['higher_models']==3 and not first.tables;first.flush()
            nexthelper=SatAssist(problem,root,'edges');objects.append(nexthelper)
            second=ProjectedRelations(problem,root,'edges',certificates=True,guided=True);objects.append(second);second.configure_cache_dir(directory)
            current=list(root)
            for _ in range(4):
                second.prune(current,nexthelper,seconds=2,max_calls=100,projection_cells=(0,1,2),max_models=64)
                if second.tables:break
            assert second.tables,'Incomplete relation enumeration after four bounded leases.'
            key,rows=next(iter(second.tables.items()));expected={tuple(board[c] for c in key[1]) for board in solutions}
            assert set(rows)==expected and first.stats['higher_models']+second.stats['higher_models']==len(rows)
            second.certificate_attempted.clear();assert second.persist_complete(seconds=2),'Incomplete completeness certificate.'
            third=ProjectedRelations(problem,root,'edges',certificates=True,guided=True);objects.append(third);third.configure_cache_dir(directory)
            assert third.tables==second.tables and third.stats['higher_queries']==0
            persisted={'complete_rows':len(rows),'prefix':dict(first.stats),'resumed':dict(second.stats),'imported':dict(third.stats),
              'prefix_certificate_work':dict(first.certificates.stats),'resumed_certificate_work':dict(second.certificates.stats),
              'independent_import_work':dict(third.certificates.stats),'baseline_cold_restart_models':3+len(rows)}
    finally:
        for obj in reversed(objects):obj.close()
    return {'semijoin':result,'durable_projection':persisted,
      'limitations':['Complete Gold tables are allowed covers/overapproximations if rows survive later Gold cuts, never falsely labelled exact final-theory projections.',
        'Only complete checked covers may prune by missing rows. Partial consequences use separate ordinary support queries.',
        'Core/missing-support/spurious-tuple selection is guidance only; no universal benefit has been established for that heuristic.',
        'Certificate production and independent import checking are additional costs, reported separately.']}


def f07():
    import random
    from test_constraint_sat_domains import fixture,admitted,ordinary_fixpoint
    from test_constraint_dfs import brute
    from constraint_lattice import IntegerLatticePruner
    from constraint_lattice_round26 import ContributionLatticePruner
    from constraint_arithmetic_round26 import IndexedPBPropagation,ArithmeticPBLearner
    from constraint_pb import primitive_rows,derive_rows,filter_rows,PBLearner
    from constraint_algebra import AffinePruner
    from constraint_deduction_chain_round26 import verify_chain
    problem,_=fixture();root=[4095]*4;answers=brute(problem,'edges');rng=random.Random(260703);contexts=[]
    for i in range(24):
        ds=root.copy();ds[i%4]=rng.randrange(1,4096);contexts.append(ds)
    tree=ContributionLatticePruner(problem);cold={};outcomes=[];verified=0
    for initial in contexts:
        original=IntegerLatticePruner(problem);a=initial.copy();b=initial.copy()
        original.prune(a,max_candidates=1000);tree.prune(b,max_candidates=1000)
        assert a==b and admitted(answers,initial)==admitted(answers,b)
        for key,value in original.stats.items():cold[key]=cold.get(key,0)+value
        for certificate in tree.drain_deduction_chains(target='edges',limit=128):verify_chain(problem,certificate,target='edges');verified+=1
        outcomes.append(b)
    base=primitive_rows(problem,tuple(enumerate(root)));rows=derive_rows(base)[0];queue=IndexedPBPropagation(rows)
    queue_contexts=[];scan_rows=0;queue_outcomes=[];rng=random.Random(260707)
    for group in range(6):
        current=root.copy()
        for step in range(4):
            if step:current[(group+step-1)%4]&=~(1<<rng.randrange(12))
            queue_contexts.append(current.copy());a=current.copy();b=current.copy()
            while True:
                scan_rows+=len(rows);changed=filter_rows(a,rows,max_passes=1)
                if not changed:break
            queue.filter(b);assert a==b and admitted(answers,current)==admitted(answers,b);queue_outcomes.append(b)
    # Assess the rule extension against the existing ordinary/affine/static-PB
    # fixpoint, not against an intentionally weaker row-only baseline.
    rng=random.Random(260706);extra=[];discovery={};completed=0
    for index in range(160):
        initial=[sum(1<<c for c in row if rng.random()<.7) for row in problem.domains]
        if not all(initial):continue
        current=initial.copy();affine=AffinePruner(problem);pb=PBLearner(problem)
        for _ in range(64):
            before=current.copy();current=ordinary_fixpoint(problem,current)
            if all(current):affine.prune(current)
            if all(current):pb.prune(current,max_values=12)
            if current==before or not all(current):break
        else:raise RuntimeError('Existing arithmetic baseline fixpoint incomplete.')
        after=current.copy();advanced=ArithmeticPBLearner(problem)
        if all(after):advanced.prune(after,max_values=12)
        assert admitted(answers,current)==admitted(answers,after)
        completed+=1
        if after!=current:extra.append({'index':index,'before':current,'after':after})
        for key,value in advanced.stats.items():discovery[key]=discovery.get(key,0)+value
    return {'lattice':{'contexts':contexts,'outcomes':outcomes,'cold':cold,'tree':dict(tree.stats),'independent_chain_checks':verified},
      'pb_queue':{'rows':len(rows),'contexts':queue_contexts,'outcomes':queue_outcomes,'cold_row_visits':scan_rows,'retained':dict(queue.stats)},
      'rule_discovery':{'seed':260706,'completed_contexts':completed,'additional_deductions':extra,'work':discovery},
      'limitations':['Integer operations, basis builds, source vectors and separator checking are distinct work units.',
        'Additional PB division/cover discovery is inactive in production unless a completed stronger discriminator establishes benefit.',
        'The retained row queue changes reuse on the same existing rows; no extra pruning is claimed for its neutral fixture.',
        'No original-puzzle solve or wall-time dominance claim.']}


RUNNERS={'F02':f02,'F03':f03,'F04':f04,'F05':f05,'F06':f06,'F07':f07}


def main(argv=None):
    require_assertions();parser=argparse.ArgumentParser()
    parser.add_argument('--family',choices=tuple(RUNNERS));parser.add_argument('--output',required=True)
    parser.add_argument('--evidence-stage',choices=('development','frozen'),default='development');args=parser.parse_args(argv)
    root=Path(__file__).resolve().parent;output=Path(args.output).resolve()
    if output.suffix.lower()!='.json' or output.exists() or not output.parent.is_dir():raise ValueError('A fresh JSON output in an existing directory is required.')
    before=source_snapshot(root);runtime_before=runtime_snapshot();repo_before=repository_snapshot(root)
    if args.evidence_stage=='frozen' and (not repo_before.get('available') or repo_before.get('dirty') or not repo_before.get('commit')):raise RuntimeError('Frozen evidence requires a readable clean Git commit.')
    results={family:run() for family,run in RUNNERS.items() if args.family is None or args.family==family}
    after=source_snapshot(root);runtime_after=runtime_snapshot();repo_after=repository_snapshot(root)
    if before!=after or runtime_before!=runtime_after:raise RuntimeError('Source/runtime changed during learning evidence; discard the run.')
    if repo_before!=repo_after:raise RuntimeError('Repository identity changed during learning evidence; discard the run.')
    report={'schema':'diamond-learning-round26-evidence-v1','status':'passed','evidence_stage':args.evidence_stage,'families':results,
      'interpretation':'Completed finite logical-work evidence; caps/unknown are not completed negative results and no whole-puzzle timing claim is made.',
      'provenance':{'sources_before':before,'sources_after':after,'runtime_before':runtime_before,'runtime_after':runtime_after,'repository_before':repo_before,'repository_after':repo_after}}
    with output.open('x',encoding='utf-8') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps({'status':'passed','families':tuple(results),'evidence_stage':args.evidence_stage,'source_runtime_stable':True}))
    return report


if __name__=='__main__':main()
