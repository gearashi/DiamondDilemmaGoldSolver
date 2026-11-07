"""Independent finite oracles for the complete round26 regional workstreams."""
import itertools,json,random,tempfile,unittest
from unittest.mock import patch
from test_constraint_separator import ring,permutation_ring
from test_constraint_advanced import domains_of,bits
from test_constraint_dfs import brute
from constraint_regions import HierarchicalRegionPruner
from constraint_region_compact import CompactFactors,GoldCompactFactors,compile_compact
from constraint_region_compiler_round26 import RegionalCompilerSession,RegionalCompiler,add_dag_sat,add_dag_cp_sat


def finish(session,lease=31):
    while not session.complete and not session.blocked:session.advance(lease)
    return session.result()

class OrderedCompilerTests(unittest.TestCase):
    def test_exact_every_order_matches_independent_full_completions(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*2);domains=domains_of(problem)
        for target in ('gold','edges'):
            answers=brute(problem,target);owner=HierarchicalRegionPruner(problem,target)
            expected=tuple(bits(row[c] for row in answers) for c in range(problem.n))
            for order in itertools.permutations(range(problem.n)):
                result=finish(RegionalCompilerSession(owner,domains,order=order))
                self.assertEqual(result['rows'],len(answers));self.assertEqual(result['supports'],expected)
                factory=GoldCompactFactors if target=='gold' else CompactFactors
                old=compile_compact(factory(problem,domains,order=order))
                self.assertEqual((old['rows'],old['supports']),(len(answers),expected))

    def test_frontier_rank_is_existence_only_and_preserves_small_completions(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*3);owner=HierarchicalRegionPruner(problem,'gold')
        answers=brute(problem,'gold');rng=random.Random(260902)
        for _ in range(32):
            domains=[bits(code for code in row if rng.random()<.7) for row in problem.domains]
            expected=any(all(mask>>code&1 for mask,code in zip(domains,row)) for row in answers)
            result=finish(RegionalCompilerSession(owner,domains,existence_only=True))
            self.assertEqual(result['feasible'],expected);self.assertIsNone(result['rows']);self.assertIsNone(result['supports'])
            if result['witness'] is not None:self.assertTrue(problem.validate(result['witness'])['valid'])

    def test_completed_transitions_survive_slice_and_snapshot_without_replay(self):
        problem=ring(6);owner=HierarchicalRegionPruner(problem,'gold');domains=domains_of(problem)
        cold=RegionalCompilerSession(owner,domains);expected=finish(cold)
        warm=RegionalCompilerSession(owner,domains)
        while not warm.complete:
            warm.advance(7)
            warm=RegionalCompilerSession(owner,domains,order=warm.order,resume=json.loads(json.dumps(warm.snapshot())))
        self.assertEqual(warm.result(),expected)
        self.assertEqual(warm.stats['regional_compiler_transitions'],cold.stats['regional_compiler_transitions'])
        self.assertEqual(warm.stats['regional_compiler_transition_attempts'],cold.stats['regional_compiler_transition_attempts'])

    def test_stop_cap_scope_and_malformed_resume_never_publish_partial_supports(self):
        problem=ring(4);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        for boundary in range(1,25):
            session=RegionalCompilerSession(owner,domains);calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop')
            try:session.advance(20,check)
            except InterruptedError:pass
            result=finish(session)
            self.assertEqual(result['rows'],24)
        limited=RegionalCompilerSession(owner,domains,max_steps=1);limited.advance(100)
        self.assertTrue(limited.blocked);self.assertIsNone(limited.result())
        snapshot=limited.snapshot()
        with self.assertRaises(ValueError):RegionalCompilerSession(owner,domains,max_steps=2,resume=snapshot)
        snapshot['body']['steps']=False
        from constraint_region_compiler_round26 import _digest
        snapshot['sha256']=_digest(snapshot['body'])
        with self.assertRaises(ValueError):RegionalCompilerSession(owner,domains,max_steps=1,resume=snapshot)
        with self.assertRaises(ValueError):RegionalCompilerSession(owner,[True,*domains[1:]])

    def test_disk_session_reuses_completed_construction_and_binding(self):
        problem=ring(5);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        with tempfile.TemporaryDirectory() as directory:
            stage=RegionalCompiler(owner);stage.configure_cache_dir(directory)
            self.assertIsNone(stage.get(domains,lease=2))
            again=RegionalCompiler(owner);again.configure_cache_dir(directory)
            result=again.get(domains,lease=10000)
            self.assertIsNotNone(result);self.assertEqual(again.stats['regional_compiler_loaded'],1)
            self.assertGreater(again.stats['regional_compiler_saved'],0)

    def test_native_dag_encodings_equal_all_independent_answers(self):
        try:
            from ortools.sat.python import cp_model
            from pysat.solvers import Solver
        except ImportError:self.skipTest('Optional SAT/CP-SAT packages unavailable.')
        problem=ring(4);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        diagram=finish(RegionalCompilerSession(owner,domains));answers=set(brute(problem,'gold'))
        model=cp_model.CpModel();codes=[model.new_int_var_from_domain(cp_model.Domain.from_values(row),f'c{c}') for c,row in enumerate(problem.domains)]
        receipt=add_dag_cp_sat(diagram,model,codes);self.assertEqual(receipt['expanded_histories'],0)
        found=set()
        class Collector(cp_model.CpSolverSolutionCallback):
            def on_solution_callback(self):found.add(tuple(self.value(v) for v in codes))
        solver=cp_model.CpSolver();solver.parameters.enumerate_all_solutions=True
        solver.solve(model,Collector());self.assertEqual(found,answers)
        literals={(c,code):i+1 for i,(c,code) in enumerate((c,v) for c,row in enumerate(problem.domains) for v in row)}
        counter=[len(literals)]
        def fresh():counter[0]+=1;return counter[0]
        with Solver(name='g3') as sat:
            for c,row in enumerate(problem.domains):
                choices=[literals[c,code] for code in row];sat.add_clause(choices)
                for a,b in itertools.combinations(choices,2):sat.add_clause([-a,-b])
            add_dag_sat(diagram,sat,lambda c,v:literals[c,v],fresh)
            found=set()
            while sat.solve():
                positive=set(v for v in sat.get_model() if v>0)
                board=tuple(next(v for v in row if literals[c,v] in positive) for c,row in enumerate(problem.domains))
                found.add(board);sat.add_clause([-literals[c,v] for c,v in enumerate(board)])
            self.assertEqual(found,answers)


class ParametricTreeTests(unittest.TestCase):
    def test_outside_widening_and_fixed_chain_changes_reuse_one_envelope(self):
        from constraint_region_interfaces_round26 import ParametricEnvelope
        problem=permutation_ring([(1,0,2),(0,2,1)]*3);owner=HierarchicalRegionPruner(problem)
        root=domains_of(problem);cells=(0,1,2);envelope=ParametricEnvelope(owner,cells,root)
        while not envelope.advance(100):pass
        work=envelope.session.steps;answers=brute(problem,'gold')
        contexts=[root.copy()]
        for code in problem.domains[4]:
            child=root.copy();child[4]=1<<code;contexts.extend((child,root.copy()))
        for domains in contexts:
            view=envelope.view(domains)
            compatible=[row for row in answers if all(d>>v&1 for d,v in zip(domains,row))]
            self.assertTrue(all(all(mask>>row[cell]&1 for cell,mask in zip(cells,view['supports'])) for row in compatible))
        self.assertEqual(envelope.session.steps,work)
        self.assertGreater(envelope.stats['regional_envelope_reuses'],1)

    def test_recursive_conditioning_and_correlated_multiway_messages_are_exact(self):
        from constraint_region_tree_round26 import RecursiveRegionSession
        rng=random.Random(2610)
        for problem in (ring(5),permutation_ring([(1,0,2),(0,2,1)]*3)):
            for target in ('gold','edges'):
                owner=HierarchicalRegionPruner(problem,target);answers=brute(problem,target)
                for trial in range(5):
                    domains=domains_of(problem) if trial==0 else [bits(v for v in row if rng.random()<.8) for row in problem.domains]
                    expected={row for row in answers if all(d>>v&1 for d,v in zip(domains,row))}
                    session=RecursiveRegionSession(owner,domains,leaf_size=2)
                    while not session.complete and not session.blocked:session.advance(53)
                    result=session.result();self.assertIsNotNone(result)
                    self.assertEqual(result['rows'],len(expected))
                    self.assertEqual(result['supports'],tuple(bits(row[c] for row in expected) for c in range(problem.n)))
                    self.assertGreater(session.stats['recursive_region_redecompositions'],0)

    def test_recursive_snapshot_restores_partial_joins_without_completed_work_replay(self):
        from constraint_region_tree_round26 import RecursiveRegionSession
        problem=ring(5);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        cold=RecursiveRegionSession(owner,domains,leaf_size=2)
        while not cold.complete:cold.advance(500)
        warm=RecursiveRegionSession(owner,domains,leaf_size=2)
        while not warm.complete:
            warm.advance(17);snapshot=json.loads(json.dumps(warm.snapshot()))
            warm=RecursiveRegionSession(owner,domains,leaf_size=2,resume=snapshot)
        self.assertEqual(warm.result(),cold.result())
        self.assertEqual(warm.stats['recursive_region_join_attempts'],cold.stats['recursive_region_join_attempts'])
        self.assertEqual(warm.stats['recursive_region_leaf_transitions'],cold.stats['recursive_region_leaf_transitions'])

    def test_recursive_cancel_unknown_and_query_binding(self):
        from constraint_region_tree_round26 import RecursiveRegionSession
        problem=ring(5);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        session=RecursiveRegionSession(owner,domains,max_steps=5);session.advance(100)
        self.assertTrue(session.blocked);self.assertIsNone(session.result())
        with self.assertRaises(ValueError):RecursiveRegionSession(owner,domains,max_steps=6,resume=session.snapshot())
        for boundary in (1,5,17,39):
            session=RecursiveRegionSession(owner,domains,leaf_size=2);calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop')
            try:session.advance(100,check)
            except InterruptedError:pass
            while not session.complete:session.advance(100)
            self.assertEqual(session.result()['rows'],120)


class RegionalProofTests(unittest.TestCase):
    def test_pure_failure_core_reuses_sufficient_assumptions_on_smaller_residuals(self):
        from constraint_region_proof_round26 import build_core,verify_core,RegionalProofReuse
        problem=ring(5);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        domains[0]=domains[1]=1<<problem.domains[0][0]
        record=build_core(owner,(0,1),domains,generalize=8);self.assertIsNotNone(record)
        assumptions=verify_core(owner,record);self.assertEqual({c for c,d in assumptions},{0,1})
        stage=RegionalProofReuse(owner);self.assertTrue(stage.import_certificate(record))
        child=domains.copy();child[3]=1<<problem.domains[3][-1]
        self.assertTrue(stage.infeasible(child));self.assertEqual(stage.stats['regional_core_reuses'],1)
        widened=domains_of(problem)
        self.assertFalse(stage.infeasible(widened,max_active=0))
        exported=stage.export_certificates();exported[0]['assumptions'].clear()
        self.assertEqual(len(stage.export_certificates()[0]['assumptions']),2)

    def test_checked_contraction_keeps_every_fixed_strand_and_rejects_fake_core(self):
        from constraint_region_proof_round26 import build_core,verify_core
        from search_problem import SearchProblem
        from test_constraint_dfs import board
        from constraint_region_certificate import CertificateError
        problem=SearchProblem([{'id':i,'segments':[[[0,2],[1,10]],[[0,10],[1,2]]]} for i in range(5)],board(5,[(i,1,(i+1)%5,0) for i in range(5)]))
        owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        domains[3]=1<<9;domains[4]=1<<12
        record=build_core(owner,(0,1,2),domains,generalize=0);self.assertIsNotNone(record)
        self.assertEqual(len(record['fixed']),2);verify_core(owner,record)
        bad=json.loads(json.dumps(record));bad['fixed'].pop()
        from constraint_region_compiler_round26 import _digest
        bad['sha256']=_digest({k:v for k,v in bad.items() if k!='sha256'})
        with self.assertRaises(CertificateError):verify_core(owner,bad)
        good=ring(5);goodowner=HierarchicalRegionPruner(good)
        self.assertIsNone(build_core(goodowner,(0,1,2),[bits((0,3,6))]*3+[1<<9,1<<12],generalize=0))

    def test_corrupt_or_capped_proofs_never_enter_store(self):
        from constraint_region_proof_round26 import build_core,RegionalProofReuse
        problem=ring(4);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem);domains[0]=domains[1]=1
        self.assertIsNone(build_core(owner,(0,1),domains,max_steps=0))
        record=build_core(owner,(0,1),domains);store=RegionalProofReuse(owner)
        record['certificate']['states'][0]['edges']=[]
        self.assertFalse(store.import_certificate(record));self.assertFalse(store.records)


class ConnectivityObligationTests(unittest.TestCase):
    def test_integer_assignment_dual_matches_independent_injections(self):
        from constraint_connectivity_round26 import assignment_upper_bound,verify_assignment_bound
        rng=random.Random(261201)
        for _ in range(100):
            n=rng.randint(1,4);m=rng.randint(n,5)
            matrix=[[rng.randint(0,9) if rng.random()<.7 else None for j in range(m)] for i in range(n)]
            legal=[sum(matrix[i][p[i]] for i in range(n)) for p in itertools.permutations(range(m),n) if all(matrix[i][p[i]] is not None for i in range(n))]
            certificate=assignment_upper_bound(matrix);verify_assignment_bound(matrix,certificate)
            if legal:self.assertEqual(certificate['bound'],max(legal))
            else:self.assertLess(certificate['bound'],0)
            forged=dict(certificate);forged['bound']-=1
            with self.assertRaises(ValueError):verify_assignment_bound(matrix,forged)

    def test_joint_demands_keep_orientation_correlations(self):
        from constraint_connectivity_round26 import joint_demands
        # Every component can individually receive2, but the one selected
        # orientation cannot supply both. Componentwise maxima would be wrong.
        rows=[[(0,(2,0)),(0,(0,2))]]
        self.assertFalse(joint_demands(rows,(2,2)))
        self.assertTrue(joint_demands(rows+ [[(1,(2,0)),(1,(0,2))]],(2,2)))
        from constraint_separator import _Budget
        with self.assertRaises(_Budget):joint_demands(rows,(2,2),max_states=1)

    def test_all_strands_refines_false_positive_chain_linkage(self):
        from constraint_region_multichain import JointChainOracle
        from constraint_connectivity_round26 import ObligationChainOracle
        graph={0:((3,((0,0),)),),1:((2,((0,0),)),),2:((1,((0,0),)),),3:((0,((0,0),)),),4:((5,((0,0),)),),5:((4,((0,0),)),)}
        chains=((0,1),(2,3));obligations={(0,0):((1,2),(0,3),(4,5))}
        self.assertTrue(JointChainOracle(graph,chains).possible())
        oracle=ObligationChainOracle(graph,chains,obligations)
        self.assertFalse(oracle.possible());self.assertEqual(oracle.refinements,1)
        self.assertIsNone(ObligationChainOracle(graph,chains,obligations,max_steps=1).possible())

    def test_dynamic_component_blocks_match_independent_fresh_traversals(self):
        from constraint_connectivity_round26 import ReversibleBlockCertificates
        from constraint_advanced import cyclic_edge_blocks
        rng=random.Random(261205);helper=ReversibleBlockCertificates();edges={(0,1),(1,2),(0,2),(3,4),(4,5),(3,5)}
        contexts=[]
        for _ in range(60):
            pair=tuple(sorted(rng.sample(range(8),2)))
            if pair in edges:edges.remove(pair)
            else:edges.add(pair)
            contexts.extend((tuple(sorted(edges)),contexts[-1] if contexts else tuple(sorted(edges))))
        for edges in contexts:
            actual=helper.update(edges);raw=cyclic_edge_blocks(8,edges,lambda:None)
            for i,left in enumerate(edges):
                self.assertEqual(actual['blocks'][left] is None,raw[i]<0)
                for j,right in enumerate(edges):
                    if raw[i]>=0 and raw[j]>=0:self.assertEqual(actual['blocks'][left]==actual['blocks'][right],raw[i]==raw[j])
        self.assertGreater(helper.stats['connectivity_graph_restores'],0)

    def test_actual_stage_preserves_all_independent_gold_answers(self):
        from constraint_connectivity_round26 import ConnectivityObligations
        rng=random.Random(2612);problem=permutation_ring([(1,0,2),(0,2,1)]*2);owner=HierarchicalRegionPruner(problem);answers=brute(problem,'gold');stage=ConnectivityObligations(owner)
        for _ in range(25):
            domains=[bits(v for v in row if rng.random()<.8) for row in problem.domains]
            expected=[row for row in answers if all(d>>v&1 for d,v in zip(domains,row))]
            stage.prune(domains)
            self.assertTrue(all(all(d>>v&1 for d,v in zip(domains,row)) for row in expected))


class IntegerRegionalTests(unittest.TestCase):
    def test_three_region_integer_master_matches_independent_gold_feasibility(self):
        from constraint_region_integer_round26 import IntegerRegionalSession
        problem=permutation_ring([(1,0,2),(0,2,1)]*3);owner=HierarchicalRegionPruner(problem);answers=brute(problem,'gold');rng=random.Random(2613)
        for _ in range(18):
            domains=[bits(v for v in row if rng.random()<.75) for row in problem.domains]
            expected=any(all(d>>v&1 for d,v in zip(domains,row)) for row in answers)
            session=IntegerRegionalSession(owner,domains,region_size=2)
            while not session.complete and not session.blocked:session.advance(31)
            result=session.result();self.assertIsNotNone(result);self.assertEqual(result['feasible'],expected)
            self.assertGreaterEqual(len(session.parts),3)
            if result['witness'] is not None:self.assertTrue(problem.validate(result['witness'])['valid'])

    def test_weighted_inventory_cut_replays_every_omitted_column(self):
        from constraint_region_integer_round26 import IntegerRegionalSession,verify_weighted_cut
        problem=ring(6);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        # Three cells demand two physical pieces; keep every local region nonempty.
        for cell in (0,2,4):domains[cell]=bits((0,3))
        session=IntegerRegionalSession(owner,domains,region_size=2)
        with patch('constraint_region_branch_price_round26.phase_one',return_value={'status':'unavailable'}):
            while not session.complete and not session.blocked:session.advance(100)
        self.assertFalse(session.result()['feasible'])
        self.assertGreater(session.stats['integer_region_bound_replays'],0)
        for key,entry in session.bounds.items():
            query,level,members,signature=key
            for certificate in entry['certificates']:
                self.assertTrue(verify_weighted_cut(owner,session.diagrams,query,tuple(range(level,len(session.parts))),members,signature,certificate))
                bad=dict(certificate);bad['required']+=1
                with self.assertRaises(ValueError):verify_weighted_cut(owner,session.diagrams,query,tuple(range(level,len(session.parts))),members,signature,bad)

    def test_integer_master_snapshot_preserves_completed_prices_columns_and_branch_work(self):
        from constraint_region_integer_round26 import IntegerRegionalSession
        problem=ring(6);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        cold=IntegerRegionalSession(owner,domains,region_size=2)
        while not cold.complete:cold.advance(100)
        warm=IntegerRegionalSession(owner,domains,region_size=2)
        while not warm.complete:
            warm.advance(11);warm=IntegerRegionalSession(owner,domains,region_size=2,resume=json.loads(json.dumps(warm.snapshot())))
        self.assertEqual(warm.result(),cold.result())
        for counter in ('integer_region_price_queries','integer_region_price_edges','integer_region_branches','integer_region_compiler_transitions'):
            self.assertEqual(warm.stats[counter],cold.stats[counter])

    def test_candidate_fixings_and_caps_never_remove_an_independent_completion(self):
        from constraint_region_integer_round26 import IntegerRegionalSession
        problem=ring(5);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem);answers=brute(problem,'gold')
        session=IntegerRegionalSession(owner,domains,region_size=2)
        while session.compiling<len(session.parts):session.advance(1)
        removed=session.candidate_fixings(limit=100)
        self.assertTrue(all(all(not removed.get(c,0)>>v&1 for c,v in enumerate(row)) for row in answers))
        limited=IntegerRegionalSession(owner,domains,max_steps=2);limited.advance(100)
        self.assertTrue(limited.blocked);self.assertIsNone(limited.result())
        with self.assertRaises(ValueError):IntegerRegionalSession(owner,domains,max_steps=3,resume=limited.snapshot())


class RegionalIntegrationTests(unittest.TestCase):
    def test_overlap_messages_preserve_exact_join_and_descendant_cache_reuse(self):
        from constraint_region_interfaces_round26 import ParametricEnvelope,OverlapConsistency
        from constraint_region_tree_round26 import RecursiveRegions
        problem=ring(6);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        views=[]
        for cells in ((0,1,2,3),(2,3,4,5)):
            env=ParametricEnvelope(owner,cells,domains)
            while not env.advance(100):pass
            views.append(env.view(domains))
        helper=OverlapConsistency(owner);count,cells,supports=helper.supports(*views,domains)
        self.assertEqual(count,len(brute(problem,'gold')));self.assertEqual(supports,tuple(domains))
        self.assertEqual(helper.supports(*views,domains),(count,cells,supports));self.assertEqual(helper.hits,1)
        stage=RecursiveRegions(owner);current=domains.copy()
        for _ in range(50):
            stage.prune(current,lease=10000,leaf_size=2)
            if stage.solution is not None:break
        child=domains.copy();child[0]=1<<problem.domains[0][0]
        for _ in range(50):
            stage.prune(child,lease=10000,leaf_size=2)
            if stage.solution is not None:break
        self.assertGreater(stage.stats['recursive_region_cross_query_hits'],0)
        self.assertTrue(problem.validate(stage.solution)['valid'])

    def test_core_callback_exceptions_propagate_without_importing_authority(self):
        from constraint_region_proof_round26 import build_core,RegionalProofReuse
        problem=ring(4);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem);domains[0]=domains[1]=1
        certificate=build_core(owner,(0,1),domains)
        for error_type in (InterruptedError,OSError,ValueError):
            store=RegionalProofReuse(owner)
            def check():raise error_type('callback stop')
            with self.assertRaises(error_type):store.import_certificate(certificate,check)
            self.assertFalse(store.records)
            with self.assertRaises(error_type):build_core(owner,(0,1),domains,check)

    def test_native_compilation_stop_reuses_exact_pending_cursor_and_disk_resume(self):
        try:from ortools.sat.python import cp_model
        except ImportError:self.skipTest('CP-SAT unavailable.')
        from constraint_region_models import add_regional_dags,_NATIVE_REGION_SESSIONS
        problem=ring(5)
        def invoke(check=lambda:None,cache_dir=None,sink=None):
            model=cp_model.CpModel();codes=[model.new_int_var_from_domain(cp_model.Domain.from_values(row),f'c{c}') for c,row in enumerate(problem.domains)]
            return add_regional_dags(problem,problem.domains,backend='cp-sat',model=model,codes=codes,max_cells=5,check=check,cache_dir=cache_dir,stats_sink=sink)
        _NATIVE_REGION_SESSIONS.clear();cold=invoke();_NATIVE_REGION_SESSIONS.clear()
        with tempfile.TemporaryDirectory() as directory:
            calls=0;sink={}
            def check():
                nonlocal calls
                calls+=1
                if calls==40:raise InterruptedError('stop')
            with self.assertRaises(InterruptedError):invoke(check,directory,sink)
            self.assertGreater(sink['regional_dag_compile_transitions'],0)
            prefix=sink['regional_dag_compile_transitions'];_NATIVE_REGION_SESSIONS.clear()
            resumed=invoke(cache_dir=directory)
            self.assertEqual(resumed['regional_dag_cursor_loads'],1)
            self.assertEqual(prefix+resumed['regional_dag_compile_transitions'],cold['regional_dag_compile_transitions'])
            repeated=invoke(cache_dir=directory)
            self.assertEqual(repeated['regional_dag_compile_transitions'],0)
            self.assertGreater(repeated['regional_dag_encoder_replays'],0)
        _NATIVE_REGION_SESSIONS.clear()

    def test_every_grouped_stage_preserves_known_gold_on_actual160_geometry(self):
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        from constraint_regions_round26 import RegionalWorkstreams,KEYS
        data,mesh,witness=planted_gold(261600,size=4);problem=SearchProblem(data,mesh)
        domains=[1<<v for v in witness]
        for cell in range(2):domains[cell]|=sum(1<<(witness[cell]//3*3+r) for r in range(3))&bits(problem.domains[cell])
        self.assertEqual(problem.n,160);self.assertTrue(problem.validate(witness)['valid'])
        for key in KEYS:
            stage=RegionalWorkstreams(problem,'gold',{key:True});current=domains.copy();stage.prune(current)
            self.assertTrue(all(d>>v&1 for d,v in zip(current,witness)),key)
            if stage.solution is not None:self.assertTrue(problem.validate(stage.solution)['valid'])


class BranchPriceTests(unittest.TestCase):
    def test_native_column_generation_uses_checked_complete_negative_prices(self):
        from constraint_region_integer_round26 import IntegerRegionalSession
        try:from ortools.linear_solver import pywraplp
        except ImportError:self.skipTest('Optional LP proposer unavailable.')
        problem=ring(6);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        for cell in (0,2,4):domains[cell]=bits((0,3))
        session=IntegerRegionalSession(owner,domains,region_size=2)
        while not session.complete and not session.blocked:session.advance(100)
        self.assertFalse(session.result()['feasible']);self.assertGreater(session.stats['integer_bp_native_calls'],0)
        self.assertGreater(session.stats['integer_bp_generated_columns'],0)
        self.assertGreater(session.stats['integer_bp_exact_replays'],0)
        self.assertGreater(session.stats['integer_bp_certified_failures'],0)
        for node in session.lp_nodes.values():
            if node.status=='infeasible':self.assertTrue(node._verify_negative(node.dual,lambda:None))

    def test_conditional_gold_cut_valid_for_every_independent_solution_even_changed_ports(self):
        from constraint_region_branch_price_round26 import loop_cut,cut_coefficient,validate_cut
        problem=permutation_ring([(1,0,2),(0,2,1)]*2);owner=HierarchicalRegionPruner(problem)
        edges=brute(problem,'edges');gold=brute(problem,'gold')
        bad=next(row for row in edges if not problem.validate(row)['valid'])
        cut=loop_cut(owner,bad);self.assertIsNotNone(cut);validate_cut(problem,cut)
        self.assertLess(sum(cut_coefficient(owner,c,v,cut) for c,v in enumerate(bad)),-2)
        self.assertTrue(all(sum(cut_coefficient(owner,c,v,cut) for c,v in enumerate(row))>=-2 for row in gold))
        with self.assertRaises(ValueError):validate_cut(problem,(cut[0],cut[1],cut[1]))

    def test_completed_native_report_survives_stop_before_return_without_query_retry(self):
        from constraint_region_branch_price_round26 import BranchPriceNode
        from constraint_region_interfaces_round26 import ParametricEnvelope
        from fractions import Fraction
        problem=ring(4);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem);diagrams=[]
        for cells in ((0,1),(2,3)):
            env=ParametricEnvelope(owner,cells,domains)
            while not env.advance(100):pass
            diagrams.append(env.view(domains))
        node=BranchPriceNode(owner,diagrams,domains,0,(),None,{})
        def native(columns,rhs,check,time_ms,telemetry):
            telemetry.update(builds=1,calls=1,result={'status':'optimal','objective':1,'dual':tuple(Fraction(1) for _ in rhs),'primal':()})
            raise InterruptedError('after native completion')
        with patch('constraint_region_branch_price_round26.phase_one',side_effect=native):
            with self.assertRaises(InterruptedError):node.advance()
        self.assertEqual(node.phase,'pricing');self.assertFalse(node.pending_native)
        snapshot=node.snapshot();restored=BranchPriceNode(owner,diagrams,domains,0,(),None,{},state=snapshot)
        with patch('constraint_region_branch_price_round26.phase_one',side_effect=AssertionError('must not rerun completed native query')):
            restored.advance()
        self.assertEqual(restored.stats['integer_bp_native_calls'],1)
        self.assertEqual(restored.stats['integer_bp_native_retries'],0)
        self.assertEqual(restored.price_index,1)

    def test_lp_unknowns_remain_optional_and_exact_integer_fallback_still_finishes(self):
        from constraint_region_integer_round26 import IntegerRegionalSession
        problem=ring(5);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        with patch('constraint_region_branch_price_round26.phase_one',return_value={'status':'unknown'}):
            session=IntegerRegionalSession(owner,domains,region_size=2)
            while not session.complete and not session.blocked:session.advance(100)
        self.assertTrue(session.result()['feasible']);self.assertTrue(problem.validate(session.witness)['valid'])
        self.assertEqual(session.stats['integer_bp_certified_failures'],0)


class RegionalLifecycleClosureTests(unittest.TestCase):
    def test_joint_order_and_early_rank_reduce_completed_transition_work(self):
        from search_problem import SearchProblem
        from test_constraint_dfs import board
        rng=random.Random(260905);positions=(1,3,6,9,11)
        selected=rng.sample(list(itertools.permutations(range(5))),7)
        data=[{'id':i,'segments':[[[0,positions[j]],[1,positions[4-p[j]]]] for j in range(5)]} for i,p in enumerate(selected)]
        sequence=(0,3,6,2,5,1,4)
        problem=SearchProblem(data,board(7,[(sequence[i],1,sequence[(i+1)%7],0) for i in range(7)]));owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        cold=RegionalCompilerSession(owner,domains,order=tuple(range(7)));ordered=RegionalCompilerSession(owner,domains);ranked=RegionalCompilerSession(owner,domains,existence_only=True)
        results=[finish(session,512) for session in (cold,ordered,ranked)]
        # A single five-strand loop induces an even5-cycle. The fixed multiset
        # has odd permutation parity, independent of the order of its pieces.
        parity=sum(sum(p[i]>p[j] for i in range(5) for j in range(i+1,5)) for p in selected)%2
        self.assertEqual(parity,1);self.assertTrue(all(not result['feasible'] for result in results))
        self.assertLess(ordered.steps,cold.steps);self.assertLess(ranked.steps,ordered.steps)
        self.assertGreater(ranked.stats['regional_compiler_rank_entries'],0)
        self.assertLess(ranked.stats['regional_compiler_rank_kept'],ranked.stats['regional_compiler_rank_rows'])

    def test_small_core_drops_irrelevant_restricted_cells(self):
        from constraint_region_proof_round26 import build_core,STAT_KEYS
        problem=ring(5);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem);domains[0]=domains[1]=1;domains[2]=1<<3
        stats=dict.fromkeys(STAT_KEYS,0);record=build_core(owner,(0,1,2),domains,stats=stats)
        self.assertIsNotNone(record);self.assertEqual({c for c,d in record['assumptions']},{0,1})
        self.assertGreater(stats['regional_core_generalized_assumptions'],0)

    def test_all_live_caches_reject_source_mutation_before_query_reuse(self):
        from constraint_regions_round26 import RegionalWorkstreams,KEYS
        for flag in KEYS:
            problem=ring(4);stage=RegionalWorkstreams(problem,'gold',{flag:True});domains=domains_of(problem);stage.prune(domains)
            problem.tiles[0]['segments']=[[[0,2],[1,10]]]
            with self.assertRaises(ValueError):stage.prune(domains_of(problem))

    def test_integer_infeasibility_is_audited_by_independent_signature_closure(self):
        from search_problem import SearchProblem
        from test_constraint_dfs import board
        from constraint_region_integer_round26 import IntegerRegionalSession
        problem=SearchProblem([{'id':i,'segments':[[[0,2],[1,10]],[[0,10],[1,2]]]} for i in range(6)],board(6,[(i,1,(i+1)%6,0) for i in range(6)]))
        owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        # Bypass optional bound proposers to exercise complete integral closure,
        # while retaining exact branch-compatible pricing and Gold joins.
        with patch('constraint_region_branch_price_round26.phase_one',return_value={'status':'unavailable'}),patch.object(IntegerRegionalSession,'_cuts',return_value=False):
            session=IntegerRegionalSession(owner,domains,region_size=2)
            while not session.complete and not session.blocked:
                session.advance(19)
                session=IntegerRegionalSession(owner,domains,region_size=2,resume=json.loads(json.dumps(session.snapshot())))
        self.assertFalse(session.result()['feasible']);self.assertTrue(session.exhausted)
        self.assertTrue(session.verifier.complete);self.assertFalse(session.verifier.feasible)
        self.assertGreater(session.stats['integer_region_closure_checks'],0)
        self.assertFalse(brute(problem,'gold'))

    def test_strict_integer_and_complete_flags_reject_malformed_local_continuations(self):
        from constraint_region_tree_round26 import RecursiveRegionSession,_pack,_unpack
        from constraint_region_integer_round26 import IntegerRegionalSession
        from constraint_region_compiler_round26 import _digest
        problem=ring(4);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        for factory in (RecursiveRegionSession,IntegerRegionalSession):
            session=factory(owner,domains);session.advance(2);record=session.snapshot();body=_unpack(record['body']);body['steps']=True
            record={'body':_pack(body),'sha256':_digest(_pack(body))}
            with self.assertRaises(ValueError):factory(owner,domains,resume=record)
            valid=session.snapshot();body=_unpack(valid['body']);body['complete']=True
            forged={'body':_pack(body),'sha256':_digest(_pack(body))}
            with self.assertRaises(ValueError):factory(owner,domains,resume=forged)


class SymbolicRegionalMessageTests(unittest.TestCase):
    @staticmethod
    def fixture():
        from constraint_region_interfaces_round26 import ParametricEnvelope,signature_messages
        from constraint_region_certificate import _Envelope
        problem=ring(6);owner=_Envelope(HierarchicalRegionPruner(problem,'gold'));domains=domains_of(problem)
        tables=[]
        for cells in ((0,1,2),(3,4,5)):
            env=ParametricEnvelope(owner,cells,domains)
            while not env.advance(1000):pass
            tables.append(signature_messages(env.view(domains,fixed_chains=False))[0])
        return problem,owner,domains,tables
    def test_exact_correlated_join_and_durable_work(self):
        from constraint_region_messages_round26 import SymbolicMessageJoin
        from constraint_region_interfaces_round26 import join_messages
        from constraint_region_tree_round26 import _pack,_unpack
        problem,owner,domains,(left,right)=self.fixture()
        baseline,links,pairs=join_messages(owner,(0,1,2),left,(3,4,5),right,domains)
        cold=SymbolicMessageJoin(owner,(0,1,2),left,(3,4,5),right,domains)
        while not cold.advance():pass
        warm=SymbolicMessageJoin(owner,(0,1,2),left,(3,4,5),right,domains)
        steps=0
        while not warm.advance():
            steps+=1
            if steps%11==0:
                state=_unpack(json.loads(json.dumps(_pack(warm.snapshot()))))
                warm=SymbolicMessageJoin(owner,(0,1,2),left,(3,4,5),right,domains,state=state)
        exact=lambda rows:{sig:(entry['count'],entry['supports']) for sig,entry in rows.items()}
        self.assertEqual(exact(cold.result),exact(baseline));self.assertEqual(exact(warm.result),exact(baseline))
        self.assertEqual(warm.stats,cold.stats)
        self.assertLess(cold.stats['recursive_region_symbolic_geometry_tests'],pairs)
        self.assertEqual(sum(entry['count'] for entry in cold.result.values()),len(brute(problem,'gold')))
        self.assertGreater(cold.stats['recursive_region_zdd_operations'],0)
    def test_stop_and_caps_cannot_publish_partial_join(self):
        from constraint_region_messages_round26 import SymbolicMessageJoin
        from constraint_separator import _Budget
        _,owner,domains,(left,right)=self.fixture()
        baseline=SymbolicMessageJoin(owner,(0,1,2),left,(3,4,5),right,domains)
        while not baseline.advance():pass
        for boundary in (1,3,9,25,80):
            join=SymbolicMessageJoin(owner,(0,1,2),left,(3,4,5),right,domains);calls=0
            def stop():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('test boundary')
            try:
                while not join.advance(stop):pass
            except InterruptedError:pass
            join=SymbolicMessageJoin(owner,(0,1,2),left,(3,4,5),right,domains,state=join.snapshot())
            while not join.advance():pass
            self.assertEqual(join.result,baseline.result)
        limited=SymbolicMessageJoin(owner,(0,1,2),left,(3,4,5),right,domains,max_nodes=1)
        with self.assertRaises(_Budget):
            while not limited.advance():pass
        self.assertFalse(limited.complete)
        bad=baseline.snapshot();bad['side']=1
        with self.assertRaises(ValueError):SymbolicMessageJoin(owner,(0,1,2),left,(3,4,5),right,domains,state=bad)

class RegionalAccountingTests(unittest.TestCase):
    def test_partial_encoding_accounts_completed_native_insertions(self):
        from types import SimpleNamespace
        problem=ring(3);owner=HierarchicalRegionPruner(problem);diagram=finish(RegionalCompilerSession(owner,domains_of(problem)))
        clauses=[];count=[0];work={};poll=[0]
        def fresh():count[0]+=1;return count[0]
        def stop():
            poll[0]+=1
            if poll[0]==4:raise InterruptedError('during native encoding')
        with self.assertRaises(InterruptedError):add_dag_sat(diagram,SimpleNamespace(add_clause=clauses.append),lambda c,v:100+c*10+v,fresh,check=stop,work=work)
        self.assertEqual(work['arcs'],3);self.assertEqual(work['constraints'],len(clauses));self.assertGreater(len(clauses),0)
    def test_disk_reload_counts_fresh_setup_not_historical_transitions(self):
        problem=ring(4);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        with tempfile.TemporaryDirectory() as directory:
            first=RegionalCompiler(owner);first.configure_cache_dir(directory);first.get(domains,lease=10000)
            second=RegionalCompiler(owner);second.configure_cache_dir(directory);second.get(domains,lease=10000)
            self.assertEqual(second.stats['regional_compiler_transitions'],0)
            self.assertEqual(second.stats['regional_compiler_model_builds'],1)
            self.assertGreater(second.stats['regional_compiler_model_placements'],0)


class RegionalSourceBindingTests(unittest.TestCase):
    def test_transitive_algorithm_bytes_are_bound_selectively(self):
        from pathlib import Path
        from constraint_region_compiler_round26 import _source
        from hashlib import sha256
        baseline=dict(_source());original=Path.read_bytes
        names=('constraint_region_symbolic.py','constraint_region_certificate.py','constraint_separator.py','constraint_pruning.py','constraint_region_integer_round26.py')
        for name in names:
            self.assertIn(name,baseline)
            def changed(path):
                raw=original(path)
                return raw+b'\n# Test source identity change\n' if path.name==name else raw
            with patch.object(Path,'read_bytes',changed):
                actual=dict(_source())
            self.assertEqual(set(actual),set(baseline))
            self.assertNotEqual(actual[name],baseline[name])
            self.assertTrue(all(actual[key]==value for key,value in baseline.items() if key!=name))
