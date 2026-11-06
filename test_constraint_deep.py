"""Independent finite checks of R23 deep propagation, proofs and traversal."""
import copy
import random
import unittest
from itertools import product,permutations
from constraint_deep import (hall_certificate,HallCertificateCache,minimize_failure_scope,
    DeepPruner,orientation_equivalences,rotate_gold_clause,CertificateStore)
from constraint_policies import ImpactPolicy,worker_options,reorder_frontier,PieceSymmetry
from constraint_pruning import values
from test_constraint_advanced import bits,domains_of
from test_constraint_dfs import (brute,board,double_face,simple_tiles,multiple_loops,tetrahedron_tiles)
from search_fixtures import planted_gold
from search_problem import SearchProblem
from geometry import build_board


def surviving(solutions,domains):
    return [s for s in solutions if all(d&(1<<c) for d,c in zip(domains,s))]


def injective_assignments(rows):
    return [row for row in product(*(tuple(values(d)) for d in rows)) if len(set(row))==len(rows)]


class HallCertificateTests(unittest.TestCase):
    def test_all_small_matching_graphs_have_independently_valid_witnesses(self):
        count=0
        for n in range(1,4):
            for rows in product(range(1<<n),repeat=n):
                expected=injective_assignments(rows);witness=hall_certificate(rows);count+=1
                self.assertEqual(witness is None,bool(expected))
                if witness is not None:
                    cells,union=witness
                    actual=0
                    for c in cells:actual|=rows[c]
                    self.assertEqual(actual,union);self.assertLess(union.bit_count(),len(cells))
        self.assertEqual(count,530)

    def test_cached_shortage_applies_only_under_its_tile_bound(self):
        cache=HallCertificateCache();self.assertIsNotNone(cache.remember((3,3,3,8)))
        self.assertTrue(cache.impossible((1,2,3,16)))
        self.assertFalse(cache.impossible((1,2,4,8)))
        self.assertFalse(cache.impossible((1,2)))
        self.assertEqual(cache.hits,1)

    def test_scope_minimization_requires_completed_failure_after_releasing_cells(self):
        root=[7,7,7,15];narrow=[3,3,3,8]
        oracle=lambda row:False if not injective_assignments(row) else True
        cells,tests=minimize_failure_scope(narrow,root,oracle,max_tests=8)
        self.assertEqual(cells,(0,1,2));self.assertEqual(tests,4)
        released=root.copy()
        for cell in cells:released[cell]=narrow[cell]
        self.assertFalse(injective_assignments(released))
        cells,_=minimize_failure_scope(narrow,root,lambda row:False if row==narrow else None)
        self.assertEqual(cells,(0,1,2,3))


class DeepPropagationTests(unittest.TestCase):
    def test_path_consistency_preserves_all_enumerated_completions_and_cache_scopes(self):
        data,mesh=tetrahedron_tiles();problem=SearchProblem(data,mesh)
        solutions=brute(problem,'edges');rng=random.Random(230013);helper=DeepPruner(problem)
        for _ in range(100):
            original=[bits(c for c in row if rng.random()<.65) for row in problem.domains]
            if not all(original):continue
            expected=surviving(solutions,original);actual=original.copy()
            helper.paths(actual,max_values=12)
            self.assertEqual(surviving(expected,actual),expected)
            repeated=original.copy();helper.paths(repeated,max_values=12)
            self.assertEqual(repeated,actual)
        self.assertGreater(helper.stats['path_pairs_removed'],0)

    def test_path_cancellation_never_applies_partial_support_unions(self):
        data,mesh=tetrahedron_tiles();problem=SearchProblem(data,mesh)
        original=[bits(range(8))]*4
        for limit in (1,3,10,30):
            helper=DeepPruner(problem);actual=original.copy();calls=0
            def check():
                nonlocal calls
                calls+=1
                if calls==limit:raise InterruptedError
            with self.assertRaises(InterruptedError):helper.paths(actual,check,max_values=12)
            self.assertEqual(actual,original);self.assertFalse(helper.cache)

    def test_conservation_preserves_all_small_edge_solutions(self):
        data,mesh=tetrahedron_tiles();problem=SearchProblem(data,mesh)
        solutions=brute(problem,'edges');rng=random.Random(230048)
        for _ in range(100):
            original=[bits(c for c in row if rng.random()<.7) for row in problem.domains]
            if not all(original):continue
            expected=surviving(solutions,original);actual=original.copy()
            DeepPruner(problem).conservation(actual)
            self.assertEqual(surviving(expected,actual),expected)

    def test_varying_segment_totals_preserve_independent_planted_completions(self):
        rng=random.Random(230481)
        for seed in (230481,230482,230483):
            data,mesh,reference=planted_gold(seed,1,extra_edges=1)
            problem=SearchProblem(data,mesh);helper=DeepPruner(problem)
            self.assertGreater(len(set(problem._segment_counts)),1)
            for _ in range(8):
                original=[bits([code,rng.randrange(30)]) for code in reference]
                solutions=[]
                for row in product(*(tuple(values(d)) for d in original)):
                    if len({c//3 for c in row})<10:continue
                    report=problem.validate(row)
                    if report['matched_edges']==len(problem.edges):solutions.append(row)
                self.assertIn(tuple(reference),solutions)
                actual=original.copy();helper.conservation(actual)
                self.assertEqual(surviving(solutions,actual),solutions)
            self.assertGreater(helper.stats['conservation_prunes'],0)

    def test_open_board_does_not_assume_closed_seam_conservation(self):
        from test_constraint_separator import ring
        problem=ring(4);helper=DeepPruner(problem);original=domains_of(problem);actual=original.copy()
        self.assertFalse(helper.conservation(actual));self.assertEqual(actual,original)


class PortableCertificateTests(unittest.TestCase):
    def test_hall_certificates_rechecked_on_import_and_scoped_on_reuse(self):
        data,mesh=tetrahedron_tiles();problem=SearchProblem(data,mesh)
        domains=[bits([0,3])]*3+[bits([6,9])]
        store=CertificateStore(problem)
        self.assertTrue(store.capture(domains));snapshot=store.snapshot()
        other=CertificateStore(problem);self.assertEqual(other.load_snapshot(snapshot),1)
        actual=domains.copy();self.assertTrue(other.prune(actual));self.assertIn(0,actual)
        self.assertFalse(other.prune(domains_of(problem)))
        damaged=copy.deepcopy(snapshot);damaged['records'][0]['cells']=[0]
        empty=CertificateStore(problem);self.assertEqual(empty.load_snapshot(damaged),0)
        damaged=copy.deepcopy(snapshot);damaged['records'][0]['cells']=[[0]]
        self.assertEqual(empty.load_snapshot(damaged),0)
        for field,value in (('target','edges'),('fingerprint','wrong'),('version',2)):
            changed=copy.deepcopy(snapshot);changed[field]=value
            with self.assertRaises(ValueError):empty.load_snapshot(changed)

    def test_proper_loop_certificate_is_gold_only_and_independently_rechecked(self):
        mesh=board(4,[(0,s,1,s) for s in range(3)]+[(2,s,3,s) for s in range(3)])
        data=[{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(4)]
        problem=SearchProblem(data,mesh);domains=domains_of(problem)
        domains[0]=1;domains[1]=1<<3
        store=CertificateStore(problem);self.assertTrue(store.capture(domains))
        self.assertEqual(store.records[0]['kind'],'loop')
        record=store.records[0]
        self.assertFalse(CertificateStore(problem,'edges').offer(record))
        self.assertFalse(CertificateStore(problem).offer({**record,'cells':[0]}))
        self.assertTrue(store.prune(domains.copy()))

    def test_rotated_gold_clauses_are_all_semantically_revalidated(self):
        mesh=build_board(1)
        data=[{'id':i,'segments':[[[0,2],[0,10]]]} for i in range(10)]
        problem=SearchProblem(data,mesh)
        a,sa,b,sb=problem.edges[0];placements=((a,3*a+sa),(b,3*b+sb))
        clauses=rotate_gold_clause(problem,placements)
        self.assertGreater(len(clauses),1)
        for clause in clauses:
            partial=[-1]*10
            for cell,code in clause:partial[cell]=code
            self.assertTrue(problem.partial_loop_impossible(partial))
        self.assertEqual(rotate_gold_clause(problem,(placements[0],)),())


class OrderingAndEquivalenceTests(unittest.TestCase):
    def test_complete_pairing_distinguishes_identical_edge_masks(self):
        original=multiple_loops()
        original[1]['segments']=[[[0,2],[1,2]],[[0,10],[1,10]],[[2,2],[2,10]]]
        problem=SearchProblem(original,double_face())
        self.assertEqual(problem.masks[0].tolist(),problem.masks[3].tolist())
        rotations,classes=orientation_equivalences(problem)
        self.assertIn((0,1,2),rotations)
        self.assertFalse(classes)

    def test_piece_ordering_preserves_a_representative_of_every_label_orbit(self):
        _,mesh=tetrahedron_tiles()
        data=[{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(4)]
        problem=SearchProblem(data,mesh);solutions=brute(problem,'gold')
        self.assertTrue(solutions)
        helper=PieceSymmetry(problem);actual=domains_of(problem);helper.prune(actual)
        survivors=surviving(solutions,actual);self.assertTrue(survivors)
        self.assertLess(len(survivors),len(solutions))
        for solution in solutions:
            positions=[cell for cell,code in enumerate(solution) if code//3 in (1,2,3)]
            canonical=list(solution)
            for tile,cell in zip((1,2,3),positions):canonical[cell]=3*tile+solution[cell]%3
            self.assertTrue(problem.validate(canonical)['valid'])
            self.assertTrue(all(d&(1<<c) for d,c in zip(actual,canonical)))
        restricted=SearchProblem(data,mesh,fixed={0:0})
        self.assertFalse(PieceSymmetry(restricted).classes)

    def test_worker_portfolio_and_impact_change_priorities_not_domains(self):
        base={'worker_portfolio':True,'impact_branching':False,'lookahead':True}
        original=base.copy();variants=[worker_options(base,i) for i in range(4)]
        self.assertEqual(base,original);self.assertEqual(len({row['ordering_goal'] for row in variants}),3)
        self.assertTrue(all(row['lookahead'] for row in variants))
        helper=ImpactPolicy(3);before=[7,7,7];after=[1,1,3]
        helper.observe(0,before,after);self.assertLess(helper.key(0,3),helper.key(1,3))
        self.assertEqual(before,[7,7,7]);self.assertEqual(after,[1,1,3])

    def test_frontier_restarts_and_discrepancy_preserve_every_owned_board(self):
        jobs={i:{'region':[hex(1<<i),hex(3)]} for i in range(6)}
        preferred=[4,0];original=copy.deepcopy(jobs)
        for discrepancy in (False,True):
            for epoch in range(16):
                queue=[5,2,4,1,3,0];before=queue.copy()
                reorder_frontier(queue,jobs,preferred,discrepancy=discrepancy,epoch=epoch)
                self.assertEqual(set(queue),set(before));self.assertEqual(len(queue),len(before))
                self.assertEqual(jobs,original)
                if epoch%8==7:self.assertEqual(queue[0],0)
                elif discrepancy:self.assertEqual(queue[0],4)


class CommittedProofWorkTests(unittest.TestCase):
    def _check_final_rejection(self,native):
        from constraint_dfs import solve
        from unittest.mock import patch
        problem=SearchProblem(multiple_loops(),double_face())
        self.assertFalse(brute(problem,'gold'))
        for feature in ('proof_certificates','compact_explanations'):
            with self.subTest(feature=feature,native=native):
                interrupted=[False];calls=[]
                def stop_in_optional_work(*args,**kwargs):
                    calls.append(args);interrupted[0]=True
                    check=args[-1] if feature=='proof_certificates' else args[3]
                    check()
                    self.fail('The solver did not observe Stop inside optional work')
                path=('constraint_dfs.CertificateStore.capture' if feature=='proof_certificates'
                      else 'constraint_dfs.minimize_failure_scope')
                with patch(path,side_effect=stop_in_optional_work):
                    outcome=solve(problem,seconds=10,stop=lambda:interrupted[0],options={
                        feature:True,'native':'compiled' if native else 'python','native_search':native})
                self.assertEqual(len(calls),1)
                self.assertEqual(outcome['search_loop_backend'],'compiled' if native else 'python')
                self.assertEqual(outcome['status'],'infeasible')
                self.assertTrue(outcome['complete']);self.assertNotIn('checkpoint',outcome)
                self.assertEqual(outcome['nodes'],1)

    def test_python_stop_during_optional_proof_work_keeps_final_rejection_committed(self):
        self._check_final_rejection(False)

    def test_native_stop_during_optional_proof_work_keeps_final_rejection_committed(self):
        from constraint_native import load_library
        from constraint_native_search import select_native_search
        try:
            load_library()
            engine,_=select_native_search(2,6,mode='compiled');engine.close()
        except (OSError,ValueError,RuntimeError) as exc:
            self.skipTest('Optional compiled controller unavailable: '+str(exc))
        self._check_final_rejection(True)


if __name__=='__main__':unittest.main()
