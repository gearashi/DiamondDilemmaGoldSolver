"""Independent finite checks of reusable proper-loop witness certificates."""
import copy
from itertools import permutations,product
import unittest
from unittest.mock import patch
from constraint_knowledge import KnowledgeStore,load,revalidate_witness,_digest,MAX_BYTES,MAX_RECORDS
from search_problem import SearchProblem
from test_constraint_dfs import board,double_face,multiple_loops,simple_tiles


def mixed_problem():
    tiles=[{'id':i,'segments':[[[0,2],[1,2]],[[0,10],[1,10]],[[2,2],[2,10]]]} for i in range(2)]
    return SearchProblem(tiles,double_face())


class KnowledgeTests(unittest.TestCase):
    def test_proper_loop_clause_never_excludes_a_finite_gold_solution(self):
        problem=mixed_problem();clause=revalidate_witness(problem,[[0,0],[1,3]])
        self.assertFalse(problem.validate([0,3])['valid'])
        gold=0
        for tiles in permutations(range(2)):
            for rotations in product(range(3),repeat=2):
                codes=[3*tile+rotation for tile,rotation in zip(tiles,rotations)]
                if problem.validate(codes)['valid']:
                    gold+=1;self.assertFalse(all(codes[cell]==code for cell,code in clause))
        self.assertGreater(gold,0)
    def test_closed_component_minimizes_unrelated_placements(self):
        tiles=[{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(4)]
        geometry=board(4,[(a,side,a+1,side) for a in (0,2) for side in range(3)])
        problem=SearchProblem(tiles,geometry)
        self.assertEqual(revalidate_witness(problem,[[2,6],[1,3],[0,0]]),((0,0),(1,3)))
    def test_does_not_trust_fast_solver_loop_verdict_or_digest(self):
        problem=mixed_problem();store=KnowledgeStore(problem);store.add([[0,0],[1,3]])
        payload=store.snapshot();payload['records']=[[[0,0],[1,4]]];payload['digest']=_digest(payload)
        self.assertTrue(problem.validate([0,4])['valid'])
        with patch.object(problem,'partial_loop_impossible',return_value=True),patch.object(problem,'gold_nogood',return_value=[(0,0)]):
            with self.assertRaisesRegex(ValueError,'closed proper'):load(problem,payload)
    def test_malformed_open_full_gold_duplicate_and_edge_witnesses_rejected(self):
        problem=SearchProblem(simple_tiles(),double_face())
        for witness in ([],[[0,0]],[[0,0],[1,3]],[[0,0],[0,1]],[[0,0],[1,1]],[[True,0]],[[0,99]]):
            with self.subTest(witness=witness),self.assertRaises(ValueError):revalidate_witness(problem,witness)
        with self.assertRaises(ValueError):revalidate_witness(mixed_problem(),[[0,0],[1,3]],target='edges')
        self.assertEqual(load(problem,KnowledgeStore(problem,target='edges').snapshot(),target='edges').records(),[])
    def test_fixed_and_original_domain_binding_are_checked(self):
        problem=SearchProblem(multiple_loops(),double_face(),fixed={0:0})
        store=KnowledgeStore(problem);store.add([[0,0],[1,3]])
        with self.assertRaises(ValueError):store.add([[0,1],[1,3]])
        payload=store.snapshot();problem.domains[1]=[4,5]
        self.assertFalse(store.matches(problem,'gold'))
        with self.assertRaises(ValueError):store.snapshot()
        with self.assertRaises(ValueError):load(problem,payload)
    def test_roundtrip_dedup_bounds_eviction_and_snapshot_ownership(self):
        problem=SearchProblem(multiple_loops(),double_face());store=KnowledgeStore(problem,limit=2)
        for rotation in range(3):store.add([[0,rotation],[1,3]])
        self.assertEqual(len(store),2);self.assertEqual(store.stats['evictions'],1)
        self.assertFalse(store.add(store.records()[0]));self.assertEqual(len(store),2)
        payload=store.snapshot();restored=load(problem,payload,limit=2)
        self.assertEqual(restored.records(),store.records())
        payload['records'][0][0][1]=999
        self.assertNotEqual(store.records()[0][0][1],999)
        with self.assertRaises(ValueError):load(problem,payload)
    def test_loaded_payload_is_bound_bounded_and_unknown_versions_rejected(self):
        problem=mixed_problem();store=KnowledgeStore(problem);store.add([[0,0],[1,3]]);original=store.snapshot()
        changes=[('knowledge_version',True),('knowledge_version',999),('fingerprint','wrong'),('target','edges'),('records',[[]]*(MAX_RECORDS+1)),('records',[[['x'*MAX_BYTES,0]]])]
        for key,value in changes:
            payload=copy.deepcopy(original);payload[key]=value;payload['digest']=_digest(payload)
            with self.subTest(key=key),self.assertRaises(ValueError):load(problem,payload)
    def test_invalid_batch_cannot_partly_modify_store(self):
        problem=mixed_problem();store=KnowledgeStore(problem)
        with self.assertRaises(ValueError):store.merge([[[0,0],[1,3]],[[0,0],[1,4]]])
        self.assertEqual(store.records(),[])

    def test_snapshot_cache_and_trusted_duplicate_do_not_reprove_but_import_does(self):
        problem=mixed_problem();store=KnowledgeStore(problem);store.add([[0,0],[1,3]])
        first=store.snapshot()
        with patch('constraint_knowledge._digest',side_effect=AssertionError('Unchanged snapshot hashed twice')):
            second=store.snapshot()
        self.assertEqual(first,second);self.assertIsNot(first,second)
        with patch('constraint_knowledge.revalidate_witness',side_effect=ValueError('Independent proof still required')):
            self.assertFalse(store.add([[1,3],[0,0]]))
            # A matching digest cannot make imported JSON into trusted state.
            with self.assertRaisesRegex(ValueError,'Independent proof'):load(problem,first)
        second['records'][0][0][1]=999
        self.assertEqual(store.snapshot(),first)
        with self.assertRaises(ValueError):store.add([[False,0],[1,3]])
    def test_original_domain_rows_can_reorder_but_cannot_change(self):
        problem=mixed_problem();store=KnowledgeStore(problem)
        problem.domains[0].reverse();self.assertTrue(store.matches(problem))
        problem.domains[0].pop();self.assertFalse(store.matches(problem))
        with self.assertRaises(ValueError):store.merge([])
    def test_verified_store_transfer_does_not_share_mutable_records(self):
        problem=mixed_problem();one=KnowledgeStore(problem);one.add([[0,0],[1,3]])
        two=KnowledgeStore(problem)
        with patch('constraint_knowledge.revalidate_witness',side_effect=AssertionError('Already independently verified')):
            self.assertEqual(two.merge_store(one),1)
        self.assertEqual(one.snapshot(),two.snapshot())
        with self.assertRaises(ValueError):two.merge_store(KnowledgeStore(problem,target='edges'))

    def test_import_reuses_binding_and_exact_retained_proofs_only(self):
        problem=mixed_problem();store=KnowledgeStore(problem);store.add([[0,0],[1,3]])
        payload=store.snapshot();target=KnowledgeStore(problem)
        with patch('constraint_knowledge._domains_hash',side_effect=AssertionError('Binding rehashed')):
            with patch('constraint_knowledge.revalidate_witness',wraps=revalidate_witness) as prove:
                self.assertEqual(target.merge_snapshot(payload),1)
                self.assertEqual(target.merge_snapshot(payload),0)
                self.assertEqual(prove.call_count,1)
                self.assertEqual(target.stats['proof_reuses'],1)
        bad=copy.deepcopy(payload);bad['records']=[[[0,0],[1,4]]];bad['digest']=_digest(bad)
        with self.assertRaises(ValueError):target.merge_snapshot(bad)
        self.assertEqual(target.snapshot(),payload)

    def test_long_witnesses_fit_actual_pretty_atomic_file_and_reload(self):
        import json,tempfile
        from pathlib import Path
        from io_utils import atomic_json,read_json_shared
        from constraint_knowledge import _disk_bytes
        n=160;split=79
        edges=[(i,1,(i+1)%split,0) for i in range(split)]
        edges += [(i,1,split+(i-split+1)%(n-split),0) for i in range(split,n)]
        tiles=[{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(n)]
        problem=SearchProblem(tiles,board(n,edges));store=KnowledgeStore(problem)
        records=[[[cell,3*((cell+shift)%n)] for cell in range(split)] for shift in range(n)]
        for record in records:store.add(record)
        self.assertGreater(store.stats['evictions'],0);self.assertLess(len(store),len(records))
        snapshot=store.snapshot();self.assertLessEqual(_disk_bytes(snapshot),MAX_BYTES)
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'knowledge.json';atomic_json(path,snapshot)
            self.assertLessEqual(path.stat().st_size,MAX_BYTES)
            self.assertEqual(load(problem,read_json_shared(path)).records(),store.records())
        oversized=store.snapshot();oversized['records']=records;oversized['digest']=_digest(oversized)
        self.assertLess(len(json.dumps(oversized,separators=(',',':')).encode()),MAX_BYTES)
        self.assertGreater(_disk_bytes(oversized),MAX_BYTES)
        with self.assertRaisesRegex(ValueError,'byte limit'):load(problem,oversized)

    def test_repeated_exact_imports_use_proof_content_not_supplied_digest(self):
        problem=mixed_problem();source=KnowledgeStore(problem);source.add([[0,0],[1,3]])
        payload=source.snapshot();target=KnowledgeStore(problem);target.merge_snapshot(payload)
        permuted=copy.deepcopy(payload);permuted['records']=[[[1,3],[0,0]]];permuted['digest']=_digest(permuted)
        with patch('constraint_knowledge.revalidate_witness',side_effect=AssertionError('Exact proof should be reused')):
            for _ in range(5):self.assertEqual(target.merge_snapshot(payload),0)
            self.assertEqual(target.merge_snapshot(permuted),0)
        self.assertEqual(target.stats['proof_reuses'],6)
        self.assertEqual(target._record_set,set(target._records))
        # A checksum copied from a known payload still cannot authenticate edits.
        forged=copy.deepcopy(payload);forged['records']=[[[0,0],[1,4]]]
        with self.assertRaisesRegex(ValueError,'digest'):target.merge_snapshot(forged)
        forged['digest']=_digest(forged)
        with self.assertRaisesRegex(ValueError,'closed proper'):target.merge_snapshot(forged)
        self.assertEqual(target.snapshot(),payload)

    def test_reused_and_novel_malformed_batch_is_transactional(self):
        problem=SearchProblem(multiple_loops(),double_face());store=KnowledgeStore(problem)
        store.add([[0,0],[1,3]]);before=store.snapshot();stats=store.stats.copy()
        with self.assertRaises(ValueError):store.merge([[[1,3],[0,0]],[[0,1],[1,3]],[[0,0]]])
        self.assertEqual(store.snapshot(),before);self.assertEqual(store.stats,stats)
        self.assertEqual(store._record_set,set(store._records))
        self.assertEqual(store.merge([[[1,3],[0,0]],[[0,1],[1,3]]]),1)
        self.assertEqual(store.stats['proof_reuses'],stats['proof_reuses']+1)
        self.assertEqual(store.stats['verified'],stats['verified']+1)

    def test_evicted_or_subsumed_claims_do_not_keep_an_unbounded_proof_cache(self):
        problem=SearchProblem(multiple_loops(),double_face());store=KnowledgeStore(problem,limit=1)
        first=[[0,0],[1,3]];second=[[0,1],[1,3]]
        store.add(first);store.add(second)
        self.assertEqual(len(store._record_set),1)
        with patch('constraint_knowledge.revalidate_witness',wraps=revalidate_witness) as proof:
            store.merge([first]);self.assertEqual(proof.call_count,1)
        self.assertEqual(store._record_set,set(store._records));self.assertEqual(len(store),1)

    def test_warm_import_still_rejects_bad_types_ownership_and_changed_bindings(self):
        problem=mixed_problem();store=KnowledgeStore(problem);store.add([[0,0],[1,3]]);payload=store.snapshot()
        for record in ([[[False,0],[1,3]]],[[[0,False],[1,3]]],[[[0,0],[0,0],[1,3]]],[[[0,0],[1,0]]]):
            bad=copy.deepcopy(payload);bad['records']=record;bad['digest']=_digest(bad)
            with self.subTest(record=record),self.assertRaises(ValueError):store.merge_snapshot(bad)
        for field,value in (('target','edges'),('fingerprint','different'),('knowledge_version',True),('domains_sha256','different')):
            bad=copy.deepcopy(payload);bad[field]=value;bad['digest']=_digest(bad)
            with self.subTest(field=field),self.assertRaises(ValueError):store.merge_snapshot(bad)
        problem.domains[0].reverse();self.assertEqual(store.merge_snapshot(payload),0)
        problem.domains[0].pop()
        with self.assertRaisesRegex(ValueError,'source domains changed'):store.merge_snapshot(payload)

    def test_noncanonical_superset_is_independently_revalidated_before_minimization(self):
        tiles=[{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(4)]
        geometry=board(4,[(a,side,a+1,side) for a in (0,2) for side in range(3)])
        problem=SearchProblem(tiles,geometry);store=KnowledgeStore(problem);store.add([[0,0],[1,3]])
        with patch('constraint_knowledge.revalidate_witness',wraps=revalidate_witness) as proof:
            self.assertEqual(store.merge([[[0,0],[1,3],[2,6]]]),0)
            self.assertEqual(proof.call_count,1)
        self.assertEqual(store.stats['verified'],2)

if __name__=='__main__':unittest.main()
