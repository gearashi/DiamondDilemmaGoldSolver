import json,unittest
from constraint_proof import RUPChecker
from constraint_proof_indexed_round26 import IndexedRUPChecker
from constraint_proof_reuse_round26 import clear,compact
from test_constraint_proof_indexed_round26 import complete

class ProofReuseTests(unittest.TestCase):
    def setUp(self):clear()
    def test_shared_prefix_different_suffix_and_assumptions_fail_closed(self):
        base=[(1,)];prefix=['1 2 0']*64
        first=IndexedRUPChecker(base,[*prefix,'0'],reuse=True);self.assertFalse(complete(first));self.assertEqual(first.proof_cursor,64)
        second=IndexedRUPChecker(base,[*prefix,'-1 0','0'],reuse=True)
        self.assertEqual(second.stats['proof_shared_steps'],64);self.assertFalse(complete(second))
        third=IndexedRUPChecker(base,[*prefix,'0'],extra_clauses=[(-1,)],reuse=True)
        self.assertEqual(third.stats['proof_shared_steps'],0);self.assertTrue(complete(third))
        fourth=IndexedRUPChecker(base,[*prefix,'0'],extra_clauses=[(-1,)],reuse=True)
        self.assertEqual(fourth.stats['proof_shared_steps'],65);self.assertTrue(complete(fourth));self.assertEqual(fourth.stats['proof_clause_visits'],0)
    def test_shared_prefix_indexing_roundtrips_every_microstep(self):
        base=[(1,),(-1,)];proof=['2 -2 0']*64+['0'];first=IndexedRUPChecker(base,proof,reuse=True);self.assertTrue(complete(first))
        current=IndexedRUPChecker(base,proof,reuse=True)
        for _ in range(500):
            current.advance(ticks=1)
            current=IndexedRUPChecker.restore_trusted_local(base,proof,json.loads(json.dumps(current.snapshot())),reuse=True)
            if current.finished:break
        self.assertTrue(current.result);self.assertEqual(current.stats['proof_clause_visits'],0)
    def test_compaction_replays_actual_dependency_closure(self):
        base=[(1,2),(1,-2),(-1,2),(-1,-2)];proof=['3 -3 0']*20+['1 0','-1 0','0']+['4 -4 0']*10;stats={}
        result=compact(base,proof,stats=stats)
        self.assertIsNotNone(result);self.assertTrue(RUPChecker(base).verify(result));self.assertLess(len(result),len(proof));self.assertEqual(stats['proof_compaction_checks'],2)
        rejected={};self.assertIsNone(compact([(1,)],['0'],stats=rejected));self.assertEqual(rejected['proof_compaction_checks'],1);self.assertGreater(rejected['proof_compaction_visits'],0)
        def stop():raise ValueError('callback')
        with self.assertRaisesRegex(ValueError,'callback'):compact(base,proof,stop)
        from constraint_proof import ProofLimit
        def capped_callback():raise ProofLimit('callback budget')
        with self.assertRaisesRegex(ProofLimit,'callback budget'):compact(base,proof,capped_callback)
    def test_public_export_compacts_and_preserves_external_stop(self):
        from unittest.mock import patch
        from constraint_proof import MaskProofContext,BoundedDPLL,ProofLimit
        from search_problem import SearchProblem
        from test_constraint_dfs import simple_tiles,double_face
        context=MaskProofContext(SearchProblem(simple_tiles(),double_face()),[63,63],indexed=True)
        self.addCleanup(context.close)
        def redundant(owner,check):owner.proof=['1 -1 0']*12+['0'];return False
        with patch.object(BoundedDPLL,'solve',redundant):result=context.prove([(0,1),(1,1)],seconds=2)
        self.assertIsNotNone(result);self.assertTrue(context.verify(result));self.assertLess(len(result['proof']),13)
        self.assertGreater(context.stats['proof_compaction_input_steps'],context.stats['proof_compaction_output_steps'])
        self.assertGreater(context.stats['proof_compaction_index_literals'],0)
        def stop():raise ProofLimit('external callback')
        with self.assertRaisesRegex(ProofLimit,'external callback'):context.prove([(0,1),(1,1)],stop)

if __name__=='__main__':unittest.main()
