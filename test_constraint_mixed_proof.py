"""Independent finite checks for mixed-theory proof DAG transport and reuse."""
import copy
import json
import unittest
from unittest.mock import patch
from constraint_mixed_proof import MixedProofContext,FORMAT
from constraint_proof import MaskProofContext
from constraint_pb import primitive_rows,combine,verify_derivation,is_integer_row
from test_constraint_separator import ring
from constraint_mask_conflicts import VerifiedMaskConflictStore
from search_problem import SearchProblem
from test_constraint_dfs import multiple_loops,double_face,simple_tiles,brute


def loop_context():
    problem=SearchProblem(multiple_loops(),double_face())
    return problem,MixedProofContext(problem,[63,63])


def mixed_comparison():
    problem,source=loop_context();edges=brute(problem,'edges');assert len(edges)==18 and not brute(problem,'gold')
    try:
        # A Hall lemma and each proper-loop lemma are separate reusable nodes.
        assert source.prove([(0,7),(1,7)],seconds=2)
        for board in edges:
            assert source.prove([(c,1<<code) for c,code in enumerate(board)],seconds=2)
        first=source.prove([],seconds=3);assert first
        second=source.prove([(0,7)],seconds=3);assert second
        certificates=[first,second];cold=0
        for certificate in certificates:
            checker=MixedProofContext(problem,[63,63])
            try:
                assert checker.verify(certificate)
                cold+=checker.stats['mixed_nodes_checked']
            finally:checker.close()
        checker=MixedProofContext(problem,[63,63])
        try:
            for certificate in certificates:assert checker.verify(certificate)
            warm=checker.stats['mixed_nodes_checked'];hits=checker.stats['mixed_nodes_reused']
        finally:checker.close()
        return {'edge_solutions':18,'gold_solutions':0,'certificates':certificates,
                'cold_checked_nodes':cold,'shared_checked_nodes':warm,'reused_nodes':hits,
                'rules':sorted({node['rule'] for cert in certificates for node in cert['nodes']})}
    finally:source.close()


class MixedProofTests(unittest.TestCase):
    def test_combined_gold_refutation_shares_checked_prefixes(self):
        result=mixed_comparison()
        self.assertLess(result['shared_checked_nodes'],result['cold_checked_nodes'])
        self.assertTrue({'hall','loop','rup'}<=set(result['rules']))

    def test_affine_and_pb_leaves_are_semantically_rechecked(self):
        problem=SearchProblem(simple_tiles(),double_face())
        context=MixedProofContext(problem,[63,63]);self.addCleanup(context.close)
        self.assertTrue(context.prepare(seconds=2))
        affine=context._affine_leaf(((0,1),(1,16)),lambda:None)
        self.assertIsNotNone(affine);self.assertTrue(context._accept(affine,lambda:None))
        receiver=MixedProofContext(problem,[63,63]);self.addCleanup(receiver.close)
        self.assertTrue(receiver.verify(context.bundle(affine['id'])))
        data=copy.deepcopy(affine['data']);data['equations'][0][2]^=1
        forged=context.make_node('affine',context.decode(affine['record']),data)
        self.assertFalse(context._accept(forged,lambda:None))
        pb=context._pb_leaf(((0,7),(1,7)),lambda:None)
        self.assertIsNotNone(pb);self.assertTrue(context._accept(pb,lambda:None))
        self.assertTrue(receiver.verify(context.bundle(pb['id'])))
        forged=context.make_node('pb',context.decode(pb['record']),{**pb['data'],'row':-1})
        self.assertFalse(context._accept(forged,lambda:None))

    def test_native_free_store_produces_and_imports_gold_only_failure(self):
        problem,producer=loop_context();self.addCleanup(producer.close)
        plain=MaskProofContext(problem,[63,63]);self.addCleanup(plain.close)
        record=((0,1),(1,8))
        self.assertIsNone(plain.prove(record,seconds=2))
        def no_native():raise AssertionError('No native SAT call is needed.')
        sender=VerifiedMaskConflictStore(problem,mixed_proof_dag=True,factory=no_native)
        receiver=VerifiedMaskConflictStore(problem,mixed_proof_dag=True,factory=no_native)
        self.addCleanup(sender.close);self.addCleanup(receiver.close)
        sender.offer(record);self.assertEqual(sender.advance(seconds=1),1)
        payload=json.loads(json.dumps(sender.snapshot()))
        receiver.import_snapshot(payload);self.assertFalse(receiver.records())
        for _ in range(100):
            receiver.advance(seconds=1)
            if receiver.records():break
        self.assertEqual(receiver.records(),sender.records())
        self.assertEqual(receiver.stats['mask_conflicts_calls'],0)
        narrowed=[1,8];receiver.prune(narrowed);self.assertIn(0,narrowed)

    def test_content_scope_parent_and_semantic_tampering_rejected(self):
        problem,source=loop_context();self.addCleanup(source.close)
        certificate=source.prove([(0,1),(1,8)],seconds=2)
        for change in ('target','initial','hash','record','parent','loop'):
            corrupt=copy.deepcopy(certificate)
            if change=='target':corrupt['target']='edges'
            elif change=='initial':corrupt['initial']=['0x7','0x38']
            elif change=='hash':corrupt['base_sha256']='0'*64
            elif change=='record':corrupt['record']=[]
            elif change=='parent':corrupt['nodes'][0]['parents']=['missing']
            else:
                record=source.decode(corrupt['nodes'][0]['record'])
                node=source.make_node('loop',record,{'placements':[[0,0]]})
                corrupt['nodes']=[node];corrupt['root']=node['id']
            receiver=MixedProofContext(problem,[63,63])
            try:self.assertFalse(receiver.verify(corrupt),change)
            finally:receiver.close()
        edge=MixedProofContext(problem,[63,63],'edges');self.addCleanup(edge.close)
        self.assertFalse(edge.verify(certificate))

    def test_partial_verification_never_admits_and_resumes(self):
        result=mixed_comparison();certificate=result['certificates'][0]
        problem,receiver=loop_context();self.addCleanup(receiver.close)
        self.assertTrue(receiver.prepare(seconds=2))
        seen_pending=False
        for _ in range(5000):
            answer=receiver.verify_incremental(certificate,ticks=1)
            if answer is None:seen_pending=True;continue
            self.assertTrue(answer);break
        else:self.fail('Proof never completed across leases.')
        self.assertTrue(seen_pending)
        self.assertGreater(receiver.stats['proof_cursor_resumptions'],0)

    def test_cancellation_keeps_only_independently_completed_nodes(self):
        result=mixed_comparison();certificate=result['certificates'][0]
        problem,receiver=loop_context();self.addCleanup(receiver.close)
        calls=[0]
        def cancel():
            calls[0]+=1
            if calls[0]==15:raise InterruptedError('stop')
        with self.assertRaises(InterruptedError):receiver.verify_incremental(certificate,cancel)
        self.assertNotIn(certificate['root'],receiver.nodes)
        self.assertTrue(receiver.verify(certificate))

    @staticmethod
    def cancellation_rows(problem,record):
        base=list(primitive_rows(problem,record))
        positive=base.index((((0,0,1),),1))
        negative_one=base.index((((1,3,-1),),-1))
        negative_two=base.index((((2,6,-1),),-1))
        scale=2**54
        first=combine(base[positive],base[negative_one],scale,1)
        second=combine(first,base[negative_two],1,scale)
        steps=((positive,negative_one,scale,1,first),
               (len(base),negative_two,1,scale,second))
        return base,steps

    def test_float_cancellation_cannot_refute_a_valid_gold_board(self):
        problem=ring(3);record=((0,1),(1,8),(2,64))
        self.assertTrue(problem.validate([0,3,6])['valid'])
        initial=tuple(sum(1<<code for code in row) for row in problem.domains)
        context=MixedProofContext(problem,initial);self.addCleanup(context.close)
        self.assertTrue(context.prepare(seconds=2))
        base,steps=self.cancellation_rows(problem,record)
        self.assertTrue(verify_derivation(base,steps))
        self.assertEqual(sum(k for _,_,k in steps[-1][-1][0]),-1)
        floating=tuple((*step[:4],(tuple((c,v,float(k)) for c,v,k in step[4][0]),step[4][1])) for step in steps)
        # Sequential floating addition loses the middle -1. Exact arithmetic
        # correctly retains -1, equal to the bound rather than greater than it.
        rounded=0.0
        for _,_,coefficient in floating[-1][-1][0]:rounded+=coefficient
        self.assertEqual(rounded,0.0)
        self.assertFalse(verify_derivation(base,floating))
        for proof in (steps,floating):
            node=context.make_node('pb',record,{'cells':[0,1,2],'steps':proof,'row':len(base)+1})
            self.assertFalse(context._accept(node,lambda:None))
            certificate={'proof_format':FORMAT,**context.binding(record),'nodes':[node],'root':node['id']}
            self.assertFalse(context.verify(certificate))
        self.assertFalse(context.nodes)

    def test_pb_proof_numbers_require_exact_integers_including_ancestors(self):
        problem=ring(3);record=((0,1),(1,8),(2,64));base,steps=self.cancellation_rows(problem,record)
        self.assertTrue(verify_derivation(base,steps))
        # Reject equal-valued floats and bools in every imported numeric role.
        valid=(((0,0,1),),1)
        malformed=( (((False,0,1),),1), (((0,False,1),),1),
                    (((0,0,True),),1), (((0,0,1.0),),1),
                    (((0,0,1),),True), (((0,0,1),),1.0) )
        self.assertTrue(is_integer_row(valid))
        for bad in malformed:
            self.assertFalse(is_integer_row(bad))
            self.assertFalse(verify_derivation([bad],()))
            self.assertFalse(verify_derivation([valid],[(0,0,1,1,bad)]))
        for position in range(4):
            broken=list(steps[0]);broken[position]=float(broken[position])
            self.assertFalse(verify_derivation(base,[tuple(broken)]))
            broken[position]=True
            self.assertFalse(verify_derivation(base,[tuple(broken)]))
        # A successor may cite only earlier independently checked rows.
        forward=list(steps[0]);forward[0]=len(base)
        self.assertFalse(verify_derivation(base,[tuple(forward)]))
        def stop():raise InterruptedError('check arithmetic replay')
        with self.assertRaises(InterruptedError):verify_derivation(base,steps,stop)

    def test_mixed_pb_import_rejects_boolean_coefficients_and_integer_over_budget(self):
        problem=SearchProblem(simple_tiles(),double_face())
        context=MixedProofContext(problem,[63,63]);self.addCleanup(context.close)
        self.assertTrue(context.prepare(seconds=2))
        record=((0,7),(1,7))
        good=context._pb_leaf(record,lambda:None)
        self.assertIsNotNone(good)
        for change in ('bool','float','huge'):
            data=copy.deepcopy(good['data'])
            if not data['steps']:
                # A primitive PB contradiction needs no derivation. Supply a
                # synthetic claimed row so numeric validation is exercised.
                data['steps']=[[0,0,1,1,[[[0,0,1]],1]]]
            if change=='huge':data['steps'][0][2]=1<<4097
            else:data['steps'][0][4][0][0][2]=True if change=='bool' else 1.0
            forged=context.make_node('pb',record,data)
            self.assertFalse(context._accept(forged,lambda:None))

    def test_cached_gold_nodes_bind_actual_segment_pairings_with_identical_masks(self):
        problem,context=loop_context();self.addCleanup(context.close)
        certificate=context.prove([(0,1),(1,8)],seconds=2)
        self.assertIsNotNone(certificate);self.assertTrue(context.verify(certificate))
        original_masks=problem.masks.copy()
        # The same occupied ports now connect across sides, joining all six
        # strands into one valid loop; old loop lemmas are no longer valid.
        problem.tiles[1]['segments']=[[[0,2],[1,10]],[[1,2],[2,10]],[[2,2],[0,10]]]
        self.assertTrue((problem.masks==original_masks).all())
        self.assertTrue(problem.validate([0,3])['valid'])
        with self.assertRaises(ValueError):context.verify(certificate)
        with self.assertRaises(ValueError):context.verify_incremental(certificate)

    def test_unhashable_roots_are_rejected_and_gold_sources_are_rechecked(self):
        problem,context=loop_context();self.addCleanup(context.close)
        certificate=context.prove([(0,1),(1,8)],seconds=2)
        for root in ([],{},None,True):
            bad=copy.deepcopy(certificate);bad['root']=root
            self.assertFalse(context.verify(bad))
            self.assertFalse(context.verify_incremental(bad))
        valid=ring(3)
        initial=tuple(sum(1<<code for code in row) for row in valid.domains)
        other=MixedProofContext(valid,initial);self.addCleanup(other.close)
        other.add_gold_witnesses([[(0,0)],[["bad",0]]])
        self.assertTrue(valid.validate([0,3,6])['valid'])
        self.assertIsNone(other.prove([],seconds=2))
        self.assertFalse(any(json.loads(node[0])['rule']=='loop' for node in other.nodes.values()))


if __name__=='__main__':unittest.main()
