import copy,itertools,json,random,unittest
from constraint_deduction_chain_round26 import FORMAT,ChainError,verify_chain,expand_placement_conflicts
from constraint_shared_conflicts import VerifiedConflictStore
from constraint_mask_conflicts import VerifiedMaskConflictStore
from constraint_knowledge import revalidate_witness
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face,multiple_loops,brute
from test_constraint_separator import ring
from validator import reverse_mask


def certificate(problem,root=None,decisions=(),target='gold'):
    original=[sum(1<<code for code in row) for row in problem.domains]
    return {'format':FORMAT,'fingerprint':problem.fingerprint,'target':target,
        'original':list(map(hex,original)),'root':list(map(hex,root or original)),
        'decisions':[[c,hex(mask)] for c,mask in decisions],'events':[],
        'terminal':{'rule':'empty','data':{'cell':0}}}


def independent_preserves(problem,record,target):
    return all(not all(mask>>board[cell]&1 for cell,mask in record) for board in brute(problem,target))


class DeductionChainTests(unittest.TestCase):
    def test_gold_only_evidence_checked_before_shared_publication(self):
        problem=SearchProblem(multiple_loops(),double_face());boards=brute(problem,'edges')
        self.assertTrue(boards);self.assertFalse(brute(problem,'gold'))
        board=boards[0];cert=certificate(problem,decisions=[(c,1<<code) for c,code in enumerate(board)])
        cert['terminal']={'rule':'loop','data':{'placements':list(enumerate(board))}}
        record=verify_chain(problem,cert);self.assertTrue(independent_preserves(problem,record,'gold'))
        self.assertFalse(independent_preserves(problem,record,'edges'))
        store=VerifiedConflictStore(problem);masks=VerifiedMaskConflictStore(problem)
        try:
            store.offer(list(enumerate(board)));work={}
            self.assertEqual(store.admit_deduction_chain(cert,stats=work),1)
            self.assertEqual(store.pending_count,0);self.assertEqual(store.stats['shared_conflicts_pending_reused'],1)
            self.assertEqual(work['chain_loop_checks'],1)
            self.assertEqual(store.records(),[[[c,v] for c,v in enumerate(board)]])
            self.assertTrue(masks.admit_deduction_chain(cert));self.assertEqual(masks.records(),(record,))
            self.assertEqual(store.stats['shared_conflicts_calls'],0)
            with self.assertRaises(ChainError):verify_chain(problem,cert,target='edges')
            changed=copy.deepcopy(cert);changed['target']='edges'
            with self.assertRaises(ChainError):verify_chain(problem,changed,target='edges')
        finally:store.close();masks.close()

    def test_root_guards_never_escape_and_expansion_is_exact_bounded(self):
        problem=ring(4);root=[sum(1<<code for code in row) for row in problem.domains]
        root[0]=root[1]=9  # physical resources 0/1; no shortage yet
        root[2]=9;cert=certificate(problem,root,target='edges');cert['terminal']={'rule':'hall','data':{'cells':[0,1,2]}}
        record=verify_chain(problem,cert,target='edges');self.assertTrue(independent_preserves(problem,record,'edges'))
        rows=expand_placement_conflicts(record,limit=8);self.assertEqual(len(rows),8)
        self.assertIsNone(expand_placement_conflicts(record,limit=7))
        for board in itertools.product(range(12),repeat=4):
            self.assertEqual(all(mask>>board[cell]&1 for cell,mask in record),any(all(board[c]==v for c,v in row) for row in rows))
        store=VerifiedConflictStore(problem,'edges')
        try:
            self.assertEqual(store.admit_deduction_chain(cert,max_clauses=7),0);self.assertFalse(store.records())
            self.assertEqual(store.admit_deduction_chain(cert,max_clauses=8),8)
        finally:store.close()

    def test_seam_removals_and_hall_candidate_condition_replay(self):
        problem=SearchProblem(simple_tiles(),double_face());n=problem.n;original=[sum(1<<c for c in row) for row in problem.domains]
        checked=0
        for a,b in itertools.product(problem.domains[0],problem.domains[1]):
            if any(int(problem.masks[a,s])!=reverse_mask(int(problem.masks[b,t])) for s,t in [(0,0),(1,1),(2,2)]):
                cert=certificate(problem,decisions=[(0,1<<a),(1,1<<b)],target='edges')
                cert['events']=[{'rule':'seam','cell':0,'removed':hex(1<<a),'data':{'neighbor':1}}]
                record=verify_chain(problem,cert,target='edges');self.assertTrue(independent_preserves(problem,record,'edges'));checked+=1
        self.assertGreater(checked,0)
        cert=certificate(problem,decisions=[(0,1),(1,1)],target='edges')
        cert['events']=[{'rule':'hall','cell':0,'removed':'0x1','data':{'cells':[1]}}]
        self.assertEqual(verify_chain(problem,cert,target='edges'),((0,1),(1,1)))
        cert['events'][0]['data']['cells']=[0]
        with self.assertRaises(ChainError):verify_chain(problem,cert,target='edges')

    def test_malformed_incomplete_omitted_premise_cannot_admit(self):
        problem=SearchProblem(simple_tiles(),double_face());cert=certificate(problem,decisions=[(0,1),(1,1)],target='edges')
        cert['events']=[{'rule':'hall','cell':0,'removed':'0x1','data':{'cells':[1]}}]
        mutations=[{'events':[]},{'events':[{'rule':'native_bits','cell':0,'removed':'0x1','data':{}}]},
            {'root':['0x0','0x3f']},{'decisions':[[False,'0x1']]},{'original':['0x1','0x3f']},
            {'terminal':{'rule':'empty','data':{'cell':1}}}]
        store=VerifiedConflictStore(problem,'edges')
        try:
            for change in mutations:
                bad=copy.deepcopy(cert);bad.update(change)
                with self.assertRaises(ChainError):store.admit_deduction_chain(bad)
                self.assertFalse(store.records())
            bad=copy.deepcopy(cert);bad['events']*=2
            with self.assertRaises(ChainError):verify_chain(problem,bad,target='edges')
        finally:store.close()

    def test_cancel_and_input_mutation_never_publish(self):
        problem=ring(4);root=[9,9,9,585];cert=certificate(problem,root,target='edges');cert['terminal']={'rule':'hall','data':{'cells':[0,1,2]}}
        for boundary in range(1,7):
            calls=0;store=VerifiedConflictStore(problem,'edges')
            def stop():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop')
            try:
                with self.assertRaises(InterruptedError):store.admit_deduction_chain(cert,stop)
                self.assertFalse(store.records())
            finally:store.close()
        calls=0
        def mutate():
            nonlocal calls
            calls+=1
            if calls==4:problem.domains[3].pop()
        with self.assertRaises(ChainError):verify_chain(problem,cert,target='edges',check=mutate)

    def test_positive_full_gold_loop_is_never_proper_loop_failure(self):
        problem=SearchProblem(simple_tiles(),double_face());board=brute(problem,'gold')[0]
        cert=certificate(problem,decisions=[(c,1<<v) for c,v in enumerate(board)])
        cert['terminal']={'rule':'loop','data':{'placements':list(enumerate(board))}}
        with self.assertRaises(ChainError):verify_chain(problem,cert)

if __name__=='__main__':unittest.main()
