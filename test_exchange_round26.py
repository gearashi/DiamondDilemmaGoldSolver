import copy,unittest
from constraint_exchange_round26 import CheckedExchange,import_for_worker
from constraint_symbolic_frontier import SymbolicFrontier
from constraint_work import WorkScope
from constraint_knowledge import KnowledgeStore
from constraint_mask_conflicts import VerifiedMaskConflictStore
from constraint_sat_session import solve
from search_options import normalize_options
from test_constraint_knowledge import mixed_problem
from test_constraint_dfs import brute
from test_dependencies import requires_sat

class CheckedExchangeTests(unittest.TestCase):
    def test_only_independently_checked_records_retire_unleased_work(self):
        problem=mixed_problem();options=normalize_options({'smart_learning':True,'proof_transport':True})
        exchange=CheckedExchange(problem,'gold',options);self.addCleanup(exchange.close)
        frontier=SymbolicFrontier(WorkScope.for_problem(problem,'gold'))
        lease=frontier.claim('worker',choose=lambda row:(1,row[1]))
        owned=frontier.leases[lease.ticket][1]
        knowledge=KnowledgeStore(problem);knowledge.add([[0,0],[1,3]])
        exchange.offer({'knowledge':knowledge.snapshot()})
        # This false claim is well formed but has no valid contradiction proof.
        source=VerifiedMaskConflictStore(problem);source.offer([])
        exchange.offer({'mask_conflicts':source.snapshot()});source.close()
        exchange.advance(seconds=.1);exchange.retire_verified(frontier)
        self.assertEqual(frontier.leases[lease.ticket][1],owned)
        self.assertFalse(exchange.masks.records())
        for answer in brute(problem,'gold'):
            point=frontier.sets.box(tuple(1<<v for v in answer))
            self.assertFalse(frontier.sets.intersection(point,frontier.completed))
        saved=exchange.snapshot();other=CheckedExchange(problem,'gold',options);self.addCleanup(other.close)
        other.offer(saved);other.advance(seconds=.1)
        self.assertEqual(other.knowledge.records(),exchange.knowledge.records())

    def test_local_root_mask_record_is_not_widened(self):
        problem=mixed_problem();options=normalize_options({'smart_learning':True})
        exchange=CheckedExchange(problem,'gold',options);self.addCleanup(exchange.close)
        source=VerifiedMaskConflictStore(problem,initial=(1,63));self.addCleanup(source.close)
        exchange.offer({'mask_conflicts':source.snapshot()})
        self.assertEqual(exchange.stats['exchange_rejected_payloads'],1)
        self.assertFalse(exchange.masks.records())

    @requires_sat
    def test_verified_gold_loops_reach_sat_owner_without_changing_answers(self):
        problem=mixed_problem();options=normalize_options({'shared_conflicts':True})
        exchange=CheckedExchange(problem,'gold',options);self.addCleanup(exchange.close)
        knowledge=KnowledgeStore(problem);knowledge.add([[0,0],[1,3]])
        exchange.offer({'knowledge':knowledge.snapshot()})
        state={};import_for_worker(state,problem,'gold',options,exchange.snapshot())
        try:
            result=solve(problem,seconds=5,learning_state=state,options=options)
            self.assertEqual(result['status'],'solved')
            self.assertIn(tuple(result['codes']),brute(problem,'gold'))
            self.assertGreater(result['stats']['session_imported_loops'],0)
        finally:
            for value in state.values():
                close=getattr(value,'close',None)
                if close:close()

if __name__=='__main__':unittest.main()


class ExchangeOwnershipRefreshTests(unittest.TestCase):
    def test_same_verified_proof_retires_newly_reclaimed_and_new_frontier_work(self):
        problem=mixed_problem();options=normalize_options({});exchange=CheckedExchange(problem,'gold',options);self.addCleanup(exchange.close)
        knowledge=KnowledgeStore(problem);knowledge.add([[0,0],[1,3]]);exchange.offer({'knowledge':knowledge.snapshot()})
        frontier=SymbolicFrontier(WorkScope.for_problem(problem,'gold'))
        lease=frontier.claim('worker',choose=lambda row:(1,row[1]))
        forbidden=frontier.sets.box((1,1<<3))
        exchange.retire_verified(frontier)
        self.assertFalse(frontier.sets.intersection(forbidden,frontier.completed))
        frontier.reclaim(lease);exchange.retire_verified(frontier)
        self.assertTrue(frontier.sets.intersection(forbidden,frontier.completed))
        self.assertFalse(frontier.sets.intersection(forbidden,frontier.pending))
        previous=frontier.snapshot();exchange.retire_verified(frontier);self.assertEqual(frontier.snapshot(),previous)
        other=SymbolicFrontier(WorkScope.for_problem(problem,'gold'));exchange.retire_verified(other)
        self.assertFalse(other.sets.intersection(other.sets.box((1,1<<3)),other.pending))
