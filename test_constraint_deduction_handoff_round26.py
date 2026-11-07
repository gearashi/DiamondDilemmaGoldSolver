"""A receiving checker must finish before a producer drops pending evidence."""
import copy,unittest
from constraint_algebra import AffinePruner
from constraint_pb import PBLearner
from constraint_conflict_pb import ConflictPBLearner
from constraint_arithmetic_round26 import ArithmeticPBLearner
from constraint_lattice_round26 import ContributionLatticePruner
from constraint_round24 import ResearchPruner
from constraint_deduction_chain_round26 import verify_chain
from test_constraint_sat_domains import fixture

class DeductionHandoffTests(unittest.TestCase):
    def test_all_arithmetic_producers_retain_checked_claim_on_consumer_stop(self):
        problem,initial=fixture();conflict=ConflictPBLearner(problem,explain=True);conflict.prune(initial.copy())
        self.assertTrue(conflict.pending)
        plain=PBLearner(problem,explain=True);plain.pending.extend(copy.deepcopy(conflict.pending))
        arithmetic=ArithmeticPBLearner(problem);arithmetic.pending.extend(copy.deepcopy(conflict.pending))
        affine=AffinePruner(problem,explain=True);affine.prune([1]*4);self.assertTrue(affine.pending)
        lattice=ContributionLatticePruner(problem);lattice.prune([1754,2435,2592,2378],max_candidates=1000);self.assertTrue(lattice.reasons)
        for stage in (plain,conflict,arithmetic,affine,lattice):
            with self.subTest(stage=type(stage).__name__):
                queue=stage.reasons if stage is lattice else stage.pending;count=len(queue)
                owner=ResearchPruner.__new__(ResearchPruner);owner.stages=[('actual stage',stage)]
                stream=owner.drain_deduction_chains(target='edges',limit=1);record=next(stream);expected=copy.deepcopy(record)
                def stop():raise InterruptedError('receiving checker stopped')
                try:
                    with self.assertRaises(InterruptedError):verify_chain(problem,record,target='edges',check=stop)
                    record['root'][0]='0x0' # Detached publication cannot corrupt the pending proof.
                finally:stream.close()
                self.assertEqual(len(queue),count)
                stream=owner.drain_deduction_chains(target='edges',limit=1);again=next(stream)
                self.assertEqual(again,expected);verify_chain(problem,again,target='edges')
                with self.assertRaises(StopIteration):next(stream)
                self.assertEqual(len(queue),count-1)
    def test_aggregator_does_not_drain_later_stages_before_acknowledgment(self):
        problem,_=fixture();first=AffinePruner(problem,explain=True);first.prune([1]*4)
        second=AffinePruner(problem,explain=True);second.prune([1]*4)
        owner=ResearchPruner.__new__(ResearchPruner);owner.stages=[('first',first),('second',second)]
        before=(len(first.pending),len(second.pending));iterator=owner.drain_deduction_chains(target='edges');next(iterator);iterator.close()
        self.assertEqual((len(first.pending),len(second.pending)),before)
        records=list(owner.drain_deduction_chains(target='edges',limit=2));self.assertEqual(len(records),2)
        self.assertFalse(first.pending);self.assertFalse(second.pending)

if __name__=='__main__':unittest.main()
