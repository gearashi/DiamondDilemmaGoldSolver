"""Scheduling changes execution only; an omitted check is never a proof."""
import unittest
from constraint_scheduling import PropagationScheduler, STAT_KEYS


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.stats=dict.fromkeys(STAT_KEYS,0)

    def test_exact_unchanged_signature_and_widening(self):
        scheduler=PropagationScheduler(adaptive=False)
        calls=[]
        for domains in ([3,12],[3,12],[1,12],[3,12],[7,12]):
            scheduler.run('check',list(domains),lambda:calls.append(tuple(domains)),self.stats)
        self.assertEqual(calls,[(3,12),(1,12),(7,12)])
        self.assertEqual(self.stats['propagation_unchanged_skips'],2)

    def test_dependency_projection_ignores_only_irrelevant_rotation_changes(self):
        scheduler=PropagationScheduler(adaptive=False);calls=[]
        for domains in ([7,56],[3,24],[1,8],[9,16],[7,56]):
            signature=tuple(sum(1<<tile for tile in range(2) if value&(7<<(3*tile))) for value in domains)
            scheduler.run('tile_inventory',list(domains),lambda:calls.append(tuple(domains)),self.stats,signature=signature)
        self.assertEqual(calls,[(7,56),(9,16)])
        self.assertEqual(self.stats['propagation_unchanged_skips'],3)

    def test_projected_key_still_measures_actual_domain_reductions(self):
        scheduler=PropagationScheduler();domains=[7]
        scheduler.run('rotations',domains,lambda:domains.__setitem__(0,3),self.stats,signature=(1,))
        self.assertEqual(list(scheduler.records['rotations']['unchanged']),[])
        scheduler.run('rotations',domains,lambda:domains.__setitem__(0,1),self.stats,signature=(1,))
        self.assertEqual(domains,[1])

    def test_a_reduction_is_not_mistaken_for_a_fixpoint(self):
        scheduler=PropagationScheduler()
        domains=[7]
        def prune():
            domains[0] &= domains[0]-1
        scheduler.run('check',domains,prune,self.stats)
        scheduler.run('check',domains,prune,self.stats)
        self.assertEqual(domains,[4])
        self.assertEqual(self.stats['propagation_runs'],2)

    def test_exception_and_incomplete_search_are_not_memoized(self):
        scheduler=PropagationScheduler()
        def interrupt(): raise RuntimeError('stop')
        with self.assertRaises(RuntimeError):
            scheduler.run('check',[3],interrupt,self.stats)
        calls=[]
        for _ in range(2):
            scheduler.run('check',[3],lambda:calls.append(1),self.stats,complete=lambda:False)
        self.assertEqual(calls,[1,1])
        self.assertEqual(self.stats['propagation_unchanged_skips'],0)

    def test_cost_based_cooldowns_retry_and_never_mutate_domains(self):
        clock=iter(range(0,100_000_000,1_000_000))
        scheduler=PropagationScheduler(incremental=False,clock=lambda:next(clock))
        calls=[]
        for _ in range(20):
            domains=[7,56]
            scheduler.run('optional',domains,lambda:calls.append(1),self.stats,optional=True)
            self.assertEqual(domains,[7,56])
        self.assertGreater(self.stats['propagation_adaptive_skips'],0)
        self.assertGreaterEqual(len(calls),5)
        # Mandatory propagators are not cooled down.
        for _ in range(6):scheduler.run('mandatory',[7],lambda:None,self.stats)
        self.assertEqual(scheduler.records['mandatory']['cooldown'],0)

    def test_proof_forces_call_and_a_useful_call_resets_cooldown(self):
        clock=iter(range(0,100_000_000,1_000_000))
        scheduler=PropagationScheduler(incremental=False,clock=lambda:next(clock))
        for _ in range(3):scheduler.run('check',[7],lambda:None,self.stats,optional=True)
        self.assertGreater(scheduler.records['check']['cooldown'],0)
        domains=[7]
        scheduler.run('check',domains,lambda:domains.__setitem__(0,3),self.stats,optional=True,force=True)
        self.assertEqual(domains,[3])
        self.assertEqual(scheduler.records['check']['cooldown'],0)


if __name__=='__main__':unittest.main()
