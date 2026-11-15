"""Whole verified GPU batches survive a repeat resize without another encoding."""
from copy import deepcopy
from unittest import TestCase
from unittest.mock import patch

from constraint_work import digest
from gpu_domain_runner import pack_native, solve, unpack_native
from test_r34_lane_resize import new_engine, pause_restore, setup_saved


class ParkedArchivesTests(TestCase):
    def test_whole_archives_reuse_compression_but_own_mutable_metadata(self):
        for balanced, forked in ((False,False), (True,False), (False,True), (True,True)):
            with self.subTest(balanced=balanced, forked=forked):
                problem, source = setup_saved(balanced,forked)
                options = {'gpu_lane_balance':balanced,'gpu_propagated_forks':forked}
                saved = pause_restore(problem,source,1,options)['checkpoint']
                calls = []
                def count(engine,snapshot=None):
                    calls.append(engine.lanes)
                    return pack_native(engine,snapshot)
                with patch('gpu_domain_runner.pack_native',side_effect=count):
                    after = pause_restore(problem,saved,1,options)['checkpoint']
                self.assertEqual(calls,[1])  # The active lane is checkpointed on pause.
                self.assertEqual(after,saved)
                after['native_queue'][0]['cubes'][0][0] = 'changed'
                self.assertNotEqual(after['native_queue'],saved['native_queue'])

    def test_aligned_parked_rollback_and_counters_checked_before_reuse(self):
        problem, initial = setup_saved()
        saved = pause_restore(problem,initial,2,{})['checkpoint']
        for case in ('mechanical','totals'):
            with self.subTest(case=case):
                bad = deepcopy(saved)
                if case == 'mechanical':
                    last = bad['native_queue'][-1]
                    cubes = [tuple(int(word,16) for word in row) for row in last['cubes']]
                    view = new_engine(problem,cubes)
                    snapshot = unpack_native(last,view)
                    snapshot['state'][1,0] = view.n+1
                    # Rehash both envelopes to exercise mechanical validation.
                    bad['native_queue'][-1] = pack_native(view,view._checkpoint_payload(snapshot['state']))
                else:
                    bad['active_stats']['decisions'] += 1
                bad['digest'] = digest({k:v for k,v in bad.items() if k != 'digest'})
                with self.assertRaises(ValueError):
                    pause_restore(problem,bad,2,{})

    def test_unrecognized_metadata_takes_canonical_repack_path(self):
        problem, initial = setup_saved()
        saved = pause_restore(problem,initial,2,{})['checkpoint']
        saved['native_queue'][0]['ignored_metadata'] = 'old private field'
        saved['digest'] = digest({k:v for k,v in saved.items() if k != 'digest'})
        result = pause_restore(problem,saved,2,{})['checkpoint']
        self.assertNotIn('ignored_metadata',result['native_queue'][0])

    def test_mixed_widths_fill_every_device_batch_and_preserve_terminal_ownership(self):
        for balanced in (False,True):
            problem, initial = setup_saved(balanced,True)
            opts = {'gpu_lane_balance':balanced,'gpu_propagated_forks':True}
            narrow = pause_restore(problem,initial,1,opts)['checkpoint']
            medium = pause_restore(problem,narrow,2,opts)['checkpoint']
            self.assertEqual([len(archive['cubes']) for archive in medium['native_queue']], [1,1])
            launched = []
            def track(*args,**kwargs):
                engine = new_engine(*args,**kwargs)
                launched.append(engine.lanes)
                return engine
            with patch('gpu_domain_round24.DomainCUDA',side_effect=track):
                completed = solve(problem,target='edges',replicas=2,resume=medium,options=opts)
            self.assertEqual(launched,[2,2])
            self.assertEqual((completed['status'],completed['nodes']),('infeasible',22))
            self.assertEqual(completed['stats']['gpu_domain_completed'],4)
            if balanced:
                self.assertEqual(completed['stats']['gpu_lane_owners_completed'],1)
