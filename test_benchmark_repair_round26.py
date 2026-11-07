"""Bounded real model comparison plus fail-closed evidence orchestration."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import benchmark_repair_round26 as benchmark

class RetainedRepairEvidenceTests(unittest.TestCase):
    def test_completed_ordinary_model_reuse_against_independent_optima(self):
        try:import ortools
        except ImportError:self.skipTest('OR-Tools unavailable')
        rows=benchmark.ordinary_finite();self.assertEqual(len(rows),2)
        for row in rows:
            old,warm,resumed=[row['variants'][name] for name in ('existing_local_models','retained_warm','retained_restore_after3')]
            self.assertTrue(row['same_completed_optima']);self.assertEqual(row['unique_contexts'],6)
            self.assertEqual(old['native']['model_constructions'],6)
            self.assertEqual(warm['native']['model_constructions'],1)
            self.assertEqual(resumed['native']['model_constructions'],2)
            self.assertEqual(warm['retained_stats']['retained_completed_hits'],1)
            self.assertGreater(resumed['checkpoint_bytes'][0],0)
    def test_frozen_dirty_guard_precedes_any_query(self):
        with tempfile.TemporaryDirectory() as temporary,patch.object(benchmark,'source_snapshot',return_value={}),patch.object(benchmark,'_runtime_snapshot',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={'available':True,'commit':'x','dirty':True}),patch.object(benchmark,'ordinary_finite',side_effect=AssertionError('query started')):
            with self.assertRaisesRegex(RuntimeError,'clean'):benchmark.main(['--output',str(Path(temporary)/'r.json'),'--evidence-stage','frozen'])
            self.assertFalse((Path(temporary)/'r.json').exists())
