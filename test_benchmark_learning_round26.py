"""Harness guards only; actual finite families run separately from frozen input."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import benchmark_learning_round26 as harness

class LearningHarnessTests(unittest.TestCase):
    def invoke(self,path,stage='development',sources=None,repo=None):
        sources=[{'a':'same'},{'a':'same'}] if sources is None else sources
        repo={'available':True,'commit':'abc','dirty':False,'status':[]} if repo is None else repo
        with patch.object(harness,'RUNNERS',{'F02':lambda:{'complete':True}}),patch.object(harness,'source_snapshot',side_effect=sources),patch.object(harness,'runtime_snapshot',return_value={'runtime':'same'}),patch.object(harness,'repository_snapshot',return_value=repo):
            return harness.main(['--family','F02','--output',str(path),'--evidence-stage',stage])
    def test_fresh_output_and_guarded_report(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'out.json';report=self.invoke(path,'frozen')
            self.assertEqual(report['status'],'passed');self.assertEqual(json.loads(path.read_text())['families'],{'F02':{'complete':True}})
            with self.assertRaises(ValueError):self.invoke(path)
    def test_changed_source_and_dirty_frozen_refuse_publication(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'out.json'
            with self.assertRaisesRegex(RuntimeError,'changed'):self.invoke(path,sources=[{'a':1},{'a':2}])
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(RuntimeError,'clean'):self.invoke(path,'frozen',repo={'available':True,'commit':'abc','dirty':True})
            self.assertFalse(path.exists())
    def test_family_failure_cannot_write_pass(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'out.json'
            def failure():raise RuntimeError('incomplete finite case')
            with patch.object(harness,'RUNNERS',{'F02':failure}),patch.object(harness,'source_snapshot',return_value={}),patch.object(harness,'runtime_snapshot',return_value={}),patch.object(harness,'repository_snapshot',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'incomplete'):harness.main(['--output',str(path)])
            self.assertFalse(path.exists())
    def test_optimized_runtime_guard_precedes_family_work(self):
        with patch.object(harness,'require_assertions',side_effect=RuntimeError('optimized')),patch.object(harness,'source_snapshot') as source:
            with self.assertRaisesRegex(RuntimeError,'optimized'):harness.main([])
            source.assert_not_called()

if __name__=='__main__':unittest.main()
