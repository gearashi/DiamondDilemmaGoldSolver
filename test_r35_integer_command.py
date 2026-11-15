"""Check the native integer CLI's reusable certified scope, conditional witnesses and opt-out."""
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from bench_r35_loss_proof import original
from r28_exchange import read_json
from r32_gold_loss import reference_groups
from r35_loss_bound import global_integer


class BoundReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path=next((f for f in (Path('data/tiles.json'),Path('../data/tiles.json')) if f.is_file()),None)
        if cls.path is None:raise unittest.SkipTest('Original physical tile data unavailable')
        cls.p,cls.ref,cls.packed,cls.req,cls.out=original(cls.path)
        _,_,allg=reference_groups(cls.p,read_json('docs/R29_INCUMBENT.json')['codes'])
        cls.extra=[g for g in allg if g not in cls.packed]

    def test_public_cli_known_cap_and_sparse_positive(self):
        for cap,overlap,expected in ((19,False,'INFEASIBLE'),(20,False,'INTEGER_RELAXATION_FEASIBLE'),(21,True,'INTEGER_RELAXATION_FEASIBLE')):
            cmd=[sys.executable,'-B','r32_gold.py','integer-bound','--tiles',str(self.path),'--upper',str(cap),'--seconds','0.01']
            if overlap:cmd.append('--all-groups')
            proc=subprocess.run(cmd,capture_output=True,text=True,check=True)
            result=json.loads(proc.stdout)
            self.assertEqual(result['status'],expected);self.assertEqual(result['work']['decisions'],0)
            if cap==19:self.assertEqual(result['certified_loss_lower_bound'],20)
            else:self.assertEqual(result['loss_cost'],cap)

    def test_real_raw_phase_checked_for_extra_constraints_and_every_uncovered_case_calls_native(self):
        p,ref,g,req,out=self.p,self.ref,self.packed,self.req,self.out
        def run(cap,**options):return global_integer(p,ref,g,req,out,cap,.05,**options)
        with patch('r35_integer_sat.InventorySAT',side_effect=AssertionError('native reached')):
            self.assertEqual(run(21,additional_groups=self.extra)['loss_cost'],21)
            for args in (dict(independent_native=True),dict(excess=True),dict(additional_groups=[set(req)])):
                with self.assertRaisesRegex(AssertionError,'native reached'):run(21,**args)
            with self.assertRaisesRegex(AssertionError,'native reached'):run(20,additional_groups=self.extra)
            with self.assertRaisesRegex(AssertionError,'native reached'):global_integer(p,ref,g,req,out[:-1],19,.01)
            with self.assertRaisesRegex(AssertionError,'native reached'):global_integer(p,ref,list(reversed(g)),req,out,19,.01)
            with self.assertRaisesRegex(AssertionError,'native reached'):global_integer(p,ref,g,req,out+out[:1],19,.01)
            with self.assertRaisesRegex(AssertionError,'native reached'):run(19,additional_groups=[{(888,999)}])
        for cap,sec in ((-1,0),(False,0),(20,float('nan'))):
            with self.assertRaises(ValueError):global_integer(p,ref,g,req,out,cap,sec)


if __name__=='__main__':unittest.main()
