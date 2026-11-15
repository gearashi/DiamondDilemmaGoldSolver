"""Active signed-phase geometry and degree/matching checks across public and ranked searches."""
import json
from importlib.util import find_spec
from math import inf
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from bench_r35_loss_boundary344 import problem_scope,verify_21
from r28_exchange import read_json
from r32_gold import run_syndrome
from r35_signed_conditions import SignedConditions
from r35_signed_mate_support import matching


class SignedMateSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if not path:raise unittest.SkipTest('Original data unavailable')
        cls.path=path
        cls.p,cls.ref,cls.groups,cls.req=problem_scope(path)
        cls.first,_,_=verify_21(cls.p,cls.ref,cls.groups,cls.req)
        cls.second=int(read_json('docs/R35_OVERLAP_BOUNDARY.json')['disjoint20']['phase'],16)

    def test_reciprocal_row_and_hall_rejections_preserve_independent_full_sewings(self):
        c=SignedConditions(self.p,mates=True)
        self.assertFalse(c.accepts(self.second))
        self.assertFalse(c.rejecting)
        self.assertFalse(c.accepts(self.first))
        self.assertEqual(len(c.rejecting),2)
        self.assertEqual({len(rule) for rule in c.rejecting},{10})
        for data in read_json('docs/R35_CONNECTED_TOURS.json').values():
            phase=sum(v<<i for i,v in enumerate(data['phases']))
            self.assertTrue(c.accepts(phase))
            self.assertTrue(c.accepts(phase^((1<<self.p.n)-1)))
        self.assertGreater(c.cached_match.cache_info().hits,0)
        self.assertFalse(SignedConditions(self.p,euler=True).accepts(self.first))
        self.assertTrue(SignedConditions(self.p,euler=True).accepts(self.second))

    def test_hall_core_has_no_independent_partner_permutation(self):
        from r32_gold_bonds import inventory
        from r35_loss_witness import raw_sides,reciprocal
        c=SignedConditions(self.p,mates=True)
        self.assertFalse(c.accepts(self.first))
        raw=raw_sides(self.p.tiles);source=inventory(self.p).possible
        # Independently enumerate all physical pairings of the fourteen-side
        # family and test the two complete critical partial assignments.
        labels={raw[9][0],reciprocal(raw[9])[0]}
        family={i for i,label in enumerate(raw) if label[0] in labels}
        self.assertEqual(len(family),14)
        def has_pairs(rest,pins):
            if not rest:return True
            u=min(rest)
            for v in rest-{u}:
                if (u,v) not in source:continue
                if u//3 in pins and v//3 in pins:
                    a=(raw[u][0],raw[u][1]^pins[u//3])
                    b=(raw[v][0],raw[v][1]^pins[v//3])
                    if reciprocal(a)!=b:continue
                if has_pairs(rest-{u,v},pins):return True
            return False
        for clause in c.rejecting:
            pins={abs(v)-1:int(v<0) for v in clause}
            self.assertFalse(has_pairs(family,pins))
            for tile in pins:
                self.assertTrue(has_pairs(family,{t:v for t,v in pins.items() if t!=tile}))

    @unittest.skipUnless(find_spec('pysat'),'independent native SAT is optional')
    def test_incremental_native_rechecks_and_learns_exact_hall_core(self):
        from pysat.solvers import Solver
        from r35_integer_sat import InventorySAT
        p=self.p;model=InventorySAT.__new__(InventorySAT);model.n=p.n
        pins=[i if self.first>>(i-1)&1 else -i for i in range(1,p.n+1)]
        try:
            model.solver=Solver(name='g4');c=SignedConditions(p,mates=True);c.install(model)
            self.assertTrue(model.decide(0,pins))
            # Simulate another persistent instance learning this phase after
            # this native formula was created.
            self.assertFalse(c.accepts(self.first))
            self.assertFalse(c.decide(model,inf,pins));self.assertEqual(len(c.rejecting),2)
            self.assertFalse(model.decide(0,pins))
        finally:model.close()

    def test_published_negative_skips_sphere_and_partner_compilation(self):
        from unittest.mock import patch
        from r32_gold import run_syndrome
        with patch.object(SignedConditions,'_prepare',side_effect=AssertionError('unneeded compilation')):
            out=run_syndrome(self.p,self.ref,self.groups[:13],20,None,':memory:',.01,
                             refine_integer=True,all_groups=True,sphere_euler=True,mate_support=True)
        self.assertEqual(out['certified_loss_lower_bound'],21)

    def test_material_weak_and_combined_public_cli(self):
        cmd=[sys.executable,'-B','r32_gold.py','integer-bound','--tiles',str(self.path),'--upper','20','--all-groups','--seconds','.005',
             '--sphere-euler','--mate-support','--claims','docs/R35_GEOGRAPHIC_LOSS_CORES.json']
        row=json.loads(subprocess.run(cmd,capture_output=True,text=True,check=True).stdout)
        self.assertEqual((row['status'],row['work']['decisions']),('INFEASIBLE',0))
        self.assertTrue(all(row['signed_conditions'].values()))
        self.assertEqual(row['verified_geographic_departures'],2)
        sub=cmd[:];sub[sub.index('20')]='21'
        if find_spec('pysat'):
            row=json.loads(subprocess.run(sub,capture_output=True,text=True,check=True).stdout)
            self.assertNotEqual(row['status'],'INTEGER_RELAXATION_FEASIBLE')
        weak=[sys.executable,'-B','r32_gold.py','integer-bound','--tiles',str(self.path),'--all-groups','--upper','22','--sphere-euler','--seconds','.001']
        positive=json.loads(subprocess.run(weak,capture_output=True,text=True,check=True).stdout)
        self.assertEqual((positive['status'],positive['loss_cost']),('INTEGER_RELAXATION_FEASIBLE',22))

    def test_ranked_public_witness_does_not_escape_to_strengthened_scope(self):
        with TemporaryDirectory() as temp:
            db=str(Path(temp)/'ranks.sqlite')
            opts=dict(refine_integer=True,all_groups=True)
            plain=run_syndrome(self.p,self.ref,self.groups[:13],21,None,db,.001,**opts)
            self.assertEqual(plain['loss_cost'],21)
            for flags in (dict(sphere_euler=True),dict(mate_support=True)):
                filtered=run_syndrome(self.p,self.ref,self.groups[:13],21,None,db,.001,**opts,**flags)
                self.assertEqual(filtered['status'],'UNKNOWN')
                confirmed=run_syndrome(self.p,self.ref,self.groups[:13],20,None,db,.001,**opts,**flags)
                self.assertEqual((confirmed['status'],confirmed['certified_loss_lower_bound']),('INFEASIBLE',21))
            self.assertEqual(run_syndrome(self.p,self.ref,self.groups[:13],21,None,db,.001,**opts)['loss_cost'],21)


if __name__=='__main__':unittest.main()
