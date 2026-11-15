"""Positive and adverse controls for explicitly requested geometric ranked-loss evidence."""
import json
from pathlib import Path
import tempfile
import unittest

from bench_r35_loss_boundary344 import problem_scope
from r27_structure_fixtures import planted
from r31_geographic import physical_bonds
from r32_gold import run_syndrome
from r35_geographic_packing_check import check_packing
from r35_geographic_replay import geographic_departures
from r35_loss_witness import physical_price
from test_r35_geographic_replay import triangle


class ConditionalGeographyTests(unittest.TestCase):
    def test_exhaustive_plants_keep_actual_gold_and_condition_departures(self):
        for seed in (35,355,815):
            p,_=planted(1,2,seed,False,True)
            codes=[3*t for t in range(p.n)]
            self.assertTrue(p.validate(codes)['valid'])
            actual=set(physical_bonds(p,codes))
            core=triangle(p)
            self.assertFalse(check_packing(p,core)['feasible'])
            self.assertTrue(check_packing(p,actual)['feasible'])
            self.assertIsNone(check_packing(p,actual,max_steps=0)['feasible'])
            outside=min(core);ref=core-{outside}
            self.assertFalse(geographic_departures(p,ref,set(),[core]))
            groups=geographic_departures(p,ref,{outside},[core])
            self.assertEqual(groups,(frozenset(ref),))
            phases=range(0,1<<p.n,2)
            def minimum(reference,charged,required):
                return min(row[0] for phase in phases if (row:=physical_price(p.tiles,reference,charged,required,phase)) is not None)
            self.assertEqual(minimum(ref,[],{outside}),0)
            self.assertEqual(minimum(ref,groups,{outside}),1)
            good=geographic_departures(p,actual,set(),[actual,core])
            self.assertFalse(good)
            self.assertEqual(minimum(actual,good,set()),0)

    def original(self):
        tiles=next((q for q in (Path('data/tiles.json'),Path('../data/tiles.json')) if q.is_file()),None)
        if tiles is None:self.skipTest('Original physical dataset unavailable')
        p,ref,groups,req=problem_scope(tiles)
        h=json.loads((Path(__file__).parent/'docs/R35_GEOGRAPHIC_LOSS_CORES.json').read_bytes())['geographic_cores']
        return p,ref,groups[:13],req,h

    def test_ranked_positives_reprice_and_pure_scope_remains_independent(self):
        p,ref,groups,req,hints=self.original()
        options=dict(refine_integer=True,all_groups=True)
        plain=run_syndrome(p,ref,groups,21,None,':memory:',0,**options)
        stronger=run_syndrome(p,ref,groups,22,None,':memory:',0,
                              geographic_hints=hints,**options)
        self.assertEqual(plain['loss_cost'],21)
        self.assertNotIn('verified_geographic_departures',plain)
        self.assertEqual(stronger['loss_cost'],22)
        self.assertEqual(stronger['verified_geographic_departures'],2)
        projected=geographic_departures(p,ref,req,hints)
        raw=physical_price(p.tiles,ref,[*groups,*projected],req,int(stronger['phase'],16))
        self.assertEqual(raw[0],stronger['loss_cost'])
        # An actually embeddable partial bond set cannot supply a geometry claim.
        actual=next(iter(ref))
        ignored=run_syndrome(p,ref,groups,21,None,':memory:',0,
                             geographic_hints=[[actual]],**options)
        self.assertEqual(ignored,plain)

    def test_real_negative_pages_transfer_one_way_and_hold_original_rank_count(self):
        p,ref,groups,req,hints=self.original()
        with tempfile.TemporaryDirectory() as folder:
            def run(which,h):
                return run_syndrome(p,ref,groups,22,[2],str(Path(folder)/which),0,
                                    refine_integer=True,all_groups=True,geographic_hints=h)
            weak=run('forward.db',())
            reuse=run('forward.db',hints)
            strong=run('reverse.db',hints)
            reverse=run('reverse.db',())
            for x in (weak,reuse,strong,reverse):
                self.assertEqual(x['status'],'INFEASIBLE')
                self.assertEqual(x['completed_prefixes'],1)
            self.assertEqual(reuse['new_pages'],0)
            self.assertEqual(reuse['work'],{})
            for x in (weak,strong,reverse):
                self.assertGreater(x['work']['joint_queries'],0)
                self.assertEqual(x['new_pages'],1)


if __name__=='__main__':unittest.main()
