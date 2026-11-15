"""Independent score, domain, witness and cap controls for rotation-graph reuse."""
from collections import Counter
import itertools
import math
import random
import unittest

import numpy as np

from r28_exchange_rotations import RotationOracle
from r35_rotation_graph import RotationThreshold


def table(number):
    return np.array([(number >> i) & 1 for i in range(9)],dtype=np.int16).reshape(3,3)


def full_stars(rows, edges):
    scores=[[0]*3 for _ in rows]
    for a,b,grid in edges:
        for r in rows[a]: scores[a][r]+=any(grid[r,t] for t in rows[b])
        for t in rows[b]: scores[b][t]+=any(grid[r,t] for r in rows[a])
    return sum(max(scores[c][r] for r in row) for c,row in enumerate(rows))


def full_reduce(domains, edges, floor):
    rows=list(domains)
    if full_stars(rows,edges)<2*floor: return None
    while True:
        before=rows.copy()
        for c,old in enumerate(rows):
            if len(old)<2: continue
            keep=[]
            for r in old:
                rows[c]=(r,)
                if full_stars(rows,edges)>=2*floor: keep.append(r)
            if not keep: return None
            rows[c]=tuple(keep)
            if full_stars(rows,edges)<2*floor: return None
        if before==rows:return rows


def all_scores(domains, edges):
    return [(sum(int(g[row[a],row[b]]) for a,b,g in edges),row) for row in itertools.product(*domains)]


def naive_plan(widths,pairs):
    factors=list(pairs)+[(i,) for i in range(len(widths))]
    remain=set(range(len(widths)))
    out=[]
    while remain:
        scopes={v:{i for f in factors if v in f for i in f} for v in remain}
        cell=min(remain,key=lambda v:(math.prod(widths[i] for i in scopes[v]),v))
        scope=tuple(sorted(scopes[cell]));out.append((cell,scope,math.prod(widths[i] for i in scope)))
        factors=[f for f in factors if cell not in f]+[tuple(set(scope)-{cell})]
        remain.remove(cell)
    return tuple(out)


class RotationGraphTests(unittest.TestCase):
    def test_distant_component_wakes_after_total_decreases(self):
        pairs=[(0,1),(1,2),(2,0),(3,4),(4,5),(5,6),(6,3),(3,5)]
        edges=[(a,b,table(code)) for (a,b),code in zip(pairs,(237,35,368,76,329,136,200,371))]
        rows=[(0,1,2),(0,1,2),(0,1),(0,2),(1,2),(0,1,2),(0,1,2)]
        work=Counter(); calculator=RotationThreshold(rows,edges,{},work)
        self.assertEqual(calculator.reduce(7),[(1,2),(0,),(1,),(0,),(2,),(0,2),(1,2)])
        self.assertGreater(work['threshold_remote_events'],0)
        self.assertGreater(work['threshold_condition_hits'],0)
        for floor in (8,6,0,7,5,7.25,float('-inf'),8):
            self.assertEqual(calculator.reduce(floor),full_reduce(rows,edges,floor))

    def test_every_surviving_witness_and_fixpoint_across_thresholds(self):
        rng=random.Random(42195)
        edgesets=[[(0,1),(1,2),(2,0),(3,4)],[(0,1),(0,1),(1,2),(2,3),(3,0)],[],[(0,0),(0,1),(1,2)]]
        for i in range(24):
            n=5;rows=[tuple(r for r in range(3) if m&(1<<r)) for m in (rng.randrange(1,8) for _ in range(n))]
            edges=[(a,b,table(rng.randrange(512))) for a,b in edgesets[i%len(edgesets)]]
            calculator=RotationThreshold(rows,edges,{},Counter());states=all_scores(rows,edges)
            for floor in reversed(range(len(edges)+2)):
                keep=calculator.reduce(floor)
                self.assertEqual(keep,full_reduce(rows,edges,floor))
                for score,witness in states:
                    if score>=floor:
                        self.assertIsNotNone(keep)
                        self.assertTrue(all(v in d for v,d in zip(witness,keep)))

    def test_all_two_rotation_path_tables(self):
        rows=[(0,1)]*3
        tables=[np.pad(np.array([(m>>i)&1 for i in range(4)],dtype=np.int16).reshape(2,2),((0,1),(0,1))) for m in range(16)]
        for a,b in itertools.product(tables,repeat=2):
            edges=[(0,1,a),(1,2,b)]
            oracle=RotationThreshold(rows,edges,{},Counter())
            states=all_scores(rows,edges)
            for floor in (3,2,1,0):
                result=oracle.reduce(floor)
                self.assertEqual(result,full_reduce(rows,edges,floor))
                self.assertTrue(all(result and all(v in d for v,d in zip(witness,result))
                                    for score,witness in states if score>=floor))

    def test_factor_budget_and_interrupted_numeric_trace(self):
        tiles=np.arange(7);domains=[(0,1,2)]*len(tiles)
        edges=[(i,i+1,table(273)) for i in range(len(tiles)-1)]
        class Stop(Exception):pass
        def stop(*_): raise Stop()
        oracle=RotationOracle(None,cache_bytes=2300,observe=stop)
        with self.assertRaises(Stop):
            oracle._eliminate(tiles,domains,edges,81,True,None)
        self.assertLessEqual(oracle.plan_bytes+oracle.cache_size,2300)
        oracle.observe=None
        score,answers=oracle._eliminate(tiles,domains,edges,81,True,None)
        self.assertEqual(score,6);self.assertEqual(len(list(answers)),3)
        self.assertLessEqual(oracle.plan_bytes+oracle.cache_size,2300)

    def test_capped_plan_extends_without_repricing_old_work_and_obeys_caps(self):
        widths=[3]*7; pairs=[(0,1),(0,2),(1,2),(2,3),(4,5)]
        expected=naive_plan(widths,pairs);oracle=RotationOracle(None)
        cap=1; first=oracle._plan(widths,pairs,cap)
        self.assertEqual(first,expected[:1]);self.assertEqual(oracle.work['plan_neighbor_products'],len(widths))
        self.assertEqual(oracle._plan(widths,list(reversed(pairs)),cap),first)
        for cap in (3,9,27,81):
            plan=oracle._plan(widths,pairs,cap)
            stop=next((i+1 for i,s in enumerate(expected) if s[2]>cap),len(widths))
            self.assertEqual(plan[:stop],expected[:stop])
        cold=RotationOracle(None);self.assertEqual(cold._plan(widths,pairs,81),expected)
        self.assertEqual(oracle.work['plan_neighbor_products'],cold.work['plan_neighbor_products'])
        self.assertGreater(oracle.work['plan_extensions'],0)
        self.assertEqual(oracle._plan([1,*widths[1:]],pairs,81),naive_plan([1,*widths[1:]],pairs))
        self.assertTrue(0<oracle.plan_bytes<=oracle.plan_limit)
        bounded=RotationOracle(None,cache_bytes=8);bounded._plan(widths,pairs,81)
        self.assertEqual(bounded.plan_bytes,0)

    def test_tied_assignments_and_directional_frames_with_reused_structure(self):
        tiles=np.arange(6);dims=[(2,0,1)]*5+[(2,)]
        pairs=[(1,0),(2,1),(2,0),(4,3)];obs=[]
        oracle=RotationOracle(None,observe=lambda key,size,product:obs.append(product))
        for shift in (0,1,2):
            edges=[(a,b,np.roll(np.eye(3,dtype=np.int16),shift,axis=1)) for a,b in pairs]
            all_rows=all_scores(dims,edges);opt=max(v for v,_ in all_rows)
            winners={tuple(3*tiles+row) for v,row in all_rows if v==opt}
            score,rows=oracle._eliminate(tiles,dims,edges,27,True,None)
            self.assertEqual((score,{tuple(row) for row in rows}),(opt,winners))
            self.assertIsNone(oracle._eliminate(tiles,dims,edges,2,True,None))
        self.assertGreater(oracle.work['plan_hits'],0);self.assertTrue(obs)


if __name__=='__main__': unittest.main()
