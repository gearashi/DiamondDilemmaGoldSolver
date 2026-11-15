"""Complete directional tie responses, cap and byte gates for rotation components."""
import itertools
import unittest

import numpy as np

from r28_exchange_rotations import RotationOracle
from r35_rotation_components import translate_responses


def scored(rows,edges):
    return [(sum(int(v[x[a],x[b]]) for a,b,v in edges),x) for x in itertools.product(*rows)]


class ComponentTests(unittest.TestCase):
    def test_all_ties_under_axes_rotations_vertex_labels_and_large_numeric_inputs(self):
        tiles=np.arange(7)
        dims=[(2,0,1),(1,0,2),(0,2,1),(1,2,0),(2,1,0),(0,1,2),(1,)]
        rng=np.random.default_rng(980)
        base=[(0,1,rng.integers(0,2,(3,3),dtype=np.int32)),(1,2,rng.integers(0,2,(3,3),dtype=np.int32))]
        oracle=RotationOracle(None)
        cold=RotationOracle(None,components=False)
        def query(engine,edges,cap=27,ties=True,limit=None):
            result=engine._eliminate(tiles,dims,edges,cap,ties,limit)
            return None if result is None else (result[0],list(result[1]))
        for ix in range(20):
            order=rng.permutation(3)
            edges=[]
            for a,b,v in base:
                ra,rb=(dims[3+order[q]] for q in (a,b))
                moved=v[np.ix_(ra,rb)]+(65536 if ix==19 else ix%4)
                edges.append((3+order[b],3+order[a],moved.T))
                edges.append((a,b,v))
            # An additive one-ended edge changes a unary without joining the components.
            if ix%4==2:
                edges.append((5,6,np.array([[0,0,0],[1,1,1],[0,0,0]],np.int32)))
            expected=scored(dims,edges)
            high=max(n for n,_ in expected)
            winners={tuple(3*tiles+np.array(row)) for n,row in expected if n==high}
            got,all_witnesses=query(oracle,edges)
            self.assertEqual(got,high)
            self.assertEqual({tuple(v) for v in all_witnesses},winners)
            self.assertEqual((got,all_witnesses),query(cold,edges))
            self.assertEqual(query(oracle,edges,ties=False),(got,all_witnesses[:1]))
            self.assertEqual(query(oracle,edges,limit=2),(got,all_witnesses[:2]))
            self.assertIsNone(query(oracle,edges,1))
        self.assertGreater(oracle.work['component_hits'],0)
        self.assertGreater(oracle.work['component_vertex_hits'],0)

    def test_directional_mask_reindex_roundtrips_nontrivial_conditioning_axes(self):
        scopes=((1,2),(2,),())
        dims=((0,1),(0,1,2),(0,1))
        perms=((1,0),(2,0,1),(1,0))
        rng=np.random.default_rng(134)
        tables=[rng.integers(1,1<<len(dims[i]),tuple(len(dims[j]) for j in axes),dtype=np.uint8)
                for i,axes in enumerate(scopes)]
        moved=translate_responses(tables,scopes,perms)
        restored=translate_responses(moved,scopes,[tuple(np.argsort(p)) for p in perms])
        for a,b in zip(tables,restored):np.testing.assert_array_equal(a,b)

    def test_budget_includes_component_records_and_interruption_cannot_admit_partial_ties(self):
        calls=0
        def stop():
            nonlocal calls
            calls+=1
            if calls==6:raise InterruptedError()
        dims=[(0,1,2)]*6
        tiles=np.arange(6)
        edges=[(0,1,np.eye(3,dtype=np.int32)),(1,2,np.eye(3,dtype=np.int32)),
               (4,5,np.eye(3,dtype=np.int32))]
        o=RotationOracle(None,cache_bytes=8000,poll=stop)
        with self.assertRaises(InterruptedError):o._eliminate(tiles,dims,edges,27,True,None)
        o.poll=None
        high,wins=o._eliminate(tiles,dims,edges,27,True,None)
        candidates=scored(dims,edges)
        self.assertEqual(high,max(q for q,_ in candidates))
        self.assertEqual({tuple(x) for x in wins},{tuple(3*tiles+np.array(x)) for q,x in candidates if q==high})
        self.assertLessEqual(o.component_cache.bytes+o.cache_size+o.plan_bytes,8000)
        self.assertTrue(all(v is not None for v in o.component_cache.rows.values()))
        self.assertIsNone(RotationOracle(None,capacity=0).component_cache)
        self.assertEqual(RotationOracle(None,capacity=1).component_cache.limit,128)
        self.assertIsNone(RotationOracle(None,observe=lambda *_:None).component_cache)
        tiny=RotationOracle(None,cache_bytes=1)
        tiny._eliminate(tiles,dims,edges,27,True,None)
        self.assertEqual(tiny.component_cache.bytes+tiny.cache_size+tiny.plan_bytes,0)

    def test_shape_match_alone_and_changed_numeric_operand_do_not_alias(self):
        d=[(0,1,2)]*4
        tiles=np.arange(4)
        o=RotationOracle(None)
        g=np.eye(3,dtype=np.int32)
        for q in (g,1-g,g+65536,np.eye(3,dtype=np.int32)*65536,g):
            factors=[(0,1,q),(2,3,g)]
            score,rows=o._eliminate(tiles,d,factors,9,True,None)
            expected=scored(d,factors)
            self.assertEqual(score,max(n for n,_ in expected))
            self.assertEqual({tuple(w) for w in rows},{tuple(3*tiles+np.array(r)) for n,r in expected if n==score})
        self.assertGreater(o.work['component_hits'],0)
