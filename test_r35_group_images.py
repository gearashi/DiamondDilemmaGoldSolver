"""Verify grouped image costs by direct, labelled free-phase assignments."""
from itertools import product
from math import inf
from random import Random
import unittest
from collections import Counter
from types import SimpleNamespace

from bench_r35_group_boundaries import GroupBoundaries
from r35_group_circuits import GroupCircuits
from r35_group_drops import GroupDropTimeout, reuse_drops
from unittest.mock import patch

from r35_group_images import GroupImages
from test_r33_gold_lookup import fixture, phase_costs


def direct(groups, required, equations, outside, width):
    errors = 1 | sum(2 << i for i in outside)
    candidates = []
    for phase in range(1 << width):
        mismatches = {bond for bond, (row, symbol) in equations.items()
                      if ((phase & (row >> 1)).bit_count() + (symbol & errors).bit_count()) % 2}
        if mismatches & required or any(not (group - required) for group in groups):
            continue
        lost = mismatches.copy()
        for group in groups:
            if not lost & group:
                lost.add(min(group - required))
        candidates.append((len(lost), phase, lost))
    return min(candidates, key=lambda e:e[0]) if candidates else (inf, None, None)


class GroupImageTests(unittest.TestCase):
    def test_every_outside_pattern_and_reconstructed_optimum(self):
        rng = Random(3517)
        for trial in range(60):
            width = 1 + trial % 7
            groups = [set(range(0, 2)),set(range(2, 5)),set(range(5, 8))]
            required = {bond for bond in range(8) if rng.randrange(6)==0}
            rows = {bond:(rng.randrange(1 << width) << 1,rng.randrange(16)) for bond in range(8)}
            method = GroupImages.compile(groups, required, rows, 3, max_rank=width)
            query = SimpleNamespace(groups=groups,required=required,_joint_bonds=rows)
            boundary = GroupBoundaries.compile(query)
            circuit = GroupCircuits(query)
            for pattern in product((0,1),repeat=3):
                flipped = tuple(i for i,value in enumerate(pattern) if value)
                optimum, phase, lost = direct(groups, required, rows, flipped, width)
                self.assertEqual(method.minimum(flipped,inf), optimum, (trial,flipped))
                self.assertEqual(boundary.minimum(flipped), optimum)
                for budget in (3,6,9):
                    expected = optimum + len(flipped) <= budget
                    self.assertEqual(circuit.solve(flipped,budget,inf,Counter()),expected)
                    self.assertEqual(reuse_drops(query,flipped,budget,inf,Counter()),expected)
                if phase is not None:
                    self.assertFalse(lost & required)
                    self.assertTrue(all(g & lost for g in groups))
                    errors = 1 | sum(2 << i for i in flipped)
                    for bond in rows.keys()-lost:
                        row, symbol = rows[bond]
                        self.assertEqual((phase & (row >> 1)).bit_count() % 2, (symbol & errors).bit_count() % 2)

    def test_fallback_rank_gate_and_all_small_physical_costs(self):
        groups = [{i} for i in range(7)]
        rows = {i: (2 << i,1) for i in range(7)}
        self.assertIsNone(GroupImages.compile(groups, set(), rows, 0, max_rank=4))
        for seed in (1,11):
            p,reference,_,query = fixture(seed)
            truth = phase_costs(p,set(reference),query)
            for use_images in (True,False):
                if not use_images:
                    query._joint_images = None
                for bits in range(1 << len(query.outside)):
                    flipped = tuple(i for i in range(len(query.outside)) if bits>>i & 1)
                    for cap in range(len(reference)+1):
                        self.assertEqual(query.joint(flipped,cap,inf),truth.get(flipped,inf)<=cap)

    def test_cache_reuses_only_the_grouped_cost_and_interruption_is_retryable(self):
        # Either outside bit can toggle the same free equation, while only
        # the second outside bit toggles the independent constant equation.
        rows = {0:(2,2|4),1:(0,4),2:(0,4)}
        method = GroupImages.compile([{0,1,2}],set(),rows,2)
        self.assertEqual(method.minimum((),inf),1)
        self.assertEqual(method.minimum((0,),inf),1)
        self.assertEqual(method.cache_hits,1)
        with patch('r35_group_images.time.monotonic', side_effect=[1,3]):
            with self.assertRaises(TimeoutError):
                method.minimum((1,),2)
        self.assertEqual(len(method.minima),1)
        self.assertEqual(method.minimum((1,),inf),2)
        with self.assertRaises(TimeoutError):
            method.minimum((1,),0)
        self.assertEqual(GroupImages.compile([],set(),{},0).minimum((),inf),0)
        self.assertEqual(GroupImages.compile([{0}],{0},{0:(0,0)},0).minimum((),inf),inf)


    def test_inner_cursor_preserves_positive_siblings_and_rejects_out_of_scope_tokens(self):
        _, _, _, q = fixture()
        q._joint_images = None
        initial = q.indexed(9,4,start=0,stop=1)
        self.assertEqual(initial['status'],'PHASE_RELAXATION_FEASIBLE')
        seen = []
        def wrapped(query, flipped, budget, deadline, work, *, resume, checkpoint):
            def stop(path):
                seen.append(path)
                if len(seen)==2:
                    raise GroupDropTimeout(path)
            return reuse_drops(query,flipped,budget,deadline,work,resume=resume,checkpoint=stop)
        with patch('r32_gold_syndrome.reuse_drops',wrapped):
            paused = q.indexed(9,4,start=0,stop=1)
        self.assertEqual(paused['status'],'UNKNOWN')
        self.assertEqual(paused['cursor'][0][-2],[1,list(seen[-1])])
        for cursor in (paused['cursor'],[paused['cursor'][0],'changed']):
            resumed = q.indexed(9,4,start=0,stop=1,cursor=cursor)
            self.assertEqual(resumed['outside_errors'],initial['outside_errors'])
        self.assertEqual(q.indexed(7,4,start=0,stop=1,cursor=paused['cursor'])['work'].get('suffix_cursors_resumed',0),0)
        self.assertEqual(reuse_drops(q,(),9,inf,Counter(),resume=[999999]),reuse_drops(q,(),9,inf,Counter()))


if __name__=='__main__':
    unittest.main()
