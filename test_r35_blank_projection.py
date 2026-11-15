"""Independent finite controls for deferred blank matchings and complete geographic lifts."""
from collections import Counter
from importlib.util import find_spec
from itertools import combinations
import random
import unittest
from unittest.mock import patch

from r27_structure_filters import domain_bits
from r27_structure_fixtures import independent, planted
from r30_gold import GoldMatching, search
from r31_geographic import physical_bonds
from r35_blank_projection import BlankProjection, ProjectionTimeout, lift
from test_r28_geometry import phases
from test_r30_exactgold import bridge_fixture


class BlankMatchingTests(unittest.TestCase):
    def test_signed_and_forced_blank_matching_agrees_with_unconditioned_enumeration(self):
        from test_r28_geometry import matchings
        rng = random.Random(351)
        sides = [(0,rng.randrange(2)) for _ in range(12)]
        complete = list(matchings(tuple(range(12))))
        for _ in range(24):
            possible = {p for p in combinations(range(12),2)
                        if p[0]//3 != p[1]//3 and rng.random() < .75}
            wanted = set(rng.sample(sorted(possible),rng.randrange(3)))
            values = [rng.randrange(2) for _ in range(4)]
            oracle = [set(map(tuple,row)) for row in complete if wanted<=set(row)<=possible
                      and all((sides[a][1]^values[a//3]) == (sides[b][1]^values[b//3]) for a,b in row)]
            projector = BlankProjection(sides,possible,wanted)
            ans = projector.matching(projector.signed(values),float('inf'),Counter())
            self.assertEqual(ans is not None,bool(oracle))
            if ans is not None:
                self.assertIn(set(ans),oracle)

    def test_deadline_and_contradictory_restrictions(self):
        sides = [(0,0)]*6
        possible = set(combinations(range(6),2))
        with self.assertRaises(ProjectionTimeout):
            BlankProjection(sides,possible,[]).matching(possible,0,Counter())
        self.assertFalse(BlankProjection(sides,possible,[(0,3),(0,4)]).possible)
        self.assertFalse(BlankProjection(sides,possible-{(0,3)},[(0,3)]).possible)


@unittest.skipUnless(find_spec('ortools'), 'optional CP backend')
class BlankProjectionTests(unittest.TestCase):
    def setUp(self):
        self.p,self.ds = planted(1,3,351)
        self.codes = [3*t for t in range(self.p.n)]
        self.pairs = physical_bonds(self.p,self.codes)
        self.pins = dict(enumerate(phases(self.p)))
        self.nonblank = [pair for pair in self.pairs if self.p.masks[3*(pair[0]//3),pair[0]%3]]
        self.blank = sorted(set(self.pairs)-set(self.nonblank))

    def test_partial_requests_and_physical_materialization(self):
        for wanted in (self.nonblank[::2], self.nonblank):
            result = search(self.p,domains=self.ds,pins=self.pins,bonds=wanted,project_blanks=True)
            self.assertEqual('CONNECTED_SEWING',result['status'])
            self.assertTrue(set(wanted)<=set(result['pairs']))
            self.assertEqual(sorted(d for pair in result['pairs'] for d in pair), list(range(3*self.p.n)))
            check = GoldMatching(self.p,self.ds,self.pins,wanted)
            self.assertTrue(set(result['pairs'])<=check.mate.keys())
        gold = independent(self.p,self.ds)
        answer = search(self.p,domains=self.ds,pins=self.pins,bonds=self.blank[:1],project_blanks=True,geography=True)
        self.assertEqual('GOLD',answer['status'])
        self.assertIn(tuple(answer['codes']),gold)
        self.assertEqual(physical_bonds(self.p,answer['codes']),answer['pairs'])

    def test_geographic_completion_uses_actual_blank_joins(self):
        ds = list(domain_bits(self.p))
        ds[0] = 1
        args = dict(domains=tuple(ds),pins=self.pins,bonds=self.nonblank,forbidden=[self.blank[0]],
                    physical_support=False,project_blanks=True,seconds=5)
        loose = search(self.p,**args)
        self.assertEqual(loose['status'],'CONNECTED_SEWING')
        self.assertNotIn(self.blank[0],loose['pairs'])
        self.assertEqual(search(self.p,geography=True,**args)['status'],'INFEASIBLE')
        self.assertEqual(search(self.p,geography=True,**(args|{'forbidden':[]}))['status'],'GOLD')

    def test_blank_deficiency_blocks_only_checked_phase(self):
        ds = domain_bits(self.p)
        model = GoldMatching(self.p,ds,self.pins,self.nonblank,physical_support=False,project_blanks=True)
        blanks = model.blank_projection
        left = set(blanks.blanks[:5])
        banned = [pair for pair in blanks.allowed if len(left.intersection(pair))==1]
        args = dict(domains=ds,pins=self.pins,bonds=self.nonblank,forbidden=banned,physical_support=False)
        self.assertEqual(search(self.p,**args)['status'],'INFEASIBLE')
        trial = search(self.p,**args,project_blanks=True)
        self.assertEqual(trial['status'],'INFEASIBLE')
        self.assertGreater(trial['blank_phase_failures'],0)
        self.assertEqual(trial['cp_calls'],1)

    def test_incomplete_circuits_and_interruption(self):
        model = GoldMatching(self.p,self.ds,self.pins,(),project_blanks=True)
        blanks = model.blank_projection
        signed = blanks.signed(self.pins)
        with self.assertRaisesRegex(ValueError,'partial nonblank'):
            lift(self.p,self.ds,self.nonblank[:-1],blanks,signed,float('inf'),Counter())
        with self.assertRaises(ProjectionTimeout):
            lift(self.p,self.ds,self.nonblank,blanks,signed,0,Counter())
        with patch('r35_blank_projection.lift',side_effect=ProjectionTimeout):
            out = search(self.p,domains=self.ds,bonds=self.nonblank,geography=True,project_blanks=True)
            self.assertEqual((out['status'],out['failed_embeddings']),('UNKNOWN',0))
        from r27_structure_fixtures import nested_loops
        from r29_gold_phase import EndpointCuts
        disconnected,frozen = nested_loops()
        sides = EndpointCuts(disconnected).sides
        pairs = physical_bonds(disconnected,[3*t for t in range(disconnected.n)])
        active = [pair for pair in pairs if sides[pair[0]][0]]
        other = BlankProjection(sides,pairs,[])
        with self.assertRaisesRegex(ValueError,'Disconnected nonblank'):
            lift(disconnected,frozen,active,other,other.allowed,float('inf'),Counter())
        p,_ = planted(1,0,23,uniform_ports=True)
        _,forbid,required,pins = bridge_fixture(p)
        args=dict(bonds=required,forbidden=forbid,pins=pins,physical_support=False)
        for projection in (False,True):
            out=search(p,**args,project_blanks=projection)
            self.assertEqual(out['status'],'INFEASIBLE')
            self.assertGreater(out['strand_cuts'],0)

    def test_requested_full_side_channels_and_retained_plain_search(self):
        args=dict(domains=self.ds,pins=self.pins,bonds=self.nonblank,geography=True,channel=True)
        control = search(self.p,**args)
        full = search(self.p,**args,project_blanks=True)
        self.assertEqual(full['status'],'GOLD')
        self.assertEqual(full['channel'],control['channel'])
        self.assertEqual(full['partner_variables'],control['partner_variables'])
        self.assertIn('blank_projection_fallback',full)
        with self.assertRaises(ValueError):
            search(self.p,matching=False,project_blanks=True)
        with self.assertRaisesRegex(ValueError,'must have gold segments'):
            from search_problem import SearchProblem
            SearchProblem([dict(id=t,segments=[] if t==0 else self.p.tiles[t]['segments'])
                           for t in range(self.p.n)],self.p.board)
