"""Exhaust the residual blank matching rule and compare complete physical search."""
from functools import lru_cache
from importlib.util import find_spec
from itertools import combinations
from unittest import TestCase, skipUnless

from r35_mip_pairing import blank_completion, sparse_forbidden


def brute(vertices,bans):
    @lru_cache(None)
    def solve(remaining):
        if not remaining:
            return True
        a=remaining[0]
        return any((a,b) not in bans and solve(tuple(x for x in remaining[1:] if x!=b))
                   for b in remaining[1:])
    return solve(tuple(sorted(vertices)))


class BlankFamilyTests(TestCase):
    def test_exhaustive_sparse_forbidden_and_residual_graphs(self):
        sides=tuple(range(6))
        edges=list(combinations(sides,2))
        for mask in range(1<<len(edges)):
            banned=frozenset(e for j,e in enumerate(edges) if mask>>j&1)
            if not sparse_forbidden(banned):
                continue
            for n in (0,2,4,6):
                for rest in combinations(sides,n):
                    expected=brute(rest,banned)
                    self.assertEqual(expected,n!=2 or rest not in banned)
                    chosen=blank_completion(rest,banned)
                    self.assertEqual(expected,chosen is not None)
                    if chosen is not None:
                        self.assertEqual(sorted(v for e in chosen for v in e),list(rest))
                        self.assertTrue(banned.isdisjoint(chosen))
        self.assertFalse(sparse_forbidden({(0,1),(0,2),(0,3)}))
        self.assertFalse(sparse_forbidden({(0,1),(0,2),(1,2)}))
        self.assertIsNone(blank_completion([0],set()))

    def test_comparison_rows_accept_exactly_the_completable_even_residuals(self):
        from bench_r35_blank_rows import post_rows
        sides=tuple(range(6))
        # The real solver already fixed the even parity of each color from
        # signed inventory. Test the optional redundant rows with that premise.
        class FixedEngine:
            def __init__(self, halves):
                self.halves=iter(halves)
                self.rows=[]
            def IntVar(self,*_):
                return next(self.halves)
            def Add(self,row):
                self.rows.append(row)
        for banned in (set(), {(0,1),(2,3),(4,5)}, {(0,1),(2,3),(4,5),(0,2)}):
            for size in (0,2,4,6):
                for remaining in combinations(sides,size):
                    active=[int(v in remaining) for v in sides]
                    flags=[active,[0]*len(active)]
                    expected=brute(remaining,banned)
                    for parity in (0,1,2):
                        for pairs,neighbors in (('weak',False),('hull',False),('weak',True),
                                                ('hull',True),('none',True)):
                            engine=FixedEngine([size//2,0])
                            post_rows(engine,flags,sides,banned,parity,pairs,neighbors)
                            self.assertEqual(all(engine.rows),expected)

    def test_general_even_size_and_both_color_buckets(self):
        sides=tuple(range(46))
        banned={(2*i,2*i+1) for i in range(23)} | {(0,44)}
        for even in range(0,47,2):
            for group in (sides[:even],sides[even:]):
                pairs=blank_completion(group,banned)
                self.assertEqual(pairs is not None,len(group)!=2 or group not in banned)
                if pairs is not None:
                    self.assertTrue(banned.isdisjoint(pairs))
                    self.assertEqual(sorted(v for e in pairs for v in e),list(group))

    def test_original_pinned_catalog_and_complement_are_checked_before_native_search(self):
        from pathlib import Path
        from random import Random
        from unittest.mock import patch
        from bench_r35_loss_boundary344 import problem_scope
        from verify_r35_contact_wide import read_rows
        from r32_gold_bonds import inventory
        from r35_mip_pairing import decide, published_witness
        tiles=next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if tiles is None:
            self.skipTest('Original prepared pieces unavailable')
        p,ref,base,req=problem_scope(tiles)
        docs=Path(__file__).resolve().parent/'docs'
        groups=[*base[:13],*read_rows(docs/'R35_GEOGRAPHIC_RESIDUALS.json','geographic_cores'),
                *read_rows(docs/'R35_EDGE_CONTACTS.json','edge_perfect_cores')]
        allowed=inventory(p).possible
        original=0xd237aba7e91101e88041413e8c589902021e5126
        mask=(1<<p.n)-1
        order=list(range(1,p.n))
        Random(35590+ord('c')).shuffle(order)
        fixed={0,*order[112:]}
        for chosen in (original,original^mask):
            pins={i:chosen>>i&1 for i in fixed}
            with patch('r35_mip_pairing.build_model',side_effect=AssertionError('Redundant physical search')):
                result=decide(p,ref,groups,req,37,1,published=True,pins=pins,sphere=True)
            self.assertEqual(result['phase'],hex(chosen))
            self.assertEqual(result['loss_cost'],37)
            self.assertEqual(result['work']['nodes'],0)
            self.assertEqual(len(result['pairs']),240)
        exact={i:original>>i&1 for i in range(p.n)}
        self.assertIsNone(published_witness(p,ref,req,groups,36,allowed,True,exact))
        self.assertIsNone(published_witness(p,ref,req,groups,37,allowed-{next(iter(req))},True,exact))

    def test_two_actual_blank_choices_preserve_named_losses_and_strand_cycles(self):
        import json
        from pathlib import Path
        from bench_r35_loss_boundary344 import problem_scope
        from r32_gold_bonds import inventory
        from r35_loss_witness import raw_sides
        from r35_phase_sewing import strand_components
        tiles=next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if tiles is None:
            self.skipTest('Original prepared pieces unavailable')
        p,reference,_,req=problem_scope(tiles)
        saved=json.loads((Path(__file__).resolve().parent/'docs/R35_MIXED_MIP_DONOR.json').read_bytes())
        pairs=set(map(tuple,saved['first_pairing']))
        phase=int(saved['phase'],16)
        blanks=[i for i,(slots,_) in enumerate(raw_sides(p.tiles)) if not slots]
        labels=raw_sides(p.tiles)
        choices=sorted(pair for pair in pairs-reference-req if pair[0] in blanks)
        allowed=inventory(p).possible
        def color(side):
            return labels[side][1]^((phase>>(side//3))&1)
        success=False
        for first,second in combinations(choices,2):
            (a,b),(c,d)=first,second
            replacement={tuple(sorted((a,c))),tuple(sorted((b,d)))}
            if not replacement<=allowed or replacement&reference or color(a)!=color(c):
                continue
            new=pairs-{first,second}|replacement
            self.assertEqual(reference-new,reference-pairs)
            self.assertEqual(strand_components(p,new),strand_components(p,pairs))
            success=True
            break
        self.assertTrue(success)


@skipUnless(find_spec('ortools'),'physical SCIP comparison needs optional OR-Tools')
class PhysicalSolverTests(TestCase):
    def test_variable_phases_and_owned_losses_survive_projection(self):
        from test_r33_gold_refine import fixture
        from r35_mip_pairing import decide
        for split,uniform in ((False,False),(True,True),(True,False)):
            p,ref,groups,req,_=fixture(split,uniform)
            for cap in (0,4,15):
                outcomes=[]
                for mode in ('explicit','blank','flow','both'):
                    row=decide(p,ref,groups,req,cap,0,sphere=True,mode=mode)
                    outcomes.append(row['status']!='INFEASIBLE')
                    if outcomes[-1]:
                        self.assertEqual(row['loss_cost'],len(set(ref)-{tuple(v) for v in row['pairs']}))
                        self.assertLessEqual(row['loss_cost'],cap)
                        self.assertEqual(len(row['pairs']),3*p.n//2)
                        if row['status']=='GOLD':
                            self.assertTrue(p.validate(row['codes'])['valid'])
                self.assertEqual(len(set(outcomes)),1)

    def test_absolute_phase_pins_preserve_both_complements(self):
        from r35_mip_pairing import decide
        from test_r33_gold_refine import fixture
        p,ref,groups,req,_=fixture(False,False)
        for pins in ({0: 1}, {p.n-1: 1}, {0: 0}):
            with self.subTest(pins=pins):
                row=decide(p,ref,groups,req,0,4,sphere=True,pins=pins)
                self.assertEqual(row['status'],'GOLD')
                phase=int(row['phase'],16)
                self.assertTrue(all(phase>>i&1==value for i,value in pins.items()))
        conflict=decide(p,ref,groups,req,0,4,sphere=True,pins={0: 1, p.n-1: 0})
        self.assertEqual(conflict['status'],'INFEASIBLE')
        for pins in ({-1: 0}, {p.n: 1}, {0: 2}, {'0': 0}):
            with self.subTest(pins=pins), self.assertRaises(ValueError):
                decide(p,ref,groups,req,0,4,pins=pins)

    def test_dense_nonmatching_bans_fall_back(self):
        from ortools.linear_solver import pywraplp
        from r35_mip_pairing import project_blanks
        sides=(0,3,6,9,12,15)
        engine=pywraplp.Solver.CreateSolver('SCIP')
        phase=[engine.BoolVar('') for _ in range(6)]
        legal=set(combinations(sides,2))-{(0,3),(0,6),(0,9)}
        self.assertIsNone(project_blanks(engine,phase,{},[((),0)]*16,sides,legal,set(),set()))

    def test_unaffordable_damaged_and_wrong_signed_pairs(self):
        from r35_mip_pairing import decide, physical_witness
        from r32_gold_bonds import inventory
        from test_r33_gold_refine import fixture
        p,ref,groups,req,_=fixture(False,False)
        result=decide(p,ref,groups,req,0,0,sphere=True)
        phase=int(result['phase'],16)
        pairs=[tuple(e) for e in result['pairs']]
        allowed=inventory(p).possible
        checked=lambda value,rows,available:physical_witness(p,value,rows,set(ref),set(req),groups,0,available,sphere=True)
        self.assertIsNotNone(checked(phase,pairs,allowed))
        self.assertIsNone(checked(phase,pairs[:-1]+[pairs[0]],allowed))
        self.assertIsNone(checked(phase,pairs,allowed-{pairs[0]}))
        self.assertIsNone(checked(1<<p.n,pairs,allowed))
