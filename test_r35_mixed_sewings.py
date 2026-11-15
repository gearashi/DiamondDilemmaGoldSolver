"""Check child-specific phases against direct enumeration and planted geographic Gold."""
from itertools import product
from importlib.util import find_spec
from unittest import TestCase,main,skipUnless
from types import SimpleNamespace
from unittest.mock import patch

from bench_r35_mixed_sewings import labels,local_geography,raw_cycles,ring_donors
from r27_structure_filters import domain_bits
from r35_alternating_sewings import Family,common_phases,complete
from r35_mixed_sewings import PhaseComponents,donor_geography,geographic_roots,mixed_solve


def accepts(f,phase,edges):
    return all((phase[a//3]^phase[b//3]) ==
               (f.inputs.sides[a][1]^f.inputs.sides[b][1]^(f.inputs.sides[a][0].bit_count()&1))
               for a,b in edges)


def exhaustive(f,excluded=(),pins=None):
    n=len(f.groups)+len(f.blanks)
    good={}
    for picks in product((0,1),repeat=n):
        if any(picks[i]!=value for i,value in (pins or {}).items()):continue
        num=sum(v<<i for i,v in enumerate(picks))
        edges=f.pairs(num%2**len(f.groups),num>>len(f.groups))
        phase=labels(f.problem,edges,f.inputs.sides)
        if (phase is not None and not any(accepts(f,p,edges) for p in excluded)
                and raw_cycles(f.problem,edges)==[f.problem.total_segments]):
            good[num]=phase
    return good


class MixedSewingTests(TestCase):
    def test_all_children_including_children_requiring_a_third_phase(self):
        for seed in (0,103,224,310,384,392):
            with self.subTest(seed=seed):
                p,parents,_=ring_donors(seed)
                common,state=common_phases(p,*parents)
                self.assertIsNone(common)
                self.assertEqual(state,'NO_COMMON_PHASE')
                f=Family(p,*parents,None)
                excluded=[r['phases'] for r in parents]
                for exclude in ((),excluded[:1],excluded):
                    expected=exhaustive(f,exclude)
                    if len(exclude)==2 and seed:self.assertTrue(expected)
                    for memo,look,repeat in ((False,0,False),(True,0,False),(True,2,False),(False,48,True),(True,48,False)):
                        row=mixed_solve(f,memo=memo,exclude_phases=exclude,lookahead=look,repeat_forcing=repeat)
                        self.assertTrue(row['complete'])
                        self.assertEqual(row['connected_full'],len(expected))
                        for example in row['examples']:
                            mask=example['mask']
                            self.assertIn(mask,expected)
                            edges=f.pairs(mask%2**len(f.groups),mask>>len(f.groups))
                            complete(p,edges,example['phases'],f.inputs)
                idx=len(f.groups)
                for side in (0,1):
                    pins={idx:side}
                    self.assertEqual(mixed_solve(f,fixed=pins)['connected_full'],len(exhaustive(f,pins=pins)))

    def test_geographic_gold_children_require_independent_tile_phases(self):
        for seed,example in ((2164,20),(3521,5),(5388,5)):
            with self.subTest(seed=seed):
                p,parents,_=ring_donors(seed)
                f=Family(p,*parents,None)
                excluded=[r['phases'] for r in parents]
                self.assertIn(example,exhaustive(f,excluded))
                result=mixed_solve(f,geography=True,exclude_phases=excluded)
                board=result['gold']
                self.assertEqual(result['geographic_status'],'GOLD')
                self.assertTrue(p.validate(board)['valid'])
                from r31_geographic import physical_bonds
                actual=physical_bonds(p,board)
                self.assertTrue(all(not accepts(f,phase,actual) for phase in excluded))
                self.assertIsNotNone(labels(p,actual,f.inputs.sides))
                self.assertEqual(raw_cycles(p,actual),[p.total_segments])

    def test_witnesses_geographic_roots_and_empty_or_limited_domains(self):
        p,parents,codes=ring_donors(103)
        f=Family(p,*parents,None)
        allowed={e for rec in parents for e in rec['pairs']}
        for board in codes:
            domains=tuple(1<<c for c in board)
            roots=geographic_roots(p,domains,allowed)
            oracle=local_geography(p,allowed,domains,max_tiles=p.n)
            self.assertEqual(roots['survivors'],oracle['survivors'])
            self.assertEqual(len(roots['survivors']),1)
            row=f.solve(geography=True,domains=domains)
            self.assertEqual(row['geographic_status'],'GOLD')
            self.assertEqual(list(row['gold']),board)
            self.assertTrue(p.validate(row['gold'])['valid'])
        unavailable=list(domain_bits(p))
        unavailable[0]=0
        row=f.solve(geography=True,domains=unavailable)
        self.assertEqual(row['geographic_status'],'INFEASIBLE')
        self.assertIsNone(row['connected_full'])
        self.assertEqual(row['work']['nodes'],0)
        limited=f.solve(max_nodes=0,geography=True)
        self.assertFalse(limited['complete'])
        self.assertEqual(limited['geographic_status'],'UNKNOWN')
        self.assertIsNone(limited['gold'])
        half=f.solve(max_nodes=1)
        self.assertFalse(half['complete'])
        self.assertLessEqual(half['connected_full'],len(exhaustive(f)))
        for args in ({'max_nodes':-1},{'fixed':{100:0}},{'exclude_phases':[[0]]}):
            with self.assertRaises(ValueError):mixed_solve(f,**args)

    def test_rejected_phase_union_does_not_require_optional_cp(self):
        p,parents,_=ring_donors(103)
        from builtins import __import__ as original_import
        def no_cp(name,*args,**kw):
            if name.startswith('ortools'):raise ModuleNotFoundError(name=name)
            return original_import(name,*args,**kw)
        with patch('builtins.__import__',side_effect=no_cp):
            self.assertEqual(common_phases(p,*parents),(None,'NO_COMMON_PHASE'))
            self.assertEqual(common_phases(p,parents[0],parents[0]),(None,'OPTIONAL_CP_UNAVAILABLE'))
            row=Family(p,parents[0],parents[0],None).solve(geography=True)
            self.assertTrue(p.validate(row['gold'])['valid'])
        bad=dict(parents[0],pairs=parents[0]['pairs'][:-1])
        with self.assertRaises(ValueError):Family(p,bad,parents[1],None)

    @skipUnless(find_spec('pysat'),'requires a SAT unit propagation engine')
    def test_independent_union_clauses_leave_valid_planted_roots(self):
        from bench_r35_mixed_sewings import union_sat
        p,parents,boards=ring_donors(103)
        edges=set().union(*(rec['pairs'] for rec in parents))
        for code in boards:
            # A singleton domain selects a directly validated board exactly.
            limited=SimpleNamespace(n=p.n,board=p.board,domains=[[v] for v in code])
            row=union_sat(limited,edges)
            self.assertEqual(row['checked_roots'],1)
            self.assertEqual(row['unresolved_roots'],1)
        no_matches=union_sat(p,set())
        self.assertEqual(no_matches['roots_rejected_by_unit_propagation'],30)

    def test_repeated_or_added_donors_preserve_real_geographic_roots(self):
        p,parents,boards=ring_donors(103)
        domains=tuple(1<<v for v in boards[0])
        first=donor_geography(p,parents[:1],domains)
        repeated=donor_geography(p,[parents[0]]*3,domains)
        self.assertEqual(first['roots'],repeated['roots'])
        extended=donor_geography(p,[*parents,parents[0]],domains)
        self.assertEqual(extended['geographic_status'],'UNKNOWN')
        self.assertEqual(len(extended['roots']['survivors']),1)
        with self.assertRaises(ValueError):donor_geography(p,[])
        with self.assertRaises(ValueError):donor_geography(p,[dict(parents[0],pairs=[])])

    def test_undo_recovers_phase_signs_after_a_partial_conflict(self):
        c=PhaseComponents(5)
        self.assertTrue(c.push([(1,0,1)]))
        mark=len(c.log)
        before=c.frontier(range(5))
        self.assertFalse(c.push([(0,3,1),(3,4,1),(1,4,0)]))
        c.restore(mark)
        self.assertEqual(c.frontier(range(5)),before)
        self.assertTrue(c.push([(1,4,1)]))


if __name__=='__main__':main()
