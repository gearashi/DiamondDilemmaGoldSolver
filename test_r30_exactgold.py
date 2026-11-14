"""Exact cycle and physical-tour cuts against exhaustive small-board oracles."""
from collections import Counter
from importlib.util import find_spec
from itertools import permutations
from pathlib import Path
import tempfile
import unittest


from r27_incumbent import BoardModel
from r27_structure_filters import Union, domain_bits
from r27_structure_fixtures import planted
from r30_exact_routes import CycleModel, cycle_type, learn, rules_in
from r30_gold import GoldMatching, core_applies, restriction_core, search
from test_r28_geometry import phases, seams


@unittest.skipUnless(find_spec("ortools"),"Optional OR-Tools")
class ExactComparisons(unittest.TestCase):
    def test_partial_crossings_preserve_planted_phase_complements_and_roots(self):
        from unittest.mock import patch
        from r29_gold import phase_search
        from r29_gold_phase import EndpointCuts
        from r30_gold import insufficient_crossings
        p,ds=planted(1,0,23,False,True)
        data=EndpointCuts(p)
        groups=[[a] for a in data.graph]
        true_phases=phases(p)
        for flip in (0,1):
            for limit in (1,p.n//2,p.n):
                pins={i:v^flip for i,v in enumerate(true_phases) if i<limit}
                self.assertIsNone(insufficient_crossings(data,pins,groups))
                result=phase_search(p,pins=pins,width=1,max_subsets=10,seconds=1)
                self.assertIn(result['status'],('FEASIBLE','OPTIMAL'))
        with patch('r30_gold.insufficient_crossings',side_effect=AssertionError("root called optional screen")):
            self.assertEqual(phase_search(p,cp=False,width=1,max_subsets=2)['status'],'UNKNOWN')

    def test_cycle_partition_against_all_cell_permutations(self):
        p,ds=planted(1,0,71)
        board=[3*i for i in range(p.n)]
        for c in range(p.n):
            p.domains[c]=[3*t for t in range(6)] if c<6 else [board[c]]
        model=BoardModel(p)
        best=Counter()
        for row in permutations(range(6)):
            k=cycle_type(row)
            codes=[3*t for t in row]+board[6:]
            best[k]=max(best[k],p.validate(codes)['matched_edges'])
        for k in [(4,2),(3,3),()]:
            q=CycleModel(model,board,range(6),k)
            out=q.solve(target=0,optimize=True,seconds=20)
            self.assertEqual((out['status'],out['upper'],out['score']),('OPTIMAL',best[k],best[k]))
            if best[k]<len(p.edges):
                self.assertEqual(q.solve(target=best[k]+1)['status'],'INFEASIBLE')
        # Four moves with seven and eight cells; fixture forces a unique physical endpoint.
        p,_=planted(1,0)
        original=[3*i for i in range(p.n)]
        model=BoardModel(p)
        q=CycleModel(model,original,range(6),(4,2))
        fallback=q.solve(target=0,predecessors=True)
        self.assertEqual(fallback['predecessor_cuts'],0)
        self.assertIn(fallback['status'],('OPTIMAL','FEASIBLE'))
        for row in ([1,2,0,4,3,6,5],[1,0,3,2,5,4,7,6]):
            q=CycleModel(model,original,range(len(row)),cycle_type(row))
            witness=q.solve(target=0,pins=list(enumerate(row)),seconds=10)
            self.assertIn(witness['status'],('OPTIMAL','FEASIBLE'))
            self.assertEqual([v//3 for v in witness['codes'][:len(row)]],row)

    def test_conditional_bound_resumes_across_permutations(self):
        p,_=planted(1,0,23,False,True)
        model=BoardModel(p)
        b=[3*i for i in range(p.n)]
        q=CycleModel(model,b,range(6),(4,2))
        pins=list(enumerate((1,2,3,0,5,4)))
        with tempfile.TemporaryDirectory() as temp:
            db=Path(temp)/"model.sqlite"
            result=learn(q,pins,13,seconds=3,max_delete=10,database=db)
            self.assertEqual(result['status'],'CERTIFIED')
            rule=result['rule']
            self.assertLess(len(rule['pins']),len(pins))
            checked=rules_in(db,q.binding)
            oracle=0
            for row in permutations(range(6)):
                if cycle_type(row)!=(4,2) or not all(row[c]==j for c,j in rule['pins']):
                    continue
                out=q.solve(target=rule['upper']+1,pins=list(enumerate(row)))
                self.assertEqual(out['status'],'INFEASIBLE')
                self.assertEqual(q.solve(target=13,pins=list(enumerate(row)),rules=checked)['cp_calls'],0)
                oracle+=1
            self.assertGreater(oracle,1)
            alt=CycleModel(model,b,range(6),(3,3))
            self.assertEqual(rules_in(db,alt.binding),[])

    def test_valid_and_unrealizable_incompatible_physical_matchings(self):
        p,ds=planted(1,0,23,False,True)
        allowed,banned,fixed,pins=bridge_fixture(p)
        choices=GoldMatching(p,domain_bits(p),pins,fixed,(),banned)
        graph=Union(choices.strands)
        for pair in allowed:
            for a,b in choices.transitions[pair]:
                graph.join(a,b)
        self.assertEqual(len({graph.find(i) for i in range(choices.strands)}),1)
        counts=pairing_oracle(choices,allowed,fixed)
        self.assertGreater(counts['matchings'],0)
        self.assertEqual(counts['connected'],0)
        result=search(p,pins=pins,bonds=fixed,forbidden=banned,seconds=5)
        self.assertEqual(result['status'],'INFEASIBLE')
        self.assertGreater(result['strand_cuts'],0)
        independent_bounds=[choices.crossing_capacity(g,2) for g in result['physical_cutsets']]
        self.assertTrue(all(r['upper']>=2 for r in independent_bounds))
        joint=[]
        for group in result['physical_cutsets']:
            joint.append(choices.crossing_capacity(group,2))
            choices.strand_cut(group)
        self.assertTrue(any(r.get('upper',0)<2 for r in joint))
        rule=restriction_core(p,pins=pins,bonds=fixed,forbidden=banned,max_delete=3,seconds=3)
        self.assertTrue(core_applies(p,None,pins,fixed,banned,rule))
        self.assertFalse(core_applies(p,ds,pins,fixed,banned,rule))
        valid=search(p,domains=ds,bonds=seams(p),geography=True,seconds=3)
        self.assertEqual(valid['status'],'GOLD')
        self.assertTrue(p.validate(valid['codes'])['valid'])
        # An empty time/iteration frontier must remain open.
        self.assertEqual(search(p,pins=pins,bonds=fixed,forbidden=banned,max_rounds=1)['status'],'UNKNOWN')


def bridge_fixture(p):
    """A planted compatible-side union with mutually exclusive bridge choices."""
    allowed={(0,9),(0,14),(0,16),(1,3),(2,14),(2,17),(2,22),(2,27),(3,7),(3,8),
             (3,9),(3,20),(7,14),(7,16),(7,22),(8,15),(8,17),(8,18),(8,20),(8,26),
             (9,23),(14,20),(14,27),(16,22),(17,18),(18,21),(18,23),(20,22),(20,26),(26,27)}
    pins=dict(enumerate(phases(p)))
    total=GoldMatching(p,domain_bits(p),pins,[],[],[])
    banned=[pair for pair in total.mate if total.transitions[pair] and pair not in allowed]
    fixed=[pair for pair in seams(p) if not total.transitions.get(tuple(sorted(pair)))]
    return allowed,banned,fixed,pins


def pairing_oracle(model,allowed,fixed):
    by={}
    for a,b in allowed:
        by.setdefault(a,[]).append(b)
        by.setdefault(b,[]).append(a)
    counts=Counter()
    def visit(left,chosen):
        if not left:
            counts['matchings']+=1
            graph=Union(model.strands)
            for pair in chosen:
                for a,b in model.transitions[tuple(sorted(pair))]:
                    graph.join(a,b)
            counts['connected']+=len({graph.find(i) for i in range(model.strands)})==1
            return
        a=min(left,key=lambda i:sum(j in left for j in by[i]))
        for b in by[a]:
            if b in left:
                visit(left-{a,b},chosen+[(a,b)])
    visit(set(by),list(fixed))
    return counts


if __name__=="__main__":
    unittest.main()
