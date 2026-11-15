"""Independent whole-matching enumeration for overlapping loss and scope tests."""
from itertools import product
from importlib.util import find_spec
from math import inf
from pathlib import Path
from random import Random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from r35_mate_price import ClassPrices
from r35_loss_witness import reciprocal
from r35_signed_conditions import SignedConditions


class MatePriceTests(unittest.TestCase):
    def test_every_phase_and_full_physical_pairing_of_three_families(self):
        families=[tuple(range(i,12,3)) for i in range(3)]
        tiles=[{}]*4
        p=SimpleNamespace(tiles=tiles)
        labels=[None]*12
        for k,fam in enumerate(families):
            for i,a in enumerate(fam):labels[a]=((() if k==0 else (3 if i%2==0 else 9,) if k==1 else (2,10)),(i&k)%2)
        allowed={(a,b) for family in families for a in family for b in family if a<b and labels[b][0]==reciprocal(labels[a])[0]}
        def pairs(rest):
            if not rest:yield set();return
            a=min(rest)
            for b in rest-{a}:
                if (a,b) in allowed:
                    for child in pairs(rest-{a,b}):yield child|{(a,b)}
        all_matchings=[set.union(*sets) for sets in product(*(list(pairs(set(family))) for family in families))]
        rng=Random(3579)
        with patch('r35_mate_price.reciprocal_classes',return_value=(labels,families)), \
             patch('r35_mate_price.inventory',return_value=SimpleNamespace(possible=allowed)):
            for trial in range(48):
                ref=set(rng.choice(all_matchings))
                if trial >= 36:
                    ref.update(rng.sample(sorted(allowed), 3))
                required=set(rng.sample(sorted(ref),rng.randrange(3)))
                groups=[set(rng.sample(sorted(ref),rng.randrange(1,5))) for _ in range(rng.randrange(1,5))]
                exact=ClassPrices(p,ref,groups,required,limit=8)
                loose=ClassPrices(p,ref,groups,required,limit=0)
                costs={}
                for ph in range(16):
                    signed=[(s,c^(ph>>(a//3)&1)) for a,(s,c) in enumerate(labels)]
                    expected=min((len(ref-m) for m in all_matchings if required<=m and all(not g<=m for g in groups)
                                  and all(signed[b]==reciprocal(signed[a]) for a,b in m)),default=inf)
                    self.assertEqual(exact.memo(ph),expected)
                    self.assertLessEqual(loose.memo(ph),expected)
                    costs[ph] = expected
                # Every one of the 3**4 partial assignments represents every
                # completion of its unset phases. The union graph is a lower bound.
                for mask in range(16):
                    for prefix in {phase & mask for phase in costs}:
                        best=min(cost for phase,cost in costs.items() if phase & mask == prefix)
                        self.assertLessEqual(exact.partial(prefix,mask), best)
                for phase,cost in costs.items():
                    if cost == 0:
                        continue
                    cap = 3 if cost == inf else cost - 1
                    for deadline in (0,inf):
                        pins=exact.conflict(phase,cap,deadline)
                        for candidate,actual in costs.items():
                            if (phase ^ candidate) & pins in (0,pins):
                                self.assertGreater(actual,cap)


                if find_spec('pysat'):
                    from pysat.card import CardEnc
                    from pysat.solvers import Solver
                    from r35_mate_native import clauses
                    # A fresh Boolean per physical reference bond. Every input
                    # group requires at least one of its optional bonds to break.
                    losses={edge:5+i for i,edge in enumerate(sorted(ref-required))}
                    native=clauses(exact,losses,max([4,*losses.values()]))
                    native.extend([[losses[e] for e in group-required] for group in groups])
                    cap=trial%(len(losses)+1)
                    if cap < len(losses):
                        top=max([4,*[abs(v) for clause in native for v in clause]])
                        native.extend(CardEnc.atmost(lits=list(losses.values()),bound=cap,top_id=top).clauses)
                    with Solver(name='g4',bootstrap_with=native) as solver:
                        for phase,cost in costs.items():
                            pins=[i+1 if phase>>i&1 else -i-1 for i in range(4)]
                            self.assertEqual(solver.solve(assumptions=pins),cost<=cap)

    def test_broad_obligations_and_node_cutoff_have_independent_witnesses(self):
        from itertools import combinations
        from r35_family_obligations import family_bound
        rng=Random(357912)
        families=[tuple(range(4)),tuple(range(4,8)),tuple(range(8,12))]
        allowed=[set(combinations(family,2)) for family in families]
        profiles=[]
        for group in families:
            a,b,c,d=group
            profiles.append([{(a,b),(c,d)},{(a,c),(b,d)},{(a,d),(b,c)}])
        all_matchings=[set.union(*rows) for rows in product(*profiles)]
        reference=set.union(*allowed)
        candidates=list(combinations(sorted(reference),2))
        for trial in range(50):
            groups=[frozenset(pair) for pair in rng.sample(candidates,15+trial%8)]
            expected=min((len(reference-m) for m in all_matchings if all(not g<=m for g in groups)),default=inf)
            families_with_edges=[(family,()) for family in families]
            exact=family_bound(families_with_edges,allowed,reference,groups,4)
            self.assertEqual(exact,expected)
            for budget in (0,2,10):
                cutoff=family_bound(families_with_edges,allowed,reference,groups,4,node_limit=budget)
                self.assertLessEqual(cutoff,expected)

    @unittest.skipUnless(find_spec('pysat'),'Native signed SAT is optional.')
    def test_partial_native_reuse_gauge_deadline_and_broader_price_scope(self):
        from pysat.solvers import Solver
        from r35_integer_sat import InventorySAT
        from bench_r35_loss_boundary344 import problem_scope
        path=next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if not path:self.skipTest('Original data unavailable.')
        problem,ref,groups,req=problem_scope(path)
        w=0xd237aba7e91101e88041413e8c589902021e5126
        cond=SignedConditions(problem,mate_cost=True).for_loss(ref,groups[:13],req,23)
        broad=SignedConditions(problem,mate_cost=True).for_loss(ref,groups,req,23)
        self.assertEqual(len(cond.native_cost_pairs()),48)
        self.assertIsNone(broad.native_cost_pairs())
        self.assertEqual(cond.price(w,0),24)
        rules=cond._price_conflict(w,inf)
        self.assertEqual({len(row) for row in rules},{25})
        self.assertEqual(len(rules),2)
        self.assertTrue(all(1 not in {abs(v) for v in row} for row in rules))
        interrupted=cond._prices.conflict(w,23,0)
        self.assertGreater(interrupted.bit_count(),25)
        looser=cond.for_loss(ref,groups[:13],req,24)
        self.assertTrue(looser.accepts(w))
        self.assertFalse(looser._price_rules)

        # Both learned patterns have completions at tile zero's representative.
        model=InventorySAT.__new__(InventorySAT);model.n=problem.n;model.clauses=[[-1]]
        model.solver=Solver(name='g4',bootstrap_with=model.clauses)
        try:
            for row in rules:
                pins=[-v for v in row]
                self.assertTrue(model.solver.solve(assumptions=pins))
            cond._post(model,rules)
            for row in rules:
                self.assertFalse(model.solver.solve(assumptions=[-v for v in row]))
        finally:model.close()

    def test_original_heldout_conditions_and_both_public_paths(self):
        from r32_gold import run_syndrome
        from r35_loss_bound import global_integer
        from bench_r35_loss_proof import original
        from bench_r35_loss_boundary344 import problem_scope
        from r32_gold_loss import read_json
        path=next((p for p in map(Path,('data/tiles.json','../data/tiles.json')) if p.is_file()),None)
        if not path:self.skipTest('Original data unavailable.')
        p,ref,g,req=problem_scope(path)
        _,_,pack,_,out=original(path)
        w=0xd237aba7e91101e88041413e8c589902021e5126
        geo=[set(map(tuple,row))-req for row in read_json('docs/R35_GEOGRAPHIC_LOSS_CORES.json')['geographic_cores']]
        from bench_r35_near_reference_tour import raw_cycles
        from r31_geographic import ribbon_genus
        from r27_structure_filters import domain_bits
        from r30_gold import embedding
        for label,retain,genus in (('physical',183,26),('geo',182,29)):
            row=read_json('docs/R35_NEAR_REFERENCE_TOURS.json')[label]
            mesh=set(map(tuple,row['pairs']));self.assertEqual(len(mesh),240)
            self.assertEqual(raw_cycles(p,mesh),[365]);self.assertEqual(ribbon_genus(p,sorted(mesh)),genus)
            self.assertEqual(len(mesh&ref),retain)
            mates=[-1]*(p.n*3)
            for a,b in mesh:mates[a]=b;mates[b]=a
            self.assertIsNone(embedding(p,domain_bits(p),mates))
        for added,expected in (([],24),(geo,25)):
            self.assertEqual(ClassPrices(p,ref,[*g,*added],req,limit=8).memo(w),expected)
            cond=SignedConditions(p,euler=True,mate_cost=True)
            free=global_integer(p,ref,pack,req,out,expected,.01,additional_groups=[*g,*added],conditions=cond)
            self.assertEqual((free['phase'],free['loss_cost'],free['work']['decisions']),(hex(w),expected,0))
        weak=run_syndrome(p,ref,g[:13],23,None,':memory:',.01,refine_integer=True,all_groups=True,sphere_euler=True,mate_support=True)
        strong=run_syndrome(p,ref,g[:13],24,None,':memory:',.01,refine_integer=True,all_groups=True,sphere_euler=True,mate_cost=True)
        self.assertEqual((weak['phase'],weak['loss_cost']),(hex(w),23))
        self.assertEqual((strong['phase'],strong['loss_cost']),(hex(w),24))
        declined=run_syndrome(p,ref,g[:13],23,None,':memory:',.01,refine_integer=True,all_groups=True,sphere_euler=True,mate_cost=True)
        self.assertEqual(declined['status'],'UNKNOWN')
        from sqlite3 import connect
        from r35_loss_witness import save_witness, checked_hits
        with connect(':memory:') as db:
            save_witness(db,'ordinary-scope',w)
            priced=SignedConditions(p,euler=True,mate_cost=True)
            for cap in (23,24):
                live=priced.for_loss(ref,g[:13],req,cap)
                hits=checked_hits(db,'ordinary-scope',p,ref,g[:13],req,out,cap,True,live)
                if cap==23:
                    self.assertFalse(hits)
                else:
                    self.assertEqual(next(iter(hits.values()))['loss_cost'],24)

        import json,subprocess,sys
        cmd=[sys.executable,'-B','r32_gold.py','integer-bound','--tiles',str(path),'--upper','25',
             '--all-groups','--sphere-euler','--mate-cost','--claims','docs/R35_GEOGRAPHIC_LOSS_CORES.json','--seconds','.005']
        case=json.loads(subprocess.run(cmd,capture_output=True,text=True,check=True).stdout)
        self.assertEqual((case['phase'],case['loss_cost'],case['work']['decisions']),(hex(w),25,0))
        self.assertTrue(all(case['signed_conditions'].values()))



if __name__=='__main__':unittest.main()
