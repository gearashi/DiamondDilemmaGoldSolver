import copy,random,unittest
from constraint_lattice import IntegerLattice,IntegerLatticePruner
from constraint_lattice_round26 import ContributionLatticePruner,separating_functional,verify_separator
from constraint_deduction_chain_round26 import verify_chain,ChainError,FORMAT
from test_constraint_sat_domains import fixture,admitted
from test_constraint_dfs import brute


class LatticeTreeTests(unittest.TestCase):
    def test_exact_modular_or_equality_separator_against_original_columns(self):
        rng=random.Random(2607)
        for _ in range(160):
            columns=[{c:v for c in range(4) if (v:=rng.randrange(-4,5))} for _ in range(rng.randrange(5))]
            basis=IntegerLattice(columns)
            for _ in range(12):
                target={c:v for c in range(4) if (v:=rng.randrange(-8,9))};member=basis.contains(target);reason=separating_functional(basis,target)
                self.assertEqual(reason is None,member)
                if reason is not None:
                    weights=dict(reason['weights']);modulus=reason['modulus']
                    products=[sum(weights.get(c,0)*v for c,v in row.items()) for row in columns]
                    target_value=sum(weights.get(c,0)*v for c,v in target.items())
                    self.assertTrue(all((v%modulus if modulus else v)==0 for v in products))
                    self.assertNotEqual(target_value%modulus if modulus else target_value,0)

    def test_actual_puzzle_matches_cold_lattice_and_preserves_all_solutions(self):
        problem,_=fixture();solutions={target:brute(problem,target) for target in ('edges','gold')};helper=ContributionLatticePruner(problem);rng=random.Random(260701)
        contexts=[[1754,2435,2592,2378]]+[[sum(1<<c for c in row if rng.random()<.75) for row in problem.domains] for _ in range(70)]
        for initial in contexts:
            old=initial.copy();new=initial.copy();IntegerLatticePruner(problem).prune(old,max_candidates=1000);helper.prune(new,max_candidates=1000)
            self.assertEqual(new,old)
            for target,rows in solutions.items():self.assertEqual(admitted(rows,initial),admitted(rows,new),target)
            for reason in helper.drain_reasons():
                trial=list(reason['domains'])
                if reason['cell'] is not None:trial[reason['cell']]=1<<reason['code']
                self.assertTrue(verify_separator(problem,trial,reason['certificate']))
        self.assertLessEqual(len(helper.cache),4);self.assertLessEqual(len(helper.nodes),1024)
        self.assertTrue(all(len(value['without'])<=8 for value in helper.cache.values()))

    def test_modular_reason_transfers_under_guards_and_rejects_forgery(self):
        problem,_=fixture();initial=[1754,2435,2592,2378];helper=ContributionLatticePruner(problem);domains=initial.copy();helper.prune(domains,max_candidates=1000)
        reason=next(row for row in helper.drain_reasons() if row['cell'] is not None);original=[hex(sum(1<<c for c in row)) for row in problem.domains]
        certificate={'format':FORMAT,'fingerprint':problem.fingerprint,'target':'edges','original':original,'root':[hex(v) for v in initial],
            'decisions':[[reason['cell'],hex(1<<reason['code'])]],'events':[],'terminal':{'rule':'lattice','data':{'certificate':reason['certificate']}}}
        self.assertTrue(verify_chain(problem,certificate,target='edges'))
        forged=copy.deepcopy(certificate);forged['terminal']['data']['certificate']['weights'][0][1]=True
        with self.assertRaises(ChainError):verify_chain(problem,forged,target='edges')
        calls=0
        def stop():
            nonlocal calls
            calls+=1
            if calls==3:raise ValueError('callback Stop')
        with self.assertRaisesRegex(ValueError,'callback Stop'):verify_chain(problem,certificate,target='edges',check=stop)

    def test_stop_caps_sibling_origins_and_full160_preserve_exact_state(self):
        problem,_=fixture();initial=[1754,2435,2592,2378]
        for limit in (1,5,20,60,180,500,1000):
            helper=ContributionLatticePruner(problem);current=initial.copy();calls=0
            def stop():
                nonlocal calls
                calls+=1
                if calls==limit:raise InterruptedError
            try:helper.prune(current,stop,max_candidates=1000)
            except InterruptedError:self.assertEqual(current,initial)
            helper.prune(current,max_candidates=1000);self.assertEqual(current,[1754,2434,2592,74])
        for caps in ({'max_operations':0},{'max_dimensions':0},{'max_placements':0}):
            current=initial.copy();ContributionLatticePruner(problem).prune(current,**caps);self.assertEqual(current,initial)
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        tiles,board,witness=planted_gold(260702,size=4);p=SearchProblem(tiles,board);current=[1<<c for c in witness]
        for cell in (0,17,76,159):current[cell]=7<<(3*(witness[cell]//3))
        helper=ContributionLatticePruner(p);helper.prune(current,max_candidates=1000)
        self.assertTrue(all(mask>>code&1 for mask,code in zip(current,witness)))
        self.assertEqual(helper.stats['lattice_caps'],0)

if __name__=='__main__':unittest.main()
