"""Finite independent oracles for elimination, lazy masters, proofs and routing."""
import importlib.util
import itertools
import random
import unittest
from fractions import Fraction
from constraint_region_elimination import (Factor,WorkBudget,eliminate,min_fill_order,
    relaxed_diagram_possible,problem_factors,EliminationPruner,support_elimination,
    refined_diagram_possible,conditioned_elimination,piece_lifetimes)
from constraint_region_master import (check_farkas,phase_one,puzzle_linear_system,
    CertifiedRelaxationPruner,diagram_max_weight,priced_inventory_master,LazyRegionMaster)
from constraint_region_symbolic import DirectRegionDiagram,DirectRegionSession,join_diagrams,subset_inventory_supports,representative_component_feasible
from constraint_region_research import RegionalResearchPruner
from constraint_region_oracles import residual_key,ResidualProofCache,LabelledRouteOracle,LabelledRoutingPruner
from constraint_regions import HierarchicalRegionPruner
from constraint_separator import _Budget
from test_constraint_advanced import bits,domains_of
from test_constraint_separator import ring,permutation_ring
from test_constraint_dfs import brute


def accepted(factors,assignment):
    return all(tuple(assignment[v] for v in f.scope) in f.rows for f in factors)


class EliminationTests(unittest.TestCase):
    def test_exact_and_relaxed_reasoning_against_enumeration(self):
        rng=random.Random(240019)
        for _ in range(80):
            domains={v:tuple(range(3)) for v in range(4)};factors=[]
            for a,b in itertools.combinations(domains,2):
                if rng.random()<.8:factors.append(Factor((a,b),frozenset(pair for pair in itertools.product(range(3),repeat=2) if rng.random()<.55)))
            feasible=any(accepted(factors,dict(enumerate(row))) for row in itertools.product(range(3),repeat=4))
            order=min_fill_order(factors,domains)
            exact,work=eliminate(factors,order)
            self.assertEqual(exact,feasible)
            for width in (2,3):
                relaxed,_=eliminate(factors,order,width=width)
                self.assertTrue(relaxed or not feasible)
            for width in (1,2,100):
                relaxed,stats=relaxed_diagram_possible(factors,domains,width=width)
                self.assertTrue(relaxed or not feasible)
                if width==100:self.assertEqual(relaxed,feasible)
    def test_elimination_sees_nonlocal_physical_inventory(self):
        problem=ring(6);domains=[bits([0,3,6])]*4+[bits([9,12,15])]*2
        factors,choices=problem_factors(problem,domains)
        result,stats=eliminate(factors,min_fill_order(factors,choices))
        self.assertFalse(result)
        self.assertFalse(any(len({x//3 for x in row})==6 for row in itertools.product(*(tuple(c) for c in choices.values()))))
    def test_budgets_never_become_negative_answers(self):
        f=Factor((0,1),frozenset(itertools.product(range(3),repeat=2)))
        with self.assertRaises(_Budget):eliminate([f],(0,1),budget=WorkBudget(0))
        with self.assertRaises(_Budget):relaxed_diagram_possible([f],{0:range(3),1:range(3)},budget=WorkBudget(0))
    def test_real_problem_pruning_preserves_every_gold_solution(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*2);solutions=brute(problem,'gold');rng=random.Random(240021)
        for _ in range(12):
            original=[bits(x for x in row if rng.random()<.8) for row in problem.domains]
            expected=[s for s in solutions if all(d>>x&1 for d,x in zip(original,s))]
            for mode in ('exact','mini','diagram'):
                actual=original.copy();EliminationPruner(problem).prune(actual,mode=mode,width=3)
                self.assertTrue(all(all(d>>x&1 for d,x in zip(actual,s)) for s in expected))


class ExtraIntegrationTests(unittest.TestCase):
    def test_bucket_tree_recovers_exact_all_value_supports(self):
        rng=random.Random(2400199)
        for _ in range(100):
            domains={v:tuple(range(3)) for v in range(5)}
            factors=[Factor((v,),frozenset((x,) for x in row)) for v,row in domains.items()]
            for a,b in itertools.combinations(domains,2):
                if rng.random()<.5:factors.append(Factor((a,b),frozenset(pair for pair in itertools.product(range(3),repeat=2) if rng.random()<.6)))
            solutions=[row for row in itertools.product(range(3),repeat=5) if accepted(factors,dict(enumerate(row)))]
            supported,work=support_elimination(factors,min_fill_order(factors,domains))
            self.assertEqual(supported,{v:{row[v] for row in solutions} for v in domains})
    def test_refinement_and_cutset_conditioning(self):
        factors=[Factor((a,b),frozenset({(0,1),(1,0)})) for a,b in ((0,1),(1,2),(0,2))]
        domains={v:(0,1) for v in range(3)}
        self.assertTrue(relaxed_diagram_possible(factors,domains,width=1)[0])
        result,stats=refined_diagram_possible(factors,domains)
        self.assertFalse(result);self.assertGreater(stats['passes'],1)
        self.assertFalse(conditioned_elimination(factors,domains,(0,))[0])
    def test_compiler_continuation_does_not_repeat_transitions(self):
        problem=ring(7);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        reference=DirectRegionDiagram(owner,range(7),domains)
        session=DirectRegionSession(owner,range(7),domains)
        with self.assertRaises(ValueError):session.result()
        attempts=0
        while not session.complete:
            attempts+=1
            if attempts%7==0:
                def stop():raise InterruptedError
                with self.assertRaises(InterruptedError):session.advance(1,stop)
            session.advance(3)
        result=session.result();self.assertEqual(result['steps'],reference.steps)
        self.assertEqual(result['nodes'],reference.nodes);self.assertEqual(result['rows'],5040)
    def test_subset_supports_and_zdd_have_same_complete_answers(self):
        rng=random.Random(2400239)
        for _ in range(16):
            groups=[{m for m in range(32) if m.bit_count()==2 and rng.random()<.5} for _ in range(3)]
            expected=[set() for _ in groups]
            for row in itertools.product(*groups):
                if all(not a&b for a,b in itertools.combinations(row,2)):
                    for i,x in enumerate(row):expected[i].add(x)
            actual,_=subset_inventory_supports(groups)
            self.assertEqual(actual,expected)
    def test_all_research_options_preserve_complete_gold_set(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*2);solutions=brute(problem,'gold');rng=random.Random(240000)
        keys=RegionalResearchPruner.KEYS
        for _ in range(8):
            original=[bits(x for x in row if rng.random()<.8) for row in problem.domains]
            expected=[s for s in solutions if all(d>>x&1 for d,x in zip(original,s))]
            for options in (*({key:True} for key in keys),dict.fromkeys(keys,True)):
                actual=original.copy();helper=RegionalResearchPruner(problem,options=options,max_steps=50000)
                helper.prune(actual)
                self.assertTrue(all(all(d>>x&1 for d,x in zip(actual,s)) for s in expected),options)
    def test_full_constraint_mode_includes_singleton_and_remote_resources(self):
        problem=ring(40)
        original=[bits([3*c,3*((c+20)%40)]) for c in range(40)]
        helper=EliminationPruner(problem,max_steps=200000)
        actual=original.copy();helper.prune(actual,full=True)
        self.assertEqual(actual,original)
        self.assertEqual(helper.stats['factor_queries'],1)
        self.assertEqual(helper.stats['factor_unknown'],0)
        self.assertEqual(helper.stats['factor_forgotten_resources'],40)
        choices={c:tuple(code for code in range(120) if original[c]>>code&1) for c in range(40)}
        factors,_=problem_factors(problem,original);order=min_fill_order(factors,choices)
        self.assertLessEqual(max(map(len,piece_lifetimes(choices,order)['live'])),2)


class ScopedResidualTests(unittest.TestCase):
    def test_different_fixed_histories_same_exact_live_problem(self):
        problem=ring(6)
        a=[bits([0,3]),bits([0,3]),1<<6,1<<9,1<<12,1<<15]
        b=a.copy();b[2],b[3]=b[3],b[2]
        self.assertNotEqual(a,b);self.assertEqual(residual_key(problem,a),residual_key(problem,b))
        self.assertNotEqual(residual_key(problem,a,'gold'),residual_key(problem,a,'edges'))
        self.assertNotEqual(residual_key(problem,a,root_scope=('a',)),residual_key(problem,a,root_scope=('b',)))
        c=b.copy();c[0]|=1<<6
        # Used pieces are correctly removed from live domains in the key.
        self.assertEqual(residual_key(problem,a),residual_key(problem,c))
        c=b.copy();c[0]|=1<<1;self.assertNotEqual(residual_key(problem,a),residual_key(problem,c))
    def test_unknown_is_never_saved_as_closed(self):
        problem=ring(4);domains=domains_of(problem);cache=ResidualProofCache(problem)
        for status in ('unknown','stopped','possible','solution'):
            self.assertFalse(cache.remember_completed(domains,status=status));self.assertFalse(cache.contains(domains))
        impossible=[bits([0,3])]*4
        self.assertFalse(any(len(set(row))==4 for row in itertools.product((0,3),repeat=4)))
        cache.remember_completed(impossible,status='unsatisfiable');self.assertTrue(cache.contains(impossible))
        self.assertFalse(cache.contains(domains))


class RoutingTests(unittest.TestCase):
    @staticmethod
    def graph(edges):
        result={}
        for a,b,labels in edges:
            result.setdefault(a,[]).append((b,labels));result.setdefault(b,[]).append((a,labels))
        return result
    def test_individual_paths_exist_but_incompatible_together(self):
        graph=self.graph([(0,1,((0,0),)),(1,3,()),(0,2,((0,1),)),(2,3,())])
        oracle=LabelledRouteOracle(graph)
        self.assertTrue(oracle.possible(0,3,count=1));self.assertFalse(oracle.possible(0,3,count=2))
        graph=self.graph([(0,1,((0,0),)),(1,3,()),(0,2,((0,0),)),(2,3,())])
        self.assertTrue(LabelledRouteOracle(graph).possible(0,3,count=2))
    def test_connected_graph_with_inconsistent_only_path(self):
        graph=self.graph([(0,1,((0,0),)),(1,2,((0,1),))])
        self.assertFalse(LabelledRouteOracle(graph).possible(0,2))
        self.assertIsNone(LabelledRouteOracle(graph,max_steps=0).possible(0,2))
    def test_routing_preserves_all_real_gold_solutions(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*2);solutions=brute(problem,'gold');rng=random.Random(240027)
        for solution in solutions[:16]:
            domains=[bits(set([code]+[x for x in choices if rng.random()<.4])) for code,choices in zip(solution,problem.domains)]
            domains[0]=1<<solution[0];actual=domains.copy()
            LabelledRoutingPruner(problem,max_steps=10000).prune(actual)
            self.assertTrue(all(d>>x&1 for d,x in zip(actual,solution)))


class RegionalMasterTests(unittest.TestCase):
    def test_lazy_master_matches_complete_gold_enumeration(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*2);solutions=brute(problem,'gold');rng=random.Random(240024)
        for _ in range(24):
            domains=[bits(x for x in row if rng.random()<.7) for row in problem.domains]
            expected=any(all(d>>x&1 for d,x in zip(domains,s)) for s in solutions)
            owner=HierarchicalRegionPruner(problem)
            master=LazyRegionMaster(owner,domains,((0,1),(2,3)),100000)
            self.assertEqual(master.solve(),expected)
            if expected:self.assertTrue(problem.validate([master.solution[c] for c in range(4)])['valid'])
    def test_lazy_master_avoids_eager_table_expansion(self):
        problem=ring(8);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        master=LazyRegionMaster(owner,domains,((0,1,2,3),(4,5,6,7)))
        self.assertTrue(master.solve());self.assertEqual(master.stats['lazy_local_rows'],2)
        eager=sum(DirectRegionDiagram(owner,r,domains).rows for r in master.regions)
        self.assertGreater(eager,1000)
        capped=LazyRegionMaster(owner,domains,master.regions,0)
        self.assertIsNone(capped.solve());self.assertFalse(capped.failed)
    def test_symbolic_overlap_join_matches_full_enumeration(self):
        problem=ring(8);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        left=DirectRegionDiagram(owner,range(6),domains);right=DirectRegionDiagram(owner,range(2,8),domains)
        supports,stats=join_diagrams(owner,left,right,domains,max_steps=100000)
        self.assertEqual(stats['histories'],40320);self.assertEqual([supports[c] for c in range(8)],domains)
        self.assertLess(stats['steps'],40320)
    def test_exact_dual_checker_rejects_corruption(self):
        # x+y=1, x=1, y=1 is impossible for x,y>=0.
        columns=[{0:1,1:1},{0:1,2:1}];rhs=(1,1,1)
        self.assertTrue(check_farkas(columns,rhs,(-1,1,1)))
        self.assertFalse(check_farkas(columns,rhs,(1,1,1)))
        self.assertFalse(check_farkas(columns,(2,1,1),(-1,1,1)))
    @unittest.skipUnless(importlib.util.find_spec('ortools'),'OR-Tools unavailable')
    def test_glop_dual_is_verified_in_exact_arithmetic(self):
        columns=[{0:1,1:1},{0:1,2:1}];rhs=(1,1,1)
        result=phase_one(columns,rhs)
        self.assertEqual(result['status'],'optimal');self.assertTrue(check_farkas(columns,rhs,result['dual']))
    @unittest.skipUnless(importlib.util.find_spec('ortools'),'OR-Tools unavailable')
    def test_missing_columns_rescue_incomplete_master(self):
        problem=ring(6);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        diagrams=[DirectRegionDiagram(owner,r,domains) for r in ((0,1),(2,3),(4,5))]
        result=priced_inventory_master(diagrams,6)
        self.assertEqual(result['status'],'possible');self.assertGreater(result['columns'],0)
        self.assertLess(result['columns'],sum(d.rows for d in diagrams))
        # All regions compete for only four pieces: complete pricing certifies failure.
        narrowed=[bits([0,3,6,9])]*6
        diagrams=[DirectRegionDiagram(owner,r,narrowed) for r in ((0,1),(2,3),(4,5))]
        result=priced_inventory_master(diagrams,6)
        self.assertEqual(result['status'],'infeasible')
    def test_pricing_matches_enumerated_weight_optimum(self):
        problem=ring(6);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        diagram=DirectRegionDiagram(owner,(0,1,2),domains);weights={t:Fraction(t*t-3,7) for t in range(6)}
        score,path,steps=diagram_max_weight(diagram,weights)
        self.assertEqual(score,max(sum(weights[x//3] for x in row) for row,_ in diagram.histories()))
        self.assertEqual(score,sum(weights[x//3] for x in path));self.assertLess(steps,diagram.rows*3)

if __name__=='__main__':unittest.main()
