"""Independent finite regression checks for R24 symbolic regional operations."""
import itertools
import random
import unittest
from constraint_region_symbolic import (DirectRegionDiagram,InventoryZDD,inventory_supports,
    subset_convolution,endpoint_matchings,single_cycle,representative_matchings)
from constraint_region_network import ExactRegionDiagram,assignment_signature
from constraint_regions import HierarchicalRegionPruner
from constraint_separator import _Budget
from test_constraint_separator import ring,permutation_ring
from test_constraint_advanced import domains_of,bits
from test_constraint_dfs import brute


class SymbolicDiagramTests(unittest.TestCase):
    def test_histories_equal_independent_enumeration(self):
        rng=random.Random(240017)
        for target in ('gold','edges'):
            problem=permutation_ring([(1,0,2),(0,2,1)]*2)
            solutions=brute(problem,target)
            for _ in range(12):
                domains=[bits(x for x in row if rng.random()<.8) for row in problem.domains]
                owner=HierarchicalRegionPruner(problem,target)
                diagram=DirectRegionDiagram(owner,range(4),domains)
                expected={row for row in solutions if all(d>>c&1 for d,c in zip(domains,row))}
                self.assertEqual({row for row,_ in diagram.histories()},expected)
                self.assertEqual(diagram.rows,len(expected))
                self.assertEqual(diagram.supports(),tuple(bits(row[c] for row in expected) for c in range(4)))
    def test_direct_construction_avoids_factorial_enumeration(self):
        problem=ring(7);domains=domains_of(problem);owner=HierarchicalRegionPruner(problem)
        with self.assertRaises(_Budget):ExactRegionDiagram(owner,range(7),domains,lambda:None,max_rows=512)
        direct=DirectRegionDiagram(owner,range(7),domains,max_rows=512)
        self.assertEqual(direct.rows,5040)
        self.assertLess(direct.states,256)
        self.assertLess(direct.steps,5040)
        self.assertGreater(direct.memo_hits,0)
    def test_partial_region_signatures_exact(self):
        problem=permutation_ring([(1,0,2),(0,2,1)]*3)
        owner=HierarchicalRegionPruner(problem)
        domains=domains_of(problem)
        for cells in ((0,1),(0,2,4),(0,1,2,3)):
            old=ExactRegionDiagram(owner,cells,domains,lambda:None)
            new=DirectRegionDiagram(owner,cells,domains)
            self.assertEqual(set(old.histories()),set(new.histories()))
    def test_budget_cancel_and_explicit_expansion_limit(self):
        problem=ring(6);owner=HierarchicalRegionPruner(problem);domains=domains_of(problem)
        for options in ({'max_rows':1},{'max_steps':0},{'max_histories':100}):
            with self.assertRaises(_Budget):DirectRegionDiagram(owner,range(6),domains,**options)
        def stop():raise InterruptedError
        with self.assertRaises(InterruptedError):DirectRegionDiagram(owner,range(6),domains,stop)


class SymbolicInventoryTests(unittest.TestCase):
    def test_zdd_products_match_explicit_disjoint_pairs(self):
        rng=random.Random(240018)
        for _ in range(60):
            left={m for m in range(64) if rng.random()<.3}
            right={m for m in range(64) if rng.random()<.3}
            zdd=InventoryZDD();a=zdd.from_sets(left);b=zdd.from_sets(right)
            self.assertEqual(set(zdd.sets(zdd.union(a,b))),left|right)
            product=zdd.disjoint_product(a,b)
            expected={x|y for x in left for y in right if not x&y}
            self.assertEqual(set(zdd.sets(product)),expected)
            self.assertEqual(zdd.count(product),len(expected))
            for forbidden in (0,3,17,63):
                self.assertEqual(zdd.exists_disjoint(product,forbidden),any(not x&forbidden for x in expected))
    def test_global_inventory_supports_match_cartesian_oracle(self):
        rng=random.Random(2400181)
        for _ in range(50):
            groups=[{m for m in range(32) if m.bit_count()<=2 and rng.random()<.2} for _ in range(4)]
            expected=[set() for _ in groups]
            for row in itertools.product(*groups):
                if all(not a&b for a,b in itertools.combinations(row,2)):
                    for i,value in enumerate(row):expected[i].add(value)
            actual,_=inventory_supports(groups)
            self.assertEqual(actual,expected)
    def test_subset_convolution_exact_integer_counts(self):
        rng=random.Random(240023)
        for n in range(6):
            a=[rng.randrange(5) for _ in range(1<<n)];b=[rng.randrange(5) for _ in a]
            expected=[sum(a[sub]*b[mask^sub] for sub in range(1<<n) if not sub&~mask) for mask in range(1<<n)]
            actual,_=subset_convolution(a,b,n)
            self.assertEqual(actual,tuple(expected))
        with self.assertRaises(ValueError):subset_convolution([],[],160)
    def test_zdd_budget_is_not_an_empty_family(self):
        with self.assertRaises(_Budget):InventoryZDD(max_nodes=0).from_sets([3])


class RepresentativeTests(unittest.TestCase):
    @staticmethod
    def reference_cycle(left,right):
        # Independent walk through alternating left/right matching edges.
        a=dict((*left,*[(v,u) for u,v in left]));b=dict((*right,*[(v,u) for u,v in right]))
        if not a:return False
        pending=[min(a)];seen=set()
        while pending:
            x=pending.pop()
            if x in seen:continue
            seen.add(x);pending.extend((a[x],b[x]))
        return len(seen)==len(a)
    def test_basis_preserves_every_possible_outside_connector(self):
        for size in (2,4,6,8):
            rows=tuple(endpoint_matchings(range(size)))
            chosen,stats=representative_matchings(rows)
            for outside in rows:
                self.assertEqual(any(self.reference_cycle(row,outside) for row in rows),
                                 any(self.reference_cycle(row,outside) for row in chosen))
                for row in rows:self.assertEqual(single_cycle(row,outside),self.reference_cycle(row,outside))
            if size>=6:self.assertLess(len(chosen),len(rows))
    def test_representatives_preserve_subfamilies(self):
        rng=random.Random(240022);rows=tuple(endpoint_matchings(range(6)))
        for _ in range(80):
            family=[row for row in rows if rng.random()<.35]
            chosen,_=representative_matchings(family)
            for outside in rows:self.assertEqual(any(self.reference_cycle(r,outside) for r in family),any(self.reference_cycle(r,outside) for r in chosen))
    def test_rank_cap_is_unknown(self):
        with self.assertRaises(_Budget):representative_matchings([tuple((i,i+1) for i in range(0,12,2))])

if __name__=='__main__':unittest.main()
