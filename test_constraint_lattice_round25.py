"""Integer-lattice reasoning checked by exact rational and arrangement oracles."""
import copy
import random
import unittest
from fractions import Fraction
from constraint_lattice import IntegerLattice,IntegerLatticePruner,LatticeLimit
from constraint_algebra import AffinePruner
from constraint_pb import PBLearner
from test_constraint_sat_domains import fixture,ordinary_fixpoint,admitted
from test_constraint_dfs import brute


def prior_fixpoint(problem,domains):
    current=list(domains);affine=AffinePruner(problem);pb=PBLearner(problem,derive=True)
    while all(current):
        old=current.copy();current=ordinary_fixpoint(problem,current)
        if all(current):affine.prune(current,max_candidates=1000)
        if all(current):pb.prune(current,max_values=12)
        if current==old:break
    return current


def lattice_comparison():
    problem,_=fixture();initial=[1759,3459,2720,2890]
    baseline=prior_fixpoint(problem,initial);helper=IntegerLatticePruner(problem);candidate=baseline.copy()
    helper.prune(candidate,max_candidates=1000)
    models={target:brute(problem,target) for target in ('edges','gold')}
    for target,solutions in models.items():assert admitted(solutions,candidate)==admitted(solutions,baseline)
    assert sum((a&~b).bit_count() for a,b in zip(baseline,candidate))>0
    return {'input':initial,'baseline':baseline,'candidate':candidate,
            'preserved_completions':{target:len(admitted(rows,baseline)) for target,rows in models.items()},'stats':dict(helper.stats)}


class IntegerBasisTests(unittest.TestCase):
    def test_integer_divisibility_catches_an_obstruction_gf2_accepts(self):
        columns=[{0:1,1:2},{0:2,1:1}];basis=IntegerLattice(columns)
        self.assertFalse(basis.contains({0:1,1:1}))
        self.assertTrue(basis.contains({0:3,1:3}))
        parity={0}
        for column in columns:
            residue=sum((value&1)<<coordinate for coordinate,value in column.items())
            parity|={value^residue for value in tuple(parity)}
        self.assertIn(3,parity)  # The target (1,1) belongs to the GF(2) span.

    def test_random_rank_deficient_lattices_match_independent_rational_coordinates(self):
        rng=random.Random(251104)
        for case in range(100):
            width=rng.randrange(1,5);rank=rng.randrange(width+1)
            matrix=[[0 if i<j else rng.randrange(1,5) if i==j else rng.randrange(-3,4)
                     for j in range(rank)] for i in range(width)]
            columns=[{i:matrix[i][j] for i in range(width) if matrix[i][j]} for j in range(rank)]
            for _ in range(5):
                coefficients=[rng.randrange(-3,4) for _ in range(rank)]
                values=[sum(matrix[i][j]*coefficients[j] for j in range(rank)) for i in range(width)]
                columns.append({i:value for i,value in enumerate(values) if value})
            rng.shuffle(columns);basis=IntegerLattice(columns)
            for query in range(40):
                if query%2:
                    coefficients=[rng.randrange(-8,9) for _ in range(rank)]
                    target=[sum(matrix[i][j]*coefficients[j] for j in range(rank)) for i in range(width)]
                else:target=[rng.randrange(-12,13) for _ in range(width)]
                coordinates=[]
                for i in range(rank):
                    coordinates.append((Fraction(target[i])-sum(matrix[i][j]*coordinates[j] for j in range(i)))/matrix[i][i])
                expected=all(x.denominator==1 for x in coordinates) and all(
                    sum(matrix[i][j]*coordinates[j] for j in range(rank))==target[i] for i in range(width))
                self.assertEqual(basis.contains({i:value for i,value in enumerate(target) if value}),expected,(case,target))

    def test_caps_cancellation_and_invalid_types_are_not_membership_answers(self):
        with self.assertRaises(LatticeLimit):IntegerLattice([{0:1}],max_operations=0)
        with self.assertRaises(LatticeLimit):IntegerLattice([{0:1<<100}],max_bits=32)
        for value in ({0:1.0},{True:1},{0:True},{-1:1}):
            with self.assertRaises(ValueError):IntegerLattice([value])
        def stop():raise InterruptedError('stop')
        with self.assertRaises(InterruptedError):IntegerLattice([{0:1}],stop)
        with self.assertRaises(InterruptedError):IntegerLattice([{0:1}]).contains({0:2},stop)


class LatticePrunerTests(unittest.TestCase):
    def test_genuine_domain_discriminator_exceeds_old_fixpoint(self):
        report=lattice_comparison()
        self.assertEqual(report['baseline'],[1754,2435,2592,2378])
        self.assertEqual(report['candidate'],[1754,2434,2592,74])
        self.assertEqual(report['preserved_completions'],{'edges':6,'gold':4})
        self.assertEqual(report['stats']['lattice_caps'],0)

    def test_random_scope_changes_preserve_every_enumerated_edge_and_gold_solution(self):
        problem,_=fixture();models={target:brute(problem,target) for target in ('edges','gold')}
        rng=random.Random(251105);helper=IntegerLatticePruner(problem)
        for _ in range(100):
            domains=[sum(1<<code for code in row if rng.random()<.75) for row in problem.domains]
            before=domains.copy();helper.prune(domains,max_candidates=1000)
            for target,rows in models.items():self.assertEqual(admitted(rows,domains),admitted(rows,before),target)
        self.assertLessEqual(len(helper.cache),4)

    def test_stop_and_caps_do_not_publish_incomplete_work(self):
        problem,_=fixture();original=[1754,2435,2592,2378]
        for limit in (1,10,40,100,300,800,1500,3000):
            helper=IntegerLatticePruner(problem);current=original.copy();ticks=[0]
            def stop():
                ticks[0]+=1
                if ticks[0]==limit:raise InterruptedError('stop')
            try:helper.prune(current,stop,max_candidates=1000)
            except InterruptedError:self.assertEqual(current,original)
            resumed=original.copy();helper.prune(resumed,max_candidates=1000)
            self.assertEqual(resumed,[1754,2434,2592,74])
        for limits in ({'max_placements':0},{'max_dimensions':0},{'max_operations':0}):
            helper=IntegerLatticePruner(problem);current=original.copy();helper.prune(current,**limits)
            self.assertEqual(current,original);self.assertGreater(helper.stats['lattice_caps'],0)

    def test_actual_research_stage_and_dfs_route_are_live(self):
        from constraint_round24 import ResearchPruner
        from constraint_dfs import solve
        from search_options import normalize_options
        from test_constraint_dfs import tetrahedron_tiles
        from search_problem import SearchProblem
        problem,_=fixture();current=[1754,2435,2592,2378]
        helper=ResearchPruner(problem,'gold',normalize_options({'integer_lattice':True}))
        helper.prune(current,lambda:None,lambda action,_rule:action())
        self.assertEqual(current,[1754,2434,2592,74])
        p=SearchProblem(*tetrahedron_tiles());answer=solve(p,seconds=10,target='edges',options={
            'integer_lattice':True,'native':'python','lookahead':False,'pair_lookahead':False,'adaptive':False})
        self.assertTrue(answer['complete']);self.assertIn(tuple(answer['codes']),brute(p,'edges'))
        self.assertGreater(answer['stats']['lattice_builds'],0)

    def test_fixed_resource_caps_are_reused_without_claiming_an_exclusion(self):
        problem,_=fixture();original=[1754,2435,2592,2378];helper=IntegerLatticePruner(problem)
        first=original.copy();helper.prune(first,max_operations=3);spent=helper.stats['lattice_integer_ops']
        second=original.copy();helper.prune(second,max_operations=3)
        self.assertEqual(first,original);self.assertEqual(second,original)
        self.assertEqual(helper.stats['lattice_integer_ops'],spent)
        helper.prune(second,max_operations=250000,max_candidates=1000)
        self.assertEqual(second,[1754,2434,2592,74])
        self.assertTrue(all(len(item.get('without',{}))<=1 for item in helper.cache.values()))

    def test_all160_cells_and_inventory_remain_in_near_completion_equations(self):
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        data,board,witness=planted_gold(251106,size=4);problem=SearchProblem(data,board)
        self.assertEqual(problem.n,160);self.assertTrue(problem.validate(witness)['valid'])
        domains=[1<<code for code in witness]
        for cell in (0,17,76,159):domains[cell]=7<<(3*(witness[cell]//3))
        helper=IntegerLatticePruner(problem);helper.prune(domains,max_candidates=1000)
        self.assertTrue(all(mask>>code&1 for mask,code in zip(domains,witness)))
        self.assertGreater(helper.stats['lattice_builds'],0);self.assertEqual(helper.stats['lattice_caps'],0)


if __name__=='__main__':unittest.main()
