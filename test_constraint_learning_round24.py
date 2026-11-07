"""Independent finite checks of round-24 relational and arithmetic reasoning."""
import copy
import itertools
import random
import unittest
from collections import Counter
from constraint_relations import ScopedRelations,capacity_supports,bits
from constraint_algebra import affine_equations,modular_affine_equations,AffinePruner
from constraint_deep import DeepPruner
from constraint_pb import row,derive_rows,verify_derivation,filter_rows,primitive_rows,PBLearner,ObstructionMiner
from constraint_interface import regional_cnf,resolution_projection,check_resolution,InterfaceProjector,ProjectionLimit,apply_interface
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,brute
from test_constraint_sat_domains import fixture,admitted
from test_constraint_sat_assist import HAS_SAT


def domains_of(problem):return [sum(1<<v for v in row) for row in problem.domains]


def satisfies_rows(rows,assignment):
    return all(sum(k for c,v,k in terms if assignment[c]==v)<=rhs for terms,rhs in rows)


def cnf_models(clauses,variables):
    return [tuple(row) for row in itertools.product((False,True),repeat=variables)
            if all(any(row[abs(lit)-1]==(lit>0) for lit in clause) for clause in clauses)]


class CapacityTests(unittest.TestCase):
    def test_joint_shortage_passes_individual_pattern_bounds(self):
        choices=[{0,1}]*3+[{2,3}]*3;caps={0:1,1:1,2:2,3:2}
        for label,total in caps.items():
            self.assertLessEqual(sum(row=={label} for row in choices),total)
            self.assertGreaterEqual(sum(label in row for row in choices),total)
        self.assertIsNone(capacity_supports(choices,caps))

    def test_exact_capacity_supports_match_independent_assignments(self):
        rng=random.Random(240013)
        for n in range(1,6):
            for _ in range(80):
                reference=[rng.randrange(3) for _ in range(n)];caps=dict(Counter(reference))
                choices=[{v for v in range(3) if rng.random()<.65} for _ in range(n)]
                models=[r for r in itertools.product(*choices) if Counter(r)==Counter(caps)]
                actual=capacity_supports(choices,caps)
                if not models:self.assertIsNone(actual)
                else:self.assertEqual(actual,tuple(frozenset(row[c] for row in models) for c in range(n)))

    def test_joint_puzzle_filter_preserves_all_edge_arrangements(self):
        problem,domain=fixture();solutions=brute(problem,'edges');rng=random.Random(240130)
        helper=DeepPruner(problem)
        for _ in range(70):
            current=[sum(1<<v for v in row if rng.random()<.8) for row in problem.domains]
            before=admitted(solutions,current);helper.joint_conservation(current)
            self.assertEqual(admitted(solutions,current),before)


class RelationalTests(unittest.TestCase):
    def test_pair_correlation_survives_without_initial_unary_deduction(self):
        helper=ScopedRelations();helper.remember(((0,3),(1,3)),0,1,((0,0),(1,1)))
        domains=[3,3];self.assertFalse(helper.prune(domains))
        domains[0]=1;self.assertTrue(helper.prune(domains));self.assertEqual(domains,[1,1])
        widened=[5,3];self.assertFalse(helper.prune(widened));self.assertEqual(widened,[5,3])

    def test_retained_path_relations_preserve_every_completion_on_narrowing(self):
        problem,domains=fixture();solutions=brute(problem,'edges');rng=random.Random(240011)
        helper=DeepPruner(problem,retain_relations=True)
        for _ in range(50):
            current=[sum(1<<v for v in row if rng.random()<.7) for row in problem.domains]
            expected=admitted(solutions,current)
            helper.paths(current,max_values=12)
            helper.retained_relations(current)
            self.assertEqual(admitted(solutions,current),expected)
            if all(current):
                cell=rng.randrange(problem.n);current[cell]&=sum(1<<v for v in bits(current[cell]) if rng.random()<.7)
                expected=admitted(solutions,current);helper.retained_relations(current)
                self.assertEqual(admitted(solutions,current),expected)
        self.assertGreater(helper.stats['relation_records'],0)


class AlgebraTests(unittest.TestCase):
    def test_every_three_bit_affine_hull_matches_independent_xor_closure(self):
        for selection in range(256):
            patterns=[v for v in range(8) if selection>>v&1]
            equations=affine_equations(patterns,3)
            actual={v for v in range(8) if all((v&m).bit_count()%2==b for m,b in equations)}
            if not patterns:expected=set()
            else:
                origin=patterns[0];span={0}
                for v in patterns:span|={x^v^origin for x in tuple(span)}
                expected={origin^v for v in span}
            self.assertEqual(actual,expected)

    def test_modular_reference_detects_relation_missed_by_binary_affine_hull(self):
        patterns=[0,7,11,13,14]
        self.assertEqual(affine_equations(patterns,4),())
        vectors=[tuple((v>>i)&1 for i in range(4)) for v in patterns]
        equations=modular_affine_equations(vectors,4,3)
        self.assertTrue(equations)
        self.assertTrue(all(sum(a*b for a,b in zip(row,coeff))%3==rhs for row in vectors for coeff,rhs in equations))
        self.assertFalse(all(sum(a*b for a,b in zip((1,1,0,0),coeff))%3==rhs for coeff,rhs in equations))

    def test_affine_puzzle_deductions_preserve_edge_solutions(self):
        problem,_=fixture();solutions=brute(problem,'edges');rng=random.Random(240014)
        helper=AffinePruner(problem)
        for _ in range(80):
            domains=[sum(1<<v for v in row if rng.random()<.75) for row in problem.domains]
            expected=admitted(solutions,domains);helper.prune(domains,max_candidates=128)
            self.assertEqual(admitted(solutions,domains),expected)
        self.assertGreater(helper.stats['affine_prunes'],0)

    def test_affine_interruption_preserves_callers_domains(self):
        problem,original=fixture()
        for limit in (1,10,100,300):
            helper=AffinePruner(problem);domains=original.copy();calls=[0]
            def check():
                calls[0]+=1
                if calls[0]==limit:raise InterruptedError
            try:helper.prune(domains,check)
            except InterruptedError:self.assertEqual(domains,original)


class PBTests(unittest.TestCase):
    def test_combined_integer_cuts_close_nonunit_capacity_obstruction(self):
        base=[row([(0,1,-1),(1,1,-1)],-1),row([(0,1,-1),(2,1,-1)],-1),
              row([(1,1,-1),(2,1,-1)],-1),row([(0,1,1),(1,1,1),(2,1,1)],1)]
        old=[3]*3;self.assertFalse(filter_rows(old,base))
        derived,steps,_=derive_rows(base,max_pairs=128)
        self.assertTrue(verify_derivation(base,steps));self.assertTrue(steps)
        result=[3]*3;filter_rows(result,derived);self.assertIn(0,result)
        self.assertFalse([v for v in itertools.product(range(2),repeat=3) if satisfies_rows(base,v)])
        damaged=list(steps);a,b,x,y,cut=damaged[0];damaged[0]=(a,b,x,y,(cut[0],cut[1]-1))
        self.assertFalse(verify_derivation(base,damaged))

    def test_random_signed_integer_derivations_preserve_all_models(self):
        rng=random.Random(240015)
        for _ in range(80):
            base=[row([(c,1,rng.randrange(-2,3)) for c in range(4)],rng.randrange(-3,4)) for _ in range(5)]
            derived,steps,_=derive_rows(base,max_pairs=48,max_new=16)
            self.assertTrue(verify_derivation(base,steps))
            expected=[v for v in itertools.product(range(2),repeat=4) if satisfies_rows(base,v)]
            self.assertEqual([v for v in itertools.product(range(2),repeat=4) if satisfies_rows(derived,v)],expected)
            domains=[3]*4;filter_rows(domains,derived)
            self.assertEqual(admitted(expected,domains),expected)

    def test_puzzle_primitive_derived_and_discovered_rules_preserve_all_solutions(self):
        problem,_=fixture();solutions=brute(problem,'edges');rng=random.Random(240032)
        helper=PBLearner(problem,derive=True,discover=True)
        for _ in range(45):
            domains=[sum(1<<v for v in row if rng.random()<.6) for row in problem.domains]
            expected=admitted(solutions,domains);helper.prune(domains,max_values=12)
            self.assertEqual(admitted(solutions,domains),expected)
        self.assertGreater(helper.stats['obstruction_rules'],0)
        for record in helper.records.values():
            for rule in record['rules']:self.assertTrue(helper.miner.verify(rule))
        bad={'kind':'clique','events':((0,1),(1,8)),'row':row([(0,0,1),(1,3,1)],0)}
        self.assertFalse(helper.miner.verify(bad))


class InterfaceTests(unittest.TestCase):
    def test_resolution_projection_matches_existential_truth_table(self):
        rng=random.Random(240025)
        for _ in range(100):
            base=[]
            for k in range(6):
                variables=rng.sample(range(1,5),rng.randrange(1,4))
                base.append(tuple(sorted(v*rng.choice((-1,1)) for v in variables)))
            base=tuple(dict.fromkeys(base))
            record=resolution_projection(base,(2,3))
            self.assertTrue(check_resolution(base,record))
            original={(r[0],r[3]) for r in cnf_models(base,4)}
            projected={(r[0],r[3]) for r in cnf_models(record['final'],4)}
            self.assertEqual(original,projected)
            if record['steps']:
                damaged=copy.deepcopy(record);steps=list(damaged['steps']);a,b,pivot,clause=steps[0]
                steps[0]=(a,b,pivot,());damaged['steps']=tuple(steps)
                if clause:self.assertFalse(check_resolution(base,damaged))

    def test_interface_reuse_is_scoped_and_preserves_all_puzzle_completions(self):
        problem,domains=fixture();solutions=brute(problem,'edges');helper=InterfaceProjector(problem)
        scope=tuple(enumerate(domains));record=helper.compile(scope,(0,3),max_steps=100000,max_clauses=10000)
        self.assertIsNotNone(record)
        self.assertTrue(all(record['decode'][abs(lit)][0] in (0,3) for clause in record['final'] for lit in clause))
        rng=random.Random(240250)
        for _ in range(50):
            current=[sum(1<<v for v in bits(mask) if rng.random()<.8) for mask in domains]
            expected=admitted(solutions,current);apply_interface(current,scope,record['decode'],record['final'])
            self.assertEqual(admitted(solutions,current),expected)
        self.assertIs(helper.compile(scope,(0,3)),record)
        self.assertEqual(helper.stats['interface_reuses'],1)
        widened=domains_of(problem);before=widened.copy()
        self.assertFalse(apply_interface(widened,scope,record['decode'],record['final']));self.assertEqual(widened,before)

    def test_budget_exhaustion_publishes_no_formula(self):
        problem,domains=fixture();helper=InterfaceProjector(problem)
        self.assertIsNone(helper.compile(tuple(enumerate(domains)),(0,),max_steps=0))
        self.assertFalse(helper.records);self.assertEqual(helper.stats['interface_unknown'],1)


@unittest.skipUnless(HAS_SAT,'optional python-sat missing')
class NativeTheoryTests(unittest.TestCase):
    def test_partial_hall_failure_produces_globally_valid_explained_clause(self):
        from constraint_sat_connectivity import gold_propagator
        problem,_=fixture();variables=[];nextid=0
        for choices in problem.domains:
            row={}
            for code in choices:nextid+=1;row[code]=nextid
            variables.append(row)
        helper=gold_propagator(problem,variables,theory_hall=True)
        helper.on_new_level()
        for cell in range(3):
            for code,lit in variables[cell].items():
                if code//3 not in (0,1):helper.on_assignment(-lit)
        helper.propagate();self.assertIsNone(helper.error)
        self.assertGreater(helper.stats['sat_external_hall_conflicts'],0)
        clauses=[]
        while helper.has_clause():clauses.append(helper.add_clause())
        for solution in brute(problem,'edges'):
            chosen={variables[c][code] for c,code in enumerate(solution)}
            self.assertTrue(all(any(lit in chosen if lit>0 else -lit not in chosen for lit in clause) for clause in clauses))
        helper.on_backtrack(0);helper.propagate();self.assertFalse(helper.has_clause())

if __name__=='__main__':unittest.main()
