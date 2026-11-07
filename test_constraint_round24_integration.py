"""Public DFS integration checks for optional research reasoning stages."""
import json,random,unittest
from constraint_dfs import solve
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,multiple_loops,double_face,tetrahedron_tiles,brute
from search_options import ROUND24_OPTIONS,normalize_options
from constraint_round24 import KEYS

class Round24IntegrationTests(unittest.TestCase):
    def test_each_optional_reasoner_preserves_finite_solutions(self):
        features=('resumable_reasoning','incremental_hall','witness_pool','support_cover','retained_relations','joint_conservation','affine_reasoning','pb_learning','interface_formulas','obstruction_rules','symbolic_regions','symbolic_inventory','elimination','mini_buckets','relaxed_diagrams','representative_connectivity','subset_joins','lazy_regions','certified_relaxation','labelled_routing','dual_piece_branching','strand_branching','completion_ordering')
        rng=random.Random(2401)
        cases=[SearchProblem(simple_tiles(),double_face()),SearchProblem(multiple_loops(),double_face()),SearchProblem(*tetrahedron_tiles())]
        for _ in range(3):
            p=SearchProblem(*tetrahedron_tiles());p.domains=[[v for v in row if rng.random()<.6] for row in p.domains];cases.append(p)
        for p in cases:
            for target in ('gold','edges'):
                expected=brute(p,target)
                for feature in features:
                    with self.subTest(n=p.n,target=target,feature=feature):
                        answer=solve(p,seconds=15,target=target,options={'native':'python',feature:True})
                        self.assertTrue(answer['complete'],answer)
                        if expected:self.assertIn(tuple(answer['codes']),expected)
                        else:self.assertEqual(answer['status'],'infeasible')
    def test_combination_warm_restart_and_normalized_dependencies(self):
        p=SearchProblem(*tetrahedron_tiles());options={key:True for key in KEYS}
        options.update(native='python',retained_relations=True,symbolic_regions=True,symbolic_inventory=True,witness_pool=True,incremental_hall=True)
        state={};first=solve(p,seconds=0,options=options,learning_state=state)
        self.assertEqual(first['status'],'timeout')
        answer=solve(p,seconds=30,resume=json.loads(json.dumps(first['checkpoint'])),options=options,learning_state=state)
        self.assertEqual(answer['status'],'solved');self.assertTrue(answer['validation']['valid'])
        self.assertTrue(normalize_options({'witness_pool':True})['sat_assist'])
        self.assertTrue(normalize_options({'proof_transport':True})['smart_learning'])

if __name__=='__main__':unittest.main()
