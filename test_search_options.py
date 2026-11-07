import unittest
from search_options import DEFAULT_OPTIONS, normalize_options, validate_workers, validate_algorithm_options


class OptionsTests(unittest.TestCase):
    def test_default_and_partial_options_do_not_mutate_callers(self):
        given = {'lookahead': False}
        result = normalize_options(given)
        self.assertFalse(result['lookahead'])
        self.assertTrue(result['regional_cache'])
        for name in ('biconnected','separator_regions','backjumping','incremental','adaptive','persistent_learning'):
            self.assertTrue(result[name])
            self.assertFalse(normalize_options({name:False})[name])
            with self.assertRaises(ValueError): normalize_options({name:1})
        self.assertEqual(given, {'lookahead': False})
        result['native'] = 'python'
        self.assertEqual(normalize_options(), DEFAULT_OPTIONS)

    def test_experimental_options_are_explicit_and_strictly_boolean(self):
        for name in ('reason_tracking','sat_assist','hierarchical_regions','value_ordering','shared_conflicts','gold_oracle','connection_branching','lcg','native_search'):
            self.assertFalse(normalize_options()[name])
            self.assertTrue(normalize_options({name:True})[name])
            with self.assertRaises(ValueError):normalize_options({name:1})

    def test_lookahead_budget_and_pair_control_are_strict(self):
        self.assertTrue(normalize_options()['pair_lookahead'])
        self.assertEqual(normalize_options()['lookahead_ms'],100)
        self.assertFalse(normalize_options({'pair_lookahead':False})['pair_lookahead'])
        for value in (True,0,1001,20.0,'20',None):
            with self.subTest(value=value),self.assertRaises(ValueError):
                normalize_options({'lookahead_ms':value})
        for value in (1,20,100,1000):
            self.assertEqual(normalize_options({'lookahead_ms':value})['lookahead_ms'],value)
        with self.assertRaises(ValueError):normalize_options({'pair_lookahead':1})

    def test_round25_controls_require_an_engine_that_executes_them(self):
        for algorithm,key in (('dfs','sat_core_lifting'),('hybrid','symbolic_coverage'),
                              ('sat','mixed_proof_dag'),('gpu-domains','symbolic_coverage')):
            with self.subTest(algorithm=algorithm,key=key),self.assertRaises(ValueError):
                validate_algorithm_options(algorithm,normalize_options({key:True}))
        for algorithm,key in (('sat','sat_core_lifting'),('dfs','symbolic_coverage'),
                              ('hybrid','mixed_proof_dag'),('parallel-dfs','shared_sat_assistant')):
            validate_algorithm_options(algorithm,normalize_options({key:True}))
        with self.assertRaises(ValueError):
            validate_algorithm_options('dfs',normalize_options({'native_explanations':True,'native':'python'}))
        with self.assertRaises(ValueError):
            validate_algorithm_options('sat',normalize_options({'symbolic_coverage':True}))
        with self.assertRaises(ValueError):
            validate_algorithm_options('sat',normalize_options({'persistent_sat':True,'sat_theory':True}))
        validate_algorithm_options('sat',normalize_options({'persistent_sat':True,'symbolic_coverage':True}))

    def test_lazy_gold_scoring_implies_gold_and_chain_generator(self):
        options=normalize_options({'lazy_gold_scoring':True,'gpu_gold_sampling':False,'ejection_chains':False})
        validate_algorithm_options('hybrid',options)
        self.assertTrue(options['gpu_gold_sampling']);self.assertTrue(options['ejection_chains'])
        for engine in ('dfs','parallel-dfs','gpu-domains','sat','cp-sat','cp'):
            with self.assertRaises(ValueError):validate_algorithm_options(engine,options)
        self.assertFalse(normalize_options()['lazy_gold_scoring'])
        with self.assertRaises(ValueError):normalize_options({'lazy_gold_scoring':1})

    def test_cycle_crossover_is_boolean_default_off_and_hybrid_only(self):
        options=normalize_options({'cycle_crossover':True})
        validate_algorithm_options('hybrid',options)
        for engine in ('dfs','parallel-dfs','gpu-domains','sat','cp-sat','cp'):
            with self.assertRaises(ValueError):validate_algorithm_options(engine,options)
        self.assertFalse(normalize_options()['cycle_crossover'])
        with self.assertRaises(ValueError):normalize_options({'cycle_crossover':1})

    def test_obstruction_chains_enable_ejection_chains_for_hybrid_only(self):
        options=normalize_options({'obstruction_chains':True,'ejection_chains':False})
        self.assertTrue(options['ejection_chains']);self.assertFalse(options['durable_hybrid'])
        validate_algorithm_options('hybrid',options)
        for algorithm in ('dfs','parallel-dfs','sat','gpu-domains'):
            with self.subTest(algorithm=algorithm),self.assertRaises(ValueError):validate_algorithm_options(algorithm,options)
        self.assertFalse(normalize_options()['obstruction_chains'])
        with self.assertRaises(ValueError):normalize_options({'obstruction_chains':1})

    def test_invalid_controls_fail_before_search(self):
        for value in ([], True, {'unexpected': True}, {'lookahead': 1}, {'native': 'gpu'}, {'regional_cache': None}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_options(value)
        for value in (True, 0, 65, 2.0, '4'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_workers(value)
        self.assertEqual(validate_workers(1), 1)
        self.assertEqual(validate_workers(64), 64)


if __name__ == '__main__':
    unittest.main()
