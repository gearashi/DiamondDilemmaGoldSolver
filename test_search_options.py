import unittest
from search_options import DEFAULT_OPTIONS, normalize_options, validate_workers


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
        for name in ('reason_tracking','sat_assist','hierarchical_regions','value_ordering','shared_conflicts'):
            self.assertFalse(normalize_options()[name])
            self.assertTrue(normalize_options({name:True})[name])
            with self.assertRaises(ValueError):normalize_options({name:1})

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
