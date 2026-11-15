"""Finite native and host checks for alternative initial GPU partitions."""
import itertools
import os
import unittest

from bench_r34_partitions import cover, partition, run
from test_gpu_domain_round25 import joint_hall_fixture, independent_solutions


class SplitPolicies(unittest.TestCase):
    def test_all_policy_partitions_cover_unequal_and_multiple_roots(self):
        root = (0b100111, 0b11001, 1, 0b11)
        source = list(itertools.product(*[[v for v in range(m.bit_length()) if m & (1 << v)]
                                          for m in root]))
        for policy in ('latest', 'oldest', 'volume', 'anchor'):
            for count in (1, 2, 7, 11, 32):
                first = partition([root], 2, policy)
                rows = partition(first if count >= 2 else [root], count, policy)
                cover(root, rows)
                for codes in source:
                    self.assertEqual(sum(all(v & (1<<c) for v,c in zip(row,codes)) for row in rows), 1)
                self.assertEqual(len(rows), min(count,len(source)))

    def test_algorithm_option_is_optional_and_limited_to_domain_cuda(self):
        from gpu_domain_runner import solve
        from search_options import normalize_options, validate_algorithm_options
        self.assertFalse(normalize_options()['gpu_small_domain_split'])
        opts = normalize_options({'gpu_small_domain_split':True})
        validate_algorithm_options('gpu-domains',opts)
        with self.assertRaises(ValueError):
            validate_algorithm_options('dfs',opts)
        with self.assertRaises(ValueError):
            normalize_options({'gpu_small_domain_split':1})
        with self.assertRaises(ValueError):
            solve(joint_hall_fixture(),options={'gpu_small_domain_split':1})

    def test_dashboard_posts_enabled_gpu_partition_and_rejects_other_engines(self):
        from test_constraint_ui_round25 import Inputs, ROOT, Round25Controls
        parser = Inputs()
        parser.feed((ROOT/'Dashboard.html').read_text(encoding='utf-8'))
        self.assertEqual(len(parser.inputs['r34_gpu_small_domain_split']),1)
        Round25Controls.node(self, r"""
const control=element('r34_gpu_small_domain_split');
select('dfs');assert.equal(control.checked,false);assert.equal(control.disabled,true);
select('gpu-domains');assert.equal(control.disabled,false);
control.checked=true;control.listeners.change();
element('seconds').value='0';element('seedInput').value='17';run("command('start')");
assert.equal(JSON.parse(context.fetchCalls.at(-1)[1].body).search_options.gpu_small_domain_split,true);
idle();select('dfs');assert.equal(element('startBtn').disabled,true);
control.checked=false;control.listeners.change();assert.equal(control.disabled,true);
""")

    @unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA') == '1', 'CUDA controls require opt-in')
    def test_switching_policy_preserves_live_native_and_resumes_the_app(self):
        from constraint_work import WorkScope
        from gpu_domain_runner import solve
        p = joint_hall_fixture()
        root = WorkScope.for_problem(p,'edges').root
        expected = independent_solutions(p,'edges')
        for before in (False,True):
            def pause(**kwargs):
                polls = iter((False,True))
                return solve(p,replicas=4,target='edges',stop=lambda:next(polls,True),**kwargs)['checkpoint']
            initial = pause(options={'gpu_small_domain_split':before})
            original = [tuple(int(v,16) for v in row) for row in initial['native']['cubes']]
            self.assertEqual(original,partition([root],4,'anchor' if before else 'latest'))
            restored = pause(options={'gpu_small_domain_split':not before},resume=initial)
            self.assertEqual(initial['frontier'],restored['frontier'])
            self.assertEqual(initial['native'],restored['native'])
            answer = solve(p,replicas=4,target='edges',resume=restored,options={'gpu_small_domain_split':not before})
            self.assertIn(tuple(answer['codes']),expected)

    @unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA') == '1', 'CUDA controls require opt-in')
    def test_native_pause_resume_preserves_entire_solution_set_and_counts(self):
        p = joint_hall_fixture()
        expected = independent_solutions(p,'edges')
        for policy in ('latest','oldest','volume','anchor'):
            for fork in (False,True):
                a, found = run(p,'edges',4,policy,fork,options={},quantum=16)
                b, restored = run(p,'edges',4,policy,fork,options={},quantum=16,resume=True)
                self.assertTrue(b['restored'])
                self.assertEqual(found,expected)
                self.assertEqual(restored,expected)
                self.assertEqual((a['status'],a['work']),('exhausted',b['work']))
