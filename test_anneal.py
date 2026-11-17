"""Targeted annealing: kernel invariants, exact completion check, runner, launch paths.

Uses planted fixtures only, so no puzzle data download is needed. The compiled
kernel is exercised when a C compiler is present and skipped otherwise; the
Python kernel always runs.
"""
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import anneal_search
import dashboard_server as dashboard
import platform_launcher as launcher
from anneal_core import AnnealModel, completion_check, cooling_temperature
from search_algorithms import ALGORITHMS, LABELS, LAUNCHABLE, SCORE_ALGORITHMS
from search_fixtures import planted_gold
from validator import validate_arrangement

ROOT = Path(__file__).resolve().parent


def compiled_available():
    data, board, _ = planted_gold(17, 1)
    try:
        return AnnealModel(data, board, native='compiled').backend == 'compiled'
    except RuntimeError:
        return False


COMPILED = compiled_available()
BACKENDS = ('python', 'compiled') if COMPILED else ('python',)


class KernelTests(unittest.TestCase):
    def test_table_score_equals_the_independent_validator(self):
        for size in (1, 2, 4):
            data, board, planted = planted_gold(291, size)
            model = AnnealModel(data, board, native='python')
            self.assertEqual(model.score(planted), model.total_edges)
            rng = random.Random(size)
            for _ in range(5):
                codes = model.random_board(rng)
                self.assertEqual(model.score(codes), validate_arrangement(data, codes, board)['matched_edges'])

    def test_each_backend_keeps_its_score_and_tile_invariants(self):
        data, board, _ = planted_gold(817, 2)
        for backend in BACKENDS:
            for targeted in (0, 512, 1024):
                with self.subTest(backend=backend, targeted=targeted):
                    model = AnnealModel(data, board, native=backend)
                    codes, state, best = model.random_board(random.Random(3)), 99, None
                    for step in range(6):
                        codes, state, score, best_codes, best_score, accepted = model.run(
                            codes, state, 3000, cooling_temperature(step, 6), targeted=targeted, best=best)
                        best = (best_codes, best_score)
                        self.assertEqual(score, model.score(codes))
                        self.assertEqual(best_score, model.score(best_codes))
                        self.assertGreaterEqual(best_score, score)
                        self.assertGreaterEqual(accepted, 0)
                    self.assertEqual(sorted(code // 3 for code in codes), list(range(model.n)))

    def test_same_inputs_reproduce_the_same_board(self):
        data, board, _ = planted_gold(817, 2)
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                model = AnnealModel(data, board, native=backend)
                start = model.random_board(random.Random(8))
                self.assertEqual(model.run(start, 1234, 5000, 0.3), model.run(start, 1234, 5000, 0.3))

    def test_annealing_reaches_every_seam_on_a_small_planted_board(self):
        data, board, _ = planted_gold(17, 1)
        for backend in BACKENDS:
            with self.subTest(backend=backend):
                model = AnnealModel(data, board, native=backend)
                found = -1
                for seed in range(1, 9):
                    codes, state, best = model.random_board(random.Random(seed)), seed, None
                    for step in range(40):
                        codes, state, _, best_codes, best_score, _ = model.run(
                            codes, state, 2000, cooling_temperature(step, 40), best=best)
                        best = (best_codes, best_score)
                    found = max(found, best[1])
                    if found == model.total_edges:
                        break
                self.assertEqual(found, model.total_edges)
                self.assertEqual(validate_arrangement(data, best[0], board)['matched_edges'], model.total_edges)

    def test_invalid_requests_are_rejected(self):
        data, board, planted = planted_gold(17, 1)
        model = AnnealModel(data, board, native='python')
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            model.run([planted[0]] * model.n, 1, 10, 0.3)
        with self.assertRaisesRegex(ValueError, 'temperature'):
            model.run(planted, 1, 10, 0)
        with self.assertRaisesRegex(ValueError, 'targeted'):
            model.run(planted, 1, 10, 0.3, targeted=2000)
        with self.assertRaisesRegex(ValueError, 'native'):
            AnnealModel(data, board, native='gpu')


class CompletionTests(unittest.TestCase):
    def setUp(self):
        self.data, self.board, self.planted = planted_gold(291, 2)
        self.model = AnnealModel(self.data, self.board, native='python')

    def test_a_matched_board_needs_no_search(self):
        self.assertEqual(completion_check(self.model, self.planted)['status'], 'already_matched')

    def test_displaced_tiles_are_put_back(self):
        # Positive control: a known completion exists, so the exact search must find one.
        codes = list(self.planted)
        codes[3], codes[20], codes[31] = codes[20], codes[31], codes[3]
        codes[10] = 3 * (codes[10] // 3) + (codes[10] + 1) % 3
        self.assertLess(self.model.score(codes), self.model.total_edges)
        result = completion_check(self.model, codes)
        self.assertEqual(result['status'], 'completion_found')
        self.assertEqual(result['validation']['matched_edges'], self.model.total_edges)
        self.assertTrue(result['validation']['unique_tiles'])
        self.assertEqual(self.model.score(result['codes']), self.model.total_edges)

    def test_an_unmatchable_tile_gives_no_completion(self):
        # One tile is redrawn with a pattern no other tile mirrors, so the cells
        # around it can never all match, whatever is placed there.
        data = json.loads(json.dumps(self.data))
        tiles = data['tiles'] if isinstance(data, dict) else data
        victim = self.planted[7] // 3
        tiles[victim]['segments'] = [[[0, 1], [0, 2]], [[0, 3], [0, 5]]]
        model = AnnealModel(data, self.board, native='python')
        self.assertLess(model.score(self.planted), model.total_edges)
        for rings in (0, 1):
            result = completion_check(model, self.planted, rings=rings)
            self.assertEqual(result['status'], 'no_completion', result)
            self.assertGreater(result['free_cells'], 0)

    def test_the_node_limit_reports_undetermined_not_a_refutation(self):
        codes = list(self.planted)
        codes[3], codes[20] = codes[20], codes[3]
        self.assertEqual(completion_check(self.model, codes, node_limit=1)['status'], 'undetermined')


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='anneal-test-', dir=ROOT)
        self.directory = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        data, self.board, self.planted = planted_gold(817, 4)
        self.data_path = self.directory / 'tiles.json'
        self.data_path.write_text(json.dumps(data), encoding='utf-8')
        self.data = data

    def run_main(self, *extra):
        return anneal_search.main(['--data', str(self.data_path), '--output', str(self.directory / 'runtime'),
                                   '--workers', '2', '--moves', '60000', '--native', 'python', *extra])

    def read(self, name):
        return json.loads((self.directory / 'runtime' / name).read_text(encoding='utf-8'))

    def test_a_short_run_saves_a_validated_best_board_and_resumes(self):
        self.assertEqual(self.run_main('--seconds', '1.5'), 0)
        status, best = self.read('status.json'), self.read('best.json')
        self.assertIn(status['state'], ('time_limit', 'solved', 'edge_perfect'))
        self.assertEqual((status['method'], status['algorithm'], status['target']), ('anneal', 'anneal', 'edges'))
        report = validate_arrangement(self.data, best['codes'], self.board)
        self.assertEqual(report['matched_edges'], best['validation']['matched_edges'])
        self.assertEqual(status['best_matched_edges'], report['matched_edges'])
        self.assertTrue(report['unique_tiles'])
        self.assertFalse((self.directory / 'runtime' / 'run.lock').exists())
        checkpoint = self.read('searches/anneal-edges/checkpoint.json')
        self.assertEqual(len(checkpoint['chains']), 2)
        history = self.read('history.json')
        self.assertEqual(history[-1]['score'], report['matched_edges'])
        self.assertEqual([p['score'] for p in history], sorted(p['score'] for p in history))
        if status['state'] != 'time_limit':
            return
        moves = checkpoint['moves_done']
        self.assertEqual(self.run_main('--seconds', '1', '--resume'), 0)
        again = self.read('status.json')
        self.assertTrue(again['resumed'])
        self.assertGreater(again['moves_done'], moves)
        self.assertGreaterEqual(again['best_matched_edges'], status['best_matched_edges'])

    def test_resume_continues_each_chain_exactly(self):
        model = AnnealModel(self.data, native='python')
        direct = anneal_search.Chain(model, 5, 0)
        saved = anneal_search.Chain(model, 5, 0)
        for step in range(4):
            for chain in (direct, saved):
                chain.codes, chain.rng_state, chain.score, chain.best_codes, chain.best_score, _ = model.run(
                    chain.codes, chain.rng_state, 400, cooling_temperature(chain.step, anneal_search.COOLING_STEPS),
                    best=(chain.best_codes, chain.best_score))
                chain.step += 1
            saved = anneal_search.Chain.from_json(model, json.loads(json.dumps(saved.to_json())))
        self.assertEqual(saved.to_json(), direct.to_json())

    def test_changed_settings_start_fresh_chains_but_keep_the_best_board(self):
        self.assertEqual(self.run_main('--seconds', '1'), 0)
        first = self.read('best.json')['validation']['matched_edges']
        self.assertEqual(self.run_main('--seconds', '1', '--resume', '--seed', '7'), 0)
        status = self.read('status.json')
        self.assertFalse(status['resumed'])
        self.assertIn('different settings', status['checkpoint_note'])
        self.assertGreaterEqual(status['best_matched_edges'], first)

    def test_a_corrupt_checkpoint_is_ignored_with_a_note(self):
        self.assertEqual(self.run_main('--seconds', '1'), 0)
        path = self.directory / 'runtime' / 'searches' / 'anneal-edges' / 'checkpoint.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        payload['chains'][0]['codes'][0] = payload['chains'][0]['codes'][1]
        path.write_text(json.dumps(payload), encoding='utf-8')
        self.assertEqual(self.run_main('--seconds', '1', '--resume'), 0)
        status = self.read('status.json')
        self.assertFalse(status['resumed'])
        self.assertIn('checkpoint', status['checkpoint_recovery'])

    def test_stop_request_ends_an_unlimited_run(self):
        runtime = self.directory / 'runtime'
        runtime.mkdir()
        (runtime / 'stop.request').write_text('stop', encoding='utf-8')
        self.assertEqual(self.run_main('--seconds', '0', '--stop-file-initialized'), 0)
        self.assertEqual(self.read('status.json')['state'], 'stopped')

    def test_an_active_lock_refuses_a_second_runner(self):
        runtime = self.directory / 'runtime'
        runtime.mkdir()
        (runtime / 'run.lock').write_text('1', encoding='utf-8')
        with self.assertRaises(SystemExit):
            self.run_main('--seconds', '1')
        self.assertTrue((runtime / 'run.lock').exists())

    def test_settings_validation(self):
        good = dict(seconds=10, seed=1, workers=4, moves=10**6, hot=0.5, cold=0.09, targeted=960)
        anneal_search.validate_settings(**good)
        for change in ({'seconds': -1}, {'seed': 2**32}, {'workers': 0}, {'workers': 65}, {'moves': 10},
                       {'cold': 0.9}, {'cold': 0}, {'targeted': 1025}, {'seconds': float('nan')}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                anneal_search.validate_settings(**{**good, **change})


class LaunchTests(unittest.TestCase):
    def test_registry_keeps_gold_engines_separate_from_score_engines(self):
        self.assertEqual(SCORE_ALGORITHMS, ('anneal',))
        self.assertNotIn('anneal', ALGORITHMS)
        self.assertEqual(LAUNCHABLE, ALGORITHMS + SCORE_ALGORITHMS)
        self.assertIn('anneal', LABELS)

    def test_launcher_builds_the_annealing_command(self):
        command = launcher.solver_command(hours=2, seed=42, algorithm='anneal', workers=8)
        self.assertEqual(Path(command[1]).name, 'anneal_search.py')
        self.assertEqual(command[2:], ['--seconds', '7200', '--seed', '42', '--workers', '8', '--resume'])
        fresh = launcher.solver_command(unlimited=True, algorithm='anneal', fresh=True)
        self.assertEqual(fresh[fresh.index('--seconds') + 1], '0')
        self.assertNotIn('--resume', fresh)
        for extra in ({'replicas': 128}, {'backend': 'cuda'}, {'search_options': {}}, {'gpu_features': {'dynamic_mrv': True}}):
            with self.subTest(extra=extra), self.assertRaisesRegex(ValueError, 'Annealing takes'):
                launcher.solver_command(algorithm='anneal', **extra)

    def test_windows_start_script_routes_annealing_to_its_runner(self):
        script = (ROOT / 'StartSolver.ps1').read_bytes()
        self.assertTrue(script.startswith(b'\xef\xbb\xbf'))
        text = script.decode('utf-8-sig')
        self.assertEqual(text.count('\n'), text.count('\r\n'))
        self.assertIn("'parallel-dfs','anneal')][string]$Algorithm", text)
        branch = "if ($Algorithm -eq 'anneal') { $SolverArgs = @((Join-Path $PSScriptRoot $SolverScript), '--stop-file-initialized', '--seconds', [string]$SolverSeconds, '--seed', [string]$Seed, '--workers', [string]$Workers) }"
        self.assertIn(branch, text)
        self.assertIn("elseif ($Algorithm -eq 'anneal') { 'anneal_search.py' }", text)

    def test_dashboard_starts_annealing_and_reports_startup(self):
        with tempfile.TemporaryDirectory(prefix='anneal-dash-test-', dir=ROOT) as temp:
            directory = Path(temp)
            child = SimpleNamespace(pid=5678, poll=lambda: None)
            with patch.object(dashboard, 'RUNTIME', directory), patch.object(dashboard, 'CHILD', None), \
                    patch.object(dashboard, 'LAUNCH_CONFIG', {}), patch.object(dashboard, 'live_pid', return_value=None), \
                    patch.object(dashboard.subprocess, 'Popen', return_value=child) as spawn:
                (directory / 'stop.request').write_text('old', encoding='utf-8')
                result = dashboard.start_solver({'algorithm': 'anneal', 'method': 'anneal', 'seconds': 600, 'seed': 9, 'workers': 2})
                command = spawn.call_args.args[0]
                self.assertEqual(Path(command[1]).name, 'anneal_search.py')
                self.assertEqual(command[2:], ['--resume', '--stop-file-initialized', '--seconds', '600.0', '--seed', '9', '--workers', '2'])
                self.assertEqual((result['method'], result['target'], result['backend'], result['workers']), ('anneal', 'edges', 'cpu', 2))
                self.assertFalse((directory / 'stop.request').exists())
                with patch.object(dashboard, 'live_pid', return_value=5678):
                    status = dashboard.state_payload()['status']
                self.assertEqual((status['state'], status['algorithm'], status['method'], status['target']),
                                 ('starting', 'anneal', 'anneal', 'edges'))
                self.assertEqual(status['startup_phase'], 'building_model')

    def test_dashboard_rejects_options_annealing_cannot_use(self):
        with tempfile.TemporaryDirectory(prefix='anneal-dash-test-', dir=ROOT) as temp, \
                patch.object(dashboard, 'RUNTIME', Path(temp)), patch.object(dashboard, 'live_pid', return_value=None), \
                patch.object(dashboard.subprocess, 'Popen') as spawn:
            for body in ({'search_options': {'lookahead': True}}, {'gpu_features': {'dynamic_mrv': True}},
                         {'seconds': 5 * 86400}, {'seed': -1}, {'workers': 0}, {'method': 'constraint'},
                         {'target': 'gold'}, {'verify_terminal': True}):
                with self.subTest(body=body), self.assertRaises(ValueError):
                    dashboard.start_solver({'algorithm': 'anneal', **body})
            spawn.assert_not_called()

    def test_dashboard_page_offers_annealing(self):
        page = (ROOT / 'Dashboard.html').read_text(encoding='utf-8')
        self.assertIn('<option value="anneal">', page)
        self.assertIn("'anneal':'Targeted annealing'", page)
        self.assertIn("method:'anneal'", page)


if __name__ == '__main__':
    unittest.main()
