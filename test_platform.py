"""Portable launcher/backend selection tests without real servers, GPUs, or browsers."""
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

import gpu_backends as backends
import platform_launcher as launcher

ROOT = Path(__file__).resolve().parent


class BackendSelectionTests(unittest.TestCase):
    def setUp(self):
        backends._automatic_backend.cache_clear()
        self.addCleanup(backends._automatic_backend.cache_clear)
        self.spec = self.enterContext(patch.object(backends.importlib.util, 'find_spec', return_value=None))
        self.which = self.enterContext(patch.object(backends.shutil, 'which', return_value=None))
        self.run = self.enterContext(patch.object(backends.subprocess, 'run'))

    def platform(self, name):
        return patch.object(backends, 'sys', SimpleNamespace(platform=name))

    def test_macos_auto_uses_webgpu_without_cuda_probes(self):
        with self.platform('darwin'):
            self.assertEqual(backends.choose_backend(), 'webgpu')
        self.spec.assert_not_called()
        self.which.assert_not_called()
        self.run.assert_not_called()

    def test_macos_explicit_cuda_is_rejected(self):
        with self.platform('darwin'):
            with self.assertRaisesRegex(ValueError, 'macOS'):
                backends.choose_backend('cuda')
            self.assertEqual(backends.choose_backend('webgpu'), 'webgpu')
        self.run.assert_not_called()

    def test_explicit_backends_do_not_probe_drivers(self):
        with self.platform('linux'):
            self.assertEqual(backends.choose_backend('cuda'), 'cuda')
            self.assertEqual(backends.choose_backend('webgpu'), 'webgpu')
        self.spec.assert_not_called()
        self.run.assert_not_called()

    def test_unknown_backends_fail_before_probing(self):
        for name in ('metal', 'CUDA', '', None, 12):
            with self.subTest(name=name), self.assertRaises(ValueError):
                backends.choose_backend(name)
        self.spec.assert_not_called()
        self.run.assert_not_called()

    def test_auto_requires_both_cupy_and_nvidia_tool(self):
        with self.platform('linux'):
            self.assertEqual(backends.choose_backend(), 'webgpu')
            self.which.assert_not_called()
            backends._automatic_backend.cache_clear()
            self.spec.return_value = object()
            self.assertEqual(backends.choose_backend(), 'webgpu')
        self.run.assert_not_called()

    def test_linux_and_windows_cuda_probe_success_and_cached_selection(self):
        self.spec.return_value = object()
        self.which.return_value = '/fake/nvidia-smi'
        self.run.return_value = SimpleNamespace(returncode=0, stdout='Example NVIDIA GPU\n')
        for platform, os_name in (('linux', 'posix'), ('win32', 'nt')):
            with self.subTest(platform=platform):
                backends._automatic_backend.cache_clear()
                self.run.reset_mock()
                with self.platform(platform), patch.object(backends, 'os', SimpleNamespace(name=os_name)), \
                     patch.object(backends.subprocess, 'CREATE_NO_WINDOW', 0x08000000, create=True):
                    self.assertEqual(backends.choose_backend(), 'cuda')
                    self.assertEqual(backends.choose_backend(), 'cuda')
                self.run.assert_called_once()
                args, kwargs = self.run.call_args
                self.assertEqual(args[0], ['/fake/nvidia-smi', '--query-gpu=name', '--format=csv,noheader'])
                self.assertEqual(kwargs['timeout'], 5)
                self.assertEqual(kwargs['creationflags'], 0x08000000 if os_name == 'nt' else 0)

    def test_failed_empty_or_timed_out_probe_falls_back_to_webgpu(self):
        self.spec.return_value = object()
        self.which.return_value = '/fake/nvidia-smi'
        cases = [SimpleNamespace(returncode=1, stdout='Driver unavailable'),
                 SimpleNamespace(returncode=0, stdout='  \n'),
                 OSError('not executable'), subprocess.TimeoutExpired('nvidia-smi', 5)]
        for case in cases:
            with self.subTest(case=case):
                backends._automatic_backend.cache_clear()
                self.run.side_effect = case if isinstance(case, Exception) else None
                self.run.return_value = case
                with self.platform('linux'):
                    self.assertEqual(backends.choose_backend(), 'webgpu')


class PortableLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='platform-test-', dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.assertEqual(self.root.parent, ROOT.resolve())
        self.enterContext(patch.object(launcher, 'ROOT', self.root))
        self.spawn = self.enterContext(patch.object(launcher.subprocess, 'Popen'))
        self.call = self.enterContext(patch.object(launcher.subprocess, 'call', return_value=0))
        self.browser = self.enterContext(patch.object(launcher.webbrowser, 'open', return_value=True))
        self.opener_factory = self.enterContext(patch.object(launcher.urllib.request, 'build_opener'))
        self.socket = self.enterContext(patch.object(launcher.socket, 'create_connection'))
        self.sleep = self.enterContext(patch.object(launcher.time, 'sleep'))
        self.output, self.errors = io.StringIO(), io.StringIO()
        self.enterContext(redirect_stdout(self.output))
        self.enterContext(redirect_stderr(self.errors))
        self.payload = {'service': 'diamond-dilemma', 'installation_root': str(self.root)}

    def install(self):
        for relative in ('venv/bin/python', 'venv/Scripts/python.exe', 'data/tiles.json'):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('{}', encoding='utf-8')

    def response(self, payload=None, raw=None):
        response = MagicMock()
        response.read.return_value = raw if raw is not None else json.dumps(payload).encode()
        self.opener_factory.return_value.open.return_value.__enter__.return_value = response
        return response

    def test_local_venv_paths_for_both_platform_families(self):
        for name, relative in (('posix', 'venv/bin/python'), ('nt', 'venv/Scripts/python.exe')):
            with self.subTest(name=name), patch.object(launcher, 'os', SimpleNamespace(name=name)):
                self.assertEqual(launcher.venv_python(), self.root / relative)

    def test_identity_requires_service_absolute_path_and_same_resolved_root(self):
        self.assertIs(launcher.validate_installation(self.payload), self.payload)
        alias = dict(self.payload, installation_root=str(self.root / 'unused' / '..'))
        self.assertIs(launcher.validate_installation(alias), alias)
        for value in (None, [], {}, {'service': 'other'},
                      {'service': 'diamond-dilemma'},
                      dict(self.payload, installation_root='relative'),
                      dict(self.payload, installation_root=str(self.root.parent))):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                launcher.validate_installation(value)

    def test_dashboard_probe_is_loopback_bounded_and_bypasses_proxies(self):
        response = self.response(self.payload)
        self.assertEqual(launcher.read_dashboard(9876), self.payload)
        handlers = self.opener_factory.call_args.args
        self.assertEqual(handlers[0].proxies, {})
        self.assertIs(handlers[1], launcher._NoRedirect)
        self.opener_factory.return_value.open.assert_called_once_with('http://127.0.0.1:9876/api/state', timeout=2)
        response.read.assert_called_once_with(1024 * 1024 + 1)
        self.socket.assert_not_called()

    def test_bad_port_is_rejected_before_network(self):
        for port in (0, 65536, -1, True, 8766.0, '8766'):
            with self.subTest(port=port), self.assertRaises(ValueError):
                launcher.read_dashboard(port)
        self.opener_factory.assert_not_called()
        self.socket.assert_not_called()

    def test_absent_dashboard_requires_refused_socket(self):
        self.opener_factory.return_value.open.side_effect = urllib.error.URLError('connection refused')
        self.socket.side_effect = ConnectionRefusedError()
        self.assertIsNone(launcher.read_dashboard(9876))
        self.socket.assert_called_once_with(('127.0.0.1', 9876), timeout=1)

    def test_occupied_unknown_port_is_not_treated_as_absent(self):
        self.opener_factory.return_value.open.side_effect = urllib.error.URLError('invalid HTTP service')
        with self.assertRaisesRegex(RuntimeError, 'occupied'):
            launcher.read_dashboard(9876)
        self.spawn.assert_not_called()

    def test_unverifiable_socket_error_does_not_start_another_server(self):
        self.opener_factory.return_value.open.side_effect = urllib.error.URLError('network error')
        self.socket.side_effect = PermissionError('permission denied')
        with self.assertRaisesRegex(RuntimeError, 'verify'):
            launcher.open_dashboard(9876)
        self.spawn.assert_not_called()
        self.browser.assert_not_called()

    def test_malformed_json_and_oversized_response_are_rejected(self):
        for raw, message in ((b'not-json', 'occupied'), (b'x' * (1024 * 1024 + 1), 'large')):
            with self.subTest(message=message):
                self.response(raw=raw)
                with self.assertRaisesRegex(RuntimeError, message):
                    launcher.read_dashboard(9876)

    def test_redirect_is_rejected_without_following_destination(self):
        with self.assertRaisesRegex(RuntimeError, 'redirect'):
            launcher._NoRedirect().redirect_request(None, None, 302, '', {}, 'https://example.invalid/')

    def test_existing_same_installation_is_reused_without_setup_or_spawn(self):
        self.response(self.payload)
        self.assertEqual(launcher.open_dashboard(9876), 'http://127.0.0.1:9876/')
        self.spawn.assert_not_called()
        self.browser.assert_called_once_with('http://127.0.0.1:9876/')
        self.assertFalse((self.root / 'runtime').exists())

    def test_foreign_or_unknown_installation_never_opens_browser_or_spawns(self):
        for payload in ({}, dict(self.payload, installation_root=str(self.root.parent)),
                        {'service': 'diamond-dilemma'}):
            with self.subTest(payload=payload):
                self.response(payload)
                with self.assertRaises(RuntimeError):
                    launcher.open_dashboard(9876)
        self.spawn.assert_not_called()
        self.browser.assert_not_called()

    def test_no_browser_option_preserves_local_url(self):
        self.response(self.payload)
        self.assertEqual(launcher.open_dashboard(9876, no_browser=True), 'http://127.0.0.1:9876/')
        self.browser.assert_not_called()

    def test_missing_venv_or_data_is_reported_before_spawn(self):
        with patch.object(launcher, 'read_dashboard', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'Setup'):
                launcher.open_dashboard(9876)
            path = launcher.venv_python()
            path.parent.mkdir(parents=True)
            path.touch()
            with self.assertRaisesRegex(RuntimeError, 'data'):
                launcher.open_dashboard(9876)
        self.spawn.assert_not_called()
        self.assertFalse((self.root / 'runtime').exists())

    def test_new_dashboard_uses_detached_process_local_cwd_and_log(self):
        self.install()
        for os_name in ('posix', 'nt'):
            with self.subTest(os_name=os_name):
                self.spawn.reset_mock()
                with patch.object(launcher, 'os', SimpleNamespace(name=os_name)), \
                     patch.object(launcher.subprocess, 'CREATE_NO_WINDOW', 0x08000000, create=True), \
                     patch.object(launcher.subprocess, 'CREATE_NEW_PROCESS_GROUP', 0x00000200, create=True), \
                     patch.object(launcher, 'read_dashboard', side_effect=[None, self.payload]):
                    launcher.open_dashboard(9876, no_browser=True)
                args, kwargs = self.spawn.call_args
                relative = 'venv/Scripts/python.exe' if os_name == 'nt' else 'venv/bin/python'
                self.assertEqual(args[0], [str(self.root / relative), str(self.root / 'dashboard_server.py'), '--port', '9876'])
                self.assertEqual(kwargs['cwd'], str(self.root))
                self.assertEqual(kwargs['stdin'], subprocess.DEVNULL)
                self.assertEqual(kwargs['stderr'], subprocess.STDOUT)
                self.assertEqual(Path(kwargs['stdout'].name), self.root / 'runtime/dashboard.log')
                if os_name == 'nt':
                    self.assertEqual(kwargs['creationflags'], 0x08000200)
                    self.assertNotIn('start_new_session', kwargs)
                else:
                    self.assertTrue(kwargs['start_new_session'])
                    self.assertNotIn('creationflags', kwargs)
        self.browser.assert_not_called()

    def test_early_child_exit_reports_log_without_opening_browser(self):
        self.install()
        self.spawn.return_value.poll.return_value = 1
        with patch.object(launcher, 'read_dashboard', return_value=None):
            with self.assertRaisesRegex(RuntimeError, 'exited.*dashboard.log'):
                launcher.open_dashboard(9876)
        self.browser.assert_not_called()
        self.spawn.return_value.terminate.assert_not_called()

    def test_startup_timeout_is_bounded_and_does_not_kill_a_process(self):
        self.install()
        self.spawn.return_value.poll.return_value = None
        with patch.object(launcher, 'read_dashboard', return_value=None), \
             patch.object(launcher.time, 'monotonic', side_effect=[0, 0, 21]):
            with self.assertRaisesRegex(RuntimeError, 'timed out'):
                launcher.open_dashboard(9876)
        self.sleep.assert_called_once_with(.2)
        self.browser.assert_not_called()
        self.spawn.return_value.terminate.assert_not_called()

    def test_solver_arguments_use_local_venv_and_default_resume(self):
        with patch.object(launcher, 'choose_backend', return_value='webgpu') as choose:
            command = launcher.solver_command(hours=2.5, replicas=128, seed=42, backend='webgpu')
        choose.assert_called_once_with('webgpu')
        self.assertEqual(command[:2], [str(launcher.venv_python()), str(self.root / 'systematic_search.py')])
        self.assertEqual(command[2:], ['--seconds', '9000.0', '--replicas', '128', '--seed', '42', '--backend', 'webgpu', '--resume'])

    def test_unlimited_fresh_arguments_and_input_bounds(self):
        with patch.object(launcher, 'choose_backend', return_value='cuda'):
            command = launcher.solver_command(unlimited=True, replicas=131072, seed=2**32 - 1, fresh=True)
            self.assertEqual(command[command.index('--seconds') + 1], '0')
            self.assertNotIn('--resume', command)
            cases = [{'hours': v} for v in (-1, True, float('inf'), float('nan'), '1')]
            cases += [{'replicas': v} for v in (127, 131073, True, 128.0)]
            cases += [{'seed': v} for v in (-1, 2**32, False, 0.0)]
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    launcher.solver_command(**kwargs)
        self.call.assert_not_called()

    def test_main_dispatches_dashboard_and_start_without_real_processes(self):
        with patch.object(launcher, 'open_dashboard') as dashboard:
            self.assertEqual(launcher.main(['dashboard', '--port', '9876', '--no-browser']), 0)
            dashboard.assert_called_once_with(9876, True)
        self.install()
        self.call.return_value = 7
        with patch.object(launcher, 'choose_backend', return_value='webgpu'):
            result = launcher.main(['start', '--hours', '4', '--unlimited', '--replicas', '128', '--seed', '17', '--backend', 'webgpu', '--fresh'])
        self.assertEqual(result, 7)
        args, kwargs = self.call.call_args
        self.assertEqual(kwargs['cwd'], self.root)
        self.assertIn('--backend', args[0])
        self.assertIn('webgpu', args[0])
        self.assertNotIn('--resume', args[0])
        self.assertEqual(args[0][args[0].index('--seconds') + 1], '0')

    def test_main_stop_and_status_use_only_local_runtime(self):
        self.assertEqual(launcher.main(['stop']), 0)
        self.assertFalse((self.root / 'runtime').exists())
        self.assertEqual(launcher.main(['status']), 0)
        runtime = self.root / 'runtime'
        runtime.mkdir()
        (runtime / 'status.json').write_text('{"state":"stopped"}', encoding='utf-8')
        self.assertEqual(launcher.main(['status']), 0)
        self.assertIn('stopped', self.output.getvalue())
        self.assertEqual(launcher.main(['stop']), 0)
        self.assertTrue((runtime / 'stop.request').is_file())
        self.spawn.assert_not_called()
        self.call.assert_not_called()
        self.browser.assert_not_called()

    def test_main_reports_controlled_errors_with_nonzero_status(self):
        with patch.object(launcher, 'open_dashboard', side_effect=RuntimeError('occupied')):
            self.assertEqual(launcher.main(['dashboard']), 1)
        self.assertIn('Error: occupied', self.errors.getvalue())
        with patch.object(launcher, 'choose_backend', return_value='webgpu'):
            self.assertEqual(launcher.main(['start', '--replicas', '1']), 1)
        self.call.assert_not_called()


if __name__ == '__main__':
    unittest.main()
