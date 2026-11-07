"""Control behavior tests; no solver or real dashboard process is launched."""
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import shutil
import subprocess
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import dashboard_server as dashboard
import gpu_backends as backends
import platform_launcher as launcher

ROOT = Path(__file__).resolve().parent
# Shared runners can take several seconds to start Node before running the
# finite VM harness. Preserve every assertion and retain a bounded hang limit.
FRONTEND_PROCESS_TIMEOUT = 30


class DashboardControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='dashboard-control-test-', dir=ROOT)
        self.directory = Path(self.temp.name).resolve()
        self.assertTrue(self.directory.parent == ROOT)
        self.runtime_patch = patch.object(dashboard, 'RUNTIME', self.directory)
        self.child_patch = patch.object(dashboard, 'CHILD', None)
        self.launch_patch = patch.object(dashboard, 'LAUNCH_CONFIG', {})
        self.runtime_patch.start()
        self.child_patch.start()
        self.launch_patch.start()
        # These lifecycle tests mock launches, independently of installed engines.
        from test_runtime_capabilities import capabilities
        self.enterContext(patch.object(dashboard, 'runtime_capabilities',
            return_value=capabilities('ortools', 'pysat', 'cupy', 'wgpu')))
        self.addCleanup(self.launch_patch.stop)
        self.addCleanup(self.child_patch.stop)
        self.addCleanup(self.runtime_patch.stop)

    def tearDown(self):
        self.assertTrue(self.directory.parent == ROOT)
        self.assertTrue(self.directory.name.startswith('dashboard-control-test-'))
        self.temp.cleanup()

    def status(self, **fields):
        data = {'state': 'running', 'pid': 4242, 'replicas': 131072}
        data.update(fields)
        (self.directory / 'status.json').write_text(json.dumps(data), encoding='utf-8')

    def test_posix_pid_probe_distinguishes_missing_and_permission_denied(self):
        for error, expected in ((None, True), (PermissionError('not owned'), True),
                                (ProcessLookupError('gone'), False), (OSError('other failure'), False)):
            with self.subTest(error=error):
                calls = []
                def probe(pid, signal):
                    calls.append((pid, signal))
                    if error is not None:
                        raise error
                with patch.object(dashboard, 'os', SimpleNamespace(name='posix', kill=probe)):
                    self.assertEqual(dashboard.pid_alive(4242), expected)
                    self.assertFalse(dashboard.pid_alive(True))
                    self.assertFalse(dashboard.pid_alive(-1))
                self.assertEqual(calls, [(4242, 0)])

    def test_exited_owned_child_is_reaped_before_stale_lock_probe(self):
        (self.directory / 'run.lock').write_text('4242', encoding='utf-8')
        calls = []
        child = SimpleNamespace(pid=4242, poll=lambda: calls.append('reaped') or 1)
        with patch.object(dashboard, 'CHILD', child), \
             patch.object(dashboard, 'pid_alive', side_effect=AssertionError('Zombie PID must not be probed')):
            self.assertIsNone(dashboard.live_pid())
            self.assertFalse(dashboard.state_payload()['process_running'])
            self.assertFalse(dashboard.request_stop()['stop_requested'])
        self.assertEqual(calls, ['reaped', 'reaped', 'reaped'])
        self.assertFalse((self.directory / 'stop.request').exists())

    def test_child_poll_precedes_other_recorded_pid_probe(self):
        (self.directory / 'run.lock').write_text('9000', encoding='utf-8')
        calls = []
        child = SimpleNamespace(pid=4242, poll=lambda: calls.append('poll') or 0)
        def alive(pid):
            calls.append(('probe', pid))
            return True
        with patch.object(dashboard, 'CHILD', child), patch.object(dashboard, 'pid_alive', side_effect=alive):
            self.assertEqual(dashboard.live_pid(), 9000)
        self.assertEqual(calls, ['poll', ('probe', 9000)])

    def test_live_owned_child_is_used_before_its_lock_is_written(self):
        child = SimpleNamespace(pid=4242, poll=lambda: None)
        with patch.object(dashboard, 'CHILD', child):
            self.assertEqual(dashboard.live_pid(), 4242)

    def test_permission_denied_pid_keeps_lock_and_prevents_duplicate_spawn(self):
        lock = self.directory / 'run.lock'
        lock.write_text('4242', encoding='utf-8')
        def denied(*args):
            raise PermissionError('Process exists but access is denied')
        with patch.object(dashboard, 'os', SimpleNamespace(name='posix', kill=denied)), \
             patch.object(dashboard.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'already running'):
                dashboard.start_solver({})
        self.assertEqual(lock.read_text(encoding='utf-8'), '4242')
        spawn.assert_not_called()

    def test_state_identifies_the_server_installation_not_saved_status(self):
        self.status(installation_root='C:\\a-different-checkout')
        with patch.object(dashboard, 'live_pid', return_value=4242):
            payload = dashboard.state_payload()
        self.assertEqual(payload['installation_root'], str(ROOT.resolve()))
        self.assertNotEqual(payload['installation_root'], str(self.directory))

    @unittest.skipUnless(shutil.which('powershell.exe'), 'Windows launcher semantics')
    def test_windows_wrapper_forwards_port_browser_and_native_failures(self):
        # HTTP identity, bound ports, proxy bypass and startup races are covered
        # by the shared-launcher tests in test_platform. Test the real PS wrapper.
        harness = r"""
param([string]$LauncherPath)
. $LauncherPath -Port 9876 -NoBrowser
$Failed = $false
try { Invoke-DiamondLauncher -Python $env:ComSpec -Arguments @('/d','/c','exit 17') } catch { $Failed = $true }
if (-not $Failed) { throw 'Native launch failure was lost' }
$script:Calls = @()
function Test-Path { param($LiteralPath) return $true }
function Invoke-DiamondLauncher { param($Python,$Arguments) $script:Calls += [pscustomobject]@{Python=$Python; Args=$Arguments} }
Open-DiamondDashboard
if ($script:Calls.Count -ne 1) { throw 'Launcher must run exactly once' }
$Call = $script:Calls[0]
if ($Call.Python -ne (Join-Path $SolverRoot 'venv\Scripts\python.exe')) { throw 'Wrong Python' }
if (($Call.Args -join '|') -ne ((Join-Path $SolverRoot 'platform_launcher.py')+'|dashboard|--port|9876|--no-browser')) { throw 'Arguments changed' }
function Test-Path { param($LiteralPath) return $false }
$Failed = $false
try { Open-DiamondDashboard } catch { $Failed = $_.Exception.Message -match 'Setup.cmd' }
if (-not $Failed -or $script:Calls.Count -ne 1) { throw 'Missing environment must fail before launch' }
"""
        script = self.directory / 'launcher-wrapper-tests.ps1'
        script.write_text(harness, encoding='utf-8')
        result = subprocess.run([shutil.which('powershell.exe'), '-NoProfile', '-ExecutionPolicy', 'Bypass',
                                 '-File', str(script), str(ROOT / 'OpenDashboard.ps1')],
                                cwd=ROOT, capture_output=True, text=True, timeout=20,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_dashboard_server_rejects_a_second_bind(self):
        # Construct sockets only; no HTTP serving thread or solver is started.
        with dashboard.DashboardHTTPServer(('127.0.0.1', 0), dashboard.Handler) as server:
            self.assertTrue(launcher._port_unavailable(server.server_port))
            with self.assertRaises(OSError):
                dashboard.DashboardHTTPServer(server.server_address, dashboard.Handler)

    def test_queued_stop_is_visible_without_waiting_for_runner_status(self):
        self.status()
        with patch.object(dashboard, 'live_pid', return_value=4242):
            result = dashboard.request_stop()
            payload = dashboard.state_payload()
        self.assertTrue(result['stop_requested'])
        self.assertTrue(payload['stop_requested'])
        self.assertTrue(payload['process_running'])
        self.assertEqual(payload['display_state'], 'stopping')
        self.assertEqual(payload['status']['state'], 'running')
        self.assertEqual(payload['status']['replicas'], 131072)

    def test_startup_recovers_replica_count_from_server_owned_child_arguments(self):
        self.status(pid=111, replicas=4096)
        (self.directory / 'checkpoint.npz').write_bytes(b'test-presence-only')
        child = SimpleNamespace(pid=333, poll=lambda: None, args=['python', 'solve.py', '--resume', '--replicas', '131072', '--seconds', '86400', '--seed', '20261007'])
        with patch.object(dashboard, 'live_pid', return_value=4242), patch.object(dashboard, 'CHILD', child):
            payload = dashboard.state_payload()
        self.assertEqual(payload['status']['state'], 'starting')
        self.assertEqual(payload['status']['pid'], 4242)
        self.assertEqual(payload['status']['replicas'], 131072)
        self.assertEqual(payload['status']['time_limit_seconds'], 86400)
        self.assertEqual(payload['status']['startup_phase'], 'restoring_checkpoint')
        self.assertIn('validating', payload['status']['phase_message'])

    def test_startup_uses_only_matching_process_configuration(self):
        self.status(pid=111, replicas=4096)
        config = {'pid': 4242, 'replicas': 131072, 'seconds': 86400}
        (self.directory / 'run-config.json').write_text(json.dumps(config), encoding='utf-8')
        with patch.object(dashboard, 'live_pid', return_value=4242):
            payload = dashboard.state_payload()
            self.assertEqual(payload['status']['replicas'], 131072)
        config['pid'] = 111
        (self.directory / 'run-config.json').write_text(json.dumps(config), encoding='utf-8')
        with patch.object(dashboard, 'live_pid', return_value=4242):
            payload = dashboard.state_payload()
        self.assertNotIn('replicas', payload['status'])

    def test_resume_uses_selected_smaller_count_and_preserves_branch_status(self):
        self.status(state='stopped', replicas=131072, method='systematic')
        systematic = self.directory / 'systematic'
        systematic.mkdir()
        (systematic / 'checkpoint.npz').write_bytes(b'test-presence-only')
        child = SimpleNamespace(pid=5678, poll=lambda: None)
        with patch.object(dashboard, 'live_pid', return_value=None), patch.object(
                dashboard.subprocess, 'Popen', return_value=child) as spawn:
            result = dashboard.start_solver({'seconds': 0, 'replicas': 128, 'method': 'systematic'})
        command = spawn.call_args.args[0]
        self.assertEqual(command[command.index('--replicas') + 1], '128')
        self.assertIn('--resume', command)
        self.assertEqual(result['replicas'], 128)
        with patch.object(dashboard, 'live_pid', return_value=5678):
            starting = dashboard.state_payload()['status']
            self.assertEqual(starting['replicas'], 128)
            self.assertEqual(starting['startup_phase'], 'restoring_checkpoint')
            self.status(pid=5678, replicas=128, method='systematic', active_jobs=128,
                        paused_jobs=130944, unassigned_jobs=6, queued_jobs=130950)
            active = dashboard.state_payload()['status']
        self.assertEqual(active['replicas'], 128)
        self.assertEqual(active['active_jobs'], 128)
        self.assertEqual(active['paused_jobs'], 130944)
        self.assertEqual(active['queued_jobs'], active['paused_jobs'] + active['unassigned_jobs'])

    def test_gpu_policy_startup_shows_active_flags_and_its_own_checkpoint(self):
        import hashlib
        legacy=self.directory/'systematic';legacy.mkdir();(legacy/'checkpoint.npz').write_bytes(b'presence')
        child=SimpleNamespace(pid=5678,poll=lambda:None)
        with patch.object(dashboard,'live_pid',return_value=None),patch.object(dashboard.subprocess,'Popen',return_value=child) as spawn:
            result=dashboard.start_solver({'seconds':0,'replicas':128,'algorithm':'gpu-dfs','gpu_features':{'dynamic_mrv':True}})
        cmd=spawn.call_args.args[0];features=json.loads(cmd[cmd.index('--gpu-features')+1])
        self.assertTrue(features['dynamic_mrv']);self.assertEqual(sum(features.values()),1)
        with patch.object(dashboard,'live_pid',return_value=5678):
            state=dashboard.state_payload()['status']
            self.assertEqual(state['gpu_features'],features)
            self.assertEqual(state['startup_phase'],'initializing_gpu')
            profile='systematic-'+hashlib.sha256(json.dumps(dashboard.feature_identity(features),sort_keys=True,separators=(',',':')).encode()).hexdigest()[:12]
            folder=self.directory/profile;folder.mkdir();(folder/'checkpoint.npz').write_bytes(b'presence')
            self.assertEqual(dashboard.state_payload()['status']['startup_phase'],'restoring_checkpoint')
        with patch.object(dashboard,'live_pid',return_value=None),patch.object(dashboard.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(ValueError,'only to GPU DFS'):
                dashboard.start_solver({'algorithm':'dfs','gpu_features':{'dynamic_mrv':True}})
            spawn.assert_not_called()

    def test_backend_selection_is_forwarded_and_available_during_startup(self):
        for backend in ('auto', 'cuda', 'webgpu'):
            with self.subTest(backend=backend):
                child = SimpleNamespace(pid=5678, poll=lambda: None)
                with patch.object(dashboard, 'live_pid', return_value=None), \
                     patch.object(dashboard.sys, 'platform', 'linux'), \
                     patch.object(dashboard.subprocess, 'Popen', return_value=child) as spawn, \
                     patch.object(backends, '_automatic_backend', return_value='cuda'):
                    result = dashboard.start_solver({'backend': backend, 'replicas': 128})
                command = spawn.call_args.args[0]
                self.assertEqual(Path(command[1]).name, 'systematic_search.py')
                self.assertEqual(command[command.index('--backend') + 1], backend)
                self.assertEqual(result['backend'], backend)
                self.assertEqual(dashboard.LAUNCH_CONFIG['backend'], backend)
                with patch.object(dashboard, 'live_pid', return_value=5678):
                    status = dashboard.state_payload()['status']
                self.assertEqual(status['state'], 'starting')
                self.assertEqual(status['backend'], backend)
                self.assertEqual(status['replicas'], 128)

    def test_invalid_backend_is_rejected_without_removing_state_or_spawning(self):
        lock = self.directory / 'run.lock'
        stop = self.directory / 'stop.request'
        lock.write_text('999', encoding='utf-8')
        stop.write_text('pending', encoding='utf-8')
        with patch.object(dashboard, 'live_pid', return_value=None), \
             patch.object(dashboard.subprocess, 'Popen') as spawn:
            for value in ('metal', 'CUDA', '', None, True, 1, [], {}):
                with self.subTest(backend=value), self.assertRaisesRegex(ValueError, 'backend'):
                    dashboard.start_solver({'backend': value})
            with self.assertRaisesRegex(ValueError, 'CUDA'):
                dashboard.start_solver({'method': 'stochastic', 'backend': 'webgpu'})
        spawn.assert_not_called()
        self.assertEqual(lock.read_text(encoding='utf-8'), '999')
        self.assertEqual(stop.read_text(encoding='utf-8'), 'pending')
        self.assertFalse((self.directory / 'solver.log').exists())

    def test_macos_cuda_and_legacy_stochastic_fail_before_spawn(self):
        with patch.object(dashboard.sys, 'platform', 'darwin'), \
             patch.object(dashboard, 'live_pid', return_value=None), \
             patch.object(dashboard.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(ValueError, 'macOS|Metal|CUDA'):
                dashboard.start_solver({'backend': 'cuda'})
            with self.assertRaisesRegex(ValueError, 'macOS|Metal|CUDA'):
                dashboard.start_solver({'backend': 'auto', 'method': 'stochastic'})
        spawn.assert_not_called()
        self.assertFalse((self.directory / 'solver.log').exists())

    def test_startup_recovers_backend_from_server_owned_child_arguments(self):
        self.status(pid=111, backend='cuda')
        child = SimpleNamespace(pid=333, poll=lambda: None, args=[
            'python', 'systematic_search.py', '--backend', 'webgpu', '--replicas', '128'])
        with patch.object(dashboard, 'live_pid', return_value=4242), \
             patch.object(dashboard, 'CHILD', child):
            status = dashboard.state_payload()['status']
        self.assertEqual(status['backend'], 'webgpu')
        self.assertEqual(status['method'], 'systematic')
        self.assertEqual(status['replicas'], 128)

    def test_backend_config_requires_matching_pid_and_live_status_is_authoritative(self):
        self.status(pid=111, backend='cuda')
        config = {'pid': 4242, 'backend': 'webgpu', 'method': 'systematic', 'replicas': 128}
        path = self.directory / 'run-config.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        with patch.object(dashboard, 'live_pid', return_value=4242):
            self.assertEqual(dashboard.state_payload()['status']['backend'], 'webgpu')
            config['pid'] = 111
            path.write_text(json.dumps(config), encoding='utf-8')
            self.assertNotIn('backend', dashboard.state_payload()['status'])
            self.status(pid=4242, backend='cuda')
            self.assertEqual(dashboard.state_payload()['status']['backend'], 'cuda')

    def test_all_algorithm_routes_and_truthful_startup_metadata(self):
        for algorithm in ('gpu-dfs', 'dfs', 'cp', 'cp-sat', 'cp-sat-circuit', 'sat', 'hybrid'):
            with self.subTest(algorithm=algorithm):
                child = SimpleNamespace(pid=5678, poll=lambda: None)
                with patch.object(dashboard.sys, 'platform', 'linux'), \
                     patch.object(dashboard, 'live_pid', return_value=None), \
                     patch.object(dashboard.subprocess, 'Popen', return_value=child) as spawn:
                    result = dashboard.start_solver({'algorithm': algorithm, 'replicas': 128, 'backend': 'webgpu'})
                command = spawn.call_args.args[0]
                expected_script = 'systematic_search.py' if algorithm == 'gpu-dfs' else 'constraint_search.py'
                self.assertEqual(Path(command[1]).name, expected_script)
                if algorithm == 'gpu-dfs': self.assertNotIn('--algorithm', command)
                else: self.assertEqual(command[command.index('--algorithm')+1], algorithm)
                self.assertIn('--resume', command)
                self.assertIn('--stop-file-initialized', command)
                cpu = algorithm in ('dfs', 'cp', 'cp-sat', 'cp-sat-circuit', 'sat')
                expected_replicas, expected_backend = (0, 'cpu') if cpu else (128, 'webgpu')
                expected_target = 'gold'
                self.assertEqual(result['algorithm'], algorithm)
                self.assertEqual(result['target'], expected_target)
                self.assertEqual(result['replicas'], expected_replicas)
                self.assertEqual(result['backend'], expected_backend)
                with patch.object(dashboard, 'live_pid', return_value=5678):
                    status = dashboard.state_payload()['status']
                self.assertEqual(status['algorithm'], algorithm)
                self.assertEqual(status['target'], expected_target)
                self.assertEqual(status['replicas'], expected_replicas)
                self.assertEqual(status['backend'], expected_backend)
                if cpu:
                    self.assertEqual(status['startup_phase'], 'building_model')
                    self.assertIn('CPU', status['phase_message'])

    def test_invalid_or_conflicting_algorithm_does_not_spawn_or_remove_stop(self):
        sentinel = self.directory / 'stop.request'
        sentinel.write_text('keep', encoding='utf-8')
        bad = [{'algorithm': value} for value in ('unknown', None, True, 42, [], {})]
        bad += [{'method': 'stochastic', 'algorithm': 'dfs'}, {'method': 'constraint'},
                {'algorithm': 'dfs', 'target': 'edges'}, {'algorithm': 'gpu-dfs', 'target': 'edges'}]
        with patch.object(dashboard, 'live_pid', return_value=None), patch.object(dashboard.subprocess, 'Popen') as spawn:
            for body in bad:
                with self.subTest(body=body), self.assertRaises(ValueError): dashboard.start_solver(body)
        spawn.assert_not_called()
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'keep')

    def test_cpu_algorithm_ignores_unused_gpu_backend_on_macos(self):
        with patch.object(dashboard.sys, 'platform', 'darwin'), \
             patch.object(dashboard, 'live_pid', return_value=None), \
             patch.object(dashboard.subprocess, 'Popen', return_value=SimpleNamespace(pid=5678)) as spawn:
            result = dashboard.start_solver({'algorithm': 'dfs', 'backend': 'cuda', 'replicas': 128})
        self.assertEqual(result['backend'], 'cpu')
        self.assertEqual(result['replicas'], 0)
        self.assertEqual(Path(spawn.call_args.args[0][1]).name, 'constraint_search.py')

    def test_persistent_sat_startup_uses_its_own_checkpoint_directory(self):
        config={'pid':4242,'algorithm':'sat','method':'constraint','search_options':{'persistent_sat':True}}
        path=self.directory/'run-config.json'
        path.write_text(json.dumps(config),encoding='utf-8')
        plain=self.directory/'searches'/'sat-gold';plain.mkdir(parents=True)
        (plain/'checkpoint.json').write_text('{}',encoding='utf-8')
        self.assertEqual(dashboard.startup_status(4242)['startup_phase'],'building_model')
        session=self.directory/'searches'/'sat-session-gold';session.mkdir(parents=True)
        (session/'checkpoint.json').write_text('{}',encoding='utf-8')
        active=dashboard.startup_status(4242)
        self.assertEqual(active['startup_phase'],'restoring_checkpoint')
        self.assertIn('exact Gold search branches',active['phase_message'])
        (plain/'checkpoint.json').unlink()
        config['search_options']['persistent_sat']=False
        path.write_text(json.dumps(config),encoding='utf-8')
        self.assertEqual(dashboard.startup_status(4242)['startup_phase'],'building_model')

    def test_constraint_child_arguments_restore_algorithm_specific_status(self):
        self.status(pid=111, method='systematic', backend='cuda')
        folder = self.directory / 'searches' / 'sat-gold'
        folder.mkdir(parents=True)
        (folder / 'checkpoint.json').write_text('{}', encoding='utf-8')
        child = SimpleNamespace(pid=333, poll=lambda: None, args=[
            'python', 'constraint_search.py', '--algorithm', 'sat', '--backend', 'webgpu',
            '--replicas', '4096', '--seconds', '3600'])
        with patch.object(dashboard, 'CHILD', child), patch.object(dashboard, 'live_pid', return_value=4242):
            status = dashboard.state_payload()['status']
        self.assertEqual(status['algorithm'], 'sat')
        self.assertEqual(status['target'], 'gold')
        self.assertEqual(status['method'], 'constraint')
        self.assertEqual(status['replicas'], 0)
        self.assertEqual(status['backend'], 'cpu')
        self.assertEqual(status['startup_phase'], 'restoring_checkpoint')
        self.assertIn('rebuilding', status['phase_message'])

    @unittest.skipUnless(shutil.which('powershell.exe'), 'Windows launcher semantics')
    def test_windows_launcher_routes_algorithms_without_launching_solver(self):
        harness = r"""
param([string]$LauncherPath, [string]$Choice)
$ErrorActionPreference = 'Stop'
$global:AlgorithmUnderTest = $Choice
function Set-Location { param($LiteralPath) }
function Join-Path {
    param($Path, $ChildPath)
    if ($ChildPath -eq 'venv\Scripts\python.exe') { return 'Invoke-FakePython' }
    return [System.IO.Path]::Combine($Path, $ChildPath)
}
function Test-Path { param($LiteralPath) return -not $LiteralPath.EndsWith('run.lock') }
function Remove-Item { param($LiteralPath, $ErrorAction) }
function Invoke-FakePython {
    $seen = @($args)
    $legacy = $global:AlgorithmUnderTest -in @('default', 'gpu-dfs')
    $expected = if ($legacy) { 'systematic_search.py' } else { 'constraint_search.py' }
    if ([System.IO.Path]::GetFileName($seen[0]) -ne $expected) { throw 'Wrong solver script' }
    if ($seen -notcontains '--resume' -or $seen -notcontains '--stop-file-initialized') { throw 'Resume/stop contract missing' }
    if ($seen[$seen.IndexOf('--replicas')+1] -ne '128') { throw 'Replica argument missing' }
    if ($seen[$seen.IndexOf('--backend')+1] -ne 'webgpu') { throw 'Backend argument missing' }
    if ($legacy -and $seen -contains '--algorithm') { throw 'Legacy launch was changed' }
    if (-not $legacy -and $seen[$seen.IndexOf('--algorithm')+1] -ne $global:AlgorithmUnderTest) { throw 'Algorithm argument missing' }
    Write-Output 'MOCKED_SOLVER_ARGUMENTS_OK'
    $global:LASTEXITCODE = 0
}
if ($Choice -eq 'default') { & $LauncherPath -Replicas 128 -Backend webgpu }
else { & $LauncherPath -Replicas 128 -Backend webgpu -Algorithm $Choice }
"""
        script = self.directory / 'algorithm-launcher-test.ps1'
        script.write_text(harness, encoding='utf-8')
        for algorithm in ('default', 'gpu-dfs', 'dfs', 'cp', 'cp-sat', 'cp-sat-circuit', 'sat', 'hybrid', 'parallel-dfs'):
            with self.subTest(algorithm=algorithm):
                result = subprocess.run([shutil.which('powershell.exe'), '-NoProfile', '-ExecutionPolicy', 'Bypass',
                    '-File', str(script), str(ROOT / 'StartSolver.ps1'), algorithm], cwd=ROOT,
                    capture_output=True, text=True, timeout=15, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('MOCKED_SOLVER_ARGUMENTS_OK', result.stdout)

    def test_parallel_controls_reach_runner_and_startup_telemetry(self):
        child = SimpleNamespace(pid=5678, poll=lambda: None)
        with patch.object(dashboard, 'live_pid', return_value=None), patch.object(dashboard.subprocess, 'Popen', return_value=child) as spawn:
            result = dashboard.start_solver({'algorithm':'parallel-dfs','workers':2,
                'search_options':{'lookahead':False,'pair_lookahead':False,'lookahead_ms':250,'native':'python'}})
        command = spawn.call_args.args[0]
        self.assertEqual(command[command.index('--workers')+1], '2')
        options = json.loads(command[command.index('--search-options')+1])
        self.assertFalse(options['lookahead'])
        self.assertFalse(options['pair_lookahead'])
        self.assertEqual(options['lookahead_ms'],250)
        self.assertTrue(options['regional_cache'])
        self.assertEqual(result['workers'], 2)
        self.assertEqual(result['replicas'], 0)
        status = dashboard.startup_status(5678)
        self.assertEqual(status['workers'], 2)
        self.assertEqual(status['algorithm'], 'parallel-dfs')

    def test_invalid_optimization_options_never_launch_or_clear_state(self):
        stop = self.directory/'stop.request'
        stop.write_text('preserve',encoding='utf-8')
        for body in ({'workers':0},{'workers':True},{'search_options':{'lookahead':1}},
                     {'search_options':{'native':'wrong'}},{'search_options':{'unknown':True}},
                     {'search_options':{'pair_lookahead':1}},{'search_options':{'lookahead_ms':0}},
                     {'search_options':{'lookahead_ms':True}},{'search_options':{'lookahead_ms':1001}}):
            with self.subTest(body=body), patch.object(dashboard, 'live_pid', return_value=None), patch.object(dashboard.subprocess,'Popen') as spawn:
                with self.assertRaises(ValueError):
                    dashboard.start_solver({'algorithm':'parallel-dfs',**body})
                spawn.assert_not_called()
                self.assertEqual(stop.read_text(encoding='utf-8'),'preserve')

    def test_checkpoint_phase_survives_friendly_stopping_overlay(self):
        checkpoint = {'phase': 'saving', 'elapsed_seconds': 1.25}
        self.status(state='stopping', checkpoint=checkpoint)
        (self.directory / 'stop.request').write_text('stop', encoding='utf-8')
        with patch.object(dashboard, 'live_pid', return_value=4242):
            payload = dashboard.state_payload()
        self.assertEqual(payload['display_state'], 'stopping')
        self.assertEqual(payload['status']['checkpoint'], checkpoint)
        self.status(state='saving_checkpoint', checkpoint=checkpoint)
        with patch.object(dashboard, 'live_pid', return_value=4242):
            payload = dashboard.state_payload()
        self.assertEqual(payload['display_state'], 'saving_checkpoint')
        self.assertEqual(payload['status']['state'], 'saving_checkpoint')

    def test_stale_stop_file_does_not_claim_a_dead_process_is_stopping(self):
        self.status(state='stopped')
        (self.directory / 'stop.request').write_text('old stop', encoding='utf-8')
        with patch.object(dashboard, 'live_pid', return_value=None):
            payload = dashboard.state_payload()
        self.assertFalse(payload['stop_requested'])
        self.assertFalse(payload['process_running'])
        self.assertEqual(payload['display_state'], 'stopped')

    def test_start_clears_old_stop_before_spawn_but_keeps_a_new_stop(self):
        sentinel = self.directory / 'stop.request'
        sentinel.write_text('old stop', encoding='utf-8')
        (self.directory / 'run.lock').write_text('123', encoding='utf-8')
        seen = []
        def spawn(command, **kwargs):
            self.assertFalse(sentinel.exists())
            self.assertFalse((self.directory / 'run.lock').exists())
            self.assertIn('--stop-file-initialized', command)
            self.assertEqual(command[command.index('--replicas') + 1], '131072')
            self.assertEqual(kwargs['cwd'], str(ROOT))
            # Simulate Stop immediately after process creation, before imports finish.
            sentinel.write_text('new stop', encoding='utf-8')
            seen.append(command)
            return SimpleNamespace(pid=5678)
        with patch.object(dashboard, 'live_pid', return_value=None), patch.object(dashboard.subprocess, 'Popen', side_effect=spawn):
            result = dashboard.start_solver({'seconds': 86400, 'replicas': 131072, 'seed': 20261007})
        self.assertTrue(result['started'])
        self.assertEqual(result['replicas'], 131072)
        self.assertEqual(len(seen), 1)
        self.assertEqual(sentinel.read_text(encoding='utf-8'), 'new stop')

    def test_stop_when_idle_does_not_create_a_stale_request(self):
        with patch.object(dashboard, 'live_pid', return_value=None):
            result = dashboard.request_stop()
        self.assertFalse(result['stop_requested'])
        self.assertFalse(result['process_running'])
        self.assertFalse((self.directory / 'stop.request').exists())

    def test_repeated_stop_is_idempotent(self):
        with patch.object(dashboard, 'live_pid', return_value=4242):
            first = dashboard.request_stop()
            second = dashboard.request_stop()
        self.assertEqual(first, second)
        self.assertTrue((self.directory / 'stop.request').is_file())

    def test_unlimited_duration_passes_zero_to_both_solver_methods(self):
        with patch.object(dashboard, 'live_pid', return_value=None), \
             patch.object(dashboard.sys, 'platform', 'linux'):
            for method, script in (('systematic', 'systematic_search.py'),
                                   ('stochastic', 'solve.py')):
                with self.subTest(method=method), patch.object(
                        dashboard.subprocess, 'Popen',
                        return_value=SimpleNamespace(pid=5678)) as spawn:
                    result = dashboard.start_solver(
                        {'seconds': 0, 'replicas': 128, 'seed': 20261007, 'method': method})
                    command = spawn.call_args.args[0]
                    self.assertEqual(command[command.index('--seconds') + 1], '0.0')
                    self.assertEqual(Path(command[1]).name, script)
                    self.assertEqual(dashboard.LAUNCH_CONFIG['time_limit_seconds'], 0)
                    self.assertTrue(result['started'])

    def test_invalid_duration_values_are_rejected_before_spawn(self):
        with patch.object(dashboard, 'live_pid', return_value=None), \
             patch.object(dashboard.subprocess, 'Popen') as spawn:
            for seconds in (-1, float('nan'), float('inf'), float('-inf'), True, False):
                with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                    dashboard.start_solver({'seconds': seconds, 'replicas': 128})
            spawn.assert_not_called()

    def test_start_refuses_active_process_and_invalid_replica_counts(self):
        with patch.object(dashboard, 'live_pid', return_value=4242):
            with self.assertRaisesRegex(ValueError, 'already running'):
                dashboard.start_solver({})
        with patch.object(dashboard, 'live_pid', return_value=None), patch.object(dashboard.subprocess, 'Popen') as spawn:
            for value in (127, 131073, 128.5, '4096', True):
                with self.subTest(replicas=value), self.assertRaises(ValueError):
                    dashboard.start_solver({'replicas': value})
            spawn.assert_not_called()

    @unittest.skipUnless(shutil.which('node'), 'Node.js is only needed for the isolated frontend control test')
    def test_frontend_stop_feedback_is_immediate_and_idle_replica_edits_survive(self):
        harness = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{value:'4096',style:{},dataset:{},classList:{add(){},remove(){},toggle(){}},setAttribute(){},listeners:{},addEventListener(type,fn){this.listeners[type]=fn},clientWidth:500,textContent:'',innerHTML:'',checked:true});return elements.get(id)}
const context=vm.createContext({document:{getElementById:element,addEventListener(){}},window:{addEventListener(){},localStorage:{values:{},getItem(key){return this.values[key]??null},setItem(key,value){this.values[key]=value}}},console,Date,Number,String,Math,Set,JSON,setTimeout(){return 1},clearTimeout(){},setInterval(){},fetchCalls:[]});
context.fetch=(url,options)=>{context.fetchCalls.push([url,options]);return new Promise(()=>{})};
let source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0];
source=source.replace('poll();setInterval(poll,1000);','');
vm.runInContext(source,context);
element('algorithmInput').value='gpu-dfs';element('algorithmInput').listeners.change();
vm.runInContext("state={process_running:true,control_token:'test',status:{state:'running',replicas:131072},history:[]}; command('stop');",context);
assert.equal(element('stopBtn').disabled,true);
assert.equal(element('stopBtn').textContent,'Stopping…');
assert.equal(element('runStatus').textContent,'Stopping');
assert.equal(element('replicaInput').value,'131072');
assert.equal(context.fetchCalls.length,1);
vm.runInContext("command('stop');",context);
assert.equal(context.fetchCalls.length,1);
vm.runInContext("state.status.checkpoint={phase:'saving'};renderState();",context);
assert.equal(element('runStatus').textContent,'Saving checkpoint');
vm.runInContext("localStopRequested=false;commandBusy=false;state={process_running:false,control_token:'test',status:{state:'stopped',replicas:131072},history:[]};",context);
element('replicaInput').value='8192';
vm.runInContext('renderState();',context);
assert.equal(element('replicaInput').value,'8192');
assert.equal(element('stopBtn').textContent,'Stop');
assert.equal(element('stopBtn').disabled,true);
vm.runInContext("configuredReplicas=null;mode='live';state={process_running:true,control_token:'test',status:{state:'starting',pid:900,run_id:'new',phase_message:'Restoring saved search and validating its cache.'},live:{codes:[0],replica:0,run_id:'old',validation:{matched_edges:112,line_components:200,placement_count:160}},history:[]};",context);
element('replicaInput').value='131072';
vm.runInContext('renderState();',context);
assert.equal(element('replicas').textContent,'131,072');
assert.equal(element('viewMeta').textContent,'Saved snapshot — waiting for live search');
assert.equal(element('scoreLabel').textContent,'Saved matched edges');
assert.equal(element('snapshotAge').textContent,'Saved snapshot');
assert.match(element('controlHint').textContent,/Restoring saved search/);
assert.equal(element('matched').textContent,112);
vm.runInContext("state.status={state:'running',pid:900,run_id:'new',replicas:131072};renderState();",context);
assert.equal(element('viewMeta').textContent,'Saved snapshot — waiting for live search');
vm.runInContext("state.live.run_id='new';renderState();",context);
assert.match(element('viewMeta').textContent,/Actual live snapshot/);
element('seconds').value='3600';
vm.runInContext('state.status.time_limit_seconds=0;renderState();',context);
assert.equal(element('seconds').value,'0');
assert.match(fs.readFileSync(process.argv[1],'utf8'),/<option value="0"[^>]*>Unlimited/);
// Editing the stopped selector must survive polling and reach the next Start payload.
vm.runInContext("state={process_running:false,control_token:'test',status:{state:'stopped',method:'systematic',replicas:131072,active_jobs:131072,paused_jobs:0,queued_jobs:6,unassigned_jobs:6},history:[]};",context);
element('replicaInput').value='128';
element('replicaInput').listeners.input();
vm.runInContext('renderState();renderState();',context);
assert.equal(element('replicaInput').value,'128');
assert.equal(context.window.localStorage.values['diamond.requestedReplicas'],'128');
assert.equal(element('replicas').textContent,'131,072');
assert.equal(element('replicasLabel').textContent,'Last run replicas');
assert.match(element('controlHint').textContent,/Stop, choose fewer or more replicas, then Start/);
vm.runInContext("command('start');",context);
assert.equal(context.fetchCalls.length,2);
assert.equal(context.fetchCalls[1][0],'/api/start');
assert.equal(JSON.parse(context.fetchCalls[1][1].body).replicas,128);
assert.equal(JSON.parse(context.fetchCalls[1][1].body).seconds,0);
// Once running, the current count is authoritative and paused work stays visible.
vm.runInContext("commandBusy=false;state={process_running:true,control_token:'test',status:{state:'running',method:'systematic',replicas:128,active_jobs:128,paused_jobs:130944,queued_jobs:130950,unassigned_jobs:6,time_limit_seconds:0},history:[]};renderState();",context);
assert.equal(element('replicaInput').value,'128');
assert.equal(element('replicaInput').disabled,true);
assert.equal(element('replicas').textContent,'128');
assert.equal(element('replicasLabel').textContent,'GPU replicas');
assert.equal(element('cacheEntries').textContent,'128');
assert.equal(element('pausedBranches').textContent,'130,944');
assert.equal(element('waitingBranches').textContent,'130,950');
assert.match(element('replicaNote').textContent,/128 active.*130,944 paused/);
assert.match(element('cacheHint').textContent,/Waiting includes paused branches and 6 unassigned branches/);
assert.equal(element('controlHint').textContent.split('To change GPU load:').length,2);
// Keep the requested GPU policy visible before the first server status poll.
vm.runInContext("commandBusy=false;pollBusy=true;state={process_running:false,control_token:'test',status:{state:'stopped'},history:[]};",context);
element('gpu_dynamic_mrv').checked=true;element('gpu_dynamic_mrv').listeners.change();
context.fetch=async()=>({ok:true,json:async()=>({backend:'cuda',pid:901,replicas:128})});
vm.runInContext("command('start');",context).then(()=>{
 assert.equal(element('gpu_dynamic_mrv').checked,true);
 assert.equal(element('gpu_dynamic_mrv').disabled,true);
 assert.equal(vm.runInContext('state.status.gpu_features.dynamic_mrv',context),true);
 console.log('Immediate stop, replica edits, unlimited duration and starting GPU policy passed');
}).catch(error=>{console.error(error);process.exitCode=1;});
"""
        result = subprocess.run([shutil.which('node'), '-e', harness, str(ROOT / 'Dashboard.html')], cwd=ROOT, capture_output=True, text=True, timeout=FRONTEND_PROCESS_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is only needed for the isolated frontend control test')
    def test_frontend_backend_preference_active_selection_and_start_payload(self):
        harness = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0].replace('poll();setInterval(poll,1000);','');
function boot(saved){
 const elements=new Map();
 function element(id){if(!elements.has(id))elements.set(id,{value:'4096',style:{},dataset:{},classList:{add(){},remove(){},toggle(){}},setAttribute(){},listeners:{},addEventListener(type,fn){this.listeners[type]=fn},clientWidth:500,textContent:'',innerHTML:'',checked:true});return elements.get(id)}
 const storage={values:{'diamond.requestedBackend':saved},getItem(key){return this.values[key]??null},setItem(key,value){this.values[key]=value}};
 const context=vm.createContext({document:{getElementById:element,addEventListener(){}},window:{addEventListener(){},localStorage:storage},console,Date,Number,String,Math,Set,JSON,setTimeout(){return 1},clearTimeout(){},setInterval(){},fetchCalls:[]});
 context.fetch=(url,options)=>{context.fetchCalls.push([url,options]);return new Promise(()=>{})};
 vm.runInContext(source,context);
 element('algorithmInput').value='gpu-dfs';element('algorithmInput').listeners.change();
 return {context,element,storage};
}
for(const saved of ['auto','cuda','webgpu'])assert.equal(boot(saved).element('backendInput').value,saved);
for(const saved of ['metal','CUDA','',null])assert.equal(boot(saved).element('backendInput').value,'auto');
const {context,element,storage}=boot('webgpu');
function render(script){vm.runInContext(script+';renderState();',context)}
function changeBackend(value){const el=element('backendInput');el.value=value;const fn=el.onchange||el.listeners.change;assert.equal(typeof fn,'function');fn({target:el});}
render("state={process_running:false,control_token:'test',status:{state:'stopped',method:'systematic',backend:'cuda',replicas:128},history:[]}");
assert.equal(element('backendInput').value,'webgpu');
assert.equal(element('backendInput').disabled,false);
changeBackend('auto');
assert.equal(storage.values['diamond.requestedBackend'],'auto');
render('state.status.backend="webgpu"');
assert.equal(element('backendInput').value,'auto');
// Active status controls the disabled display, without replacing the saved preference.
render("state.process_running=true;state.status.state='running';state.status.backend='cuda'");
assert.equal(element('backendInput').value,'cuda');
assert.equal(element('backendInput').disabled,true);
assert.equal(storage.values['diamond.requestedBackend'],'auto');
render("state.status.state='starting';state.status.backend='webgpu'");
assert.equal(element('backendInput').value,'webgpu');
assert.equal(element('backendInput').disabled,true);
assert.doesNotMatch(element('gpuName').textContent,/CUDA/);
render("state.process_running=false;state.status.state='stopped'");
assert.equal(element('backendInput').value,'auto');
assert.equal(element('backendInput').disabled,false);
changeBackend('webgpu');
element('replicaInput').value='128';element('seconds').value='3600';element('seedInput').value='17';
render('');
assert.equal(element('backendInput').value,'webgpu');
vm.runInContext("command('start');",context);
assert.equal(element('backendInput').disabled,true);
assert.equal(context.fetchCalls.length,1);
assert.equal(context.fetchCalls[0][0],'/api/start');
const request=context.fetchCalls[0][1];
assert.equal(request.headers['X-Diamond-Control'],'test');
assert.deepEqual(JSON.parse(request.body),{algorithm:'gpu-dfs',method:'systematic',backend:'webgpu',seconds:3600,replicas:128,seed:17});
assert.equal(storage.values['diamond.requestedBackend'],'webgpu');
console.log('Stored backend preference, active backend display, idle restoration, and Start payload passed');
"""
        result = subprocess.run([shutil.which('node'), '-e', harness, str(ROOT / 'Dashboard.html')],
                                cwd=ROOT, capture_output=True, text=True, timeout=FRONTEND_PROCESS_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is only needed for isolated frontend control tests')
    def test_frontend_constraint_modes_preserve_preferences_and_do_not_show_old_live_boards(self):
        harness = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{value:'4096',style:{},dataset:{},classList:{add(){},remove(){},toggle(){}},setAttribute(){},listeners:{},addEventListener(type,fn){this.listeners[type]=fn},clientWidth:500,textContent:'',innerHTML:'',checked:true});return elements.get(id)}
const storage={values:{'diamond.requestedBackend':'cuda'},getItem(key){return this.values[key]??null},setItem(key,value){this.values[key]=value}};
const context=vm.createContext({document:{getElementById:element,addEventListener(){}},window:{addEventListener(){},localStorage:storage},console,Date,Number,String,Math,Set,JSON,setTimeout(){return 1},clearTimeout(){},setInterval(){},fetchCalls:[]});
context.fetch=(url,options)=>{context.fetchCalls.push([url,options]);return new Promise(()=>{})};
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0].replace('poll();setInterval(poll,1000);','');
vm.runInContext(source,context);
function render(script){vm.runInContext(script+';renderState();',context)}
function select(value){element('algorithmInput').value=value;element('algorithmInput').listeners.change();}
render("state={process_running:false,control_token:'test',status:{state:'stopped',method:'systematic'},history:[]}");
assert.equal(element('algorithmInput').value,'dfs');
assert.equal(element('replicaInput').disabled,true);assert.equal(element('backendInput').disabled,true);
select('sat');assert.equal(storage.values['diamond.requestedAlgorithm'],'sat');
element('replicaInput').value='128';element('seconds').value='3600';element('seedInput').value='17';
vm.runInContext("command('start');",context);
assert.equal(element('algorithmInput').disabled,true);
const sent=JSON.parse(context.fetchCalls[0][1].body);
assert.equal(sent.algorithm,'sat');assert.equal(sent.method,'constraint');assert.equal(sent.replicas,128);
render("commandBusy=false;state={process_running:true,control_token:'test',status:{state:'running',method:'constraint',algorithm:'dfs',target:'gold',backend:'cpu',replicas:0,run_id:'new',nodes_checked:42,stats:{loop_prunes:3}},live:{method:'systematic',run_id:'old',codes:[0],validation:{matched_edges:207}},history:[]}");
assert.equal(element('algorithmInput').value,'dfs');assert.equal(element('algorithmInput').disabled,true);
assert.equal(element('replicaInput').value,'128');assert.equal(element('replicaInput').disabled,true);
assert.equal(element('replicasLabel').textContent,'CPU search workers');assert.equal(element('replicas').textContent,'1');
assert.equal(element('viewMeta').textContent,'No live placement yet');
assert.equal(element('validationStatus').textContent,'No candidate');
assert.match(element('controlHint').textContent,/one gold loop/);
assert.equal(element('cacheHits').textContent,'42');assert.equal(element('diskEntries').textContent,'3');
render("puzzle={tiles:Array.from({length:160},(_,i)=>({id:i,segments:[]})),geometry:{cells:[],vertices:[]}};state.live={method:'constraint',algorithm:'dfs',run_id:'new',codes:[0,...Array(159).fill(-1)],assigned_tiles:1,is_partial:true};");
assert.equal(element('matched').textContent,1);
assert.equal(element('validationStatus').textContent,'Partial branch');
assert.equal(element('viewMeta').textContent,'Actual partial branch');
render("state.status.algorithm='hybrid';state.status.backend='webgpu';state.status.replicas=256");
assert.equal(element('algorithmInput').value,'hybrid');
assert.equal(element('backendInput').value,'webgpu');
assert.equal(element('replicaInput').value,'256');
assert.equal(element('replicasLabel').textContent,'GPU sampling replicas');
assert.equal(storage.values['diamond.requestedBackend'],'cuda');
render("state.process_running=false;state.status.state='stopped'");
assert.equal(element('algorithmInput').value,'sat');
assert.equal(element('backendInput').value,'cuda');
assert.equal(element('backendInput').disabled,true);
assert.equal(element('replicaInput').disabled,true);
render("state.status={state:'error',method:'constraint',algorithm:'sat',target:'gold',backend:'cpu',replicas:0,error:'SAT requires optional python-sat'}");
assert.equal(element('runStatus').textContent,'Search error');
assert.match(element('controlHint').textContent,/python-sat/);
render("commandBusy=false;state={process_running:false,control_token:'test',status:{state:'stopped',method:'constraint',algorithm:'dfs'},history:[]}");
assert.equal(element('nativeBackend').textContent,'Not recorded');
select('parallel-dfs');element('workersInput').value='2';element('lookaheadInput').checked=false;element('nativeInput').value='compiled';element('lookaheadInput').listeners.change();
vm.runInContext("command('start');",context);
const parallelSent=JSON.parse(context.fetchCalls.at(-1)[1].body);
assert.equal(parallelSent.workers,2);assert.equal(parallelSent.algorithm,'parallel-dfs');assert.equal(parallelSent.search_options.lookahead,false);assert.equal(parallelSent.search_options.pair_lookahead,true);assert.equal(parallelSent.search_options.lookahead_ms,100);assert.equal(parallelSent.search_options.native,'compiled');assert.equal(parallelSent.search_options.regional_cache,true);for(const key of ['biconnected','separator_regions','backjumping','incremental','adaptive','persistent_learning'])assert.equal(parallelSent.search_options[key],true,key);
assert.equal(JSON.parse(storage.values['diamond.searchOptions']).lookahead,false);for(const key of ['reason_tracking','sat_assist','hierarchical_regions','value_ordering','shared_conflicts','gold_oracle','connection_branching','lcg','native_search'])assert.equal(parallelSent.search_options[key],false,key);
render("commandBusy=false;state={process_running:true,control_token:'test',status:{state:'running',method:'constraint',algorithm:'parallel-dfs',workers:2,native_backend:'compiled',parallel:{pending_jobs:7},stats:{lookahead_prunes:3,strand_prunes:4,cut_prunes:5,regional_prunes:6,regional_cache_hits:11,biconnected_prunes:12,separator_prunes:13,separator_merges:14,backjumps:15,propagation_unchanged_skips:16,propagation_adaptive_skips:17,native_support_hits:19,reason_conflicts:20,sat_assist_unsat:21,hierarchical_prunes:22,hierarchical_cache_hits:23,hierarchical_subtable_hits:24,value_order_changed:25,hall_explanation_witnesses:26,shared_conflicts_entries:27,shared_conflicts_pending:28,gold_oracle_infeasible:29,connection_branches:30,lcg_learned:31,native_search_trail_writes:32,native_fast_rejects:33,native_fast_support_hits:34,lcg_clause_skips:35,gold_oracle_contracted_ports:36,gold_oracle_subsumption_hits:37,native_basic_passes_skipped:38,native_fast_seeded_nodes:39,lcg_assertions:40,lcg_asserting_jumps:41,gold_oracle_template_hits:42,gold_oracle_proofs_subsumed:43,lookahead_common_prunes:44,lookahead_probe_propagations:45,cut_matching_failures:46,native_learning_clauses:47,native_learning_checks:48,lookahead_cache_hits:49,pair_lookahead_probes:50,pair_lookahead_failures:51,pair_lookahead_propagations:52,cut_global_matching_failures:53,cut_global_matching_steps:54,sat_assist_gold_clauses:55},search_loop_backend:'compiled',knowledge:{records:18}},history:[]}");
assert.equal(element('replicas').textContent,'2');assert.equal(element('nativeBackend').textContent,'compiled');assert.equal(element('searchLoopBackend').textContent,'Compiled');assert.equal(element('cpuJobs').textContent,'7');assert.equal(element('regionalHits').textContent,'11');assert.equal(element('lookaheadPrunes').textContent,'3');assert.equal(element('strandPrunes').textContent,'4');assert.equal(element('cutPrunes').textContent,'5');assert.equal(element('regionalPrunes').textContent,'6');assert.equal(element('workersInput').disabled,true);assert.equal(element('nativeInput').disabled,true);
for(const [id,value] of Object.entries({bottleneckPrunes:'12',separatorPrunes:'13',separatorMerges:'14',backjumps:'15',skippedChecks:'16',deferredChecks:'17',knownLoops:'18',supportHits:'19',reasonConflicts:'20',satAssistUnsat:'21',hierarchicalPrunes:'22',hierarchicalHits:'23'}))assert.equal(element(id).textContent,value,id);
for(const [id,count]of [['subtableHits','24'],['orderChanged','25'],['hallReasons','26'],['sharedConflicts','27'],['pendingConflicts','28'],['goldOraclePrunes','29'],['connectionBranches','30'],['lcgLearned','31'],['nativeTrailWrites','32'],['nativeFastRejects','33'],['nativeFastSupports','34'],['lcgSkipped','35'],['goldContracted','36'],['goldSubsumption','37'],['nativeBasicSkipped','38'],['nativeSeeded','39'],['lcgAssertions','40'],['lcgAssertingJumps','41'],['goldTemplates','42'],['goldProofsSubsumed','43'],['lookaheadCommonPrunes','44'],['lookaheadProbePasses','45'],['cutTileConflicts','46'],['nativeReplayClauses','47'],['nativeReplayChecks','48'],['probeCacheHits','49'],['pairProbes','50'],['pairFailures','51'],['pairPasses','52'],['globalCutFailures','53'],['globalCutSteps','54'],['goldSatClauses','55']])assert.equal(element(id).textContent,count);
render("commandBusy=false;state={process_running:false,control_token:'test',status:{state:'stopped',method:'constraint',algorithm:'cp-sat-circuit'},history:[]}");
select('cp-sat-circuit');assert.match(element('algorithmHint').textContent,/Experimental direct single-loop/);
assert.equal(element('backendInput').disabled,true);assert.equal(element('nativeInput').disabled,true);
vm.runInContext("command('start');",context);
const circuitSent=JSON.parse(context.fetchCalls.at(-1)[1].body);
assert.equal(circuitSent.algorithm,'cp-sat-circuit');assert.equal(circuitSent.method,'constraint');assert.equal(circuitSent.search_options,undefined);
render("commandBusy=false;state={process_running:true,control_token:'test',status:{state:'running',method:'constraint',algorithm:'cp-sat-circuit',stats:{circuit_nodes:5280,circuit_arcs:147840}},history:[]}");
assert.equal(element('cacheEntriesLabel').textContent,'Circuit ports / arcs');assert.equal(element('cacheEntries').textContent,'5,280 / 147,840');
assert.match(element('controlHint').textContent,/Direct circuit constraints/);assert.match(element('loadingText').textContent,/Searching internally/);
render("state.process_running=false;state.status.state='error';state.status.error='Experimental model is unsupported: arc limit'");
assert.equal(element('runStatus').textContent,'Search error');assert.match(element('controlHint').textContent,/arc limit/);
console.log('Gold-mode routing, advanced controls, parallel telemetry, and preserved preferences passed');
"""
        result = subprocess.run([shutil.which('node'), '-e', harness, str(ROOT / 'Dashboard.html')],
                                cwd=ROOT, capture_output=True, text=True, timeout=FRONTEND_PROCESS_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    @unittest.skipUnless(shutil.which('node'), 'Node.js is only needed for isolated frontend control tests')
    def test_frontend_active_search_options_restore_saved_next_run_preferences(self):
        harness = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
function element(id){if(!elements.has(id))elements.set(id,{value:'4096',style:{},dataset:{},classList:{add(){},remove(){},toggle(){}},setAttribute(){},listeners:{},addEventListener(type,fn){this.listeners[type]=fn},clientWidth:500,textContent:'',innerHTML:'',checked:true});return elements.get(id)}
const saved={failure_guided:false,lookahead:true,pair_lookahead:false,lookahead_ms:250,strand_connectivity:false,multi_seam_cuts:true,regional_cache:false,biconnected:false,separator_regions:true,backjumping:false,incremental:true,adaptive:false,persistent_learning:true,reason_tracking:false,sat_assist:false,hierarchical_regions:false,value_ordering:false,shared_conflicts:false,native:'python'};
const storage={values:{'diamond.requestedAlgorithm':'parallel-dfs','diamond.searchOptions':JSON.stringify(saved),'diamond.workers':'2'},getItem(key){return this.values[key]??null},setItem(key,value){this.values[key]=value}};
const originalStorage=JSON.stringify(storage.values);
const context=vm.createContext({document:{getElementById:element,addEventListener(){}},window:{addEventListener(){},localStorage:storage},console,Date,Number,String,Math,Set,JSON,setTimeout(){return 1},clearTimeout(){},setInterval(){}});
context.fetch=()=>new Promise(()=>{});
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0].replace('poll();setInterval(poll,1000);','');
vm.runInContext(source,context);
const ids={failure_guided:'failureGuidedInput',lookahead:'lookaheadInput',pair_lookahead:'pairLookaheadInput',strand_connectivity:'strandInput',multi_seam_cuts:'cutsInput',regional_cache:'regionalInput',biconnected:'biconnectedInput',separator_regions:'separatorInput',backjumping:'backjumpInput',incremental:'incrementalInput',adaptive:'adaptiveInput',persistent_learning:'knowledgeInput',reason_tracking:'reasonsInput',sat_assist:'satAssistInput',hierarchical_regions:'hierarchicalInput',value_ordering:'valueOrderInput',shared_conflicts:'sharedConflictsInput',gold_oracle:'goldOracleInput',connection_branching:'connectionBranchInput',lcg:'lcgInput',native_search:'nativeSearchInput'};
function render(script){vm.runInContext(script+';renderAlgorithmControls();',context)}
function assertOptions(expected){for(const[key,id]of Object.entries(ids))assert.equal(element(id).checked,expected[key]??false,key);assert.equal(element('nativeInput').value,expected.native);assert.equal(Number(element('lookaheadBudgetInput').value),expected.lookahead_ms??100);assert.equal(element('orderingGoalInput').value,expected.ordering_goal??'auto')}
// Opening this browser during a CLI-started run must display its actual settings.
render("state={process_running:true,status:{state:'running',method:'constraint',algorithm:'parallel-dfs',workers:3,search_options:{failure_guided:true,lookahead:false,pair_lookahead:true,lookahead_ms:1000,strand_connectivity:true,multi_seam_cuts:false,regional_cache:true,biconnected:true,separator_regions:false,backjumping:true,incremental:false,adaptive:true,persistent_learning:false,reason_tracking:true,sat_assist:true,hierarchical_regions:true,value_ordering:true,shared_conflicts:true,native:'compiled'}}}");
assertOptions({failure_guided:true,lookahead:false,pair_lookahead:true,lookahead_ms:1000,strand_connectivity:true,multi_seam_cuts:false,regional_cache:true,biconnected:true,separator_regions:false,backjumping:true,incremental:false,adaptive:true,persistent_learning:false,reason_tracking:true,sat_assist:true,hierarchical_regions:true,value_ordering:true,shared_conflicts:true,native:'compiled'});
assert.equal(element('workersInput').value,'3');assert.equal(element('activeWorkersOption').value,'3');assert.equal(element('activeWorkersOption').hidden,false);
for(const id of [...Object.values(ids),'nativeInput','lookaheadBudgetInput','workersInput'])assert.equal(element(id).disabled,true);
assert.equal(JSON.stringify(storage.values),originalStorage);
render("state.status.workers=8;state.status.search_options.native='auto'");
assert.equal(element('workersInput').value,'8');assert.equal(element('activeWorkersOption').hidden,true);assert.equal(element('nativeInput').value,'auto');
render("state.process_running=false;state.status.state='stopped'");
assertOptions(saved);assert.equal(element('workersInput').value,'2');assert.equal(element('activeWorkersOption').hidden,true);
for(const id of [...Object.values(ids),'nativeInput','lookaheadBudgetInput','workersInput'])assert.equal(element(id).disabled,false);
assert.equal(JSON.stringify(storage.values),originalStorage);
// An idle edit replaces only the next-run preference; later active settings do not.
element('lookaheadInput').checked=false;element('lookaheadBudgetInput').value='400';element('nativeInput').value='auto';element('workersInput').value='4';element('lookaheadInput').listeners.change();
const edited={...saved,lookahead:false,lookahead_ms:400,native:'auto',ordering_goal:'auto',gold_oracle:false,connection_branching:false,lcg:false,native_search:false,"smart_learning":false,"group_lookahead":false,"deep_paths":false,"conservation":false,"signature_branching":false,"impact_branching":false,"objective_ordering":false,"frontier_restarts":false,"worker_portfolio":false,"limited_discrepancy":false,"compact_explanations":false,"conditional_hall":false,"wide_cuts":false,"piece_symmetry":false,"proof_certificates":false,"rotated_learning":false,"extended_regions":false,"gpu_gold_sampling":false,"dependency_repair":false,"dirty_neighborhood":false,"parity_root_reuse":false};
for(const key of vm.runInContext('round24Keys',context))edited[key]=false;
for(const key of vm.runInContext('round25Keys',context))edited[key]=false;
assert.deepEqual(JSON.parse(storage.values['diamond.searchOptions']),edited);assert.equal(storage.values['diamond.workers'],'4');
render("state.process_running=true;state.status.workers=1;state.status.search_options={failure_guided:true,lookahead:true,strand_connectivity:true,multi_seam_cuts:true,regional_cache:true,native:'python'}");
assert.equal(element('nativeInput').value,'python');assert.equal(element('workersInput').value,'1');assert.equal(element('lookaheadInput').checked,true);
render("state.process_running=false");assertOptions(edited);assert.equal(element('workersInput').value,'4');
console.log('Active CPU settings, arbitrary CLI worker count, and next-run preference restoration passed');
"""
        result = subprocess.run([shutil.which('node'), '-e', harness, str(ROOT / 'Dashboard.html')],
                                cwd=ROOT, capture_output=True, text=True, timeout=FRONTEND_PROCESS_TIMEOUT)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_http_stop_preserves_token_origin_and_host_protection(self):
        self.status()
        server = ThreadingHTTPServer(('127.0.0.1', 0), dashboard.Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_port
        def request(method, path, headers=None, payload=None):
            connection = HTTPConnection('127.0.0.1', port, timeout=3)
            try:
                connection.request(method, path, body=payload, headers=headers or {})
                response = connection.getresponse()
                return response.status, json.loads(response.read())
            finally:
                connection.close()
        good = {'Content-Type': 'application/json', 'X-Diamond-Control': dashboard.TOKEN, 'Origin': f'http://127.0.0.1:{port}'}
        try:
            with patch.object(dashboard, 'live_pid', return_value=4242):
                status, _ = request('POST', '/api/stop', {**good, 'X-Diamond-Control': 'bad-token'}, '{}')
                self.assertEqual(status, 403)
                status, _ = request('POST', '/api/stop', {**good, 'Origin': 'https://unrelated.example'}, '{}')
                self.assertEqual(status, 403)
                status, _ = request('POST', '/api/stop', {**good, 'Host': 'unrelated.example'}, '{}')
                self.assertEqual(status, 403)
                self.assertFalse((self.directory / 'stop.request').exists())
                status, result = request('POST', '/api/stop', good, '{}')
                self.assertEqual(status, 200)
                self.assertTrue(result['stop_requested'])
                status, payload = request('GET', '/api/state')
                self.assertEqual(status, 200)
                self.assertTrue(payload['stop_requested'])
                self.assertEqual(payload['display_state'], 'stopping')
                self.assertEqual(payload['status']['state'], 'running')
                self.assertEqual(payload['installation_root'], str(ROOT.resolve()))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == '__main__':
    unittest.main()
