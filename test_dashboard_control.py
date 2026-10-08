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

ROOT = Path(__file__).resolve().parent


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
    def test_launcher_refuses_foreign_or_unknown_installations_without_process_actions(self):
        harness = r"""
param([string]$LauncherPath)
. $LauncherPath
$script:Calls = [System.Collections.Generic.List[object]]::new()
$script:ProbeFails = $false
$script:Occupied = $false
$script:FilesPresent = $true
$script:Reply = $null
function Invoke-RestMethod {
    param($Uri, $TimeoutSec)
    if ($script:ProbeFails) { throw 'Simulated HTTP failure' }
    return $script:Reply
}
function Test-DiamondDashboardPortListening { return $script:Occupied }
function Test-Path { param($LiteralPath) return $script:FilesPresent }
function Start-Sleep { param($Milliseconds) }
function Stop-Process { throw 'The launcher must never stop another process' }
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, $WindowStyle)
    $script:Calls.Add(@{FilePath=$FilePath; WindowStyle=$WindowStyle})
    if ($FilePath -like '*python.exe') {
        $script:ProbeFails = $false
        $script:Reply = [pscustomobject]@{service='diamond-dilemma'; installation_root=$SolverRoot}
    }
}
function Expect-Failure {
    param([scriptblock]$Action, [string]$Pattern)
    $Message = $null
    try { & $Action } catch { $Message = $_.Exception.Message }
    if ($null -eq $Message -or $Message -notmatch $Pattern) {
        throw "Expected failure matching '$Pattern', got '$Message'"
    }
}
function Assert-True {
    param([bool]$Condition, [string]$Message)
    if (-not $Condition) { throw $Message }
}

# Windows identity matching is case-insensitive and normalizes slash/dot suffixes.
$script:Reply = [pscustomobject]@{
    service='diamond-dilemma'
    installation_root=$SolverRoot.ToUpperInvariant().Replace('\','/') + '/./'
}
Open-DiamondDashboard
Assert-True ($script:Calls.Count -eq 1) 'Existing matching server should only open its browser'
Assert-True ($script:Calls[0].FilePath -eq $DashboardUrl) 'Wrong browser target'
$script:Calls.Clear()

$script:Reply = [pscustomobject]@{service='diamond-dilemma'; installation_root=$SolverRoot + '-other'}
Expect-Failure { Open-DiamondDashboard } 'another Diamond Dilemma installation'
Assert-True ($script:Calls.Count -eq 0) 'Different installation caused a process launch'

foreach ($Response in @(
    $null,
    [pscustomobject]@{service='diamond-dilemma'},
    [pscustomobject]@{service='diamond-dilemma'; installation_root=123},
    [pscustomobject]@{service='diamond-dilemma'; installation_root='relative-folder'},
    [pscustomobject]@{service='unrelated'; installation_root=$SolverRoot}
)) {
    $script:Reply = $Response
    Expect-Failure { Open-DiamondDashboard } 'unknown dashboard identity|invalid installation path'
    Assert-True ($script:Calls.Count -eq 0) 'Unknown installation caused a process launch'
}

# HTTP errors do not imply that a port is free: a non-JSON or older service
# must be left alone. No real HTTP or TCP request is made by this harness.
$script:ProbeFails = $true
$script:Occupied = $true
Expect-Failure { Open-DiamondDashboard } 'occupied'
Assert-True ($script:Calls.Count -eq 0) 'Occupied unverified port caused a process launch'

$script:Occupied = $false
$script:FilesPresent = $false
Expect-Failure { Open-DiamondDashboard } 'Puzzle data is missing.*Setup.cmd'
Assert-True ($script:Calls.Count -eq 0) 'Missing puzzle data caused a process launch'

# A verified absent server retains the original preflight and hidden startup.
$script:FilesPresent = $true
Open-DiamondDashboard
Assert-True ($script:Calls.Count -eq 2) 'Absent server should start once then open its verified browser'
Assert-True ($script:Calls[0].WindowStyle -eq 'Hidden') 'Dashboard server was not launched hidden'
Assert-True ($script:Calls[1].FilePath -eq $DashboardUrl) 'Browser opened before correct installation was verified'
Write-Output 'Installation identity, occupied-port refusal, preflight, and hidden startup passed'
"""
        script = self.directory / 'launcher-probe-tests.ps1'
        script.write_text(harness, encoding='utf-8')
        result = subprocess.run(
            [shutil.which('powershell.exe'), '-NoProfile', '-ExecutionPolicy', 'Bypass',
             '-File', str(script), str(ROOT / 'OpenDashboard.ps1')],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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
console.log('Immediate stop, persistent replica edits, 128-lane resume, paused counts, and unlimited duration passed');
"""
        result = subprocess.run([shutil.which('node'), '-e', harness, str(ROOT / 'Dashboard.html')], cwd=ROOT, capture_output=True, text=True, timeout=10)
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
assert.deepEqual(JSON.parse(request.body),{method:'systematic',backend:'webgpu',seconds:3600,replicas:128,seed:17});
assert.equal(storage.values['diamond.requestedBackend'],'webgpu');
console.log('Stored backend preference, active backend display, idle restoration, and Start payload passed');
"""
        result = subprocess.run([shutil.which('node'), '-e', harness, str(ROOT / 'Dashboard.html')],
                                cwd=ROOT, capture_output=True, text=True, timeout=10)
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
