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
        with patch.object(dashboard, 'live_pid', return_value=None):
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
