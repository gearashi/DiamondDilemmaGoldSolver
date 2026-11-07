"""Availability is interpreter-local; rejected launches must preserve saved state."""
import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import runtime_capabilities as runtime

ROOT = Path(__file__).resolve().parent


def capabilities(*modules, platform='win32'):
    with patch.object(runtime, '_installed', side_effect=lambda name: name in modules), \
         patch.object(runtime.sys, 'platform', platform):
        return runtime.runtime_capabilities()


class RuntimeCapabilityTests(unittest.TestCase):
    def test_package_probe_does_not_import_optional_packages_or_probe_hardware(self):
        with patch.object(runtime.importlib.util, 'find_spec', return_value=None) as find, \
             patch.object(runtime.sys, 'platform', 'win32'), \
             patch('subprocess.run', side_effect=AssertionError('No driver probes')):
            result = runtime.runtime_capabilities()
        self.assertEqual({call.args[0] for call in find.call_args_list}, {'ortools', 'pysat', 'cupy', 'wgpu'})
        self.assertEqual(result['detection'], 'installed_packages')
        self.assertTrue(result['algorithms']['dfs']['available'])
        self.assertFalse(result['algorithms']['cp-sat']['available'])

    def test_native_explanation_probe_checks_build_identity_without_loading_library(self):
        import hashlib,platform,sys
        with tempfile.TemporaryDirectory(prefix='native-capability-',dir=ROOT) as directory:
            root=Path(directory);folder=root/'.native';folder.mkdir()
            self.assertFalse(runtime._native_explanation_build(root)['available'])
            source=root/'constraint_native_search.c';library=folder/'example.dll'
            source.write_bytes(b'finite source');library.write_bytes(b'not an executable')
            metadata={'abi':1,'platform':sys.platform,'machine':platform.machine(),'library':library.name,
                      'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                      'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest()}
            manifest=folder/'search-build.json'
            manifest.write_text(json.dumps(metadata),encoding='utf-8')
            with patch('ctypes.CDLL',side_effect=AssertionError('No native code loading during availability detection')):
                self.assertTrue(runtime._native_explanation_build(root)['available'])
                for key,value in [('abi',2),('platform','foreign'),('machine','foreign'),('library','../outside.dll')]:
                    manifest.write_text(json.dumps({**metadata,key:value}),encoding='utf-8')
                    self.assertFalse(runtime._native_explanation_build(root)['available'])
                manifest.write_text(json.dumps(metadata),encoding='utf-8')
                library.write_bytes(b'changed')
                self.assertFalse(runtime._native_explanation_build(root)['available'])
                library.write_bytes(b'not an executable');source.write_bytes(b'changed')
                self.assertFalse(runtime._native_explanation_build(root)['available'])
                manifest.write_text('[]',encoding='utf-8')
                self.assertFalse(runtime._native_explanation_build(root)['available'])

    def test_missing_native_explanation_build_fails_before_launch_and_keeps_preferences(self):
        missing={'available':False,'reason':'Run python build_native.py first.'}
        with patch.object(runtime,'_native_explanation_build',return_value=missing):
            caps=capabilities('wgpu')
        options={'native_explanations':True,'native':'auto'};before=dict(options)
        with self.assertRaisesRegex(ValueError,'build_native.py'):
            runtime.validate_runtime_selection('dfs',options=options,capabilities=caps)
        self.assertEqual(options,before)
        runtime.validate_runtime_selection('dfs',options={'native_explanations':False},capabilities=caps)

    def test_webgpu_profile_supports_exact_dfs_and_gpu_without_optional_solvers(self):
        caps = capabilities('wgpu')
        for algorithm in ('dfs', 'parallel-dfs', 'gpu-dfs', 'hybrid'):
            runtime.validate_runtime_selection(algorithm, 'auto', capabilities=caps)
        runtime.validate_runtime_selection('hybrid', 'webgpu', capabilities=caps)
        self.assertFalse(caps['features']['repair']['available'])
        for algorithm in ('cp', 'cp-sat', 'cp-sat-circuit', 'sat'):
            with self.subTest(algorithm=algorithm), self.assertRaisesRegex(ValueError, 'unavailable'):
                runtime.validate_runtime_selection(algorithm, capabilities=caps)
        with self.assertRaisesRegex(ValueError, 'CuPy'):
            runtime.validate_runtime_selection('hybrid', 'cuda', capabilities=caps)
        with self.assertRaisesRegex(ValueError, 'CuPy'):
            runtime.validate_runtime_selection('gpu-dfs', stochastic=True, capabilities=caps)

    def test_explicit_optional_flags_fail_without_changing_preferences(self):
        caps = capabilities('wgpu')
        for key in ('gold_oracle', 'sat_assist', 'gold_pricing'):
            options = {key: True, 'native_search': True}
            original = dict(options)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                runtime.validate_runtime_selection('dfs', options=options, capabilities=caps)
            self.assertEqual(options, original)
            runtime.validate_runtime_selection('dfs', options={key: False}, capabilities=caps)
        # GPU-only fields and irrelevant exact flags do not affect CPU model modes.
        full = capabilities('ortools', 'pysat')
        runtime.validate_runtime_selection('cp-sat', 'cuda', {'sat_assist': True}, capabilities=full)

    def test_full_profile_and_platform_restriction(self):
        caps = capabilities('ortools', 'pysat', 'cupy', 'wgpu')
        for algorithm in caps['algorithms']:
            runtime.validate_runtime_selection(algorithm, 'cuda', {'gold_oracle': True, 'sat_assist': True}, capabilities=caps)
        mac = capabilities('ortools', 'pysat', 'cupy', 'wgpu', platform='darwin')
        self.assertFalse(mac['backends']['cuda']['available'])
        self.assertIn('macOS', mac['backends']['cuda']['reason'])
        self.assertTrue(mac['backends']['webgpu']['available'])

    def test_gpu_clause_sharing_requires_its_independent_sat_verifier(self):
        missing=capabilities('cupy');full=capabilities('cupy','pysat')
        with self.assertRaisesRegex(ValueError,'gpu_shared_clauses'):
            runtime.validate_runtime_selection('gpu-domains','cuda',{'gpu_shared_clauses':True},capabilities=missing)
        runtime.validate_runtime_selection('gpu-domains','cuda',{'gpu_shared_clauses':False},capabilities=missing)
        runtime.validate_runtime_selection('gpu-domains','cuda',{'gpu_shared_clauses':True},capabilities=full)
        import dashboard_server as dashboard
        import constraint_search
        with tempfile.TemporaryDirectory(prefix='gpu-clause-capability-',dir=ROOT) as directory:
            folder=Path(directory)
            for name in ('run.lock','stop.request','saved.json'):(folder/name).write_text('preserved',encoding='utf-8')
            with patch.object(dashboard,'RUNTIME',folder),patch.object(dashboard,'live_pid',return_value=None),patch.object(dashboard,'runtime_capabilities',return_value=missing),patch.object(dashboard.subprocess,'Popen') as spawn:
                with self.assertRaisesRegex(ValueError,'gpu_shared_clauses'):dashboard.start_solver({'algorithm':'gpu-domains','search_options':{'gpu_shared_clauses':True}})
                spawn.assert_not_called()
            self.assertEqual({path.name:path.read_text(encoding='utf-8') for path in folder.iterdir()},dict.fromkeys(('run.lock','stop.request','saved.json'),'preserved'))
            output=folder/'must-not-exist'
            with patch.object(runtime,'runtime_capabilities',return_value=missing),contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:constraint_search.main(['--algorithm','gpu-domains','--search-options','{"gpu_shared_clauses":true}','--output',str(output)])
            self.assertEqual(error.exception.code,2);self.assertFalse(output.exists())

    def test_missing_modules_and_invalid_module_specs_are_absent(self):
        for error in (ImportError('missing'), ValueError('missing spec')):
            with patch.object(runtime.importlib.util, 'find_spec', side_effect=error):
                self.assertFalse(runtime._installed('cupy'))

    def test_dashboard_rejection_preserves_lock_stop_and_saved_search(self):
        import dashboard_server as dashboard
        with tempfile.TemporaryDirectory(prefix='runtime-capabilities-', dir=ROOT) as directory:
            path = Path(directory)
            for name in ('run.lock', 'stop.request', 'saved.json'):
                (path / name).write_text('preserved', encoding='utf-8')
            with patch.object(dashboard, 'RUNTIME', path), \
                 patch.object(dashboard, 'live_pid', return_value=None), \
                 patch.object(dashboard, 'runtime_capabilities', return_value=capabilities('wgpu')), \
                 patch.object(dashboard.subprocess, 'Popen') as spawn:
                with self.assertRaisesRegex(ValueError, 'OR-Tools'):
                    dashboard.start_solver({'algorithm': 'cp-sat'})
                with self.assertRaisesRegex(ValueError, 'gold_oracle'):
                    dashboard.start_solver({'algorithm': 'dfs', 'search_options': {'gold_oracle': True}})
                result = dashboard.state_payload()
                self.assertFalse(result['capabilities']['algorithms']['cp-sat']['available'])
                spawn.assert_not_called()
            self.assertEqual({file.name: file.read_text(encoding='utf-8') for file in path.iterdir()},
                             dict.fromkeys(('run.lock', 'stop.request', 'saved.json'), 'preserved'))

    def test_cli_rejection_happens_before_output_creation(self):
        import constraint_search
        with tempfile.TemporaryDirectory(prefix='runtime-capabilities-', dir=ROOT) as directory:
            output = Path(directory) / 'must-not-be-created'
            with patch.object(runtime, 'runtime_capabilities', return_value=capabilities('wgpu')), \
                 contextlib.redirect_stderr(io.StringIO()) as error:
                with self.assertRaises(SystemExit) as caught:
                    constraint_search.main(['--algorithm', 'cp-sat', '--output', str(output)])
            self.assertEqual(caught.exception.code, 2)
            self.assertIn('OR-Tools', error.getvalue())
            self.assertFalse(output.exists())

    @unittest.skipUnless(shutil.which('node'), 'Node verifies the dashboard availability controls')
    def test_frontend_preserves_saved_choices_and_allows_clearing_unavailable_flags(self):
        source = (ROOT / 'Dashboard.html').read_text(encoding='utf-8')
        function = source[source.index('function renderRuntimeAvailability('):source.index('function renderAlgorithmControls(')]
        harness = r"""
const assert=require('assert');
const controls={};
function $(id){return controls[id]??= {checked:false,disabled:false,title:'',value:'',textContent:'',classList:{toggle(){}}}}
const optionIds={gold_oracle:'oracle',sat_assist:'sat'};
function readSearchOptions(){return Object.fromEntries(Object.entries(optionIds).map(([key,id])=>[key,$(id).checked]));}
function searchOptionIssues(){return [];} // Full dependency/scoping integration is tested in test_constraint_ui_round25.
let commandBusy=false;
let state={capabilities:JSON.parse(process.argv[1])};
$('algorithmInput').options=Object.keys(state.capabilities.algorithms).map(value=>({value}));
$('backendInput').options=Object.keys(state.capabilities.backends).map(value=>({value}));
$('algorithmInput').value='dfs';$('backendInput').value='webgpu';
$('oracle').checked=true;
__FUNCTION__
renderRuntimeAvailability('dfs',true,false,false);
assert.strictEqual($('oracle').checked,true,'Saved selection was rewritten');
assert.strictEqual($('oracle').disabled,false,'An unavailable saved flag must be clearable');
assert.strictEqual($('sat').disabled,true);
assert.strictEqual($('startBtn').disabled,true);
assert.match($('runtimeHint').textContent,/Uncheck/);
$('oracle').checked=false;$('startBtn').disabled=false;
renderRuntimeAvailability('dfs',true,false,false);
assert.strictEqual($('oracle').disabled,true);
assert.strictEqual($('startBtn').disabled,false);
assert.strictEqual($('algorithmInput').value,'dfs');
assert.strictEqual($('algorithmInput').options.find(x=>x.value==='cp-sat').disabled,true);
$('algorithmInput').value='cp-sat';$('startBtn').disabled=false;
renderRuntimeAvailability('cp-sat',false,false,false);
assert.strictEqual($('algorithmInput').value,'cp-sat','Saved algorithm was rewritten');
assert.strictEqual($('startBtn').disabled,true);
$('oracle').checked=true;$('startBtn').disabled=true;
renderRuntimeAvailability('dfs',true,false,true);
assert.strictEqual($('oracle').checked,true,'Active options were rewritten');
assert.strictEqual($('oracle').disabled,true);
renderRuntimeAvailability('hybrid',true,true,true);
assert.match($('runtimeHint').textContent,/repair is unavailable/);
console.log('ok');
""".replace('__FUNCTION__', function)
        result = subprocess.run([shutil.which('node'), '-e', harness, json.dumps(capabilities('wgpu'))],
                                capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(result.stdout.strip(), 'ok')


if __name__ == '__main__':
    unittest.main()
