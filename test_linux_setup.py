"""Run Linux setup behavior tests with fake interpreters; no installs/network/GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
BASH = shutil.which('bash')

FAKE_PYTHON = r'''
import json, os
from pathlib import Path
import shutil, sys
args = sys.argv[1:]
entry = {'args': args, 'executable': str(Path(__file__)), 'cwd': os.getcwd(),
         'pip_target': os.environ.get('PIP_TARGET'), 'pip_prefix': os.environ.get('PIP_PREFIX'),
         'pip_user': os.environ.get('PIP_USER'), 'pip_config': os.environ.get('PIP_CONFIG_FILE')}
with open(os.environ['SETUP_TEST_LOG'], 'a') as output:
    output.write(json.dumps(entry) + '\n')
if args and args[0] == '-c':
    if 'sys.version_info' in args[1]:
        bad = Path(__file__).name in json.loads(os.environ.get('SETUP_BAD_NAMES', '[]'))
        bad |= Path(__file__).parent.name == 'bin' and Path(__file__).parent.parent.name == 'venv' and os.environ.get('SETUP_BAD_VENV') == '1'
        raise SystemExit(1 if bad else 0)
    if 'sys.prefix' in args[1]:
        raise SystemExit(1 if os.environ.get('SETUP_BAD_ISOLATION') == '1' else 0)
if args[:2] == ['-m', 'venv']:
    if os.environ.get('SETUP_FAIL_VENV') == '1': raise SystemExit(3)
    target = Path(args[2]) / 'bin' / 'python'
    target.parent.mkdir(parents=True)
    shutil.copyfile(__file__, target)
    target.chmod(0o755)
    raise SystemExit(0)
if args[:2] == ['-m', 'pip']:
    raise SystemExit(4 if os.environ.get('SETUP_FAIL_PIP') == '1' else 0)
if args and args[0].endswith('prepare_data.py'):
    raise SystemExit(5 if os.environ.get('SETUP_FAIL_DATA') == '1' else 0)
raise SystemExit('Unexpected fake interpreter arguments: ' + repr(args))
'''


@unittest.skipUnless(os.name == 'posix' and BASH, 'Requires POSIX Bash; run on Linux/macOS CI')
class LinuxSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="diamond setup's ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / "release $copy's directory"
        self.caller = self.root / 'caller'
        self.bin = self.root / 'interpreters'
        for folder in (self.repo, self.caller, self.bin): folder.mkdir()
        shutil.copyfile(ROOT / 'Setup.sh', self.repo / 'Setup.sh')
        for name in ('requirements.txt', 'requirements-common.txt', 'requirements-webgpu.txt'):
            (self.repo / name).write_text('fixture==1.0\n')
        (self.repo / 'prepare_data.py').write_text('# fake interpreter handles this path\n')
        self.log = self.root / 'calls.jsonl'
        self.env = dict(os.environ, SETUP_TEST_LOG=str(self.log), PATH=str(self.bin) + ':/usr/bin:/bin')
        self.env.pop('BASH_ENV', None)
        for name in ('python3.12', 'python3'): self.make_python(name)
        self.make_command('uname', 'printf Linux')
        self.make_command('nvidia-smi', 'exit 1')

    def make_command(self, name, body):
        target = self.bin / name
        target.write_text('#!/bin/sh\n' + body + '\n')
        target.chmod(0o755)
        return target

    def make_python(self, name):
        target = self.bin / name
        target.write_text('#!' + sys.executable + '\n' + FAKE_PYTHON)
        target.chmod(0o755)
        return target

    def run_setup(self, *args, **environment):
        return subprocess.run([BASH, str(self.repo / 'Setup.sh'), *args], cwd=self.caller,
                              env=dict(self.env, **environment), text=True, capture_output=True, timeout=20)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_syntax_and_help_do_not_invoke_python(self):
        result = subprocess.run([BASH, '-n', str(self.repo / 'Setup.sh')], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_setup('--help')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--skip-data', result.stdout)
        self.assertEqual(self.calls(), [])

    def test_explicit_python_and_source_paths_preserve_spaces_apostrophes_and_caller(self):
        python = self.make_python("python's custom interpreter")
        source = self.caller / "source $(touch SHOULD_NOT_EXIST) ' files"
        source.mkdir()
        result = self.run_setup('--python', str(python.relative_to(self.root.parent)) if False else os.path.relpath(python, self.caller),
                                '--source-dir', source.name)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        prepare = next(call for call in calls if call['args'][0].endswith('prepare_data.py'))
        self.assertEqual(prepare['args'], [str(self.repo / 'prepare_data.py'), '--source-dir', str(source.resolve())])
        self.assertTrue(all(call['cwd'] == str(self.caller) for call in calls))
        self.assertFalse((self.caller / 'SHOULD_NOT_EXIST').exists())
        pip = next(call for call in calls if call['args'][:2] == ['-m', 'pip'])
        self.assertEqual(pip['executable'], str(self.repo / 'venv/bin/python'))
        self.assertIn('--require-virtualenv', pip['args'])
        self.assertIn('--no-user', pip['args'])
        self.assertEqual(pip['args'][-1], str(self.repo / 'requirements-webgpu.txt'))

    def test_detection_falls_back_to_python3_and_skip_data_skips_only_data(self):
        result = self.run_setup('--skip-data', SETUP_BAD_NAMES=json.dumps(['python3.12']))
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        creation = next(call for call in calls if call['args'][:2] == ['-m', 'venv'])
        self.assertEqual(Path(creation['executable']).name, 'python3')
        self.assertTrue(any(call['args'][:2] == ['-m', 'pip'] for call in calls))
        self.assertFalse(any(call['args'][0].endswith('prepare_data.py') for call in calls))

    def selected_requirements(self):
        pip = next(call for call in self.calls() if call['args'][:2] == ['-m', 'pip'])
        return [Path(pip['args'][i + 1]).name for i, value in enumerate(pip['args']) if value == '-r']

    def test_auto_adds_cuda_only_for_a_successful_nonempty_nvidia_query(self):
        for command, expected in (
                ('exit 1', ['requirements-webgpu.txt']),
                ('exit 0', ['requirements-webgpu.txt']),
                ("printf '   '; exit 0", ['requirements-webgpu.txt']),
                ("printf 'RTX fixture'; exit 1", ['requirements-webgpu.txt']),
                ("printf 'RTX fixture'; exit 0", ['requirements-webgpu.txt', 'requirements.txt'])):
            with self.subTest(command=command):
                self.make_command('nvidia-smi', command)
                if self.log.exists(): self.log.unlink()
                result = self.run_setup('--skip-data')
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.selected_requirements(), expected)

    def test_explicit_backend_does_not_depend_on_gpu_detection(self):
        self.make_command('nvidia-smi', "printf 'RTX fixture'")
        for backend, expected in (('cuda', ['requirements.txt']), ('webgpu', ['requirements-webgpu.txt'])):
            with self.subTest(backend=backend):
                if self.log.exists(): self.log.unlink()
                result = self.run_setup('--skip-data', '--backend', backend)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.selected_requirements(), expected)

    def test_macos_uses_webgpu_and_rejects_explicit_cuda_before_changes(self):
        self.make_command('uname', 'printf Darwin')
        self.make_command('nvidia-smi', "printf 'NVIDIA fixture'")
        rejected = self.run_setup('--backend', 'cuda', '--skip-data')
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn('CUDA is not supported on macOS', rejected.stderr)
        self.assertFalse((self.repo / 'venv').exists())
        self.assertEqual(self.calls(), [])
        result = self.run_setup('--skip-data')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.selected_requirements(), ['requirements-webgpu.txt'])

    def test_existing_environment_is_reused(self):
        self.assertEqual(self.run_setup('--skip-data').returncode, 0)
        self.log.unlink()
        result = self.run_setup('--skip-data')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(any(call['args'][:2] == ['-m', 'venv'] for call in self.calls()))

    def test_invalid_explicit_interpreter_does_not_fall_back(self):
        result = self.run_setup('--python', 'python3.12', SETUP_BAD_NAMES=json.dumps(['python3.12']))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('64-bit Python 3.12', result.stderr)
        self.assertEqual(len(self.calls()), 1)

    def test_missing_or_unknown_arguments_fail_before_changes(self):
        for args in (('--python',), ('--source-dir',), ('--backend',), ('--backend', 'unknown'), ('--unknown',)):
            with self.subTest(args=args):
                result = self.run_setup(*args)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(self.calls(), [])
        self.assertFalse((self.repo / 'venv').exists())

    def test_missing_source_directory_fails_before_package_install(self):
        result = self.run_setup('--source-dir', 'missing')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Source directory does not exist', result.stderr)
        self.assertEqual(self.calls(), [])

    def test_incomplete_or_windows_environment_is_preserved(self):
        scripts = self.repo / 'venv/Scripts'
        scripts.mkdir(parents=True)
        sentinel = scripts / 'python.exe'
        sentinel.write_bytes(b'keep this environment')
        result = self.run_setup('--skip-data')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('separate checkout for this operating system', result.stderr)
        self.assertEqual(sentinel.read_bytes(), b'keep this environment')
        self.assertEqual(self.calls(), [])

    def test_venv_creation_failure_does_not_install_packages(self):
        result = self.run_setup('--skip-data', SETUP_FAIL_VENV='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('venv/ensurepip', result.stderr)
        self.assertFalse(any(call['args'][:2] == ['-m', 'pip'] for call in self.calls()))

    def test_isolation_failure_prevents_package_install(self):
        result = self.run_setup('--skip-data', SETUP_BAD_ISOLATION='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('not an isolated environment', result.stderr)
        self.assertFalse(any(call['args'][:2] == ['-m', 'pip'] for call in self.calls()))

    def test_pip_destination_overrides_do_not_escape_environment(self):
        result = self.run_setup('--skip-data', PIP_TARGET='/unwanted', PIP_PREFIX='/unwanted', PIP_USER='1')
        self.assertEqual(result.returncode, 0, result.stderr)
        pip = next(call for call in self.calls() if call['args'][:2] == ['-m', 'pip'])
        self.assertIsNone(pip['pip_target'])
        self.assertIsNone(pip['pip_prefix'])
        self.assertIsNone(pip['pip_user'])
        self.assertEqual(pip['pip_config'], '/dev/null')

    def test_pip_and_data_errors_propagate(self):
        result = self.run_setup(SETUP_FAIL_PIP='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Dependency installation failed', result.stderr)
        self.assertFalse(any(call['args'][0].endswith('prepare_data.py') for call in self.calls()))
        self.log.unlink()
        result = self.run_setup(SETUP_FAIL_DATA='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Data preparation failed', result.stderr)


if __name__ == '__main__':
    unittest.main()
