"""Windows setup behavior with an isolated venv and fake pip; no installs/GPU."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent
POWERSHELL = shutil.which('powershell.exe')


@unittest.skipUnless(os.name == 'nt' and POWERSHELL, 'Windows PowerShell setup')
class WindowsSetupTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="diamond setup's ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.repo = self.root / 'release copy'
        self.repo.mkdir()
        shutil.copyfile(ROOT / 'Setup.ps1', self.repo / 'Setup.ps1')
        for name in ('requirements.txt', 'requirements-common.txt', 'requirements-webgpu.txt'):
            (self.repo / name).write_text('fixture==1.0\n')
        self.fake = self.root / 'fake modules'
        (self.fake / 'pip').mkdir(parents=True)
        (self.fake / 'pip/__init__.py').write_text('')
        (self.fake / 'pip/__main__.py').write_text(
            'import json,os,sys\n'
            'from pathlib import Path\n'
            "Path(os.environ['SETUP_TEST_PIP_LOG']).write_text(json.dumps({'args':sys.argv[1:],'env':{key:os.environ.get(key) for key in ['PIP_TARGET','PIP_PREFIX','PIP_USER','PIP_ROOT','PIP_CONFIG_FILE']}}))\n"
            "sys.exit(int(os.environ.get('SETUP_TEST_PIP_EXIT','0')))\n")
        (self.repo / 'prepare_data.py').write_text(
            "import os,sys,json\nfrom pathlib import Path\nPath(os.environ['SETUP_TEST_DATA_LOG']).write_text(json.dumps(sys.argv[1:]))\n")
        self.bin = self.root / 'fake tools'
        self.bin.mkdir()
        (self.bin / 'nvidia-smi.cmd').write_text(
            '@echo off\n'
            'if "%SETUP_TEST_NVIDIA%"=="working" echo Fixture GPU\n'
            'if "%SETUP_TEST_NVIDIA%"=="failed" exit /b 1\n'
            'exit /b 0\n')
        self.pip_log = self.root / 'pip.json'
        self.data_log = self.root / 'data.json'
        self.env = dict(os.environ, PYTHONPATH=str(self.fake),
                        PATH=str(self.bin) + os.pathsep + os.environ.get('PATH', ''),
                        SETUP_TEST_PIP_LOG=str(self.pip_log), SETUP_TEST_DATA_LOG=str(self.data_log),
                        SETUP_TEST_NVIDIA='failed')

    def create_venv(self):
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(self.repo / 'venv')],
                       check=True, capture_output=True, text=True, timeout=30,
                       creationflags=subprocess.CREATE_NO_WINDOW)

    def run_setup(self, *args, **environment):
        return subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass',
                               '-File', str(self.repo / 'Setup.ps1'), *args], cwd=self.root,
                              env=dict(self.env, **environment), capture_output=True, text=True,
                              timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)

    def requirements(self):
        args = json.loads(self.pip_log.read_text())['args']
        self.assertIn('--require-virtualenv', args)
        self.assertIn('--no-user', args)
        return [Path(args[i + 1]).name for i, value in enumerate(args) if value == '-r']

    def test_auto_adds_cuda_only_for_successful_nonempty_driver_query(self):
        self.create_venv()
        for response, expected in (
                ('failed', ['requirements-webgpu.txt']),
                ('empty', ['requirements-webgpu.txt']),
                ('working', ['requirements-webgpu.txt', 'requirements.txt'])):
            with self.subTest(response=response):
                result = self.run_setup('-SkipData', SETUP_TEST_NVIDIA=response)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.requirements(), expected)
                self.assertFalse(self.data_log.exists())

    def test_explicit_backend_and_pip_destination_isolation(self):
        self.create_venv()
        for backend, expected in (('webgpu', ['requirements-webgpu.txt']), ('cuda', ['requirements.txt'])):
            with self.subTest(backend=backend):
                result = self.run_setup('-SkipData', '-Backend', backend, SETUP_TEST_NVIDIA='working',
                                        PIP_TARGET='outside-fixture', PIP_PREFIX='outside-fixture', PIP_USER='1', PIP_ROOT='outside-fixture')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(self.requirements(), expected)
                environment = json.loads(self.pip_log.read_text())['env']
                self.assertEqual(environment.pop('PIP_CONFIG_FILE'), 'NUL')
                self.assertTrue(all(value is None for value in environment.values()))

    def test_relative_source_path_and_data_preparation(self):
        self.create_venv()
        source = self.root / "source's files"
        source.mkdir()
        result = self.run_setup('-SourceDir', source.name, '-Backend', 'webgpu')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(self.data_log.read_text()), ['--source-dir', str(source)])

    def test_foreign_environment_is_preserved_before_any_package_install(self):
        sentinel = self.repo / 'venv/bin/python'
        sentinel.parent.mkdir(parents=True)
        sentinel.write_bytes(b'preserve foreign environment')
        result = self.run_setup('-SkipData')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('separate Windows checkout', result.stdout + result.stderr)
        self.assertEqual(sentinel.read_bytes(), b'preserve foreign environment')
        self.assertFalse(self.pip_log.exists())

    def test_package_install_failure_does_not_prepare_data(self):
        self.create_venv()
        result = self.run_setup('-Backend', 'webgpu', SETUP_TEST_PIP_EXIT='4')
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stdout + result.stderr, r'Dependency\s+installation\s+failed')
        self.assertFalse(self.data_log.exists())


if __name__ == '__main__':
    unittest.main()
