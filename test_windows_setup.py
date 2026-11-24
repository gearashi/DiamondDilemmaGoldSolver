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
        for name in ('requirements.txt', 'requirements-common.txt', 'requirements-webgpu.txt', 'requirements-py315.txt'):
            (self.repo / name).write_text('fixture==1.0\n')
        (self.repo / 'build_native.py').write_text(
            "import sys\nassert sys.argv[1:] == ['--install-windows-bundle']\n")
        (self.repo / 'windows_profile.py').write_text(
            "import os,sys\nsys.exit(int(os.environ.get('SETUP_TEST_PROFILE_EXIT','0')))\n")
        (self.repo / 'cuda_probe.py').write_text(
            "import os,sys\nassert sys.argv[1:] == ['--record']\nsys.exit(int(os.environ.get('SETUP_TEST_CUDA_EXIT','0')))\n")
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
        (self.fake / 'sitecustomize.py').write_text(
            'import collections,os,sys,sysconfig\n'
            'from pathlib import Path\n'
            "if sys.orig_argv[1:3] == ['-m', 'venv']:\n"
            "    import venv\n"
            "    venv.EnvBuilder._setup_pip = lambda self, context: None  # pip is the fixture above\n"
            "version=os.environ.get('SETUP_TEST_VERSION', '3.12')\n"
            "if (Path(sys.prefix)/'old-profile.flag').is_file(): version='3.15'\n"
            "if version: sys.version_info=collections.namedtuple('version_info','major minor micro releaselevel serial')(*map(int,version.split('.')),0,'final',0)\n"
            "if os.environ.get('SETUP_TEST_FREE_THREADED')=='1':\n"
            "    original=sysconfig.get_config_var\n"
            "    sysconfig.get_config_var=lambda name: 1 if name=='Py_GIL_DISABLED' else original(name)\n")
        self.bin = self.root / 'fake tools'
        self.bin.mkdir()
        (self.bin / 'py.cmd').write_text(
            '@echo off\nif not "%1"=="-3.12" exit /b 1\n"' + sys.executable + '" %2 %3\n')
        self.pip_log = self.root / 'pip.json'
        self.data_log = self.root / 'data.json'
        self.env = dict(os.environ, PYTHONPATH=str(self.fake),
                        PATH=str(self.bin) + os.pathsep + os.environ.get('PATH', ''),
                        SETUP_TEST_PIP_LOG=str(self.pip_log), SETUP_TEST_DATA_LOG=str(self.data_log))

    def create_venv(self):
        subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(self.repo / 'venv')],
                       check=True, capture_output=True, text=True, timeout=30,
                       creationflags=subprocess.CREATE_NO_WINDOW)

    def run_setup(self, *args, **environment):
        return subprocess.run([POWERSHELL, '-NoProfile', '-ExecutionPolicy', 'Bypass',
                               '-File', str(self.repo / 'Setup.ps1'), *args], cwd=self.root,
                              env=dict(self.env, **environment), capture_output=True, text=True,
                              timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)

    def requirements(self):
        args = json.loads(self.pip_log.read_text())['args']
        self.assertIn('--require-virtualenv', args)
        self.assertIn('--no-user', args)
        self.assertIn('--only-binary=:all:', args)
        return [Path(args[i + 1]).name for i, value in enumerate(args) if value == '-r']

    def test_auto_installs_full_software_even_if_driver_is_missing(self):
        self.create_venv()
        result = self.run_setup('-SkipData', SETUP_TEST_CUDA_EXIT='2')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('missing NVIDIA display driver', result.stdout)
        self.assertEqual(self.requirements(), ['requirements-webgpu.txt', 'requirements.txt'])
        self.assertFalse(self.data_log.exists())

    def test_explicit_backend_and_pip_destination_isolation(self):
        self.create_venv()
        for backend, expected in (('webgpu', ['requirements-webgpu.txt']), ('cuda', ['requirements-webgpu.txt', 'requirements.txt'])):
            with self.subTest(backend=backend):
                result = self.run_setup('-SkipData', '-Backend', backend,
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
        self.assertRegex(result.stdout + result.stderr, r'separate\s+Windows\s+checkout')
        self.assertEqual(sentinel.read_bytes(), b'preserve foreign environment')
        self.assertFalse(self.pip_log.exists())

    def test_explicit_python_creates_local_environment_and_same_version_reuses_it(self):
        result = self.run_setup('-SkipData', '-Python', sys.executable, '-Backend', 'webgpu')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        config = self.repo / 'venv/pyvenv.cfg'
        before = config.read_bytes()
        result = self.run_setup('-SkipData', '-Python', sys.executable, '-Backend', 'webgpu')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(config.read_bytes(), before)
        self.assertEqual(self.requirements(), ['requirements-webgpu.txt'])

    def test_315_request_selects_full_profile_for_every_backend(self):
        requested = self.bin / "requested python's version.cmd"
        requested.write_text('@echo off\nset "SETUP_TEST_VERSION=3.15"\n"' + sys.executable + '" %*\n')
        for backend in ('auto', 'cuda', 'webgpu'):
            with self.subTest(backend=backend):
                result = self.run_setup('-SkipData', '-Python', str(requested), '-Backend', backend)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn('Python 3.15 detected', result.stdout)
                expected = ['requirements-webgpu.txt']
                if backend != 'webgpu':
                    expected.append('requirements.txt')
                self.assertEqual(self.requirements(), expected)

    def test_existing_315_venv_migrates_with_contents_preserved(self):
        self.create_venv()
        (self.repo / 'venv/old-profile.flag').write_bytes(b'original')
        result = self.run_setup('-SkipData')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.repo / 'venv/old-profile.flag').exists())
        backups = list((self.repo / '.solver-setup').glob('previous-venv-*'))
        self.assertEqual(len(backups), 1)
        self.assertEqual((backups[0] / 'old-profile.flag').read_bytes(), b'original')

    def test_failed_315_migration_restores_original_environment(self):
        self.create_venv()
        marker = self.repo / 'venv/old-profile.flag'
        marker.write_bytes(b'original')
        result = self.run_setup('-SkipData', SETUP_TEST_PROFILE_EXIT='1')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Restored the previous', result.stdout)
        self.assertEqual(marker.read_bytes(), b'original')

    def test_false_full_profile_and_broken_cuda_fail_setup(self):
        self.create_venv()
        for backend, profile_exit, cuda_exit in (
                ('webgpu', '1', '0'), ('auto', '0', '1'), ('cuda', '0', '2')):
            with self.subTest(backend=backend):
                result = self.run_setup('-Backend', backend, SETUP_TEST_PROFILE_EXIT=profile_exit,
                                        SETUP_TEST_CUDA_EXIT=cuda_exit)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.data_log.exists())

    def test_unsupported_and_free_threaded_interpreters_fail_before_install(self):
        self.create_venv()
        for environment in ({'SETUP_TEST_VERSION': '3.14'}, {'SETUP_TEST_FREE_THREADED': '1'}):
            with self.subTest(environment=environment):
                result = self.run_setup('-SkipData', **environment)
                self.assertNotEqual(result.returncode, 0)
                self.assertRegex(result.stdout + result.stderr, r'regular\s+64-bit\s+Python\s+3\.12\s+or\s+3\.15')
                self.assertFalse(self.pip_log.exists())

    def test_package_install_failure_does_not_prepare_data(self):
        self.create_venv()
        result = self.run_setup('-Backend', 'webgpu', SETUP_TEST_PIP_EXIT='4')
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stdout + result.stderr, r'Dependency\s+installation\s+failed')
        self.assertFalse(self.data_log.exists())


if __name__ == '__main__':
    unittest.main()
