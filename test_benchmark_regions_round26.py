import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import benchmark_regions_round26 as benchmark

class RegionalEvidenceGuardTests(unittest.TestCase):
    def test_existing_output_refused_before_any_work(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'existing.json';output.write_text('{}',encoding='utf-8')
            with patch.dict(benchmark.RUNNERS,{'F09':lambda: self.fail('must not run')},clear=True):
                with self.assertRaises(ValueError):benchmark.main(['--output',str(output)])
            self.assertEqual(output.read_text(encoding='utf-8'),'{}')
    def test_dirty_frozen_and_changed_sources_refuse_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'fresh.json'
            with patch.object(benchmark,'source_snapshot',return_value={'source':'same'}),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={'available':True,'dirty':True,'commit':'abc'}):
                with self.assertRaises(RuntimeError):benchmark.main(['--output',str(output),'--evidence-stage','frozen'])
            self.assertFalse(output.exists())
            with patch.dict(benchmark.RUNNERS,{'F09':lambda:{'complete':True}},clear=True),patch.object(benchmark,'source_snapshot',side_effect=({'source':'first'},{'source':'changed'})),patch.object(benchmark,'runtime_snapshot',return_value={}),patch.object(benchmark,'repository_snapshot',return_value={'available':True,'dirty':False,'commit':'abc'}):
                with self.assertRaises(RuntimeError):benchmark.main(['--output',str(output)])
            self.assertFalse(output.exists())
    def test_optimized_runtime_refused_before_native_imports(self):
        import subprocess,sys
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'refused.json'
            result=subprocess.run([sys.executable,'-O',str(Path(benchmark.__file__)),'--output',str(output)],capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0);self.assertIn('assert',result.stderr.lower());self.assertFalse(output.exists())
