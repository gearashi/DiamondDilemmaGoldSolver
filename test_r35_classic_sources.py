"""Scope-restricted host reopen of locally authenticated classic results."""
import hashlib
import json
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import r35_classic_sources as sources
import systematic_evidence as proof
import systematic_search as runner
from test_systematic_runner import Scenario, scratch


class ClassicDormantTests(unittest.TestCase):
    def test_terminal_open_across_dormant_change_keeps_full_origin_and_rejects_active_change(self):
        with scratch() as (tmp,data):
            scenario = Scenario(tmp/'run',data)
            scenario.execute(replicas=40,target='gold')
            root = scenario.output
            marker = root/'systematic/exhaustion.json'
            before = marker.read_bytes()
            record = json.loads(before)
            archived = root/'runs'/record['run_id']
            changed = tmp/'other-source'
            changed.mkdir()
            for name in record['source_sha256']:
                shutil.copyfile(archived/name,changed/name)
            leaf = changed/'constraint_work.py'
            with leaf.open('ab') as f:
                f.write(b'\n# dormant endgame-only change\n')
            constructions = scenario.constructions
            with patch.object(runner,'ROOT',changed), patch.object(proof,'__file__',str(changed/'systematic_evidence.py')):
                scenario.execute(resume=True,replicas=20,target='gold')
                status = proof.read(root/'status.json')
                self.assertTrue(status['terminal_reopened_on_host'])
                self.assertTrue(status['verified_coverage_origin'])
                self.assertEqual(status['exhausted_jobs'],480)
                self.assertEqual(scenario.constructions,constructions)
                self.assertEqual(scenario.steps_this_run,0)
                self.assertEqual(marker.read_bytes(),before)
                shader = changed/'dfs_kernels.wgsl'
                with shader.open('ab') as f:
                    f.write(b'\n// active on cross-backend resume\n')
                with self.assertRaisesRegex(SystemExit,'source'):
                    scenario.execute(resume=True,replicas=20,target='gold')
            self.assertEqual(marker.read_bytes(),before)
            (archived/'constraint_work.py').write_bytes(b'corrupted archived evidence')
            with self.assertRaisesRegex(SystemExit,'source'):
                scenario.execute(resume=True,replicas=20,target='gold')

    def test_a_foreign_dispatcher_or_cpu_policy_keeps_full_authority(self):
        with scratch() as (tmp,_):
            root = Path(sources.__file__).parent
            image_dir=tmp/'source'
            image_dir.mkdir()
            full=proof.capture(root,image_dir,'dummy')
            full.pop('tiles.json')
            normalized=proof.archived_sources(image_dir,full)
            selected=sources.active_image(image_dir,normalized)
            self.assertIn(sources.image_digest(selected),sources.REVIEWED)
            self.assertIn('gpu_cpu_assist.py',selected)
            self.assertNotIn('constraint_dfs.py',selected)
            self.assertIn('constraint_dfs.py',normalized)
            flags={'cpu_probe':False,'cpu_endgame':False}
            for field in ('cpu_endgame','cpu_probe','proof_probes','unknown_mode'):
                self.assertFalse(sources.compatible(image_dir,normalized,image_dir,normalized,{**flags,field:True}))
            changed=image_dir/'gpu_cpu_assist.py'
            changed.write_bytes(changed.read_bytes()+b'\n# unreviewed dispatcher\n')
            normalized[changed.name]=hashlib.sha256(changed.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            self.assertFalse(sources.compatible(image_dir,normalized,image_dir,normalized,flags))
