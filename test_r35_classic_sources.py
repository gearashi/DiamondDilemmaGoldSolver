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
                shader.write_bytes(shader.read_bytes()+b'\n// dormant in this pure CUDA history\n')
                scenario.execute(resume=True,replicas=20,target='gold')
                self.assertEqual(scenario.constructions,constructions)
                hashes={**record['source_sha256'],leaf.name:hashlib.sha256(leaf.read_bytes()).hexdigest(),
                        shader.name:hashlib.sha256(shader.read_bytes()).hexdigest()}
                scope={k:record[k] for k in ('input_sha256','target','gpu_policy','frontier_sha256','proof_backends')}
                args=root,record['run_id'],record['origin_binding'],hashes,scope
                self.assertEqual(proof.verify_origin(*args,required_backend='cuda'),record['source_sha256'])
                for incoming in ('webgpu','unknown'):
                    with self.assertRaisesRegex(ValueError,'classic|changed'):
                        proof.verify_origin(*args,required_backend=incoming)
                # A checkpoint whose binding predates backend tracking trusts
                # both shader families despite its final displayed backend.
                legacy_scope={k:v for k,v in scope.items() if k!='proof_backends'}
                legacy_binding=proof.origin_binding(record['source_sha256'],legacy_scope)
                with self.assertRaisesRegex(ValueError,'changed'):
                    proof.verify_origin(root,record['run_id'],legacy_binding,hashes,legacy_scope)
                final=archived/'final-status.json'
                original=final.read_bytes()
                for final_state in ('error','unfamiliar',None):
                    if final_state is None:
                        final.unlink()
                    else:
                        final.write_text(json.dumps({'state':final_state}),encoding='utf8')
                    with self.assertRaisesRegex(ValueError,'errored or unknown'):
                        proof.verify_origin(*args)
                final.write_bytes(original)
                shader = changed/'dfs_kernels.cu'
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
            self.assertIn(sources.image_digest(selected),sources.GUARDED[sources.BACKENDS])
            self.assertIn('gpu_cpu_assist.py',selected)
            self.assertNotIn('constraint_dfs.py',selected)
            self.assertIn('constraint_dfs.py',normalized)
            flags={'cpu_probe':False,'cpu_endgame':False}
            for field in ('cpu_endgame','unknown_mode'):
                self.assertFalse(sources.compatible(image_dir,normalized,image_dir,normalized,{**flags,field:True}))
            for field in ('cpu_probe','proof_probes'):
                self.assertTrue(sources.compatible(image_dir,normalized,image_dir,normalized,{**flags,field:True}))
            for bad in ([],['unknown'],['webgpu','cuda'],['cuda','cuda'],['cuda',9]):
                self.assertFalse(sources.compatible(image_dir,normalized,image_dir,normalized,flags,backends=bad))
            changed=image_dir/'gpu_cpu_assist.py'
            changed.write_bytes(changed.read_bytes()+b'\n# unreviewed dispatcher\n')
            normalized[changed.name]=hashlib.sha256(changed.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            self.assertFalse(sources.compatible(image_dir,normalized,image_dir,normalized,flags))

    def test_legacy_fallback_and_invalid_contribution_claims(self):
        import numpy as np
        scope={'target':'gold','gpu_policy':{'cpu_probe':False,'cpu_endgame':False}}
        self.assertEqual(proof.checkpoint_scope({},scope),scope)
        for value in ([],['cpu'],['webgpu','cuda'],['cuda','cuda']):
            with self.assertRaises(ValueError):
                proof.checkpoint_scope({'exact_proof_backends':np.array(json.dumps(value))},scope)
        updated=proof.checkpoint_scope({'exact_proof_backends':np.array('["cuda"]')},scope)
        self.assertNotEqual(proof.origin_binding({'file':'hash'},updated),
                            proof.origin_binding({'file':'hash'},scope))

    def test_probe_ownership_is_checked_before_any_narrowed_terminal_source_alias(self):
        import numpy as np
        from gpu_cpu_assist import CPUAssist
        def payload(rows,stats):
            return {'cpu_assist_state':np.array(json.dumps({'version':1,'entries':rows,'stats':stats}))}
        flags={'cpu_probe':True,'cpu_endgame':False}
        proof.check_cpu_scope(payload([],{'cpu_endgame_calls':0}),flags)
        for state in ({},payload([{}],{}),payload([],{'cpu_endgame_calls':1}),
                      payload([],{'cpu_endgame_nodes':False})):
            with self.assertRaisesRegex(ValueError,'CPU'):
                proof.check_cpu_scope(state,flags)
        with self.assertRaisesRegex(ValueError,'CPU'):
            proof.check_cpu_scope(payload([],{}),{'cpu_probe':False,'cpu_endgame':False})
