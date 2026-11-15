"""Classic systematic proof provenance and independently runnable migration controls."""
import json
import sys
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest

import numpy as np

import dashboard_server
import systematic_search as runner
import systematic_evidence as evidence
from test_systematic_runner import Scenario, read, scratch


def edit_checkpoint(path, transform):
    with np.load(path,allow_pickle=False) as archive:
        fields = {name:archive[name] for name in archive.files}
    transform(fields)
    with path.open('wb') as stream:
        np.savez(stream,**fields)


def legacy(fields):
    for key in ('exact_source_origin','exact_source_binding','exact_exhaustion_record'):
        fields.pop(key)


class ClassicEvidenceTests(unittest.TestCase):
    def run_without_backend(self,scenario,target):
        args=['systematic_search.py','--data',str(scenario.source),'--output',str(scenario.output),
              '--target',target,'--backend','auto','--resume']
        with patch.object(sys,'argv',args), patch.object(runner,'choose_backend',side_effect=AssertionError('GPU resolution')), \
                patch.object(runner,'DFS',side_effect=AssertionError('GPU allocation')), patch.object(runner.signal,'signal'):
            runner.main()

    def test_terminal_replay_recovers_lost_marker_and_rejects_rehashed_omission(self):
        with scratch() as (root, source):
            scenario = Scenario(root/'run',source)
            scenario.execute(replicas=40)
            marker = scenario.output/'systematic'/'exhaustion.json'
            checkpoint = marker.with_name('checkpoint.npz')
            origin = read(marker)
            before = checkpoint.read_bytes()
            self.assertIn('r32_gold_bonds.py', origin['source_sha256'])
            self.assertIn('r34_coverage_index.py', origin['source_sha256'])
            self.assertIn('dfs_kernels.wgsl', origin['source_sha256'])
            marker.unlink()
            constructions = scenario.constructions
            self.run_without_backend(scenario,'edges')
            for _ in range(2):
                scenario.execute(resume=True,replicas=20)
                self.assertTrue(read(scenario.output/'status.json')['terminal_reopened_on_host'])
                self.assertEqual(scenario.constructions,constructions)
                self.assertEqual(read(marker),origin)
                self.assertEqual(scenario.steps_this_run,0)
                self.assertEqual(checkpoint.read_bytes(),before)
            del origin['source_sha256']['r32_gold_bonds.py']
            origin['sha256'] = evidence.checksum({k:v for k,v in origin.items() if k!='sha256'})
            marker.write_text(json.dumps(origin))
            with self.assertRaisesRegex(SystemExit,'exhaustion record'):
                scenario.execute(resume=True)
            self.assertEqual(scenario.steps_this_run,0)
            self.assertEqual(checkpoint.read_bytes(),before)

    def test_dashboard_without_gpu_package_selects_terminal_verification_only(self):
        from test_runtime_capabilities import capabilities
        with scratch() as (root,source):
            scenario=Scenario(root/'run',source)
            scenario.execute(replicas=40,target='gold')
            with patch.object(dashboard_server,'RUNTIME',scenario.output), patch.object(dashboard_server,'CHILD'), \
                    patch.object(dashboard_server,'LAUNCH_CONFIG'),patch.object(dashboard_server,'live_pid',return_value=None), \
                    patch.object(dashboard_server,'runtime_capabilities',return_value=capabilities()), \
                    patch.object(dashboard_server.subprocess,'Popen',return_value=SimpleNamespace(pid=1234)) as spawn:
                evidence._terminal_header.cache_clear()
                with patch.object(np,'load',wraps=np.load) as load:
                    for _ in range(3):
                        self.assertEqual(dashboard_server.state_payload()['systematic_terminal_profiles'],[()])
                    self.assertEqual(load.call_count,1)
                dashboard_server.start_solver({'algorithm':'gpu-dfs','backend':'auto','replicas':128})
                command=spawn.call_args.args[0]
                self.assertIn('--host-terminal-only',command)
                with patch.object(sys,'argv',command[1:]+['--data',str(scenario.source),'--output',str(scenario.output)]), \
                        patch.object(runner,'choose_backend',side_effect=AssertionError('GPU resolution')),patch.object(runner.signal,'signal'):
                    runner.main()
                self.assertTrue(read(scenario.output/'status.json')['terminal_reopened_on_host'])
                marker=scenario.output/'systematic'/'checkpoint.npz'
                marker.unlink()
                with self.assertRaises(ValueError):
                    dashboard_server.start_solver({'algorithm':'gpu-dfs','backend':'auto','replicas':128})
                self.assertEqual(spawn.call_count,1)

    @unittest.skipUnless(shutil.which('node'),'Node.js runs the isolated dashboard controls')
    def test_browser_click_without_gpu_submits_only_the_matching_verified_profile(self):
        script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements=new Map();
function el(id){if(!elements.has(id))elements.set(id,{value:'128',options:[],dataset:{},style:{},classList:{add(){},remove(){},toggle(){}},setAttribute(){},listeners:{},addEventListener(k,f){this.listeners[k]=f},clientWidth:500,checked:false,innerHTML:'',textContent:''});return elements.get(id)}
const storage={getItem(){return null},setItem(){}}, requests=[];
const context=vm.createContext({document:{getElementById:el,addEventListener(){}},window:{addEventListener(){},localStorage:storage},console,Date,Number,String,Math,Set,JSON,setTimeout(){},setInterval(){},clearTimeout(){}});
context.fetch=async(url,args)=>{if(args?.method==='POST')requests.push(JSON.parse(args.body));throw Error('captured')};
const source=fs.readFileSync(process.argv[1],'utf8').split('<script>')[1].split('</script>')[0].replace('poll();setInterval(poll,1000);','');
vm.runInContext(source,context);
const evaljs=s=>vm.runInContext(s,context);
el('algorithmInput').options=[{value:'gpu-dfs'}];el('backendInput').options=[{value:'auto'},{value:'cuda'}];
el('algorithmInput').value='gpu-dfs';el('backendInput').value='auto';el('seconds').value='0';el('seedInput').value='1';el('replicaInput').value='128';
evaljs("requestedAlgorithm='gpu-dfs';state={process_running:false,control_token:'test',status:{method:'systematic',state:'unverified_exhaustion',coverage_note:'Unverified source'},history:[],capabilities:{python:'3.12',algorithms:{'gpu-dfs':{available:false,reason:'Missing GPU'}},backends:{auto:{available:false,reason:'Missing GPU'},cuda:{available:false,reason:'Missing GPU'}},options:{}},systematic_terminal_profiles:[]};renderState()");
assert.equal(el('startBtn').disabled,true);assert.equal(el('algorithmInput').options[0].disabled,true);
assert.match(el('runStatus').textContent,/unverified/i);assert.match(el('controlHint').textContent,/Unverified source/);
evaljs("state.systematic_terminal_profiles=[['hall']];renderState()");
assert.equal(el('algorithmInput').options[0].disabled,false);assert.equal(el('startBtn').disabled,true);
el('gpu_hall').checked=true;el('gpu_hall').listeners.change();
assert.equal(el('startBtn').disabled,false);assert.equal(el('backendInput').options[0].disabled,false);
assert.equal(el('startBtn').textContent,'Verify saved exhaustion');
(async()=>{await el('startBtn').onclick();assert.equal(requests.length,1);assert.equal(requests[0].verify_terminal,true);assert.equal(requests[0].gpu_features.hall,true);
evaljs("state.process_running=true;renderState()");assert.equal(el('startBtn').disabled,true);assert.equal(el('startBtn').textContent,'Start search');})().catch(e=>{console.error(e);process.exitCode=1});
"""
        completed=subprocess.run([shutil.which('node'),'-e',script,str(Path(__file__).with_name('Dashboard.html'))],
                                 text=True,capture_output=True)
        self.assertEqual(completed.returncode,0,completed.stderr)

    def test_changed_origin_source_while_running_preserves_work_but_cannot_claim_proof(self):
        with scratch() as (root,source):
            scenario = Scenario(root/'run',source)
            scenario.stop_after=2
            scenario.execute(replicas=40)
            status=read(scenario.output/'status.json')
            origin=scenario.output/'runs'/status['run_id']/'r32_gold_bonds.py'
            origin.write_bytes(origin.read_bytes()+b'\n# changed\n')
            scenario.stop_after=None
            scenario.execute(resume=True,replicas=40)
            self.assertEqual(len(scenario.events),len(set(scenario.events)))
            self.assertFalse((scenario.output/'systematic'/'exhaustion.json').exists())
            for _ in range(2):
                current=read(scenario.output/'status.json')
                self.assertEqual(current['state'],'unverified_exhaustion')
                self.assertEqual(current['exhausted_jobs'],480)
                self.assertFalse(current['verified_coverage_origin'])
                self.assertIn('unverified',current['coverage_note'])
                with patch.object(dashboard_server,'RUNTIME',scenario.output), patch.object(dashboard_server,'live_pid',return_value=None), patch.object(dashboard_server,'runtime_capabilities',return_value={}):
                    self.assertEqual(dashboard_server.state_payload()['status']['state'],'unverified_exhaustion')
                scenario.execute(resume=True)
                self.assertEqual(scenario.steps_this_run,0)

    def test_unbound_legacy_partial_can_still_save_a_validated_gold_board(self):
        from test_gpu_gold import gold_fixture
        with scratch() as (root,source):
            data,_=gold_fixture()
            for i,tile in enumerate(data['tiles']): tile.update(number=i+1, group='silver')
            source.write_text(json.dumps(data))
            scenario=Scenario(root/'run',source)
            scenario.stop_after=1
            scenario.execute(replicas=1,target='gold')
            checkpoint=scenario.output/'systematic'/'checkpoint.npz'
            edit_checkpoint(checkpoint,legacy)
            scenario.stop_after=None
            scenario.mode='pending'
            scenario.execute(resume=True,replicas=1,target='gold')
            self.assertEqual(read(scenario.output/'status.json')['state'],'solved')
            self.assertFalse(read(scenario.output/'status.json')['verified_coverage_origin'])
            self.assertTrue(read(scenario.output/'solution.json')['validation']['valid'])
            self.run_without_backend(scenario,'gold')
            self.assertEqual(read(scenario.output/'status.json')['state'],'solved')

    def test_missing_legacy_terminal_origin_stays_unverified_and_preserves_checkpoint(self):
        with scratch() as (root,source):
            scenario=Scenario(root/'run',source)
            scenario.execute(replicas=40)
            marker=scenario.output/'systematic'/'exhaustion.json'
            checkpoint=marker.with_name('checkpoint.npz')
            edit_checkpoint(checkpoint,legacy)
            marker.unlink()
            before=checkpoint.read_bytes()
            scenario.execute(resume=True)
            self.assertEqual(read(scenario.output/'status.json')['state'],'unverified_exhaustion')
            self.assertFalse(marker.exists())
            self.assertEqual(before,checkpoint.read_bytes())
            self.assertEqual(scenario.steps_this_run,0)

    def test_prior_partial_replays_after_marker_was_written_but_checkpoint_was_not(self):
        with scratch() as (root,source):
            scenario = Scenario(root/'run',source)
            scenario.stop_after = 2
            scenario.execute(replicas=40)
            folder=scenario.output/'systematic'
            partial = (folder/'checkpoint.npz').read_bytes()
            scenario.stop_after = None
            scenario.execute(resume=True,replicas=40)
            first=read(folder/'exhaustion.json')
            (folder/'checkpoint.npz').write_bytes(partial)
            scenario.execute(resume=True,replicas=40)
            self.assertGreater(scenario.steps_this_run,0)
            self.assertEqual(read(scenario.output/'status.json')['state'],'exhausted')
            self.assertNotEqual(read(folder/'exhaustion.json')['run_id'],first['run_id'])

    def test_missing_native_receipt_does_not_reset_restored_progress(self):
        with scratch() as (root,source):
            scenario=Scenario(root/'run',source)
            scenario.stop_after=2
            scenario.execute(replicas=40)
            status=read(scenario.output/'status.json')
            config_path=scenario.output/'runs'/status['run_id']
            (config_path/f"checkpoint-{status['checkpoint']['completed_generation']}.json").unlink()
            scenario.stop_after=None
            scenario.execute(resume=True,replicas=40)
            self.assertEqual(len(scenario.events),len(set(scenario.events)))
            self.assertEqual(read(scenario.output/'status.json')['state'],'unverified_exhaustion')
            self.assertFalse((scenario.output/'systematic'/'exhaustion.json').exists())

    def test_lost_terminal_flag_cannot_bypass_origin_validation(self):
        with scratch() as (root,source):
            scenario=Scenario(root/'run',source)
            scenario.execute(replicas=40)
            folder=scenario.output/'systematic'
            edit_checkpoint(folder/'checkpoint.npz',lambda fields: fields.update(exact_terminal=np.array('')))
            origin=read(folder/'exhaustion.json')
            (scenario.output/'runs'/origin['run_id']/'r32_gold_bonds.py').unlink()
            with self.assertRaisesRegex(SystemExit,'source evidence failed'):
                scenario.execute(resume=True)
            self.assertEqual(scenario.steps_this_run,0)


if __name__=='__main__':
    unittest.main()
