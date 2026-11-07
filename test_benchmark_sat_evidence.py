"""SAT evidence guards, using mocks and optimized-Python refusal only.

No native SAT proof logger or backend capability probe is executed by this
module, including on Windows builds where such probes can crash in native code.
"""
from contextlib import contextmanager,ExitStack,redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import benchmark_sat_capabilities_round25 as capabilities
import benchmark_sat_persistence_round25 as persistence

ROOT=Path(__file__).resolve().parent
MODULES=(capabilities,persistence)
SOURCES={name:{'raw_sha256':str(index)*64,'lf_sha256':str(index)*64}
         for index,name in enumerate(('constraint_proof.py','constraint_proof_cursor.py','constraint_sat_persistence.py'),1)}
RUNTIME={'python':'test','sat_native':{'name':'pysolvers.native','sha256':'a'*64},
         'sat_wrapper':{'name':'pysat/solvers.py','sha256':'b'*64},'packages':{'python-sat':'test'},'optimization_level':0,'assertions_enabled':True}
REPOSITORY={'available':True,'commit':'c'*40,'dirty':False,'status':[]}


def comparison(*,gain=True,exit_code=0):
    rows=[]
    for name in ('pairwise','sequence','finite-puzzle-72'):
        rows.append({'case':name,'produced':{'counters':{}},'restarts':{
            mode:{'native':{'conflicts':5 if gain and mode=='all_base' else 10}}
            for mode in ('baseline','placement','all_base')}})
    return rows,{'exit_code':exit_code,'stdout':'mock test output','stderr':'mock failure' if exit_code else ''}


@contextmanager
def mocked(module,output,*,stage='frozen',sources=None,runtimes=None,repositories=None,answer=None,error=None):
    runner='run_capability_checks' if module is capabilities else 'run_comparisons'
    if answer is None:
        answer=({'proposal':5,'python':'test','platform':'test','python_sat':'test','results':[],
                 'safe_proof_backends':[],'source_sha256':{}} if module is capabilities else comparison())
    with ExitStack() as stack:
        stack.enter_context(patch.object(sys,'argv',[module.__file__,'--output',str(output),'--evidence-stage',stage]))
        source=stack.enter_context(patch.object(module,'source_snapshot',side_effect=sources or [SOURCES,SOURCES]))
        runtime=stack.enter_context(patch.object(module,'runtime_snapshot',side_effect=runtimes or [RUNTIME,RUNTIME]))
        repository=stack.enter_context(patch.object(module,'repository_snapshot',side_effect=repositories or [REPOSITORY,REPOSITORY]))
        run=stack.enter_context(patch.object(module,runner,side_effect=error,return_value=answer))
        capability=stack.enter_context(patch.object(persistence,'require_capability')) if module is persistence else None
        stack.enter_context(patch('importlib.metadata.version',return_value='test'))
        stack.enter_context(patch.object(module.platform,'platform',return_value='test-platform'))
        stack.enter_context(redirect_stdout(io.StringIO()))
        yield SimpleNamespace(run=run,capability=capability,source=source,runtime=runtime,repository=repository)


class SATEvidenceTests(unittest.TestCase):
    def folder(self):return tempfile.TemporaryDirectory(prefix='sat-evidence-guard-',dir=ROOT)

    def test_frozen_dirty_or_unreadable_checkout_never_reaches_native_work(self):
        for module in MODULES:
            for repository in ({},{**REPOSITORY,'dirty':True}):
                with self.subTest(module=module.__name__,repository=repository),self.folder() as folder:
                    output=Path(folder)/'result.json'
                    with mocked(module,output,repositories=[repository]) as calls:
                        with self.assertRaisesRegex(RuntimeError,'clean, readable committed'):module.main()
                    calls.run.assert_not_called()
                    if calls.capability is not None:calls.capability.assert_not_called()
                    self.assertFalse(output.exists())

    def test_existing_or_non_json_output_fails_before_provenance_or_native_work(self):
        for module in MODULES:
            for name in ('existing.json','source.py'):
                with self.subTest(module=module.__name__,name=name),self.folder() as folder:
                    output=Path(folder)/name;output.write_text('preserve source bytes',encoding='utf-8')
                    with mocked(module,output) as calls:
                        with self.assertRaisesRegex(ValueError,'fresh .json'):module.main()
                    calls.run.assert_not_called();calls.source.assert_not_called()
                    if calls.capability is not None:calls.capability.assert_not_called()
                    self.assertEqual(output.read_text(encoding='utf-8'),'preserve source bytes')

    def test_frozen_success_records_full_proof_and_native_runtime_bindings(self):
        for module in MODULES:
            with self.subTest(module=module.__name__),self.folder() as folder:
                output=Path(folder)/'result.json'
                with mocked(module,output) as calls:module.main()
                calls.run.assert_called_once()
                report=json.loads(output.read_text(encoding='utf-8'))
                self.assertEqual(report['evidence_stage'],'frozen')
                self.assertEqual(report['source_before'],SOURCES);self.assertEqual(report['source_after'],SOURCES)
                self.assertEqual(report['runtime_before'],RUNTIME);self.assertEqual(report['runtime_after'],RUNTIME)
                self.assertEqual(report['repository_before'],REPOSITORY);self.assertEqual(report['repository_after'],REPOSITORY)
                self.assertTrue(report['source_unchanged']);self.assertTrue(report['runtime_unchanged'])
                self.assertEqual(report['evidence_status'],'completed_diagnostic' if module is capabilities else 'passed')
                self.assertIn('parent',report['child_evidence_scope'])

    def test_source_runtime_or_git_changes_leave_no_report(self):
        changes=(('source',{'sources':[SOURCES,{**SOURCES,'new-proof.py':{'raw_sha256':'d'*64}}]}),
                 ('runtime',{'runtimes':[RUNTIME,{**RUNTIME,'sat_native':{'sha256':'d'*64}}]}),
                 ('head',{'repositories':[REPOSITORY,{**REPOSITORY,'commit':'d'*40}]}),
                 ('dirty',{'repositories':[REPOSITORY,{**REPOSITORY,'dirty':True,'status':[' M proof.py']}]}))
        for module in MODULES:
            for name,kwargs in changes:
                with self.subTest(module=module.__name__,change=name),self.folder() as folder:
                    output=Path(folder)/'result.json'
                    with mocked(module,output,**kwargs) as calls:
                        with self.assertRaisesRegex(RuntimeError,'changed'):module.main()
                    calls.run.assert_called_once();self.assertFalse(output.exists())

    def test_development_allows_dirty_sources_but_rejects_head_change(self):
        dirty={**REPOSITORY,'dirty':True,'status':[' M development.py']}
        for module in MODULES:
            for changed in (False,True):
                with self.subTest(module=module.__name__,changed=changed),self.folder() as folder:
                    output=Path(folder)/'result.json';after={**dirty,'commit':'e'*40} if changed else dirty
                    with mocked(module,output,stage='development',repositories=[dirty,after]):
                        if changed:
                            with self.assertRaisesRegex(RuntimeError,'HEAD changed'):module.main()
                        else:module.main()
                    self.assertEqual(output.exists(),not changed)
                    if not changed:self.assertEqual(json.loads(output.read_text(encoding='utf-8'))['evidence_stage'],'development')

    def test_comparison_or_unit_failure_cannot_leave_a_success_report(self):
        for description,answer,error in (
                ('tests',comparison(exit_code=1),None),
                ('no gain',comparison(gain=False),None),
                ('finite preservation',None,RuntimeError('Fresh-process finite solution set changed'))):
            with self.subTest(description=description),self.folder() as folder:
                output=Path(folder)/'result.json'
                with mocked(persistence,output,answer=answer,error=error):
                    with self.assertRaises(RuntimeError):persistence.main()
                self.assertFalse(output.exists())
        with self.folder() as folder:
            output=Path(folder)/'result.json'
            with mocked(capabilities,output,error=RuntimeError('probe orchestration failed')):
                with self.assertRaisesRegex(RuntimeError,'orchestration'):capabilities.main()
            self.assertFalse(output.exists())

    def test_late_output_collision_is_never_overwritten(self):
        for module in MODULES:
            with self.subTest(module=module.__name__),self.folder() as folder:
                output=Path(folder)/'result.json'
                def collision():
                    output.write_text('another owner',encoding='utf-8')
                    return ({'python':'test','platform':'test','python_sat':'test','safe_proof_backends':[]} if module is capabilities else comparison())
                with mocked(module,output) as calls:
                    calls.run.side_effect=collision
                    with self.assertRaises(FileExistsError):module.main()
                self.assertEqual(output.read_text(encoding='utf-8'),'another owner')

    def test_persistence_child_is_guarded_without_independent_frozen_snapshot(self):
        payload={'mode':'mock-only'}
        with patch.object(sys,'argv',[persistence.__file__,'--child','--evidence-stage','frozen']),patch.object(sys,'stdin',io.StringIO(json.dumps(payload))),patch.object(persistence,'require_capability') as capability,patch.object(persistence,'run_child') as run,patch.object(persistence,'source_snapshot') as sources,patch.object(persistence,'runtime_snapshot') as runtime:
            persistence.main()
        capability.assert_called_once();run.assert_called_once_with(payload);sources.assert_not_called();runtime.assert_not_called()

    def test_mocked_native_crash_or_missing_cleanup_is_not_accepted(self):
        printed=json.dumps({'status':'solved_before_cleanup','answer':False,'proof':['0']})
        with patch.object(capabilities.subprocess,'run',return_value=SimpleNamespace(returncode=17,stdout=printed,stderr='native cleanup failed')):
            result=capabilities.probe('mock',True)
        self.assertFalse(result['safe_completed']);self.assertNotIn('independently_checked',result)
        for completed in (SimpleNamespace(returncode=17,stdout='{}\ncleanup_returned\n',stderr='failed'),
                          SimpleNamespace(returncode=0,stdout='{}\n',stderr='')):
            with patch.object(persistence.subprocess,'run',return_value=completed):
                with self.assertRaises(RuntimeError):persistence.child({'mode':'mock'})
        with patch.object(persistence.subprocess,'run',return_value=SimpleNamespace(returncode=0,stdout='{"mock":true}\ncleanup_returned\n',stderr='')) as called:
            self.assertEqual(persistence.child({'mode':'mock'}),{'mock':True})
        self.assertEqual(called.call_args.kwargs['cwd'],ROOT)

    def test_optimized_parent_and_child_refuse_before_native_imports(self):
        commands=([sys.executable,'-B','-O',str(ROOT/'benchmark_sat_capabilities_round25.py')],
                  [sys.executable,'-B','-O',str(ROOT/'benchmark_sat_persistence_round25.py')],
                  [sys.executable,'-B','-O',str(ROOT/'benchmark_sat_persistence_round25.py'),'--child'])
        for command in commands:
            with self.folder() as folder:
                output=Path(folder)/'result.json'
                result=subprocess.run([*command,'--output',str(output)],input='{}',cwd=ROOT,capture_output=True,text=True,timeout=20)
                self.assertNotEqual(result.returncode,0);self.assertIn('require assertions',result.stderr)
                self.assertFalse(output.exists())
        result=subprocess.run([sys.executable,'-B','-O','-c',capabilities.CHILD,'mock','1'],cwd=ROOT,capture_output=True,text=True,timeout=20)
        self.assertNotEqual(result.returncode,0);self.assertIn('require assertions',result.stderr)
        self.assertNotIn('solved_before_cleanup',result.stdout)


if __name__=='__main__':unittest.main()
