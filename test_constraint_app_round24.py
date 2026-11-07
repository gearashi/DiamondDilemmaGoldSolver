"""Application ownership/routing tests for portable R24 work."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import constraint_search as runner
from constraint_work import WorkScope,frontier_snapshot
from test_constraint_search import pair_problem

class ApplicationRound24Tests(unittest.TestCase):
    def test_imported_empty_work_never_claims_full_puzzle_exhausted(self):
        data,problem=pair_problem()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);run=root/'runtime';source=root/'tiles.json';work=root/'work.json'
            source.write_text(json.dumps(data));work.write_text(json.dumps(frontier_snapshot(WorkScope.for_problem(problem),[])))
            args=['--algorithm','parallel-dfs','--seconds','5','--data',str(source),'--output',str(run),'--work-frontier',str(work)]
            result={'status':'infeasible','codes':None,'nodes':0,'stats':{},'complete':True}
            with patch.object(runner,'SearchProblem',return_value=problem),patch.object(runner,'validate_runtime_selection'),patch.object(runner,'run_solver',return_value=result):runner.main(args)
            state=json.loads((run/'status.json').read_text());self.assertFalse(state['full_puzzle_exhausted']);self.assertEqual(state['work_scope']['kind'],'imported_frontier');self.assertIn('unproven',state['phase_message'])
            with patch.object(runner,'SearchProblem',return_value=problem),patch.object(runner,'validate_runtime_selection'),patch.object(runner,'run_solver') as execute:
                runner.main(args[:-2]+['--resume']);execute.assert_not_called()
            state=json.loads((run/'status.json').read_text());self.assertFalse(state['full_puzzle_exhausted']);self.assertIn('unproven',state['phase_message'])
            self.assertFalse((run/'run.lock').exists())
            with patch.object(runner,'SearchProblem',return_value=problem),patch.object(runner,'validate_runtime_selection'),self.assertRaisesRegex(SystemExit,'Completed work already exists'):
                runner.main(args+['--resume'])
            self.assertFalse((run/'run.lock').exists())
    def test_invalid_plain_sat_export_does_not_create_run_lock(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch.object(runner,'validate_runtime_selection'),self.assertRaises(SystemExit):
                runner.main(['--algorithm','sat','--output',folder,'--search-options','{"portable_frontier":true}'])
            self.assertFalse((Path(folder)/'run.lock').exists())
    def test_shared_coverage_disabled_still_resumes_parallel_dfs_checkpoint(self):
        _,problem=pair_problem();checkpoint={'parallel_version':1}
        with patch('constraint_parallel.solve',return_value={'status':'timeout'}) as execute:
            result=runner.run_solver(problem,'dfs',seconds=0,resume=checkpoint,options={'shared_coverage':False},workers=4)
        self.assertEqual(execute.call_args.kwargs['workers'],1);self.assertEqual(result['status'],'timeout')

if __name__=='__main__':unittest.main()
