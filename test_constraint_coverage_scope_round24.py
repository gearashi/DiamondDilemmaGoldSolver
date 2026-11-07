"""Regression: imported restricted DFS work cannot close its broader puzzle."""
import tempfile
import unittest
from pathlib import Path
from constraint_dfs import _solve_core,_checkpoint_digest,solve as dfs
from constraint_parallel import solve as parallel,_root_copy
from constraint_work import CoverageLedger,WorkScope,checkpoint_boxes,subtract_many
from test_constraint_separator import ring


class ImportedCoverageScopeTests(unittest.TestCase):
    def setup_child(self,partly_consumed=False):
        problem=ring(4);options={'shared_coverage':True,'queue_pruning':True}
        base=_solve_core(problem,seconds=0,options=options)['checkpoint']
        child=_root_copy(base,[1<<0,1<<0,(1<<6)|(1<<9),(1<<6)|(1<<9)] if partly_consumed else [1<<0,1<<0,1<<6,1<<9])
        if partly_consumed:
            # The already consumed choice is independently impossible because
            # cells0 and1 use the same physical piece, regardless of later cells.
            child['stack'][0].update(cell=2,choices=[6,9],next=1);child['nodes']=1
            child['digest']=_checkpoint_digest(child)
        dfs(problem,seconds=0,resume=child,options=options) # actual structure validation
        self.assertTrue(problem.validate([0,3,6,9])['valid'])
        return problem,options,child
    def test_infeasible_child_never_excludes_outside_gold_solution(self):
        for partly_consumed in (False,True):
            problem,options,child=self.setup_child(partly_consumed)
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'coverage.sqlite3'
                result=parallel(problem,seconds=5,resume=child,workers=1,options=options,learning_state={'_coverage_path':str(path)})
                self.assertEqual(result['status'],'infeasible')
                with CoverageLedger(path,WorkScope.for_problem(problem)) as ledger:rows=ledger.exclusions()
                self.assertTrue(rows)
                for row in rows:
                    self.assertFalse(all(mask>>code&1 for mask,code in zip(row,[0,3,6,9])))
                    self.assertTrue(all(not mask&~int(allowed,16) for mask,allowed in zip(row,child['stack'][0]['domains'])))
    def test_stopped_or_zero_budget_import_closes_nothing(self):
        for stopped in (False,True):
            problem,options,child=self.setup_child()
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'coverage.sqlite3'
                result=parallel(problem,seconds=5 if stopped else 0,resume=child,workers=1,
                    stop=lambda:stopped,options=options,learning_state={'_coverage_path':str(path)})
                self.assertIn(result['status'],('stopped','timeout'))
                with CoverageLedger(path,WorkScope.for_problem(problem)) as ledger:self.assertEqual(ledger.exclusions(),())
    def test_capped_or_cancelled_subtraction_keeps_original_pending_cover(self):
        original=((15,15,15),)
        self.assertEqual(subtract_many(original,((1,1,1),),max_boxes=1),(original,False))
        def stop():raise InterruptedError
        with self.assertRaises(InterruptedError):subtract_many(original,((1,1,1),),check=stop)
        self.assertEqual(original,((15,15,15),))

if __name__=='__main__':unittest.main()
