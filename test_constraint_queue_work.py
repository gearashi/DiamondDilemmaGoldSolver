import tempfile,unittest
from pathlib import Path
from constraint_parallel import solve
from constraint_dfs import _checkpoint_digest
from constraint_knowledge import KnowledgeStore
from search_problem import SearchProblem
from test_constraint_dfs import multiple_loops,double_face,brute
from test_constraint_parallel import legacy_root,OPTIONS

class QueueWorkTests(unittest.TestCase):
    def fixed(self):
        p=SearchProblem(multiple_loops(),double_face());codes=brute(p,'edges')[0]
        cp=legacy_root(p);cp['stack'][0]['domains']=[hex(1<<c) for c in codes];cp['digest']=_checkpoint_digest(cp)
        return p,codes,cp
    def test_verified_loop_retires_queue_before_worker_start(self):
        p,codes,cp=self.fixed();knowledge=KnowledgeStore(p);knowledge.add(p.gold_nogood(codes))
        ans=solve(p,seconds=5,resume=cp,workers=2,options={**OPTIONS,'queue_pruning':True,'persistent_learning':True},learning_state={'_knowledge':knowledge})
        self.assertEqual(ans['status'],'infeasible',ans)
        self.assertEqual(ans['stats']['parallel_workers_started'],0)
        self.assertEqual(ans['stats']['queue_proof_retirements'],1)
    def test_completed_coverage_survives_policy_change_and_target_scope(self):
        p,codes,cp=self.fixed()
        with tempfile.TemporaryDirectory() as directory:
            state={'_coverage_path':str(Path(directory)/'coverage.sqlite')}
            opts={**OPTIONS,'shared_coverage':True}
            first=solve(p,seconds=10,resume=cp,workers=1,options=opts,learning_state=state)
            self.assertEqual(first['status'],'infeasible',first)
            second=solve(p,seconds=5,resume=cp,workers=2,options={**opts,'deep_paths':True},learning_state=state)
            self.assertEqual(second['status'],'infeasible',second);self.assertEqual(second['stats']['parallel_workers_started'],0)
            edgecp=dict(cp,target='edges');edgecp['digest']=_checkpoint_digest(edgecp)
            edge=solve(p,seconds=10,resume=edgecp,target='edges',workers=1,options=opts,learning_state={'_coverage_path':state['_coverage_path']})
            self.assertEqual(edge['status'],'edge_perfect',edge)

if __name__=='__main__':unittest.main()
