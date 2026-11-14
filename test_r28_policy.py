from pathlib import Path
import sqlite3
from contextlib import closing
import tempfile
import unittest
import numpy as np

from bench_r28_policy import symbolic
from r27_incumbent import BoardModel
from r28_cache_policy import offline_bound, replay
from r28_patch_cache import exterior, transplant
from r28_sampling import candidate, discrepancy, jump, JumpChain
from r28_swap_walk import PairScores, walk
from search_fixtures import planted_gold
from search_problem import SearchProblem
from test_constraint_dfs import double_face, simple_tiles


class PolicyTests(unittest.TestCase):
    def test_every_oriented_swap_on_multiseam_and_planted_boards(self):
        data, geo, sol = planted_gold(17,1,3)
        for problem, codes in [(SearchProblem(simple_tiles(),double_face()),[2,4]),(SearchProblem(data,geo),sol)]:
            problem.domains[0]=[codes[0],codes[1]]
            model=BoardModel(problem)
            scores=PairScores(model)
            board=model.legal_board(codes)
            grid=scores.all(board)
            for index, value in enumerate(grid.ravel()):
                child=candidate(scores,board,index)
                legal=model.legal[np.arange(model.n),child].all()
                expected=problem.validate(child.tolist())['matched_edges'] if legal else -1
                self.assertEqual(value,expected)

    def test_walk_restart_and_scope(self):
        data, geo, sol = planted_gold(17,1,3)
        model=BoardModel(SearchProblem(data,geo))
        board=sol.copy()
        board[:4]=reversed(board[:4])
        with tempfile.TemporaryDirectory() as folder:
            db=Path(folder)/'walk.sqlite'
            out=Path(folder)/'best.json'
            args=dict(goal=len(model.a),floor=0,width=3,limit=1)
            first=walk(model,[board],db,out,**args)
            second=walk(model,[board],db,out,**args)
            self.assertEqual((first['expanded_total'],second['expanded_total']),(1,2))
            with self.assertRaisesRegex(ValueError,'policy'):
                walk(model,[board],db,out,**dict(args,width=2))
            with closing(sqlite3.connect(db)) as cx:
                self.assertEqual(cx.execute('SELECT count(*) FROM centers WHERE done=1').fetchone()[0],2)

    def test_cache_upperbound_and_spill(self):
        trace=[(key,2,cost) for key,cost in [(b'a',4),(b'b',2),(b'c',3),(b'a',4),(b'b',2),(b'a',4)]]
        for weighted in (False,True):
            self.assertLessEqual(replay(trace,4,weighted=weighted)['saved_work'],offline_bound(trace,4))
            self.assertEqual(replay(trace,2,weighted=weighted,spill=':memory:')['saved_work'],10)

    def test_zero_temperature_never_degrades_and_clock_is_bounded(self):
        data,geo,codes=planted_gold(17,1,3)
        model=BoardModel(SearchProblem(data,geo))
        scores=PairScores(model)
        board=np.asarray(codes)
        for _ in range(5):
            candidate_board,elapsed,accepted=jump(scores,board,np.random.default_rng(_),0,10)
            self.assertTrue(1<=elapsed<=10)
            self.assertEqual(model.matches(candidate_board).sum(),len(model.a))
        self.assertEqual(discrepancy(0,1),6)
        self.assertEqual(discrepancy(1,4),2)
        self.assertEqual(discrepancy(1,1),0)

    def test_incremental_jump_spectrum_matches_fresh_rebuild(self):
        data,geo,codes=planted_gold(7,1,3)
        m=BoardModel(SearchProblem(data,geo))
        rng=np.random.default_rng(17)
        chain=JumpChain(m,codes,2.)
        for _ in range(4):
            board=chain.advance(rng,100)
            rebuilt=chain.pairs.all(board)
            np.testing.assert_array_equal(chain.deltas,np.where(rebuilt>=0,rebuilt-chain.score,-1000))
        self.assertGreater(chain.accepted,0)

    def test_symbolic_sifts_preserve_live_ownership(self):
        results=symbolic()
        self.assertLess(results[-1]['nodes_after'],results[0]['nodes_before'])

    def test_equivalent_gold_witness_and_boundary_adverse(self):
        # Far-away changes may still alter exactly which boundary endpoints
        # connect. Check all local outcomes on two independently scored exteriors.
        data,geo,sol=planted_gold(29,1,2)
        p=SearchProblem(data,geo)
        a=sol.copy()
        for c in (4,5,6):
            a[c]=a[c]//3*3+(a[c]+1)%3
        patch=[0]
        matches=0
        adverse=0
        for c in range(1,p.n):
            b=a.copy()
            b[c]=b[c]//3*3+(b[c]+1)%3
            old=max((a[:0]+[v]+a[1:] for v in range(3*(a[0]//3),3*(a[0]//3)+3) if v!=a[0]),key=lambda x:p.validate(x)['matched_edges'])
            r=transplant(p,{'board':b,'patch':patch},[dict(board=a,patch=patch,candidate=old,status='OPTIMAL')],target='gold',loop_aware=True)
            if r:
                high=max(p.validate([v]+b[1:])['matched_edges'] for v in range(3*(b[0]//3),3*(b[0]//3)+3) if v!=b[0])
                self.assertEqual(p.validate(r['candidate'])['matched_edges'],high)
                matches+=1
            else:
                adverse+=1
        self.assertGreater(matches,0)
        self.assertGreater(adverse,0)
        self.assertNotEqual(exterior(p,a,[0]),exterior(p,a,[1]))
