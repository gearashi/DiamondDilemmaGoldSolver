"""Cross-threshold graph, legacy recovery, move-set proofs and BFS differentials."""
from contextlib import closing
import importlib.util
import itertools
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import domain_binding, run_shell
from r28_exchange_plateau import walk
from r29_move_cores import decide, learn, load_cores, permitted_sets
from r29_plateau import identity
from search_fixtures import planted_gold
from search_problem import SearchProblem


class ExactRoutes(unittest.TestCase):
    def setUp(self):
        data, board, codes = planted_gold(19,1,3)
        self.problem = SearchProblem(data,board,{i:codes[i] for i in range(4,len(codes))})
        self.model = BoardModel(self.problem)
        self.board = list(codes)
        self.start = self.board.copy()
        self.start[0], self.start[1] = self.start[1], self.start[0]
        self.start[2], self.start[3] = self.start[3], self.start[2]
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = Path(self.tmp.name)/"walk.sqlite"

    def exhaustive(self):
        best = {}
        for perm in itertools.permutations([code//3 for code in self.start[:4]]):
            scores = []
            for rot in itertools.product(range(3),repeat=4):
                codes = [3*t+r for t,r in zip(perm,rot)] + self.start[4:]
                scores.append(int(self.model.matches(np.array(codes)).sum()))
            best[perm] = max(scores)
        source = tuple(code//3 for code in self.start[:4])
        for level in range(best[source],-1,-1):
            opened = {source}
            seen = set()
            depth = 0
            while opened:
                if any(best[p]==len(self.problem.edges) for p in opened):
                    return level, depth
                seen.update(opened)
                following = set()
                for p in opened:
                    for a,b in itertools.combinations(range(4),2):
                        q = list(p)
                        q[a],q[b] = q[b],q[a]
                        if tuple(q) not in seen and best[tuple(q)] >= level:
                            following.add(tuple(q))
                opened = following
                depth += 1

    def test_widest_and_shortest_with_cross_program_reuse(self):
        level, depth = self.exhaustive()
        edges = len(self.problem.edges)
        run_shell(self.problem,self.start,2,0,self.db)
        options = dict(max_centers=100, widest=True,tie_limit=1)
        info = walk(self.problem,self.start,self.db,0,edges,**options)
        self.assertEqual((info['widest_floor'],info['shortest_at_floor']),(level,depth))
        same = walk(self.problem,self.start,self.db,0,edges,**options)
        self.assertEqual((same['solved'],same['star_evaluations'],same['centers']),(0,0,0))
        board = self.start.copy()
        for a,b in info['swaps']:
            board[a],board[b] = board[b],board[a]
        self.assertEqual(identity(board),identity(info['codes']))
        self.assertEqual(self.problem.validate(info['codes'])['matched_edges'],edges)
        self.assertGreater(info['reused'],0)

    def test_lower_floor_and_legacy_score_with_null_witness(self):
        key = domain_binding(self.model)
        with closing(sqlite3.connect(self.db)) as db, db:
            db.execute("CREATE TABLE optimum(scope TEXT,perm TEXT,score INT,comps INT,codes TEXT)")
            db.execute("INSERT INTO optimum VALUES (?,?,?,?,?)",(key+'/15/16/8',identity(self.start),
                       max(int(self.model.matches(np.array([3*(self.start[i]//3)+r for i,r in enumerate(rot)]+self.start[4:])).sum())
                           for rot in itertools.product(range(3),repeat=4)), None,'null'))
        args = dict(max_centers=1,tie_limit=1)
        first = walk(self.problem,self.start,self.db,2,15,**args)
        second = walk(self.problem,self.start,self.db,1,15,**args)
        self.assertEqual(first['star_evaluations'],6)
        self.assertEqual(second['cached_stars'],6)
        self.assertEqual(second['star_evaluations'],0)

    @unittest.skipUnless(importlib.util.find_spec("ortools"),"optional OR-Tools")
    def test_movement_cores_and_real_shell_filter(self):
        target = len(self.problem.edges)
        self.assertEqual(decide(self.model,self.start,set(range(self.model.n)),target,15),"INFEASIBLE")
        self.assertIn(decide(self.model,self.board,[],target,15),("OPTIMAL","FEASIBLE"))
        result = learn(self.model,self.start,self.db,target,list(range(self.model.n)),seconds=5)
        self.assertTrue(result['stored'])
        with closing(sqlite3.connect(self.db)) as db, db:
            cores = load_cores(db,self.model,self.start,target)
            self.assertEqual(len(cores),1)
            changed = np.array(list(itertools.combinations(range(4),2)))
            self.assertTrue(np.any(~permitted_sets(changed,cores)))
            self.assertEqual(load_cores(db,self.model,self.start,target-1),[])
            for perm in itertools.permutations(range(4)):
                valid = self.start.copy()
                valid[:4] = [self.start[i] for i in perm]
                active = {i for i,j in enumerate(perm) if i!=j}
                if not active.intersection(cores[0]):
                    self.assertEqual(decide(self.model,valid,set(range(self.model.n)),target,5),"INFEASIBLE")
        first = run_shell(self.problem,self.start,2,target,self.db)
        control = run_shell(self.problem,self.start,2,target,Path(self.tmp.name)/'cold.sqlite')
        self.assertEqual((first['hits'],first['tested']),(control['hits'],control['tested']))
        self.assertGreater(first['core_skips'],0)
