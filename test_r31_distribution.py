import json
from itertools import product
from pathlib import Path
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor

from r27_incumbent import BoardModel
from r31_defect_bands import bands, inventory_lower, solve
from r31_sampling import _claim, _schema, ranked_start, run_seeds
from search_fixtures import planted_gold
from search_problem import SearchProblem
from test_dependencies import requires_sat


class IndependentJobs(unittest.TestCase):
    def test_distinct_ranks_and_legal_fallback(self):
        data, mesh, reference = planted_gold(37, 1, 3)
        m = BoardModel(SearchProblem(data, mesh))
        self.assertEqual(len({tuple(m.legal_board(ranked_start(m, i))) for i in range(250)}), 250)
        pinned = BoardModel(SearchProblem(data, mesh, {0:reference[0]}))
        self.assertEqual(ranked_start(pinned, 5)[0], reference[0])
        self.assertEqual(ranked_start(m, 7).tolist(), ranked_start(m, 7).tolist())

    def test_concurrent_and_restart(self):
        barrier = threading.Barrier(2)
        calls = []
        def worker(seed):
            calls.append(seed)
            barrier.wait(timeout=10)
            return dict(gold=False, score=seed, components=2, codes=[seed])
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp)/'jobs.sqlite'
            with ThreadPoolExecutor(2) as pool:
                futures=[pool.submit(run_seeds,db,'scope',0,2,worker) for _ in range(2)]
                results=[f.result() for f in futures]
            self.assertEqual(sorted(calls), [0,1])
            self.assertEqual(sum(r['completed'] for r in results), 2)
            self.assertTrue(all(r['best']['seed']==1 for r in results))
            def interrupted(seed, saved, save):
                self.assertIsNone(saved)
                save(dict(round=9))
                raise RuntimeError('power failure')
            with self.assertRaisesRegex(RuntimeError,'power failure'):
                run_seeds(db,'second',5,1,interrupted,checkpointed=True)
            def restored(seed, saved, save):
                self.assertEqual(saved['round'],9)
                return dict(gold=True,score=10,components=1,codes=[1])
            resume=run_seeds(db,'second',5,1,restored,checkpointed=True)
            self.assertTrue(resume['best']['gold'])
            self.assertEqual(run_seeds(db,'second',6,0,None)['completed'],0)
            output=Path(temp)/'best.json'
            def published(item):
                self.assertTrue(output.exists())
                self.assertGreaterEqual(json.loads(output.read_text())['matched_edges'],item['score'])
            run_seeds(db,'export',0,4,lambda seed:dict(score=4-seed,gold=False,components=1,codes=[seed]),
                      output=output,progress=published)
            self.assertEqual(json.loads(output.read_text())['matched_edges'],4)

    def test_only_one_worker_computes_a_shared_sampler_prefix(self):
        from contextlib import closing
        from r31_sampling import _open, _prefix_claim
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'streams.sqlite'
            with closing(_open(path)) as conn:
                _schema(conn)
            start=threading.Barrier(2)
            def compute():
                with closing(_open(path)) as conn:
                    start.wait(timeout=5)
                    with _prefix_claim(conn,path,'gpu',0,64):
                        if conn.execute("SELECT count(*) FROM r31_sampler").fetchone()[0]:
                            return 0
                        with conn:
                            conn.execute("INSERT INTO r31_sampler VALUES ('gpu',0,64,'state')")
                        return 1
            with ThreadPoolExecutor(2) as pool:
                calls=[pool.submit(compute) for _ in range(2)]
                self.assertEqual(sum(f.result() for f in calls),1)

    def test_expired_owner_loses_publication_right(self):
        with sqlite3.connect(':memory:') as conn:
            _schema(conn)
            before=_claim(conn,'a',1,5)
            conn.execute("UPDATE r31_jobs SET expires=0")
            after=_claim(conn,'a',1,5)
            self.assertNotEqual(before,after)
            self.assertEqual(conn.execute("UPDATE r31_jobs SET gold=1 WHERE owner=?",(before,)).rowcount,0)


@requires_sat
class GeographicDefects(unittest.TestCase):
    def test_symbol_and_defect_equivalence(self):
        from pysat.solvers import Glucose4
        from r27_reason_maxsat import SeamFormula
        data,mesh,reference=planted_gold(29,1,1)
        problem=SearchProblem(data,mesh)
        model=BoardModel(problem)
        for codes in (reference, [3*(c//3)+(c+1)%3 for c in reference]):
            with Glucose4() as sat:
                formula=SeamFormula(problem,[[v] for v in codes],sat,lambda:None,exact_defects=True)
                for lit,matched in zip(formula.defects,model.matches(model.legal_board(codes))):
                    self.assertFalse(sat.solve(assumptions=[lit if matched else -lit]))
                self.assertTrue(sat.solve())

    def test_exact_counts_resume_and_bounds(self):
        data,mesh,reference = planted_gold(19,1,3)
        problem=SearchProblem(data,mesh)
        rows=[[c,(c//3)*3+(c+1)%3] for c in reference[:4]] + [[c] for c in reference[4:]]
        rows[0]=[rows[0][1]]
        candidates=[list(p) for p in product(*rows)]
        scores=[problem.validate(p)['matched_edges'] for p in candidates]
        groups=bands(problem,3)
        self.assertEqual(sorted(x for g in groups for x in g),list(range(len(problem.edges))))
        model=BoardModel(problem)
        for g in groups:
            self.assertLessEqual(inventory_lower(problem,rows,g),
                                 min(sum(not flag for flag in model.matches(model.legal_board(p))[g]) for p in candidates))
        with tempfile.TemporaryDirectory() as temp:
            for target in (max(scores)-2, max(scores), max(scores)+1):
                if target > len(problem.edges):
                    continue
                path=Path(temp)/'defects.sqlite'
                initial=solve(problem,path,domains=rows,target=target,width=1,max_calls=0)
                self.assertEqual(initial['status'],'open')
                out=solve(problem,path,domains=rows,target=target,width=1)
                self.assertEqual(out['status'],'witness' if max(scores)>=target else 'infeasible')
                if out['witness']:
                    self.assertEqual(sum(out['witness']['defects']),len(problem.edges)-out['witness']['score'])
                self.assertEqual(solve(problem,path,domains=rows,target=target,width=1)['calls'],0)
                baseline=solve(problem,path,domains=rows,target=target,split=False)
                self.assertEqual(baseline['status'],out['status'])
