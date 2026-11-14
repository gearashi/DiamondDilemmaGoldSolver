"""Round 29 binary, durable regional and partitioned resolution regressions."""
from collections import Counter
from itertools import product
from pathlib import Path
from random import Random
import sqlite3
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from r27_structure_fixtures import planted
from r28_geo_regions import paged_trie
from r28_logic_tail import binary_search, cut_bound
from r29_logic_binary import Implications, priced_cut
from r29_logic_proof import proof_interpolants
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles
from test_constraint_models import pair_problem
from test_dependencies import requires_ortools, requires_sat


class RegionalTests(unittest.TestCase):
    @requires_ortools
    def test_priced_and_split_bounds_against_oracle(self):
        rng = Random(26)
        p = SearchProblem(*tetrahedron_tiles())
        gains = Counter()
        for _ in range(125):
            rows = [rng.sample(r,rng.choice([1,2,2,2])) for r in p.domains]
            valid = {b:p.validate(list(b))['matched_edges'] for b in product(*rows)
                     if len({x//3 for x in b})==p.n}
            old = cut_bound(p,rows)
            for use in (True,False):
                price = [rng.randint(-8,8) for _ in rows]
                out = priced_cut(p,rows,prices=price,contract=use)
                if valid:
                    best = max(valid.values())
                    self.assertLessEqual(best,out['upper'])
                    self.assertLessEqual(out['upper'],old['upper'])
                    gains['improved'] += out['upper']<old['upper']
                if out['codes']:
                    self.assertEqual(valid[tuple(out['codes'])],out['score'])
                if out.get('status')=='infeasible':
                    self.assertFalse(valid)
        self.assertGreater(gains['improved'],0)

    def test_binary_model_and_conflict_explanation(self):
        graph = Implications(3,[(1,2),(3,4),(3,5)],lambda:None)
        self.assertIsNone(graph.assign(0,2))
        path = graph.path(2,3)
        self.assertTrue(all(b in graph.graph[a] for a,b in zip(path,path[1:])))
        p = pair_problem(multiple_loops=True)
        rng = Random(12)
        work = Counter()
        for _ in range(100):
            rows = [rng.sample(r,2) for r in p.domains]
            old = binary_search(p,rows,'gold',reuse_graph=False)
            new = binary_search(p,rows,'gold')
            self.assertEqual((old[0],new[0]),('infeasible','infeasible'))
            work.update(old=old[-1],new=new[-1])
        self.assertLess(work['new'],work['old'])

    def test_durable_tables_prefixes_and_restart(self):
        p,ds = planted(1,5,97,uniform_ports=True)
        parts = ((0,1,2),(3,4,5),(6,7,8,9))
        old = paged_trie(p,parts,ds)
        with TemporaryDirectory() as tmp:
            db = Path(tmp)/'paging.sqlite3'
            cursor = None
            counts = Counter()
            while True:
                r = paged_trie(p,parts,ds,max_steps=3,table_work=7,cursor=cursor,cache_path=db)
                cursor = r['cursor']
                counts.update(r.get('work',{}))
                if r['complete']:
                    break
            self.assertEqual(r['solutions'],old['solutions'])
            self.assertEqual(r['checks'],old['checks'])
            self.assertEqual(counts['table_starts'],3)
            self.assertEqual(counts['region_nodes'],old['work']['region_nodes'])
            self.assertGreater(counts['index_hits'],0)
            narrowed = list(ds)
            c = next(i for i,mask in enumerate(ds) if mask.bit_count()>1)
            narrowed[c] ^= narrowed[c] & -narrowed[c]
            child = paged_trie(p,parts,narrowed,cache_path=db)
            self.assertEqual(child['solutions'],paged_trie(p,parts,narrowed)['solutions'])
            self.assertEqual(child['work']['table_starts'],1)
            self.assertEqual(child['work']['table_hits'],2)
            with self.assertRaises(ValueError):
                paged_trie(p,parts,narrowed,cursor=cursor,cache_path=db)

    def test_interrupted_region_transaction_rolls_back(self):
        p,ds = planted(1,4)
        parts = ((0,1,2),(3,4,5),(6,7,8,9))
        import r29_regional_cache
        with TemporaryDirectory() as tmp:
            db = Path(tmp)/'paging.sqlite3'
            with patch.object(r29_regional_cache.RegionalCache,'labels',side_effect=RuntimeError("interrupted")):
                with self.assertRaises(RuntimeError):
                    paged_trie(p,parts,ds,cache_path=db)
            with sqlite3.connect(db) as con:
                self.assertEqual(con.execute('SELECT count(*) FROM r29_region').fetchone()[0],0)
            con.close()
            self.assertEqual(paged_trie(p,parts,ds,cache_path=db)['solutions'],paged_trie(p,parts,ds)['solutions'])

    @requires_sat
    def test_checked_resolution_and_hierarchy(self):
        from r29_logic_proof import Resolution
        from r28_logic_region import regional_interpolant
        groups = [[[1,2],[-1,2]],[[-2,3]],[[-3]]]
        out = proof_interpolants(groups,[(0,),(0,1)])
        self.assertEqual(len(out['boundaries']),2)
        self.assertEqual(out['proof_builds'],1)
        from r29_logic_proof import reuse_boundary
        self.assertTrue(reuse_boundary(groups[0],groups[1]+groups[2]+[[4]],out['boundaries'][0]))
        self.assertFalse(reuse_boundary(groups[0],groups[1]+[[3]],out['boundaries'][0]))
        self.assertFalse(reuse_boundary([[4]],groups[1]+groups[2],out['boundaries'][0]))
        self.assertEqual(proof_interpolants([[[1]],[[1,2]]],[(0,)])['status'],'joined')
        self.assertEqual(proof_interpolants(groups,[(0,)],max_work=1)['status'],'unknown')
        trace = Resolution([[[1,2],[-1,2]],[[1,-2],[-1,-2]]],1000)
        self.assertIsNotNone(trace.rup([1]))
        self.assertIsNotNone(trace.rup([]))
        # A falsely asserted clause cannot enter the proof.
        self.assertIsNone(Resolution([[[1,2]],[[-1,3]]],100).rup([4]))
        p = pair_problem(multiple_loops=True)
        result = regional_interpolant(p,p.domains,[0],proof=True)
        self.assertEqual(result['status'],'separated')
        self.assertTrue(result['verified'])


if __name__ == '__main__':
    unittest.main()
