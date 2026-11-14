"""Independent comparison of diagram bounds, page ownership, and shared local stars."""
from collections import Counter
from contextlib import closing
import itertools
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r29_exchange_clusters import StarClusters
from r30_donor_dd import diagram, screened_diagram
from r30_gpu_filters import contact_counts, core_passes
from r30_multicenter import CountingTerms, SharedSupports, cycles, orders, released_six, run, sparse_cycle_type
from search_fixtures import planted_gold
from search_problem import SearchProblem


class SharedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tiles,geo,cls.codes = planted_gold(31,1,4)
        cls.problem = SearchProblem(tiles,geo)
        cls.model = BoardModel(cls.problem)

    def test_diagram_reference_and_relaxation(self):
        bound = ExchangeBound(self.model,self.codes)
        rng = np.random.default_rng(4)
        for fam in ((2,4),(3,3),(2,2,3),(2,2,2,2)):
            cells = tuple(map(int,rng.choice(bound.n,sum(fam),replace=False)))
            reference = max(int(ExchangeBound(self.model, bound.witness_seed(cells,o)).total) for o in orders(fam))
            exact = diagram(bound,cells,fam)
            relaxed = diagram(bound,cells,fam,width=1)
            unmerged = diagram(bound,cells,fam,merge=False)
            self.assertEqual(unmerged['upper_twice'],reference)
            self.assertLessEqual(exact['transitions'],unmerged['transitions'])
            self.assertEqual(exact['upper_twice'],reference)
            self.assertGreaterEqual(relaxed['upper_twice'],reference)
            self.assertEqual(screened_diagram(bound,cells,fam,0,width=1)['upper_twice'],reference)

    def test_shared_values_match_independent_prices(self):
        c = self.codes
        other = list(c)
        other[0],other[1],other[3] = other[1],other[3],other[0]
        graph = SharedSupports(self.model,[c,other],[(2,4)])
        stats = Counter()
        for cells in list(itertools.combinations(range(len(c)),6))[::23]:
            out = {}
            cache = CountingTerms(stats)
            for i,b in enumerate(graph.bounds):
                upper = graph.upper(i,cells,out,cache,stats)
                self.assertEqual(upper,StarClusters(b).support_upper_twice(cells))
            # Replaying this partial-boundary cache must give the identical score.
            self.assertEqual(graph.clusters[0].support_upper_twice(cells,cache),out[0])
        self.assertGreater(stats['local_term_hits'],0)
        c0 = np.asarray(c)//3
        # One three-cycle between the centers permits intersection of their even shells.
        identity = tuple(map(int,np.argsort(np.asarray(other)//3)[c0]))
        graph.copies[1] = [{i:j for i,j in enumerate(identity) if i!=j}]
        seen = set()
        for o in orders((2,4)):
            cells = tuple(range(6))
            dest = np.asarray(other)//3
            dest[list(cells)] = dest[[cells[j] for j in o]]
            direct = cycles(np.argsort(c0)[dest])==(2,4)
            self.assertEqual(graph.covered_by_earlier(1,cells,o),direct)
            seen.add(direct)
        self.assertEqual(seen,{False,True})
        self.assertEqual(sparse_cycle_type({1:2,2:1},{1:2,2:1}),())

    def test_ranges_rollback_witness_and_filter_order(self):
        start = list(self.codes)
        idx = [0,1,2,3,4,5]
        order = (1,0,3,4,5,2)
        for i,j in zip(idx,order):
            start[i] = self.codes[j]
        args = dict(families=((2,4),),target=len(self.problem.edges),page=1)
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp)/'route.sqlite'
            with patch('r30_multicenter.rotation_optimum',side_effect=RuntimeError('interrupted')):
                with self.assertRaises(RuntimeError):
                    run(self.problem,[start],db,stop=1,filters=(),**args)
            with closing(sqlite3.connect(db)) as con:
                self.assertEqual(con.execute('SELECT count(*) FROM r30_pages').fetchone()[0],0)
            head = run(self.problem,[start],db,start=1,stop=3,filters=('star',),**args)
            after = run(self.problem,[start],db,start=0,stop=4,filters=('diagram','star'),**args)
            self.assertEqual((head['covered_supports'],after['covered_supports'],after['new_pages']),(2,4,2))
            self.assertEqual(after['best']['score'],len(self.problem.edges))
            final = run(self.problem,[start],db,stop=4,filters=(),**args)
            self.assertEqual(final['work'],{})
            self.assertEqual(final['best'],after['best'])

    def test_concurrent_overlapping_intervals_commit_each_support_once(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp)/'joint.sqlite'
            def worker(start,stop):
                return run(self.problem,[self.codes],db,start=start,stop=stop,page=2,
                           target=len(self.problem.edges),filters=('star','family'))
            with ThreadPoolExecutor(max_workers=2) as pool:
                one = pool.submit(worker,0,5)
                two = pool.submit(worker,3,7)
                a,b = one.result(),two.result()
            self.assertEqual(sum(row['work'].get('raw_center_endpoints',0) for row in (a,b)),7*130)
            self.assertEqual(worker(0,7)['new_pages'],0)

    def test_entire_budget_on_unrelated_center(self):
        from r30_fresh import fresh_start
        from r30_bounds import budget_upper, defect_packings
        from unittest.mock import patch
        tiles,mesh,_ = planted_gold(61,2,4)
        problem = SearchProblem(tiles,mesh)
        model = BoardModel(problem)
        codes = fresh_start(model,np.random.default_rng(53))
        bound = ExchangeBound(model,codes)
        target = len(problem.edges)
        self.assertLess(budget_upper(bound,defect_packings(bound),4),target)
        with tempfile.TemporaryDirectory() as tmp:
            database = Path(tmp)/'budget.sqlite'
            with patch('r30_multicenter.SharedSupports',side_effect=AssertionError('enumerated a support')):
                row = run(problem,[codes],database,target=target)
            with patch('r30_bounds.budget_upper',side_effect=AssertionError('recalculated the budget')):
                warm = run(problem,[codes],database,families=((2,2,3),),target=target)
            self.assertTrue(warm['complete'])
        self.assertTrue(row['complete'])
        self.assertEqual(row['remaining_supports'],0)
        self.assertEqual(row['new_pages'],0)

    def test_shipped_scope_requires_every_center_and_cycle_type(self):
        from r28_exchange import read_json
        archive = read_json(Path(__file__).parent/'docs/R29_EXCHANGE_CENTERS.json')
        centers = [r['pieces'] for r in archive['centers']]
        families = ((4,2),(3,3))
        self.assertTrue(released_six(archive['domain'],centers,families,208))
        other = centers[0].copy()
        other[0],other[1] = other[1],other[0]
        self.assertFalse(released_six(archive['domain'],[other,*centers],families,208))
        self.assertFalse(released_six(archive['domain'],centers,families,207))
        self.assertFalse(released_six(archive['domain'],centers,((2,2,3),),208))
        self.assertFalse(released_six('other',centers,families,208))

    def test_device_filter_scalar_reference(self):
        cells = np.array([[1,2,3],[0,4,5],[0,2,4]])
        close = np.eye(6,dtype=bool)
        close[1,2] = close[2,1] = close[2,3] = close[3,2] = True
        self.assertEqual(contact_counts(cells,close).tolist(),[2,0,0])
        masks = [[5,16],[],[0]]
        self.assertEqual(core_passes(cells,masks,6).tolist(),[[False,True,True],[True]*3,[False]*3])


if __name__=='__main__':
    unittest.main()
