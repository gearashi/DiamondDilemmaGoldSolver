"""Exact work sharing, expiry and geographic evidence reuse across related requests."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from itertools import product
import json
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from r27_structure_fixtures import planted
from r32_gold_loss import geographic, reference_groups, save_query
from r34_query_ownership import LostQueryLease, query_lease
from test_dependencies import requires_sat


class QueryTests(unittest.TestCase):
    def test_exact_workers_share_one_call_and_reject_stale_owner(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'queries.sqlite'
            barrier=threading.Barrier(2)
            calls=[]
            def expensive():
                calls.append(None)
                time.sleep(.15)
                return dict(status='INFEASIBLE')
            def worker():
                barrier.wait(timeout=5)
                return save_query(path,'identical',expensive)
            with ThreadPoolExecutor(2) as pool:
                jobs=[pool.submit(worker) for _ in range(2)]
                answers=[job.result() for job in jobs]
            self.assertEqual(len(calls),1)
            self.assertEqual(sum(bool(r.get('reused')) for r in answers),1)
            with closing(sqlite3.connect(path)) as a,closing(sqlite3.connect(path)) as b:
                with query_lease(a,path,'recovery') as old:
                    with b:b.execute("UPDATE r31_jobs SET expires=0 WHERE scope='recovery'")
                    with query_lease(b,path,'recovery') as successor:
                        with self.assertRaises(LostQueryLease),a:
                            a.execute('BEGIN IMMEDIATE')
                            old()
                        with b:
                            b.execute('BEGIN IMMEDIATE')
                            successor()
            never=save_query(path,'other',lambda:dict(status='UNKNOWN'))
            self.assertEqual(never['status'],'UNKNOWN')
            self.assertEqual(save_query(path,'other',expensive)['status'],'INFEASIBLE')

    @requires_sat
    def test_band_workers_share_formula_and_witness(self):
        from r31_defect_bands import solve
        from r27_reason_maxsat import SeamFormula
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        tiles,mesh,board=planted_gold(19,1,3)
        p=SearchProblem(tiles,mesh)
        barrier=threading.Barrier(2)
        def build(*args,**kwargs):
            time.sleep(.1)
            return SeamFormula(*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp,patch('r31_defect_bands.SeamFormula',side_effect=build) as builds:
            path=Path(temp)/'bands.sqlite'
            def worker():
                barrier.wait(timeout=5)
                return solve(p,path,target=len(p.edges),domains=[[v] for v in board])
            with ThreadPoolExecutor(2) as pool:
                jobs=[pool.submit(worker) for _ in range(2)]
                answers=[job.result() for job in jobs]
            self.assertEqual(builds.call_count,1)
            self.assertEqual(sum(r['calls'] for r in answers),1)
            self.assertEqual([r['witness']['codes'] for r in answers],[list(board)]*2)

    def test_geographic_ancestor_sibling_and_positive_loss_checks(self):
        from importlib.util import find_spec
        if find_spec('ortools') is None:self.skipTest('full CPU profile')
        p,_=planted(1,0,23,False,True)
        board=[3*i for i in range(p.n)]
        ref,cert,g=reference_groups(p,board)
        fixed=[1<<c for c in board]
        bad=fixed.copy()
        bad[0]=1<<board[1]
        wider=bad.copy()
        wider[-1]|=1<<(board[-1]+1)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'geo.sqlite'
            def ask(ds,lower=0,upper=0,**kwargs):
                return geographic(p,ref,g,cert.rows,[],lower,upper,None,4,path,ds,**kwargs)
            self.assertEqual(ask(wider)['status'],'INFEASIBLE')
            with patch('r30_gold.search',side_effect=AssertionError('duplicate negative')):
                self.assertEqual(ask(bad)['reuse_kind'],'ancestor domains')
            good=ask(fixed)
            self.assertTrue(p.validate(good['codes'])['valid'])
            with patch('r30_gold.search',side_effect=AssertionError('duplicate positive')):
                relaxed=ask(fixed,upper=len(ref),pins={0:good['phases'][0]^1})
                self.assertEqual(relaxed['reuse_kind'],'validated board')
                self.assertEqual(relaxed['phases'][0],good['phases'][0]^1)
            self.assertEqual(ask(fixed,lower=1,upper=len(ref))['status'],'INFEASIBLE')

    @requires_sat
    def test_distinct_widths_share_one_builder_and_corrupt_clauses_are_rejected(self):
        from r31_defect_bands import solve
        from r27_reason_maxsat import SeamFormula
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        tiles,mesh,board=planted_gold(19,1,3)
        p=SearchProblem(tiles,mesh)
        domains=[[v] for v in board]
        barrier=threading.Barrier(2)
        def build(*args,**kwargs):
            time.sleep(.15)
            return SeamFormula(*args,**kwargs)
        with tempfile.TemporaryDirectory() as temp,patch('r31_defect_bands.SeamFormula',side_effect=build) as builds:
            path=Path(temp)/'bands.sqlite'
            def worker(width):
                barrier.wait(timeout=5)
                return solve(p,path,target=len(p.edges),domains=domains,width=width)
            with ThreadPoolExecutor(2) as pool:
                answers=list(pool.map(worker,[6,7]))
            self.assertEqual(builds.call_count,1)
            for part in ('structural','counter'):
                self.assertEqual(sum(a.get(part+'_builds',0) for a in answers),1)
                self.assertEqual(sum(a.get(part+'_replays',0) for a in answers),1)
            with closing(sqlite3.connect(path)) as db,db:
                db.execute("UPDATE r34_band_cnf SET checksum='broken'")
            with self.assertRaisesRegex(ValueError,'checksum'):
                solve(p,path,target=len(p.edges),domains=domains,width=8)

    @requires_sat
    def test_legacy_json_null_reopens_and_all_thresholds_agree_with_enumeration(self):
        from r31_defect_bands import solve
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        tiles,mesh,board=planted_gold(19,1,3)
        p=SearchProblem(tiles,mesh)
        domains=[[3*(v//3)+(v+j)%3 for j in range(2 if i<5 else 1)] for i,v in enumerate(board)]
        domains[0]=domains[0][1:]
        best=max(p.validate(list(row))['matched_edges'] for row in product(*domains))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'all.sqlite'
            solve(p,path,domains=domains,target=0,width=1,max_calls=0)
            with closing(sqlite3.connect(path)) as db,db:
                db.execute("UPDATE r31_bands SET witness='null'")
            for target in range(len(p.edges)+1):
                answer=solve(p,path,domains=domains,target=target,width=1)
                self.assertEqual(answer['status'],'witness' if target<=best else 'infeasible')
                if target<=best:
                    w=answer['witness']
                    self.assertEqual(w['score'],p.validate(w['codes'])['matched_edges'])
                    self.assertEqual(sum(w['defects']),len(p.edges)-w['score'])
            # Previously returned terminal boards must pass all saved band counts again.
            with closing(sqlite3.connect(path)) as db,db:
                key,raw=db.execute("SELECT scope,witness FROM r31_bands WHERE witness IS NOT NULL LIMIT 1").fetchone()
                row=json.loads(raw)
                row['defects'][0]+=1
                db.execute("UPDATE r31_bands SET witness=? WHERE scope=?",(json.dumps(row),key))
            with self.assertRaisesRegex(ValueError,'independent validation'):
                solve(p,path,domains=domains,target=0,width=1)
