"""Regression controls for completed seed and route intervals and their exact owners."""
from contextlib import closing
from itertools import combinations
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from r27_incumbent import BoardModel
from r30_multicenter import SharedSupports, cycles, orders, run
from r31_route_coverage import inherited_end, register
from r31_sampling import _claim, _schema, run_seeds
from r32_route_store import PageProofs, colex, publish_page
from search_fixtures import planted_gold
from search_problem import SearchProblem


class RangeTests(unittest.TestCase):
    def test_full_and_partial_seed_pages_with_holes_and_ties(self):
        def job(seed):
            return dict(score=seed % 9, components=(seed // 9) % 3, gold=False, codes=[seed])
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / 'seeds.sqlite'
            with closing(sqlite3.connect(db)) as conn:
                _schema(conn)
                with conn:
                    conn.executemany("INSERT INTO r31_jobs VALUES(?,?,NULL,NULL,?,0)",
                                     [('s', i, json.dumps(dict(job(i), seed=i))) for i in range(400) if i != 163])
            for start, end in ((0,384),(70,139),(21,31),(161,162),(8,9)):
                expected = max(range(start,end), key=lambda i:(job(i)['score'], -job(i)['components'], -i))
                seen = []
                def visit(i):
                    seen.append(i)
                    return job(i)
                result = run_seeds(db,'s',start,end-start,visit)
                self.assertEqual(result['best']['seed'], expected)
                self.assertEqual(seen, [163] if start == 0 else [])
                self.assertEqual(result['reused'] + result['completed'], end-start)
            with patch('r31_sampling.json.loads', wraps=json.loads) as loads:
                result = run_seeds(db, 's', 0, 384, None)
                self.assertEqual(loads.call_count, 3)
                self.assertEqual(result['reused'],384)

    def test_unlimited_revisits_a_lease_that_expires_behind_it(self):
        with tempfile.TemporaryDirectory() as temp:
            db = Path(temp) / 'seeds.sqlite'
            with closing(sqlite3.connect(db)) as conn:
                _schema(conn)
                _claim(conn, 's', 0, 3600)
                with conn:
                    conn.executemany("INSERT INTO r31_jobs VALUES(?,?,NULL,NULL,?,0)",
                        [('s', i, json.dumps(dict(seed=i,score=0,components=4,gold=False,codes=[i]))) for i in range(1,127)])
            def job(i):
                self.assertIn(i,(0,127))
                if i == 127:
                    with closing(sqlite3.connect(db)) as conn, conn:
                        conn.execute("UPDATE r31_jobs SET expires=0 WHERE scope='s' AND seed=0")
                return dict(score=10,components=1,gold=i==0,codes=[i])
            result = run_seeds(db, 's', 0, 0, job)
            self.assertEqual((result['best']['seed'], result['completed']), (0,2))

    def test_mixed_owner_evidence_and_late_dependency(self):
        with closing(sqlite3.connect(':memory:')) as db:
            db.execute('CREATE TABLE r30_pages(scope,lo,hi,work,best)')
            f=((2,2,3),)
            register(db, 'legacy','p',f,208,['a'])
            register(db, 'union','p',f,208,['b','c'])
            db.execute("INSERT INTO r30_pages VALUES('legacy',0,10,'{}','null')")
            proofs=PageProofs(db,208)
            for lo in (0,1):
                db.execute("INSERT INTO r30_pages VALUES('union',?,?, '{}','null')", (lo,lo+1))
                publish_page(db,'union',lo,lo+1,['b','c'],{'b':{('c',25)}})
            def covered(lo):
                return inherited_end(db,'new','p',f,208,['a','b'],lo,10,proofs=proofs)
            self.assertEqual(covered(0),0)
            db.execute("INSERT INTO r30_pages VALUES('union',25,26,'{}','null')")
            publish_page(db,'union',25,26,['b','c'],{})
            self.assertEqual(covered(0),1)
            trace = []
            db.set_trace_callback(trace.append)
            self.assertEqual(covered(1),2)
            self.assertEqual(sum('SELECT c.hi' in q and "owner='c'" in q for q in trace),0)
            self.assertEqual(covered(2),2)

    def test_warm_interval_preserves_positive_and_holes(self):
        tiles,mesh,board = planted_gold(31,1,4)
        problem=SearchProblem(tiles,mesh)
        center=board.copy()
        for i,j in enumerate((1,0,3,4,5,2)):
            center[i]=board[j]
        with tempfile.TemporaryDirectory() as temp:
            db=Path(temp)/'routes.sqlite'
            def search(start,stop):
                return run(problem,[center],db,start=start,stop=stop,families=((2,4),),
                           target=len(problem.edges),page=1,filters=('star',))
            first=search(0,1)
            self.assertEqual(first['best']['score'],len(problem.edges))
            search(2,4)
            with patch('r30_multicenter.ExchangeBound',side_effect=AssertionError('rebuilt route')):
                self.assertEqual(search(2,3)['best'],first['best'])
            self.assertEqual(search(0,4)['new_pages'],1)
            self.assertEqual(search(0,4)['new_pages'],0)

    def test_dependency_ranks_against_physical_reconstruction(self):
        tiles,mesh,codes=planted_gold(31,1,4)
        for pins in ({},{9:codes[9]}):
            model=BoardModel(SearchProblem(tiles,mesh,pins))
            center=list(codes)
            center[0],center[1],center[3]=center[1],center[3],center[0]
            graph=SharedSupports(model,[codes,center],[(2,4)])
            previous=np.asarray(codes)//3
            given=np.asarray(center)//3
            rotations=graph.transforms
            hits=0
            for cells in list(combinations(map(int,model.free),6))[::17]:
                for order in orders((2,4)):
                    result=graph.earlier_support(1,cells,order)
                    endpoint=given.copy()
                    endpoint[list(cells)]=given[[cells[j] for j in order]]
                    expected=None
                    for transform in rotations:
                        rel=np.argsort(previous)[endpoint[list(transform)]]
                        if cycles(rel)==(2,4):
                            changed=np.flatnonzero(rel!=np.arange(model.n))
                            expected=(0,colex(changed,graph.indices))
                            break
                    self.assertEqual(result,expected)
                    hits += result is not None
            self.assertGreater(hits,0)

    def test_native_free_zero_dispatch_replay_and_checksum(self):
        from r31_sampling import sample_archive
        tiles,mesh,_=planted_gold(28,1,3)
        p=SearchProblem(tiles,mesh)
        opts=dict(seed=5,replicas=128,backend='cuda',dispatches=0)
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'archive.sqlite'
            expected=sample_archive(p,db,**opts)
            with patch('gpu_sampling_resume_round25.SamplerSession',side_effect=AssertionError('sampler allocation')):
                self.assertEqual(sample_archive(p,db,**opts),expected)
                with self.assertRaisesRegex(AssertionError,'sampler allocation'):
                    sample_archive(p,db,**dict(opts,backend='webgpu'))
                with closing(sqlite3.connect(db)) as con, con:
                    con.execute("UPDATE r34_final_archives SET checksum='damaged'")
                with self.assertRaisesRegex(ValueError,'archive'):
                    sample_archive(p,db,**opts)

    def test_coverage_changes_across_scopes_and_external_delete(self):
        from constraint_work import CoverageLedger, WorkScope
        from test_constraint_work import leaves
        scope=WorkScope('p','gold',(15,15),(15,15))
        other=WorkScope('p','edges',scope.root,scope.root)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'coverage.sqlite'
            with CoverageLedger(path,scope,symbolic=True) as main, CoverageLedger(path,other) as unrelated, CoverageLedger(path,scope) as peer:
                for i in range(4):
                    for target in (unrelated,peer):
                        target.record_completed((1<<i,1),status='infeasible',complete=True)
                rows,done=main.remaining([scope.root])
                self.assertTrue(done)
                self.assertEqual((len(leaves(rows)),main.stats['coverage_records_checked']),(12,4))
                peer.record_completed((1,2),status='infeasible',complete=True)
                self.assertEqual(len(leaves(main.remaining([scope.root])[0])),11)
                self.assertEqual(main.stats['coverage_records_checked'],5)
                self.assertEqual(main.stats['coverage_symbolic_builds'],1)
                with peer.db:
                    peer.db.execute("DELETE FROM closed WHERE json_extract(scope,'$.target')='gold' AND box=?",
                                    (json.dumps(['0x1','0x2']),))
                self.assertEqual(len(leaves(main.remaining([scope.root])[0])),12)
                self.assertEqual(main.stats['coverage_symbolic_builds'],2)

    def test_phase_clause_complements_and_reverified_witness(self):
        from r27_structure_fixtures import planted
        from r32_fresh_phases import lift_cube,phase_cube
        from r32_gold_bonds import inventory
        from r34_phase_cache import negative,schema,checked_witness
        p,_=planted(1,4,37)
        from importlib.util import find_spec
        if find_spec('ortools') is None:
            self.skipTest('full CPU profile')
        fixed=[1<<(3*i) for i in range(p.n)]
        required=inventory(p).required
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'phases.sqlite'
            rows=[lift_cube(p,required,phase_cube(p,2,i),fixed,2,db) for i in range(4)]
            good=next(r for r in rows if r['status']=='GOLD')
            pins={i:v^1 for i,v in enumerate(good['phases'])}
            with patch('r32_fresh_phases.search',side_effect=AssertionError('native lift on replay')):
                warm=lift_cube(p,required,pins,fixed,2,db)
                self.assertEqual((warm['codes'],warm['phases'],warm['lift_calls']),(good['codes'],list(pins.values()),0))
            self.assertIsNone(checked_witness(p,warm['codes'],warm['phases'],fixed,{},forbidden=[warm['pairs'][0]]))
            self.assertIsNone(checked_witness(p,warm['codes'],warm['phases'],[2,*fixed[1:]],{}))
            with closing(sqlite3.connect(db)) as conn,conn:
                schema(conn)
                conn.execute("INSERT INTO r32_phase_cuts VALUES('control','[[2,0],[4,1]]')")
                self.assertEqual(negative(conn,'control',{2:1,4:0,5:1}),[[2,1],[4,0]])
                self.assertIsNone(negative(conn,'control',{2:1,4:1}))
                self.assertIsNone(negative(conn,'control',{4:0}))

    def test_positive_page_preserves_one_family_and_closes_another(self):
        tiles,mesh,gold=planted_gold(31,1,4)
        problem=SearchProblem(tiles,mesh)
        center=list(gold)
        for i,j in enumerate((1,0,3,4,5,2)):
            center[i]=gold[j]
        with tempfile.TemporaryDirectory() as temp:
            db=Path(temp)/'route.sqlite'
            opts=dict(start=0,stop=1,target=len(problem.edges),filters=())
            combined=run(problem,[center],db,families=((2,4),(3,3)),**opts)
            self.assertEqual(combined['best']['score'],len(problem.edges))
            warm=run(problem,[center],db,families=((3,3),),**opts)
            cold=run(problem,[center],Path(temp)/'cold.sqlite',families=((3,3),),**opts)
            self.assertEqual(warm['best'],cold['best'])
            self.assertEqual(warm['work'].get('first_prior_page'),40)
            positive=run(problem,[center],db,families=((2,4),),**opts)
            self.assertEqual(positive['best']['score'],len(problem.edges))
            self.assertEqual(positive['work'].get('first_prior_page',0),0)

    def test_fully_covered_frontier_stops_fetching_more_records(self):
        from constraint_work import subtract_many
        def exclusions():
            yield (1,3)
            yield (2,3)
            raise AssertionError('read another certificate after exhausting the source')
        self.assertEqual(subtract_many([(3,3)],exclusions()),((),True))
        self.assertEqual(subtract_many([],exclusions()),((),True))
