from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r29_plateau import SwapGraph,walk
from r34_query_ownership import LostQueryLease
from r35_plateau_pages import ExpansionStopped,advance,exact_parent,materialize,schema,stored_vector
from search_fixtures import planted_gold
from search_problem import SearchProblem


def init(path,problem):
    db=sqlite3.connect(path,timeout=30)
    graph=SwapGraph(db,BoardModel(problem))
    schema(db)
    db.execute('CREATE TABLE IF NOT EXISTS walk_legacy(board TEXT PRIMARY KEY,score INTEGER,seed TEXT)')
    graph.work['swap_candidates']=0
    return db,graph


class DurablePlateaus(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.db=Path(tmp.name)/'test.sqlite'
        data,shape,codes=planted_gold(19,1,3)
        self.problem=SearchProblem(data,shape,{i:codes[i] for i in range(4,len(codes))})
        self.board=codes.copy()
        self.board[0],self.board[1]=self.board[1],self.board[0]
        self.board[2],self.board[3]=self.board[3],self.board[2]

    def test_revoked_query_cannot_publish_solver_result_or_cursor(self):
        conn,g=init(self.db,self.problem)
        with closing(conn),closing(sqlite3.connect(self.db)) as peer:
            key=g.key(self.board)
            g.optimum(key,self.board,0)
            conn.commit()
            actual=g.oracle.optimum
            def steal(*args,**kwargs):
                item=actual(*args,**kwargs)
                with peer:
                    peer.execute("UPDATE r31_jobs SET owner='new-owner',expires=0 WHERE scope=?",('r35-plateau:'+key,))
                return item
            with patch.object(g.oracle,'optimum',steal),self.assertRaises(LostQueryLease):
                advance(g,self.db,key,self.board,0)
            self.assertEqual(conn.execute('SELECT count(*) FROM optima').fetchone(),(1,))
            self.assertEqual(conn.execute('SELECT next_pair FROM walk_band_cursor').fetchone(),(0,))
            self.assertEqual(conn.execute('SELECT count(*) FROM walk_edges').fetchone(),(0,))
            self.assertTrue(advance(g,self.db,key,self.board,0))
            self.assertEqual(len(list(g.known_neighbors(key,0))),6)

    def test_concurrent_witness_publication_cannot_restore_worse_components_or_tie_depth(self):
        conn,g=init(self.db,self.problem)
        with closing(conn),closing(sqlite3.connect(self.db)) as peer:
            worse=[1,16,14,5,21,20,26,27,6,9]
            better=[1,16,13,4,21,20,26,27,6,9]
            key=g.key(worse)
            self.assertEqual(g.check(key,worse,11)['line_components'],5)
            self.assertEqual(g.check(key,better,11)['line_components'],4)
            def delayed(*args,**kwargs):
                with peer:
                    item=dict(score=11,components=4,codes=better,gold=False)
                    peer.execute('INSERT INTO optima VALUES(?,?)',(key,json.dumps(item)))
                    peer.execute('INSERT INTO walk_ties VALUES(?,?)',(key,12))
                return 11,[worse]
            with patch.object(g.oracle,'optimum',delayed):
                result=g.optimum(key,worse,0,tie_limit=2)
            self.assertEqual(result['codes'],better)
            self.assertEqual(conn.execute('SELECT requested FROM walk_ties WHERE board=?',(key,)).fetchone(),(12,))

    def test_cold_root_deadline_preserves_an_unclosed_start(self):
        with patch('r29_plateau.time.monotonic',side_effect=[0,2]):
            paused=walk(self.problem,self.board,self.db,0,15,seconds=1,tie_limit=1)
        self.assertTrue(paused['initialization_pending'])
        self.assertEqual((paused['codes'],paused['closed'],paused['score']),
                         (self.board,0,self.problem.validate(self.board)['matched_edges']))
        resumed=walk(self.problem,self.board,self.db,0,15,max_centers=32,tie_limit=1)
        self.assertTrue(resumed['reached'])

    def test_positive_at_last_candidate_and_different_depth_resumes(self):
        target=len(self.problem.edges)
        # Sort keys once and choose the last candidate for a one-swap start.
        data,shape,witness=planted_gold(19,1,3)
        near=witness.copy()
        near[2],near[3]=near[3],near[2]
        options=dict(max_centers=8,max_swaps=1,tie_limit=1,potentials=False)
        first=walk(self.problem,near,self.db,0,target,max_candidates=5,**options)
        self.assertFalse(first['reached'])
        with closing(sqlite3.connect(self.db)) as conn:
            self.assertEqual(conn.execute('SELECT next_pair FROM walk_band_cursor').fetchone(),(5,))
            self.assertEqual(conn.execute('SELECT lowest FROM walk_vectors').fetchone(),(target+1,))
        frozen=walk(self.problem,near,self.db,0,target,max_centers=8,max_swaps=0,tie_limit=1,potentials=False)
        self.assertFalse(frozen['reached'])
        hit=walk(self.problem,near,self.db,0,target,max_candidates=1,**options)
        self.assertTrue(hit['reached'])
        self.assertEqual(hit['swaps'],[[2,3]])
        self.assertTrue(self.problem.validate(hit['codes'])['valid'])

    def test_concurrent_same_frontier_serializes_rotations(self):
        first=walk(self.problem,self.board,self.db,0,15,max_centers=0,tie_limit=1)
        self.assertFalse(first['reached'])
        def go(_):
            return walk(self.problem,self.board,self.db,0,15,max_centers=100,tie_limit=1)
        with ThreadPoolExecutor(2) as pool:
            rows=list(pool.map(go,range(2)))
        control=walk(self.problem,self.board,self.db.with_name('cold.sqlite'),0,15,max_centers=100,tie_limit=1)
        for row in rows:
            self.assertEqual(row['swaps'],control['swaps'])
            self.assertEqual(row['codes'],control['codes'])
        self.assertEqual(first['solved']+sum(r['solved'] for r in rows),control['solved'])

    def test_interrupted_vector_reuses_exact_page_and_restricts_domains(self):
        data,shape,center=planted_gold(135,2,10)
        p=SearchProblem(data,shape,{39:center[39]})
        p.domains[0]=[x for x in p.domains[0] if x//3!=center[1]//3]
        conn,g=init(self.db,p)
        with closing(conn):
            key=g.key(center)
            count=0
            real=ExchangeBound.bounds
            def cut(self,*args,**kw):
                nonlocal count
                count+=1
                if count==2:raise ExpansionStopped()
                return real(self,*args,**kw)
            with patch.object(ExchangeBound,'bounds',cut),self.assertRaises(ExpansionStopped):
                materialize(g,key,center,lambda:None,float('inf'))
            self.assertEqual(conn.execute('SELECT count(*) FROM walk_star_pages').fetchone(),(1,))
            before=g.work['star_evaluations']
            materialize(g,key,center,lambda:None,float('inf'))
            self.assertEqual(g.work['star_evaluations']-before,len(g.pairs)-512)
            self.assertEqual(conn.execute('SELECT count(*) FROM walk_star_pages').fetchone(),(0,))
            perm=center.copy()
            self.assertIsNone(ExchangeBound(g.model,center).witness_seed((0,1),(1,0)))
            for a,b in ((6,9),(2,10),(7,22),(21,32)):
                perm=center.copy()
                perm[a],perm[b]=perm[b],perm[a]
                child=g.key(perm)
                materialize(g,child,perm,lambda:None,float('inf'),parent=key)
                _,actual=stored_vector(g,child)
                expected=ExchangeBound(g.model,perm).bounds(g.pairs,(1,0),doubled=True)
                np.testing.assert_array_equal(actual,expected)
            perm[2],perm[3]=perm[3],perm[2]
            self.assertIsNone(exact_parent(g,g.key(perm),perm,key))

    def test_legacy_floor_without_parity_is_not_used_for_transport(self):
        conn,g=init(self.db,self.problem)
        with closing(conn):
            key=g.key(self.board)
            upper=ExchangeBound(g.model,self.board).bounds(g.pairs,(1,0)).astype('<i4')
            conn.execute('INSERT INTO walk_bounds VALUES(?,?,?)',(key,5,zlib.compress(upper.tobytes())))
            conn.commit()
            near=self.board.copy()
            near[0],near[1]=near[1],near[0]
            self.assertIsNone(exact_parent(g,g.key(near),near,key))
            low,old=materialize(g,key,self.board,lambda:None,float('inf'))
            self.assertEqual(low,5)
            np.testing.assert_array_equal(old,upper)

    def test_legacy_membership_import_does_not_consume_lower_band(self):
        # The old center table lists qualified members of a specific floor.
        m=BoardModel(self.problem)
        from r28_exchange import domain_binding
        from r29_plateau import identity
        scope=domain_binding(m)+'/8/15/1'
        with closing(sqlite3.connect(self.db)) as conn,conn:
            conn.executescript('CREATE TABLE centers(scope TEXT,perm TEXT,codes TEXT,closed INT); CREATE TABLE optimum(scope TEXT,perm TEXT,score INT,comps INT,codes TEXT);')
            conn.execute('INSERT INTO centers VALUES(?,?,?,1)',(scope,identity(self.board),json.dumps(self.board)))
        conn,g=init(self.db,self.problem)
        with closing(conn):
            key=g.key(self.board)
            self.assertFalse(advance(g,self.db,key,self.board,8,candidate_limit=3))
            self.assertFalse(conn.execute('SELECT * FROM walk_imported').fetchall())
            self.assertTrue(advance(g,self.db,key,self.board,8))
            self.assertEqual(conn.execute('SELECT floor FROM walk_imported').fetchall(),[(8,)])
            self.assertTrue(advance(g,self.db,key,self.board,7))
            self.assertEqual(conn.execute('SELECT lowest FROM walk_vectors').fetchone(),(7,))
