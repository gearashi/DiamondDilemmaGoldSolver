from collections import deque
from contextlib import closing
import itertools
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np

from r27_incumbent import BoardModel
from r29_plateau import walk
from search_fixtures import planted_gold
from search_problem import SearchProblem


class IncrementalDistances(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data,shape,ref=planted_gold(19,1,3)
        fixed={i:ref[i] for i in range(4,len(ref))}
        fixed[4]=3*(ref[4]//3)+(ref[4]+1)%3
        cls.problem=SearchProblem(data,shape,fixed)
        cls.start=(0,4,1,5)
        cls.codes=[3*t for t in cls.start]+list(fixed.values())
        model=BoardModel(cls.problem)
        cls.scores={}
        for perm in itertools.permutations(cls.start):
            cls.scores[perm]=max(int(model.matches(np.array([3*t+r for t,r in zip(perm,rot)]+list(fixed.values()))).sum())
                                 for rot in itertools.product(range(3),repeat=4))
        assert max(cls.scores.values())==12

    def reference(self,level,cap):
        paths={self.start:(self.start,)}
        queue=deque([self.start])
        while queue:
            vertex=queue.popleft()
            if len(paths[vertex])-1>=cap:continue
            outgoing=[]
            for a,b in itertools.combinations(range(4),2):
                q=list(vertex)
                q[a],q[b]=q[b],q[a]
                outgoing.append(tuple(q))
            for q in sorted(outgoing):
                if q not in paths and self.scores[q]>=level:
                    paths[q]=paths[vertex]+(q,)
                    queue.append(q)
        return paths

    def test_shortcut_and_exact_reachability_against_all_permutations(self):
        H=self.reference(9,4)
        L=self.reference(8,4)
        self.assertEqual((len(H[(4,1,0,5)])-1,len(L[(4,1,0,5)])-1),(4,2))
        def bare(key):
            return tuple(int(x) for x in key.split(':',1)[1].split(',')[:4])
        with tempfile.TemporaryDirectory() as tmp:
            for limit in (0,2,3,4):
                fn=Path(tmp)/f'walk-{limit}.sqlite'
                opts=dict(incremental=True,max_swaps=limit,potentials=False,widest=True,tie_limit=1)
                seen=set()
                for _ in range(150):
                    out=walk(self.problem,self.codes,fn,8,15,max_centers=1,**opts)
                    with closing(sqlite3.connect(fn)) as db:
                        current={bare(k):tuple(bare(v) for v in path.split('|')) for k,path in
                                 db.execute('SELECT board,path FROM walk_distances')}
                        if (4,1,0,5) in current:seen.add(len(current[(4,1,0,5)])-1)
                        done=db.execute('SELECT done FROM walk_widen').fetchone()[0]
                    if done:break
                else:self.fail('No finite exhaustion.')
                self.assertFalse(out['reached'])
                self.assertEqual(current,self.reference(8,limit))
                if limit==4:self.assertTrue({2,4}.issubset(seen),seen)
                check=walk(self.problem,self.codes,fn,8,15,max_centers=100,**opts)
                self.assertEqual((check['centers'],check['edge_rows'],check['swap_candidates']),(0,0,0))

    def test_legacy_graph_is_indexed_once_and_shortcut_uses_new_reverse_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            db=Path(tmp)/'migration.sqlite'
            opts=dict(max_swaps=4,potentials=False,widest=True,tie_limit=1)
            released=walk(self.problem,self.codes,db,9,15,max_centers=250,**opts)
            self.assertFalse(released['reached'])
            new=walk(self.problem,self.codes,db,8,15,incremental=True,max_centers=250,**opts)
            self.assertFalse(new['reached'])
            with closing(sqlite3.connect(db)) as conn:
                missing=conn.execute("""SELECT e.board,e.neighbor FROM walk_edges e JOIN walk_scored_centers c ON c.board=e.board
                     EXCEPT SELECT a.board,b.board FROM walk_scored_edges s JOIN walk_vertex_labels a ON a.id=s.board JOIN walk_vertex_labels b ON b.id=s.neighbor""").fetchall()
                self.assertFalse(missing)
                old=conn.execute('SELECT COUNT(*) FROM walk_scored_edges').fetchone()
            again=walk(self.problem,self.codes,db,8,15,incremental=True,max_centers=250,**opts)
            self.assertEqual(again['edge_rows'],0)
            with closing(sqlite3.connect(db)) as conn:
                self.assertEqual(old,conn.execute('SELECT COUNT(*) FROM walk_scored_edges').fetchone())
