from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import numpy as np

from bench_r35_center_schedule import compare,fixture
from r35_center_schedule import interleave


class ScheduleTests(unittest.TestCase):
    def test_normal_public_search_finds_group_two_and_reopens_the_same_union(self):
        with TemporaryDirectory() as tmp:r=compare(tmp)
        self.assertEqual(r['serial_turn_to_planted_gold'],5)
        for name in ('fair','yield'):
            self.assertEqual(r[name]['improvements'][-1]['turn'],2)
            self.assertEqual(r[name]['covered'],[4,4])
            self.assertEqual(r[name]['warm'],[0,0])

    def test_new_group_gets_a_turn_and_discretionary_yield_cannot_starve(self):
        p,boards,family=fixture()
        c={str(i):{'board':np.asarray(board)} for i,board in enumerate([*boards,boards[0]])}
        def runner(problem,rows,db,**kw):
            ix=0 if np.array_equal(rows[0],boards[0]) else 1
            return dict(scope='fake'+str(ix),best=None,complete=False,range_finished=True,new_pages=1,
                        work={'raw_center_endpoints':100,'first_owned':100*ix})
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'schedule.db'
            opt=dict(start=0,stop=24,page=1,max_pages=2)
            interleave(p,c,[['0'],['1']],path,family,42,opt,runner,quota=1)
            states,w=interleave(p,c,[['0'],['1'],['2']],path,family,42,dict(opt,max_pages=12),runner,quota=1,policy='yield')
            self.assertEqual(w['first_turns'][0],2)
            self.assertEqual(w['group_turns'],[12,12,12])
            for i in range(3):
                visits=[j for j,t in enumerate(w['first_turns']) if t==i]
                self.assertLessEqual(max(b-a for a,b in zip(visits,visits[1:])),5)
            self.assertTrue(all(not s['range_finished'] for s in states))


if __name__=='__main__':unittest.main()
