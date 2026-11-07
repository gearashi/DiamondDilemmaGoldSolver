import itertools,random,unittest
from constraint_incremental_hall import IncrementalHall


def exhaustive(rows):
    supports=[0]*len(rows);count=0
    for p in itertools.permutations(range(len(rows))):
        if all(row>>tile&1 for row,tile in zip(rows,p)):
            count+=1
            for c,t in enumerate(p):supports[c]|=1<<t
    return tuple(supports) if count else None


class IncrementalHallTests(unittest.TestCase):
    def test_random_shrink_widen_repair_and_backtrack(self):
        rng=random.Random(2409);engine=IncrementalHall();previous=None
        for n in range(2,7):
            engine=IncrementalHall()
            history=[]
            for trial in range(140):
                if history and trial%7==0:rows=rng.choice(history)
                elif history and trial%3:
                    rows=list(history[-1]);cell=rng.randrange(n);rows[cell]&=~(1<<rng.randrange(n));rows=tuple(rows)
                else:rows=tuple(rng.randrange(1,1<<n) for _ in range(n))
                history.append(rows)
                got,matching,failed=engine.filter(rows)
                self.assertEqual(got,exhaustive(rows),(n,rows))
    def test_unaffected_components_skip_traversals(self):
        engine=IncrementalHall();rows=(3,3,12,12,48,48)
        engine.filter(rows);before=engine.stats.copy()
        changed=(1,3,12,12,48,48)
        got,_,_=engine.filter(changed)
        self.assertEqual(got,exhaustive(changed))
        self.assertEqual(engine.stats['hall_scc_vertices']-before['hall_scc_vertices'],2)
        self.assertEqual(engine.stats['hall_scc_reused_vertices']-before['hall_scc_reused_vertices'],4)
        before=engine.stats['hall_scc_vertices'];engine.filter(rows)
        self.assertEqual(engine.stats['hall_scc_vertices'],before)
    def test_interrupt_does_not_commit_incomplete_state(self):
        engine=IncrementalHall();rows=(7,7,7);calls=[0]
        def check():
            calls[0]+=1
            if calls[0]==4:raise InterruptedError()
        with self.assertRaises(InterruptedError):engine.filter(rows,check=check)
        self.assertIsNone(engine.previous)
        self.assertEqual(engine.filter(rows)[0],exhaustive(rows))

if __name__=='__main__':unittest.main()
