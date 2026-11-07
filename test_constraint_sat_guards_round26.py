"""Finite guard reuse contracts, including actual native lifecycle and scopes."""
import itertools,json,time,unittest
from unittest.mock import patch
from constraint_sat_guards_round26 import DomainGuardPool
from constraint_sat_session import PersistentSatSession,solve
from constraint_work import WorkScope,restore_frontier,digest
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,multiple_loops,double_face,tetrahedron_tiles,brute
from test_constraint_sat_assist import HAS_SAT

class GuardPoolTests(unittest.TestCase):
    def make(self,limit=8192):
        clauses=[];counter=[6]
        def fresh():counter[0]+=1;return counter[0]
        return DomainGuardPool((7,7),({0:1,1:2,2:3},{0:4,1:5,2:6}),fresh,clauses.append,limit=limit),clauses

    def test_every_small_domain_box_and_stale_guards_are_existential(self):
        # Independent truth evaluation: unused auxiliary guards are false.
        # Enumerate ALL one-hot cell assignments under every nonempty mask pair.
        for limit in (0,2,8192):
            pool,clauses=self.make(limit)
            for domains in itertools.product(range(1,8),repeat=2):
                assumptions,meanings=pool.assumptions(domains)
                for values in itertools.product(range(3),repeat=2):
                    true={1+values[0],4+values[1],*(a for a in assumptions if a>6)}
                    cnf=all(any((lit in true) if lit>0 else (-lit not in true) for lit in row) for row in clauses)
                    units=all((lit in true) if lit>0 else (-lit not in true) for lit in assumptions)
                    self.assertEqual(cnf and units,all(mask>>code&1 for mask,code in zip(domains,values)))
            self.assertLessEqual(len(pool.entries),limit)
            self.assertEqual(len(clauses),len(pool.entries))
            if limit==2:self.assertGreater(pool.stats['session_guard_fallback_literals'],0)

    def test_failure_poison_prevents_using_partial_native_write(self):
        pool,clauses=self.make()
        def broken(row):clauses.append(row);raise OSError('ambiguous native write')
        pool.post=broken
        with self.assertRaises(OSError):pool.assumptions((1,1))
        self.assertTrue(pool.poisoned);self.assertFalse(pool.entries)
        with self.assertRaises(RuntimeError):pool.assumptions((2,2))

    def test_scope_and_budget_validation(self):
        pool,_=self.make()
        for bad in ((0,7),(8,7),(True,7),(1,),(-1,7)):
            with self.assertRaises(ValueError):pool.assumptions(bad)
        for bad in (-1,True,65537):
            with self.assertRaises(ValueError):self.make(bad)

@unittest.skipUnless(HAS_SAT,'optional python-sat missing')
class NativeGuardTests(unittest.TestCase):
    def test_actual_native_all_answers_match_complete_enumeration(self):
        for problem in [SearchProblem(simple_tiles(),double_face()),SearchProblem(multiple_loops(),double_face())]:
            for target in ('gold','edges'):
                complete=brute(problem,target);scope=WorkScope.for_problem(problem,target)
                for limit in (0,2,8192):
                    s=PersistentSatSession(problem,scope,guard_reuse=True,guard_limit=limit)
                    try:
                        self.assertTrue(s.prepare(lambda:None,5))
                        # Different overlapping contexts are queries, not a claimed
                        # disjoint work frontier. Reset only that test-owned cube.
                        for a,b in itertools.product((1,3,7,15,63),repeat=2):
                            s._deactivate();s.boxes=[(a,b)];answer=s.advance(seconds=5)
                            expected=[row for row in complete if a>>row[0]&1 and b>>row[1]&1]
                            self.assertTrue(answer['complete'])
                            if expected:self.assertIn(tuple(answer['codes']),expected)
                            else:self.assertEqual(answer['status'],'infeasible')
                    finally:s.close()

    def test_complete_finite_reuse_reduces_added_clauses(self):
        results=compare_guard_reuse()
        self.assertEqual(results['baseline']['status'],'infeasible')
        self.assertEqual(results['reused']['status'],'infeasible')
        self.assertEqual(results['baseline']['native_calls'],results['reused']['native_calls'])
        self.assertLess(results['reused']['guard_clauses'],results['baseline']['guard_clauses'])
        self.assertEqual(results['reused']['guard_creates'],12)

    def test_stop_restart_and_legacy_snapshot_preserve_unfinished_union(self):
        p=SearchProblem(multiple_loops(),double_face());scope=WorkScope.for_problem(p)
        boxes=[(1<<a,1<<b) for a in range(6) for b in range(6)]
        s=PersistentSatSession(p,scope,boxes,guard_reuse=True,guard_limit=2)
        try:
            stopped=[False];first=s.advance(seconds=5,stop=lambda:stopped[0],progress=lambda e:stopped.__setitem__(0,True))
            self.assertEqual(first['status'],'stopped');saved=json.loads(json.dumps(first['checkpoint']))
            self.assertEqual(restore_frontier(saved['frontier'],scope),tuple(s.boxes))
            restored=PersistentSatSession.restore(p,saved)
            try:
                self.assertTrue(restored.guard_reuse);self.assertEqual(restored.guard_limit,2)
                self.assertIsNone(restored._guards)
                self.assertEqual(restored.advance(seconds=5)['status'],'infeasible')
            finally:restored.close()
            for key in ('guard_reuse','guard_limit'):saved.pop(key)
            saved['digest']=digest({k:v for k,v in saved.items() if k!='digest'})
            restored=PersistentSatSession.restore(p,saved)
            try:self.assertFalse(restored.guard_reuse)
            finally:restored.close()
        finally:s.close()

    def test_core_lifting_with_direct_literal_cap_is_sound(self):
        p=SearchProblem(*tetrahedron_tiles());scope=WorkScope.for_problem(p)
        boxes=[(1,1,1<<code,scope.root[3]) for code in range(12)]
        for limit in (0,1,8192):
            s=PersistentSatSession(p,scope,boxes,core_lifting=True,guard_reuse=True,guard_limit=limit)
            try:
                result=s.advance(seconds=5)
                self.assertEqual(result['status'],'infeasible');self.assertEqual(s.boxes,[])
                self.assertGreater(result['stats']['session_core_verified'],0)
            finally:s.close()

    def test_policy_toggle_does_not_disable_interned_guard(self):
        p=SearchProblem(simple_tiles(),double_face());scope=WorkScope.for_problem(p,'edges')
        s=PersistentSatSession(p,scope,guard_reuse=True)
        try:
            self.assertTrue(s.prepare(lambda:None,5));s.boxes=[(1,scope.root[1])];s._activate();prior=s.selector
            s.configure_guards(False);self.assertEqual(s.stats['session_guard_retirements'],0)
            s._activate();s.configure_guards(True);self.assertEqual(s.stats['session_guard_retirements'],1)
            s._activate();self.assertEqual(s.selector,prior)
            self.assertEqual(s.advance(seconds=5)['status'],'edge_perfect' if not p.validate([0,3])['valid'] else 'solved')
        finally:s.close()

    def test_public_solver_option_and_checkpoint_are_wired(self):
        p=SearchProblem(multiple_loops(),double_face());state={}
        try:
            result=solve(p,seconds=5,options={'sat_guard_reuse':True},learning_state=state)
            self.assertEqual(result['status'],'infeasible');self.assertTrue(result['checkpoint']['guard_reuse'])
        finally:state['_sat_session'].close()


def compare_guard_reuse():
    p=SearchProblem(multiple_loops(),double_face());scope=WorkScope.for_problem(p)
    boxes=[(1<<a,1<<b) for a in range(6) for b in range(6)];results={}
    for enabled in (False,True):
        s=PersistentSatSession(p,scope,boxes,guard_reuse=enabled)
        try:
            r=s.advance(seconds=5);v=r['stats']
            results['reused' if enabled else 'baseline']={'status':r['status'],'native_calls':v['session_native_calls'],
                'decisions':r['nodes'],'guard_clauses':v['session_guard_clauses'],'guard_creates':v['session_guard_creates'],
                'guard_hits':v['session_guard_hits'],'root_boxes':len(boxes),'remaining_boxes':len(s.boxes)}
        finally:s.close()
    return results

if __name__=='__main__':unittest.main()
