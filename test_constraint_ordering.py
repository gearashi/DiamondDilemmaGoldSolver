"""Ordering changes visits, never the finite set of legal choices."""
import json,random,unittest
from unittest.mock import patch
from constraint_ordering import SupportOrdering,STAT_KEYS
import constraint_dfs as dfs
from search_options import BOOLEAN_OPTIONS
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles,double_face,simple_tiles,brute
from validator import reverse_mask

class SupportOrderingTests(unittest.TestCase):
    def helper(self,problem):
        domains=[sum(1<<c for c in row) for row in problem.domains]
        return domains,SupportOrdering(dfs._build_support_tables(problem,domains,'gold',lambda:None)),dict.fromkeys(STAT_KEYS,0)
    def test_raw_masks_joint_parallel_seams_hints_and_widened_neighbors(self):
        problem=SearchProblem(simple_tiles(),double_face());domains,order,stats=self.helper(problem)
        rank={c:c for c in range(6)};choices=list(dfs._values(domains[0]))
        def expected(hinted):
            scores={}
            for code in choices:
                support=[b for b in dfs._values(domains[1]) if b//3!=code//3 and all(int(problem.masks[code,s])==reverse_mask(int(problem.masks[b,s])) for s in range(3))]
                scores[code]=len(support)
            return sorted(choices,key=lambda c:(0,hinted[c],0) if c in hinted else (1,-scores[c],rank[c]))
        for neighbor in (domains[1],1<<3,domains[1]):
            domains[1]=neighbor
            for hinted in ({},{choices[-1]:0}):
                self.assertEqual(order.order(0,domains,choices,hinted,rank,lambda:None,stats),expected(hinted))
        self.assertGreater(stats['value_order_cache_hits'],0)
        self.assertEqual(set(order.order(0,domains,choices[:2],{},rank,lambda:None,stats)),set(choices[:2]))
        self.assertEqual(set(order.order(0,domains,choices[2:],{},rank,lambda:None,stats)),set(choices[2:]))
        def cancel():raise InterruptedError()
        with self.assertRaises(InterruptedError):order.order(0,domains,choices,{},rank,cancel,stats)
    def test_bounded_fallback_and_lru_preserve_all_values(self):
        problem=SearchProblem(*tetrahedron_tiles());domains,order,stats=self.helper(problem)
        rank={c:11-c for c in range(12)};choices=list(dfs._values(domains[0]));order.max_choices=2
        self.assertEqual(order.order(0,domains,choices,{},rank,lambda:None,stats),sorted(choices,key=rank.get))
        order.max_choices=48;order.limit=2
        for neighbor in (1,2,4,8):
            domains[1]=neighbor;self.assertEqual(set(order.order(0,domains,choices,{},rank,lambda:None,stats)),set(choices))
        self.assertLessEqual(len(order.cache),2)
    def test_tiny_gold_and_edges_agree_with_independent_enumeration(self):
        rng=random.Random(711)
        for case in range(12):
            tiles,board=tetrahedron_tiles();rng.shuffle(tiles)
            if case%3==1:tiles[0]['segments'][0][0][1]=1
            problem=SearchProblem(tiles,board)
            if case%4==2:problem.domains[0]=rng.sample(problem.domains[0],6)
            for target in ('gold','edges'):
                expected=brute(problem,target)
                result=dfs.solve(problem,seconds=5,target=target,seed=case,options={'value_ordering':True,'reason_tracking':True})
                self.assertTrue(result['complete'])
                if expected:self.assertIn(tuple(result['codes']),expected)
                else:self.assertEqual(result['status'],'infeasible')
    def test_pause_and_switch_ordering_preserve_saved_siblings(self):
        problem=SearchProblem(*tetrahedron_tiles());seen=[];original=SupportOrdering.order
        def order(*args,**kwargs):
            value=original(*args,**kwargs);seen.append(value);return value
        opts={k:False for k in BOOLEAN_OPTIONS};opts.update(value_ordering=True,native='python')
        with patch.object(SupportOrdering,'order',order):
            first=dfs.solve(problem,seconds=5,options=opts,stop=lambda:bool(seen))
        self.assertEqual(first['status'],'stopped');saved=json.loads(json.dumps(first['checkpoint']))
        self.assertTrue(any(frame['choices'] for frame in saved['stack']))
        unchanged=dfs.solve(problem,seconds=0,resume=saved,options={**opts,'value_ordering':False})
        self.assertEqual(saved['stack'],unchanged['checkpoint']['stack'])
        final=dfs.solve(problem,seconds=5,resume=unchanged['checkpoint'],options={**opts,'value_ordering':False})
        self.assertTrue(final['complete']);self.assertIn(tuple(final['codes']),brute(problem,'gold'))

if __name__=='__main__':unittest.main()
