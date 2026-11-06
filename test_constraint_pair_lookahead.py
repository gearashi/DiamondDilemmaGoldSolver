"""Independent pair-pruning, scoped-learning and reusable completed-probe checks."""
from itertools import product
import copy,json,os,unittest
from unittest.mock import patch
import constraint_dfs as dfs
from constraint_lookahead import CompletedProbeCache,paired_lookahead,CACHE_STAT_KEYS,PAIR_STAT_KEYS
from search_options import BOOLEAN_OPTIONS
from search_problem import SearchProblem
from test_constraint_dfs import tetrahedron_tiles
from test_constraint_lookahead import cube_problem


def stats():return dict.fromkeys(dfs.GUIDANCE_STAT_KEYS,0)

def turn_problem(satisfiable=True):
    _,board=tetrahedron_tiles()
    tiles=[{'id':i,'segments':[[[0,2],[0,10]],[[1,2],[2,2]],[[1,10],[2,10]]]} for i in range(4)]
    if satisfiable:tiles[0]['segments']=[[[0,2],[1,2]],[[0,10],[2,2]],[[1,10],[2,10]]]
    problem=SearchProblem(tiles,board)
    problem.domains=[[3*i+r for r in range(3)] for i in range(4)]
    return problem

def exact_solutions(problem,target='gold'):
    return {codes for codes in product(*problem.domains)
            if ((report:=problem.validate(list(codes)))['valid'] or
                target=='edges' and report['unique_tiles'] and report['matched_edges']==len(problem.edges))}

def options(pairs=True,native=False):
    result=dict.fromkeys(BOOLEAN_OPTIONS,False)
    result.update(native='compiled' if native else 'python',native_search=native,lookahead=True,
                  pair_lookahead=pairs,lookahead_ms=1000)
    return result

class CompletedProbeTests(unittest.TestCase):
    def test_complete_results_reused_without_mutable_aliasing(self):
        cache=CompletedProbeCache();counts=stats();calls=[]
        def probe(domains,cell):calls.append(tuple(domains));domains[1]&=3;return True
        first=[1,7];self.assertTrue(cache.run(('root','gold'),first,0,probe,lambda:None,counts))
        first[1]=0
        second=[1,7];self.assertTrue(cache.run(('root','gold'),second,0,probe,lambda:None,counts))
        self.assertEqual(second,[1,3]);self.assertEqual(len(calls),1)
        self.assertEqual(counts['lookahead_cache_hits'],1)

    def test_widening_and_all_context_changes_require_new_proof(self):
        cache=CompletedProbeCache();counts=stats();calls=[]
        def probe(domains,cell):calls.append((tuple(domains),cell));return True
        for context,domains in [(('root','gold','pathA',100),[1,3]),(('root','gold','pathA',100),[1,7]),
                                (('root','edges','pathA',100),[1,3]),(('root','gold','pathB',100),[1,3]),
                                (('other-root','gold','pathA',100),[1,3]),(('root','gold','pathA',20),[1,3])]:
            cache.run(context,domains,0,probe,lambda:None,counts)
        self.assertEqual(len(calls),6);self.assertEqual(counts['lookahead_cache_hits'],0)

    def test_interrupted_or_malformed_probe_is_not_cached(self):
        cache=CompletedProbeCache();counts=stats()
        def interrupted(domains,cell):domains[1]=1;raise InterruptedError('unfinished proof')
        with self.assertRaises(InterruptedError):cache.run('scope',[1,3],0,interrupted,lambda:None,counts)
        self.assertFalse(cache.entries)
        def invalid(domains,cell):domains[1]=7;return True
        with self.assertRaises(ValueError):cache.run('scope',[1,3],0,invalid,lambda:None,counts)
        self.assertFalse(cache.entries)
        calls=[]
        def fail(domains,cell):calls.append(1);domains[1]=0;return False
        for _ in range(2):self.assertFalse(cache.run('scope',[1,3],0,fail,lambda:None,counts))
        self.assertEqual(len(calls),1)

    def test_cache_bounds_evict_only_optional_completed_work(self):
        cache=CompletedProbeCache(limit=2,max_slots=8);counts=stats()
        for mask in (1,3,7):cache.run('scope',[1,mask],0,lambda *args:True,lambda:None,counts)
        self.assertLessEqual(cache.slots,8);self.assertEqual(len(cache.entries),2)
        self.assertEqual(counts['lookahead_cache_evictions'],1)

    def test_cancelled_partition_reuses_completed_probes_without_partial_union(self):
        cache=CompletedProbeCache();counts=stats();completed=[];stop=[True]
        def raw(trial,cell):
            code=trial[cell].bit_length()-1
            if code==1 and stop[0]:raise InterruptedError('second alternative unfinished')
            completed.append(code);trial[1]=1 if code==0 else 2;return True
        def probe(trial,cell):return cache.run('same-root-path-options',trial,cell,raw,lambda:None,counts)
        domains=[3,7]
        with self.assertRaises(InterruptedError):dfs._constructive_lookahead(domains,0,[0,1],probe,lambda:None,lambda code:None,counts)
        self.assertEqual(domains,[3,7]);self.assertEqual(completed,[0])
        stop[0]=False
        dfs._constructive_lookahead(domains,0,[0,1],probe,lambda:None,lambda code:None,counts)
        self.assertEqual(domains,[3,3]);self.assertEqual(completed,[0,1])
        self.assertEqual(counts['lookahead_cache_hits'],1)

class PairRoutineTests(unittest.TestCase):
    def test_independent_relation_pair_rules_preserve_every_solution(self):
        solutions={row for row in product(range(2),repeat=4) if row[0]!=0 or row[1]!=0}
        domains=[3]*4;counts=stats();failed=[]
        def pair(trial,cell):return not (trial[0]==1 and trial[1]==1)
        paired_lookahead(domains,0,[0,1],lambda *args:True,pair,lambda:None,
                         lambda a,x,b,y:failed.append(((a,x),(b,y))),lambda _:None,counts)
        self.assertEqual(failed,[((0,0),(1,0))]);self.assertEqual(domains,[3]*4)
        self.assertTrue(all(not all(row[c]==v for c,v in clause) for row in solutions for clause in failed))

    def test_pair_budget_cannot_prune_an_incomplete_failed_row(self):
        domains=[3,3];counts=stats()
        paired_lookahead(domains,0,[0,1],lambda *args:True,lambda *args:False,lambda:None,
                         lambda *args:None,lambda _:self.fail('unfinished row'),counts,max_pairs=1)
        self.assertEqual(domains,[3,3]);self.assertEqual(counts['pair_lookahead_complete_passes'],0)

    def test_stop_preserves_previously_completed_pair_rows(self):
        domains=[3,3];counts=stats();first=[];calls=[0]
        def pair(trial,cell):
            calls[0]+=1
            if calls[0]==3:raise InterruptedError('second row unfinished')
            return False
        with self.assertRaises(InterruptedError):
            paired_lookahead(domains,0,[0,1],lambda *args:True,pair,lambda:None,
                             lambda *args:None,first.append,counts)
        self.assertEqual(domains,[2,3]);self.assertEqual(first,[0])

class ActualPairIntegrationTests(unittest.TestCase):
    def require_native(self):
        engine,reason=dfs.select_native_search(4,12)
        if engine is None:
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':self.fail(str(reason))
            return False
        engine.close();return True

    def test_real_pair_contradiction_requires_ancestor_and_preserves_all_gold_solutions(self):
        problem=turn_problem();expected=exact_solutions(problem);self.assertEqual(len(expected),42)
        for native in (False,True):
            if native and not self.require_native():continue
            captures=[];original=dfs.paired_lookahead;learning={}
            def wrapped(domains,cell,choices,single,pair,check,on_failure,on_first,counters,**kwargs):
                before=domains.copy()
                def failed(a,x,b,y):
                    singles=[]
                    for c,value in ((a,x),(b,y)):
                        trial=before.copy();trial[c]=1<<value;singles.append(single(trial,c))
                    captures.append((a,x,b,y,all(singles)))
                    on_failure(a,x,b,y)
                return original(domains,cell,choices,single,pair,check,failed,on_first,counters,**kwargs)
            with self.subTest(native=native),patch.object(dfs,'paired_lookahead',side_effect=wrapped),patch.object(dfs.time,'monotonic',return_value=1.):
                result=dfs.solve(problem,seconds=5,seed=0,options=options(native=native),learning_state=learning)
            self.assertIn(tuple(result['codes']),expected)
            # At least one pair survives both independent singleton checks and
            # is legal in another ancestor context: retaining that context is essential.
            self.assertTrue(any(both and any(row[a]==x and row[b]==y for row in expected)
                                for a,x,b,y,both in captures))
            clauses=learning['_nogoods'].clauses;self.assertTrue(clauses)
            self.assertTrue(all(not all(row[c]==v for c,v in clause) for row in expected for clause in clauses))
            self.assertGreater(result['stats']['pair_lookahead_failures'],0)
            if native:self.assertEqual(result['search_loop_backend'],'compiled')

    def test_actual_unsat_search_uses_fewer_nodes_and_edge_mode_retains_every_board(self):
        problem=turn_problem(False);self.assertFalse(exact_solutions(problem))
        answers=[]
        for enabled in (False,True):
            with patch.object(dfs.time,'monotonic',return_value=1.):
                answers.append(dfs.solve(problem,seconds=5,seed=0,options=options(enabled)))
        self.assertTrue(all(answer['status']=='infeasible' for answer in answers))
        self.assertLess(answers[1]['nodes'],answers[0]['nodes'])
        edges=exact_solutions(problem,'edges');self.assertEqual(len(edges),81)
        learning={}
        with patch.object(dfs.time,'monotonic',return_value=1.):
            answer=dfs.solve(problem,seconds=5,target='edges',options=options(),learning_state=learning)
        self.assertIn(tuple(answer['codes']),edges);self.assertEqual(answer['stats']['pair_lookahead_failures'],0)
        self.assertTrue(all(not all(row[c]==v for c,v in clause) for row in edges for clause in learning['_nogoods'].clauses))

    def test_stop_resume_reuses_completed_probes_on_python_and_native_paths(self):
        for native in (False,True):
            if native and not self.require_native():continue
            problem=cube_problem();expected=exact_solutions(problem);learning={};stop=[False];calls=[0]
            original=dfs._constructive_lookahead
            def interrupted(domains,cell,choices,probe,check,on_failure,counters):
                def wrapped(trial,selected):
                    answer=probe(trial,selected);calls[0]+=1
                    if calls[0]==2:stop[0]=True
                    return answer
                return original(domains,cell,choices,wrapped,check,on_failure,counters)
            opts=options(False,native)
            with patch.object(dfs,'_constructive_lookahead',side_effect=interrupted),patch.object(dfs.time,'monotonic',return_value=1.):
                first=dfs.solve(problem,seconds=5,options=opts,learning_state=learning,stop=lambda:stop[0])
            self.assertEqual(first['status'],'stopped');self.assertEqual(first['stats']['lookahead_common_prunes'],0)
            self.assertGreaterEqual(len(learning['_runtime'].probes.entries),2)
            checkpoint=json.loads(json.dumps(first['checkpoint']))
            self.assertTrue(all(any(all(int(mask,16)&(1<<code) for mask,code in zip(frame['domains'],row))
                                       for frame in checkpoint['stack']) for row in expected))
            with patch.object(dfs.time,'monotonic',return_value=1.):
                hot=dfs.solve(problem,seconds=5,resume=copy.deepcopy(checkpoint),options=opts,learning_state=learning)
                cold=dfs.solve(problem,seconds=5,resume=copy.deepcopy(checkpoint),options=opts)
            self.assertIn(tuple(hot['codes']),expected);self.assertIn(tuple(cold['codes']),expected)
            self.assertGreater(hot['stats']['lookahead_cache_hits'],cold['stats']['lookahead_cache_hits'])
            self.assertLess(hot['stats']['lookahead_probe_propagations'],cold['stats']['lookahead_probe_propagations'])

    def test_stop_inside_pair_probe_keeps_all_gold_completions_pending(self):
        for native in (False,True):
            if native and not self.require_native():continue
            problem=turn_problem();expected=exact_solutions(problem);stop=[False];original=dfs.paired_lookahead
            def interrupted(domains,cell,choices,single,pair,check,on_failure,on_first,counters,**kwargs):
                def wrapped(trial,selected):
                    answer=pair(trial,selected);stop[0]=True;return answer
                return original(domains,cell,choices,single,wrapped,check,on_failure,on_first,counters,**kwargs)
            learning={};opts=options(native=native)
            with patch.object(dfs,'paired_lookahead',side_effect=interrupted),patch.object(dfs.time,'monotonic',return_value=1.):
                first=dfs.solve(problem,seconds=5,options=opts,learning_state=learning,stop=lambda:stop[0])
            self.assertEqual(first['status'],'stopped')
            self.assertTrue(all(any(all(int(mask,16)&(1<<code) for mask,code in zip(frame['domains'],row))
                                       for frame in first['checkpoint']['stack']) for row in expected))
            with patch.object(dfs.time,'monotonic',return_value=1.):
                final=dfs.solve(problem,seconds=5,resume=first['checkpoint'],options=opts,learning_state=learning)
            self.assertIn(tuple(final['codes']),expected)

if __name__=='__main__':unittest.main()
