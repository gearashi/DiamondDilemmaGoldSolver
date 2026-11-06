"""Independent finite tests of complete look-ahead unions and interruption."""
from itertools import product
import copy,json,os,random,unittest
from unittest.mock import patch
import constraint_dfs as dfs
from search_options import BOOLEAN_OPTIONS
from search_problem import SearchProblem
from test_constraint_dfs import brute,tetrahedron_tiles


def counters():return dict.fromkeys(dfs.GUIDANCE_STAT_KEYS,0)


class ConstructiveUnionTests(unittest.TestCase):
    def test_shared_deduction_without_any_failed_alternative(self):
        domains=[3,7,7,7];stats=counters();failed=[]
        # Independent finite relation: x=0 permits y in {0,1}; x=1 only y=1.
        solutions={row for row in product(range(2),range(3),range(3),range(3))
                   if row[1] in ((0,1) if row[0]==0 else (1,))}
        def propagate(trial,cell):
            trial[1]&=3 if trial[cell]==1 else 2
            return True
        changed=dfs._constructive_lookahead(domains,0,[0,1],propagate,lambda:None,failed.append,stats)
        self.assertTrue(changed);self.assertEqual(domains,[3,3,7,7]);self.assertEqual(failed,[])
        self.assertEqual(stats['lookahead_failures'],0);self.assertEqual(stats['lookahead_common_prunes'],1)
        self.assertEqual(stats['lookahead_probes'],2);self.assertEqual(stats['lookahead_complete_passes'],1)
        self.assertTrue(all(all(domains[c]>>code&1 for c,code in enumerate(row)) for row in solutions))

    def test_fourth_alternative_prevents_false_common_deletion(self):
        domains=[15,3];stats=counters();seen=[]
        def propagate(trial,cell):
            code=trial[cell].bit_length()-1;seen.append(code);trial[1]=2 if code==3 else 1
            return True
        self.assertFalse(dfs._constructive_lookahead(domains,0,[0,1,2,3],propagate,lambda:None,lambda _:None,stats))
        self.assertEqual(domains,[15,3]);self.assertEqual(seen,[0,1,2,3])
        self.assertEqual(stats['lookahead_probes'],4)

    def test_incomplete_choice_partition_is_rejected_before_probing(self):
        for choices in ([0,1,2],[0,1,2,2]):
            with self.subTest(choices=choices),self.assertRaisesRegex(ValueError,'complete choice partition'):
                dfs._constructive_lookahead([15,3],0,choices,lambda *_:self.fail('unexpected probe'),lambda:None,lambda _:None,counters())

    def test_budget_or_stop_discards_union_and_keeps_only_proved_failures(self):
        for error in (TimeoutError,KeyboardInterrupt):
            domains=[15,7];stats=counters();failed=[]
            def propagate(trial,cell):
                code=trial[cell].bit_length()-1
                if code==0:return False
                if code==3:raise error('interrupted fourth alternative')
                trial[1]=1
                return True
            with self.subTest(error=error),self.assertRaises(error):
                dfs._constructive_lookahead(domains,0,[0,1,2,3],propagate,lambda:None,failed.append,stats)
            self.assertEqual(domains,[14,7]);self.assertEqual(failed,[0])
            self.assertEqual(stats['lookahead_common_prunes'],0);self.assertEqual(stats['lookahead_complete_passes'],0)
            self.assertEqual(stats['lookahead_failures'],1)

    def test_every_callback_boundary_keeps_all_relation_solutions(self):
        expected={(0,0),(1,0),(1,1),(2,1),(3,0)}
        for stop_at in range(1,10):
            domains=[15,7];stats=counters();calls=[0]
            def check():
                calls[0]+=1
                if calls[0]==stop_at:raise InterruptedError('stop')
            def propagate(trial,cell):
                x=trial[cell].bit_length()-1
                trial[1]=sum(1<<y for a,y in expected if a==x)
                return True
            try:dfs._constructive_lookahead(domains,0,[0,1,2,3],propagate,check,lambda _:None,stats)
            except InterruptedError:pass
            self.assertTrue(all(domains[0]>>x&1 and domains[1]>>y&1 for x,y in expected))

    def test_random_exact_relations_retain_all_and_only_supported_values(self):
        rng=random.Random(20261025)
        for _ in range(80):
            dimensions=(rng.randrange(2,5),3,3)
            all_rows=list(product(*(range(d) for d in dimensions)))
            allowed={row for row in all_rows if rng.randrange(4)}
            domains=[(1<<d)-1 for d in dimensions];stats=counters()
            def probe(trial,cell):
                x=trial[cell].bit_length()-1
                surviving=[row for row in allowed if row[0]==x]
                if not surviving:return False
                for c in range(3):trial[c]=sum(1<<v for v in {row[c] for row in surviving})
                return True
            dfs._constructive_lookahead(domains,0,list(range(dimensions[0])),probe,lambda:None,lambda _:None,stats)
            if allowed:
                expected=[sum(1<<v for v in {row[c] for row in allowed}) for c in range(3)]
                self.assertEqual(domains,expected)
            else:self.assertFalse(domains[0])

    def test_invalid_propagator_widening_does_not_publish_union(self):
        domains=[3,3]
        def probe(trial,_):trial[1]=7;return True
        with self.assertRaisesRegex(ValueError,'invalid domains'):
            dfs._constructive_lookahead(domains,0,[0,1],probe,lambda:None,lambda _:None,counters())
        self.assertEqual(domains,[3,3])


def cube_problem():
    neighbors=[set() for _ in range(8)]
    for i in range(4):
        for a,b in ((i,(i+1)%4),(i+4,(i+1)%4+4),(i,i+4)):
            neighbors[a].add(b);neighbors[b].add(a)
    ports=[{other:side for side,other in enumerate(sorted(row))} for row in neighbors]
    edges=[(a,ports[a][b],b,ports[b][a]) for a in range(8) for b in sorted(neighbors[a]) if a<b]
    from test_constraint_dfs import board
    problem=SearchProblem([{'id':i,'segments':[[[0,6],[1,6]]]} for i in range(8)],board(8,edges))
    problem.domains=[[3*i+r for r in range(3) if bits>>r&1] for i,bits in enumerate((3,7,7,7,7,7,7,6))]
    return problem


class ConstructiveIntegrationTests(unittest.TestCase):
    def test_tiny_actual_gold_and_edge_results_agree_with_independent_enumeration(self):
        rng=random.Random(31020)
        for case in range(10):
            tiles,board=tetrahedron_tiles();rng.shuffle(tiles)
            for tile in tiles:
                rotation=rng.randrange(3)
                tile['segments']=[[[((side+rotation)%3),p] for side,p in segment] for segment in tile['segments']]
            if case%2:tiles[0]['segments'][0][0][1]=1
            problem=SearchProblem(tiles,board)
            if case%3==1:problem.domains[case%4]=[rng.randrange(12)]
            for target in ('gold','edges'):
                expected=brute(problem,target)
                options={key:False for key in BOOLEAN_OPTIONS};options.update(native='python',lookahead=True,lcg=True,reason_tracking=True)
                with patch.object(dfs.time,'monotonic',return_value=1.):
                    result=dfs.solve(problem,seconds=5,target=target,seed=case,options=options)
                self.assertTrue(result['complete'],result)
                if expected:self.assertIn(tuple(result['codes']),expected)
                else:self.assertEqual(result['status'],'infeasible')
                self.assertGreaterEqual(result['stats']['propagations'],result['stats']['lookahead_probe_propagations'])


class ConstructivePuzzleIntegrationTests(unittest.TestCase):
    def options(self,native_search=False):
        result={key:False for key in BOOLEAN_OPTIONS}
        result.update(native='compiled' if native_search else 'python',native_search=native_search,
                      lookahead=True,lcg=True,reason_tracking=True)
        return result

    def native_available(self):
        from constraint_native_search import select_native_search
        engine,reason=select_native_search(8,24)
        if engine is None:
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':
                self.fail('Required native search is unavailable: '+str(reason))
            return False
        engine.close()
        return True

    def test_common_deductions_learn_only_original_root_valid_lcg_clauses(self):
        problem=cube_problem()
        expected={codes for codes in product(*problem.domains) if problem.validate(list(codes))['valid']}
        self.assertTrue(expected)
        for native_search in (False,True):
            if native_search and not self.native_available():continue
            observed=[];original=dfs.LCGState.record
            def capture(state,domains,deps,rule):
                # The real integration must record common deductions using the
                # actual path's pre-probe facts, never a trial's singleton facts.
                before=copy.deepcopy(state) if rule=='constructive look-ahead' else None
                added=original(state,domains,deps,rule)
                if before is not None and added:
                    new_facts=state.facts[len(before.facts):]
                    self.assertTrue(all(not fact.decision for fact in new_facts))
                    self.assertTrue(all(tuple(fact.parents)==tuple(before.current) for fact in new_facts))
                    self.assertEqual(state.level,before.level)
                    observed.append((before,list(domains)))
                return added
            with self.subTest(native_search=native_search),patch.object(dfs.LCGState,'record',capture),patch.object(dfs.time,'monotonic',return_value=1.):
                result=dfs.solve(problem,seconds=5,seed=0,options=self.options(native_search))
            self.assertIn(tuple(result['codes']),expected)
            self.assertGreater(result['stats']['lookahead_common_prunes'],0)
            self.assertTrue(observed)
            self.assertTrue(any(before.level>0 for before,_ in observed))
            if native_search:self.assertEqual(result['search_loop_backend'],'compiled')
            initial=observed[0][0].initial_domains
            store=dfs.LCGStore(problem.fingerprint,'gold',initial)
            learned_count=0
            for before,after in observed:
                actual_decisions=[(fact.cell,fact.mask) for fact in before.facts if fact.decision]
                compatible=[codes for codes in expected
                            if all(mask & (1<<codes[cell]) for cell,mask in actual_decisions)]
                self.assertTrue(compatible)
                self.assertTrue(all(all(mask & (1<<codes[cell]) for cell,mask in enumerate(after))
                                    for codes in compatible))
                # Challenge each published deletion in a separate legitimate
                # child of the PRE-deduction state. Reapplying that checked
                # consequence contradicts the selected value. This forces
                # first-UIP analysis after common deductions even when the
                # ordinary fixture finds its solution before a later conflict.
                for cell,(old,new) in enumerate(zip(before.domains,after)):
                    for code in dfs._values(old & ~new):
                        trial=copy.deepcopy(before).fork(cell,1<<code)
                        trial.record_cell(cell,0,range(len(after)),'checked common consequence')
                        clause=trial.first_uip()
                        self.assertIsNotNone(clause)
                        self.assertTrue(all(not all(mask & (1<<codes[c]) for c,mask in clause.literals)
                                            for codes in expected),(native_search,clause))
                        store.learn(clause);learned_count+=1
            self.assertGreater(learned_count,0)
            root=list(initial);store.prune(root,lambda:None)
            self.assertTrue(all(all(mask & (1<<codes[c]) for c,mask in enumerate(root)) for codes in expected))

    def test_pruning_preserves_every_independently_enumerated_cube_solution(self):
        problem=cube_problem()
        expected={codes for codes in product(*problem.domains) if problem.validate(list(codes))['valid']}
        self.assertTrue(expected)
        for native_search in (False,True):
            if native_search and not self.native_available():continue
            visited=[];original=dfs._constructive_lookahead
            def checked(domains,cell,choices,probe,check,on_failure,stats):
                before=domains.copy()
                answer=original(domains,cell,choices,probe,check,on_failure,stats)
                for codes in expected:
                    if all(before[c]>>code&1 for c,code in enumerate(codes)):
                        self.assertTrue(all(domains[c]>>code&1 for c,code in enumerate(codes)))
                visited.append((before,domains.copy()))
                return answer
            with self.subTest(native_search=native_search),patch.object(dfs,'_constructive_lookahead',side_effect=checked),patch.object(dfs.time,'monotonic',return_value=1.):
                result=dfs.solve(problem,seconds=5,seed=0,options=self.options(native_search))
            self.assertIn(tuple(result['codes']),expected)
            self.assertGreater(result['stats']['lookahead_common_prunes'],0)
            self.assertGreater(result['stats']['lookahead_complete_passes'],0)
            self.assertEqual(result['stats']['lookahead_probe_propagations'],result['stats']['lookahead_probes'])
            self.assertTrue(visited)
            if native_search:self.assertEqual(result['search_loop_backend'],'compiled')

    def test_actual_stop_during_surviving_probe_resumes_with_all_pending_work(self):
        for native_search in (False,True):
            if native_search and not self.native_available():continue
            problem=cube_problem();expected={codes for codes in product(*problem.domains) if problem.validate(list(codes))['valid']}
            original=dfs._constructive_lookahead;stop=[False];probes=[0]
            def interrupted(domains,cell,choices,probe,check,on_failure,stats):
                def wrapped(trial,selected):
                    answer=probe(trial,selected);probes[0]+=1
                    if probes[0]==2:stop[0]=True
                    return answer
                return original(domains,cell,choices,wrapped,check,on_failure,stats)
            with patch.object(dfs,'_constructive_lookahead',side_effect=interrupted),patch.object(dfs.time,'monotonic',return_value=1.):
                first=dfs.solve(problem,seconds=5,seed=0,options=self.options(native_search),stop=lambda:stop[0])
            self.assertEqual(first['status'],'stopped');self.assertEqual(probes[0],2)
            self.assertEqual(first['stats']['lookahead_common_prunes'],0)
            saved=json.loads(json.dumps(first['checkpoint']))
            # For every original Gold solution at least one pending rectangle
            # still contains it; the interrupted root had no completed leaves.
            self.assertTrue(all(any(all(int(value,16)>>code&1 for value,code in zip(frame['domains'],codes))
                                        for frame in saved['stack']) for codes in expected))
            with patch.object(dfs.time,'monotonic',return_value=1.):
                last=dfs.solve(problem,seconds=5,resume=saved,options=self.options(native_search))
            self.assertTrue(last['validation']['valid']);self.assertIn(tuple(last['codes']),expected)


if __name__=='__main__':unittest.main()
