"""Exact native/Python equivalence and truthful backend/stop behavior."""
from collections import deque
from dataclasses import replace
import os,random,unittest
from unittest.mock import patch
import constraint_dfs as dfs
import constraint_native as native
from search_problem import SearchProblem
from search_fixtures import planted_gold
from test_constraint_dfs import simple_tiles,double_face

def counters():return dict.fromkeys((*native.STAT_KEYS,'arc_revisions'),0)
def reference(tables,domains,initial=None):
    queue=deque(range(len(tables.arcs)) if initial is None else initial);pending=set(queue)
    if not all(domains):return False
    while queue:
        index=queue.popleft();pending.remove(index)
        a,sa,b,sb=tables.arcs[index]
        value=dfs._filter_arc(domains[a],domains[b],tables.groups[sa][sb],tables.compatible[sa][sb],tables.tile_bits)
        if value!=domains[a]:
            domains[a]=value
            if not value:return False
            for other in tables.incoming[a]:
                if other not in pending:queue.append(other);pending.add(other)
    return True

def independent_events(problem,tables,domains,initial):
    """Raw mask reversal and pair enumeration; no support-table/filter reuse."""
    def reverse(value):
        return sum(((int(value)>>i)&1)<<(10-i) for i in range(11))
    queue=deque(initial);pending=set(queue);events=[];count=3*problem.n
    while queue:
        index=queue.popleft();pending.remove(index)
        a,sa,b,sb=tables.arcs[index]
        value=sum(1<<code for code in range(count) if domains[a]>>code&1 and
                  any(domains[b]>>other&1 and code//3!=other//3 and
                      int(problem.masks[code,sa])==reverse(problem.masks[other,sb])
                      for other in range(count)))
        if value!=domains[a]:
            domains[a]=value;events.append((a,value,b,tuple(domains)))
            if not value:return False,events
            for other in tables.incoming[a]:
                if other not in pending:queue.append(other);pending.add(other)
    return True,events

class NativeAvailabilityTests(unittest.TestCase):
    def test_auto_fallback_is_truthful_and_explicit_compiled_fails(self):
        with patch.object(native,'load_library',side_effect=OSError('unavailable test build')):
            engine,reason=native.select_native(None,'auto')
            self.assertIsNone(engine);self.assertIn('unavailable test build',reason)
            with self.assertRaisesRegex(RuntimeError,'build_native.py'):native.select_native(None,'compiled')
        with patch.object(native,'load_library',side_effect=AssertionError('No load for Python')):
            self.assertIsNone(native.select_native(None,'python')[0])
    def test_graph_fallback_and_explicit_python_mode(self):
        with patch.object(native,'load_library',side_effect=OSError('test missing build')):
            self.assertIsNone(native.select_graph_helpers('auto')[0])
            with self.assertRaisesRegex(RuntimeError,'build_native.py'):native.select_graph_helpers('compiled')
        with patch.object(native,'load_library',side_effect=AssertionError('No load')):
            self.assertIsNone(native.select_graph_helpers('python')[0])

    def test_hall_rejects_unsupported_byte_order_before_dispatch(self):
        with patch.object(native.sys,'byteorder','big'):
            self.assertIsNone(native.select_hall('auto')[0])
            with self.assertRaisesRegex(RuntimeError,'little-endian'):native.select_hall('compiled')

    def test_manifest_rejects_path_escape_and_stale_source(self):
        for metadata in ({'abi':1,'platform':native.sys.platform,'library':'../other.dll'},
                         {'abi':1,'platform':native.sys.platform,'library':'safe.dll','source_sha256':'0'*64}):
            import json
            with patch.object(native.Path,'read_text',return_value=json.dumps(metadata)):
                with self.assertRaises(RuntimeError):native.load_library()

class NativeKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:cls.library=native.load_library()
        except (OSError,ValueError,RuntimeError) as exc:
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':raise
            raise unittest.SkipTest('Optional native library not built: '+str(exc))
    def tables(self,problem):
        return dfs._build_support_tables(problem,tuple(sum(1<<c for c in row) for row in problem.domains),'gold',lambda:None)
    def test_every_tiny_domain_pair_and_side_pair_matches_independent_kernel(self):
        problem=SearchProblem(simple_tiles(),double_face());base=self.tables(problem)
        for sa in range(3):
            for sb in range(3):
                tables=replace(base,arcs=((0,sa,1,sb),),incoming=((),(0,)))
                engine=native.NativePropagation(tables,self.library)
                for left in range(1,64):
                    for right in range(1,64):
                        expected=[left,right];actual=expected.copy()
                        passed=reference(tables,expected)
                        stats=counters();found=engine.propagate(actual,[0],lambda:None,stats)
                        self.assertEqual((found,actual),(passed,expected),(sa,sb,left,right))
                        self.assertGreater(stats['native_calls'],0)
    def test_seeded_small_and_full_size_queue_fixpoints_match(self):
        rng=random.Random(40421)
        for size in (1,2,4):
            data,board,solution=planted_gold(197,size,extra_edges=0)
            problem=SearchProblem(data,board);tables=self.tables(problem)
            engine=native.NativePropagation(tables,self.library)
            for case in range(12):
                rows=[]
                for cell,domain in enumerate(problem.domains):
                    choices=domain if case==0 else sorted(set(rng.sample(domain,min(len(domain),case*3)))|{solution[cell]})
                    rows.append(sum(1<<code for code in choices))
                expected=rows.copy();actual=rows.copy();passed=reference(tables,expected)
                found=engine.propagate(actual,range(len(tables.arcs)),lambda:None,counters(),batch=7)
                self.assertEqual((found,actual),(passed,expected),(size,case))
                self.assertTrue(all(value&(1<<code) for value,code in zip(actual,solution)))
    def test_cancel_after_native_batch_commits_only_sound_reductions(self):
        tables=self.tables(SearchProblem(simple_tiles(),double_face()));engine=native.NativePropagation(tables,self.library)
        original=[63,8];actual=original.copy();calls=[0]
        class Cancel(Exception):pass
        def stop():
            calls[0]+=1
            if calls[0]==2:raise Cancel
        stats=counters()
        with self.assertRaises(Cancel):engine.propagate(actual,range(len(tables.arcs)),stop,stats,batch=1)
        self.assertEqual(stats['native_calls'],1);self.assertNotEqual(actual,original)
        expected=original.copy();passed=reference(tables,expected)
        found=engine.propagate(actual,range(len(tables.arcs)),lambda:None,stats,batch=1)
        self.assertEqual((found,actual),(passed,expected))
    def test_malformed_domains_and_queues_fail_before_native_dispatch(self):
        tables=self.tables(SearchProblem(simple_tiles(),double_face()));engine=native.NativePropagation(tables,self.library)
        for domains,queue in (([1],[]),([-1,8],[0]),([64,8],[0]),([True,8],[0]),([1,8],[0,0]),([1,8],[100])):
            with self.assertRaises(ValueError):engine.propagate(domains,queue,lambda:None,counters())
        self.assertFalse(engine.propagate([0,8],[0],lambda:None,counters()))
    def test_verified_native_buffers_reuse_across_live_slices(self):
        from test_smart_dfs import mixed_loop_problem,LEGACY_OPTIONS
        problem=mixed_loop_problem();cache={};seen=[]
        first=dfs.solve(problem,seconds=3,hints=[0,3],learning_state=cache,
                        options={**LEGACY_OPTIONS,'native':'compiled'},stop=lambda:bool(seen),
                        progress=lambda event:seen.append(event) if event['event']=='candidate' else None)
        self.assertEqual(first['status'],'stopped')
        engine=cache['_native']
        with patch.object(dfs,'select_native',side_effect=AssertionError('Reuse verified native buffers')):
            result=dfs.solve(problem,seconds=3,resume=first['checkpoint'],learning_state=cache)
        self.assertEqual(result['status'],'solved')
        self.assertIs(cache['_native'],engine)
        with patch.object(native,'build_signature',return_value=('changed',)):
            self.assertFalse(engine.matches(cache['_support']))
        cache['_native']={'serialized':'never trusted'}
        with self.assertRaisesRegex(ValueError,'native propagation cache'):
            dfs.solve(problem,seconds=0,learning_state=cache)

    def test_remembered_supports_are_rechecked_after_widening_and_stale_hints(self):
        problem=SearchProblem(simple_tiles(),double_face());tables=self.tables(problem)
        engine=native.NativePropagation(tables,self.library)
        self.assertTrue(engine.use_residues)
        total_hits=0
        for rows in ([63,63],[1,8],[63,63],[2,16],[63,8],[63,63]):
            expected=list(rows);actual=list(rows)
            passed=reference(tables,expected);stats=counters()
            found=engine.propagate(actual,range(len(tables.arcs)),lambda:None,stats,batch=1)
            self.assertEqual((found,actual),(passed,expected))
            total_hits+=stats['native_support_hits']
        self.assertGreater(total_hits,0)
        # Even a stale in-range hint is checked against the immutable support
        # relation as well as the neighbor domain; it cannot create support.
        for hint in (0,1,5,32000,-32768):
            for i in range(len(engine.residues)):engine.residues[i]=hint
            for rows in ([1,1],[2,8],[63,63]):
                expected=list(rows);actual=list(rows)
                passed=reference(tables,expected)
                found=engine.propagate(actual,range(len(tables.arcs)),lambda:None,counters())
                self.assertEqual((found,actual),(passed,expected),(hint,rows))

    def test_cached_and_uncached_native_paths_reach_identical_full_size_fixpoints(self):
        data,board,solution=planted_gold(711,4,extra_edges=0)
        problem=SearchProblem(data,board);tables=self.tables(problem)
        cached=native.NativePropagation(tables,self.library)
        uncached=native.NativePropagation(tables,self.library);uncached.use_residues=False
        rng=random.Random(551)
        for count in (480,10,30,480,2,25):
            domains=[sum(1<<c for c in set(rng.sample(row,min(count,len(row))))|{solution[i]})
                     for i,row in enumerate(problem.domains)]
            a=domains.copy();b=domains.copy()
            left=cached.propagate(a,range(len(tables.arcs)),lambda:None,counters(),batch=11)
            right=uncached.propagate(b,range(len(tables.arcs)),lambda:None,counters(),batch=11)
            self.assertEqual((left,a),(right,b))
            self.assertTrue(all(v&(1<<c) for v,c in zip(a,solution)))

    def test_legacy_export_preserves_original_two_counter_boundary(self):
        problem=SearchProblem(simple_tiles(),double_face());tables=self.tables(problem)
        engine=native.NativePropagation(tables,self.library);engine._sync([63,8],counters())
        engine.queue[0]=0;engine.pending[0]=1;engine.state[:]=(0,1,1)
        # Last two words model adjacent caller storage outside the ABI buffer.
        guard=(native.C.c_uint64*4)(0,0,0x123456789,0xabcdef012)
        self.library.dd_arc_batch(engine.buffer,engine.support,engine.arcs,engine.incoming,
            engine.n,engine.count,engine.words,engine.arc_count,engine.queue,engine.pending,
            engine.state,1,guard)
        self.assertEqual(tuple(guard)[2:],(0x123456789,0xabcdef012))

    def test_workspace_reuses_buffers_and_transfers_only_changed_rows(self):
        problem=SearchProblem(simple_tiles(),double_face());tables=self.tables(problem)
        engine=native.NativePropagation(tables,self.library)
        names=('buffer','queue','pending','state','counts','dirty')
        identities=tuple(id(getattr(engine,name)) for name in names)
        rows=[63,63];stats=counters()
        self.assertTrue(engine.propagate(rows,range(len(tables.arcs)),lambda:None,stats))
        first=stats['native_input_rows']
        engine.propagate(rows,range(len(tables.arcs)),lambda:None,stats)
        self.assertEqual(stats['native_input_rows'],first)
        rows[0]=1
        engine.propagate(rows,tables.incoming[0],lambda:None,stats)
        self.assertEqual(stats['native_input_rows']-first,1)
        self.assertEqual(identities,tuple(id(getattr(engine,name)) for name in names))
        self.assertIsNone(engine._event_arcs)
        self.assertEqual(stats['native_reason_events'],0)

    def test_ordered_events_match_independent_mask_oracle_exhaustively(self):
        problem=SearchProblem(simple_tiles(),double_face());tables=self.tables(problem)
        engine=native.NativePropagation(tables,self.library)
        for left in range(1,64):
            for right in range(1,64):
                expected=[left,right];actual=expected.copy();queue=range(len(tables.arcs))
                passed,events=independent_events(problem,tables,expected,queue)
                received=[]
                def reason(a,value,b):received.append((a,value,b,tuple(actual)))
                stats=counters()
                found=engine.propagate(actual,queue,lambda:None,stats,batch=3,reason=reason)
                self.assertEqual((found,actual,received),(passed,expected,events),(left,right))
                self.assertEqual(stats['native_reason_events'],len(events))

    def test_events_random_asymmetric_domains_and_legacy_exports(self):
        data,board,solution=planted_gold(1801,1,extra_edges=0)
        problem=SearchProblem(data,board);tables=self.tables(problem);rng=random.Random(881)
        engines=[native.NativePropagation(tables,self.library) for _ in range(3)]
        engines[1].use_workspace=False;engines[2].use_workspace=False;engines[2].use_residues=False
        total=0
        for case in range(12):
            rows=[sum(1<<code for code in set(rng.sample(row,min(7,len(row))))|{solution[cell]})
                  for cell,row in enumerate(problem.domains)]
            if case%3==0:rows[case%problem.n]=1<<solution[case%problem.n]
            expected=rows.copy();queue=range(len(tables.arcs))
            passed,events=independent_events(problem,tables,expected,queue);total+=len(events)
            for engine in engines:
                actual=rows.copy();received=[]
                found=engine.propagate(actual,queue,lambda:None,counters(),batch=7,
                    reason=lambda a,v,b:received.append((a,v,b,tuple(actual))))
                self.assertEqual((found,actual,received),(passed,expected,events))
        self.assertGreater(total,10)

    def test_callback_cancel_mid_batch_commits_prefix_and_restores_siblings(self):
        data,board,solution=planted_gold(1801,1,extra_edges=0)
        problem=SearchProblem(data,board);tables=self.tables(problem)
        engine=native.NativePropagation(tables,self.library)
        initial=[sum(1<<code for code in row) for row in problem.domains];initial[0]=1<<solution[0]
        expected=initial.copy();queue=range(len(tables.arcs))
        passed,events=independent_events(problem,tables,expected,queue)
        self.assertGreater(len(events),2)
        actual=initial.copy();seen=[]
        class Cancel(Exception):pass
        def reason(a,v,b):
            seen.append((a,v,b,tuple(actual)))
            if len(seen)==2:raise Cancel
        with self.assertRaises(Cancel):engine.propagate(actual,queue,lambda:None,counters(),batch=128,reason=reason)
        self.assertEqual(seen,events[:2]);self.assertEqual(tuple(actual),events[1][3])
        self.assertTrue(all(v is None for v in engine._shadow));self.assertFalse(engine._active)
        for rows in (actual,initial,[sum(1<<c for c in row) for row in problem.domains],initial):
            expected=rows.copy();expected_passed=reference(tables,expected);candidate=rows.copy()
            found=engine.propagate(candidate,queue,lambda:None,counters())
            self.assertEqual((found,candidate),(expected_passed,expected))

    def test_event_capacity_is_bounded_and_malformed_batch_is_not_exposed(self):
        problem=SearchProblem(simple_tiles(),double_face());tables=self.tables(problem)
        engine=native.NativePropagation(tables,self.library);engine._events(3)
        self.assertEqual(engine._event_capacity,3)
        for count,index,value in ((4,0,1),(1,999,1),(1,0,64),(1,0,63)):
            engine._event_count.value=count;engine.counts[1]=count
            engine._event_arcs[0]=index;engine._event_domains[0]=value
            rows=[63,8];calls=[]
            with self.assertRaises(RuntimeError):engine._replay(rows,lambda *args:calls.append(args),3)
            self.assertEqual(rows,[63,8]);self.assertEqual(calls,[])
        with self.assertRaises(ValueError):engine.propagate([63,8],[0],lambda:None,counters(),reason=True)
        with self.assertRaises(ValueError):engine.propagate([63,8],[0],lambda:None,counters(),batch=4097)

    def test_reentrant_callback_is_rejected_then_workspace_recovers(self):
        problem=SearchProblem(simple_tiles(),double_face());tables=self.tables(problem)
        engine=native.NativePropagation(tables,self.library);rows=[63,8]
        def recurse(*event):engine.propagate([63,63],[],lambda:None,counters())
        with self.assertRaisesRegex(RuntimeError,'already active'):
            engine.propagate(rows,range(len(tables.arcs)),lambda:None,counters(),reason=recurse)
        self.assertTrue(engine.propagate([63,63],range(len(tables.arcs)),lambda:None,counters()))

    def test_dfs_reports_actual_compiled_execution(self):
        problem=SearchProblem(simple_tiles(),double_face())
        reports=[dfs.solve(problem,seconds=2,options={'native':mode}) for mode in ('python','compiled')]
        self.assertEqual(reports[0]['codes'],reports[1]['codes'])
        self.assertEqual([r['native_backend'] for r in reports],['python','compiled'])
        self.assertEqual(reports[0]['stats']['native_calls'],0)
        self.assertGreater(reports[1]['stats']['native_calls'],0)
        self.assertEqual(reports[1]['stats']['python_arcs_checked'],0)



def independent_cycle_partition(n,edges):
    """Enumerate simple cycles and union co-cyclic edge IDs; no Tarjan logic."""
    parents=list(range(len(edges)));cyclic=set();adj=[[] for _ in range(n)]
    for eid,(a,b) in enumerate(edges):
        if a!=b:adj[a].append((b,eid));adj[b].append((a,eid))
    def find(x):
        while parents[x]!=x:x=parents[x]
        return x
    def walk(root,vertex,visited,path):
        for other,eid in adj[vertex]:
            if eid in path:continue
            if other==root and len(path)>=1:
                cycle=[*path,eid];cyclic.update(cycle)
                for item in cycle[1:]:parents[find(item)]=find(cycle[0])
            elif other>root and other not in visited:
                walk(root,other,visited|{other},[*path,eid])
    for root in range(n):walk(root,root,{root},[])
    return tuple(find(i) if i in cyclic else -1 for i in range(len(edges)))


def partition(labels):
    return {frozenset(i for i,value in enumerate(labels) if value==label)
            for label in set(labels) if label>=0}


class NativeGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:cls.engine=native.NativeGraphHelpers()
        except (OSError,ValueError,RuntimeError) as exc:
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':raise
            raise unittest.SkipTest('Optional native graph library unavailable: '+str(exc))
    def test_all_five_vertex_simple_graphs_against_independent_cycles(self):
        from itertools import combinations
        from constraint_advanced import cyclic_edge_blocks
        possible=list(combinations(range(5),2))
        for mask in range(1<<len(possible)):
            edges=[edge for i,edge in enumerate(possible) if mask>>i&1]
            actual=self.engine.cyclic_edge_blocks(5,edges)
            expected=independent_cycle_partition(5,edges)
            self.assertEqual(partition(actual),partition(expected),(mask,actual,expected))
            self.assertEqual(actual,cyclic_edge_blocks(5,edges))
    def test_packed_edges_match_cycle_oracle_and_exact_tuple_labels(self):
        from array import array
        from itertools import chain,combinations
        possible=list(combinations(range(5),2))
        for mask in range(1<<len(possible)):
            edges=[edge for i,edge in enumerate(possible) if mask>>i&1]
            packed=array('i',chain.from_iterable(edges)).tobytes()
            actual=self.engine.cyclic_edge_blocks_packed(5,packed)
            self.assertEqual(actual,self.engine.cyclic_edge_blocks(5,edges))
            self.assertEqual(partition(actual),partition(independent_cycle_partition(5,edges)))

    def test_packed_shape_types_counts_and_endpoints_are_checked(self):
        from array import array
        invalid=[(-1,b''),(5281,b''),(True,b''),(2,bytearray(8)),(2,memoryview(bytes(8))),
                 (2,b'1234'),(2,bytes(8*262145)),(0,bytes(8))]
        with patch.object(self.engine.lib,'dd_cyclic_blocks',side_effect=AssertionError('Invalid shape reached C')):
            for n,data in invalid:
                with self.subTest(n=n,type=type(data)),self.assertRaises(ValueError):
                    self.engine.cyclic_edge_blocks_packed(n,data)
        for edge in ((-1,0),(0,2),(2147483647,0),(-2147483648,0)):
            with self.subTest(edge=edge),self.assertRaisesRegex(ValueError,'endpoint'):
                self.engine.cyclic_edge_blocks_packed(2,array('i',edge).tobytes())
        self.assertEqual(self.engine.cyclic_edge_blocks_packed(0,b''),())

    def test_packed_parallel_self_edges_widening_and_owned_bytes(self):
        from array import array
        from itertools import chain
        from constraint_advanced import cyclic_edge_blocks
        source=array('i',[0,1,0,1,0,0]);packed=source.tobytes()
        def mutate_source():source[0]=-999
        self.assertEqual(self.engine.cyclic_edge_blocks_packed(2,packed,mutate_source),
                         cyclic_edge_blocks(2,[(0,1),(0,1),(0,0)]))
        rng=random.Random(8213)
        for n in (2,33,160,5280):
            edges=[(i,(i+1)%n) for i in range(n)]+[(0,1),(0,0)]
            for subset in (edges,edges[:len(edges)//2],[],edges):
                packed=array('i',chain.from_iterable(subset)).tobytes()
                self.assertEqual(self.engine.cyclic_edge_blocks_packed(n,packed),cyclic_edge_blocks(n,subset))
        for _ in range(20):
            edges=[(rng.randrange(33),rng.randrange(33)) for _ in range(rng.randrange(200))]
            self.assertEqual(self.engine.cyclic_edge_blocks_packed(33,array('i',chain.from_iterable(edges)).tobytes()),
                             cyclic_edge_blocks(33,edges))

    def test_packed_native_cancellation_during_traversal_discards_results(self):
        from array import array
        from itertools import chain
        n=5280;edges=[(i,(i+1)%n) for i in range(n)];packed=array('i',chain.from_iterable(edges)).tobytes()
        calls=[0];fail_at=2+(len(edges)+255)//256+1
        class Cancel(Exception):pass
        def check():
            calls[0]+=1
            if calls[0]>=fail_at:raise Cancel
        with self.assertRaises(Cancel):self.engine.cyclic_edge_blocks_packed(n,packed,check)
        self.assertEqual(calls[0],fail_at)
        self.assertEqual(self.engine.cyclic_edge_blocks_packed(n,packed),self.engine.cyclic_edge_blocks(n,edges))

    def test_advanced_packs_exact_same_edge_sequence_and_keeps_python_fallback(self):
        from array import array
        from constraint_advanced import AdvancedPruner
        data,board,known=planted_gold(871,1);problem=SearchProblem(data,board);rng=random.Random(1813)
        for count in (3,7,30):
            rows=[sum(1<<code for code in set(rng.sample(row,min(count,len(row))))|{known[cell]})
                  for cell,row in enumerate(problem.domains)]
            python=AdvancedPruner(problem,options={'native':'python'})
            compiled=AdvancedPruner(problem,options={'native':'compiled'})
            graphs=[];packed_graphs=[];plain=python._cyclic_edge_blocks;packed=compiled.native_graph.cyclic_edge_blocks_packed
            def capture(n,edges,check):
                graphs.append((n,tuple(edges)));return plain(n,edges,check)
            def capture_packed(n,data,check):
                values=array('i');values.frombytes(data)
                packed_graphs.append((n,tuple(zip(values[::2],values[1::2]))));return packed(n,data,check)
            python._cyclic_edge_blocks=capture
            compiled.native_graph.cyclic_edge_blocks_packed=capture_packed
            a=rows.copy();b=rows.copy()
            self.assertEqual(python.prune_biconnected(a),compiled.prune_biconnected(b))
            # Local 33-port certificates now stay in the prepared native
            # helper; the whole-board graph still crosses the packed boundary.
            global_graphs=[graph for graph in graphs if graph[0]==33*problem.n]
            self.assertTrue(global_graphs);self.assertEqual(global_graphs,packed_graphs)
            for domain in rows:
                self.assertEqual(python._local_block_certificate(domain,lambda:None),
                                 compiled._local_block_certificate(domain,lambda:None))
            self.assertEqual(a,b)
            self.assertTrue(all(value&(1<<code) for value,code in zip(b,known)))
            self.assertIsNone(python.native_graph)

    def test_flat_global_graphs_keep_non_32bit_array_fallback(self):
        from array import array
        from unittest.mock import patch
        from constraint_advanced import AdvancedPruner
        data, board, known = planted_gold(9107, 1)
        problem = SearchProblem(data, board)
        domains = [sum(1 << c for c in set(row[:6]) | {known[i]})
                   for i, row in enumerate(problem.domains)]
        for method in ('prune_strands', 'prune_biconnected'):
            expected = domains.copy()
            getattr(AdvancedPruner(problem, options={'native': 'python'}), method)(expected)
            helper = AdvancedPruner(problem, options={'native': 'compiled'})
            actual = domains.copy()
            # A host with wider C ints must use the ordered Python traversal;
            # handing its 64-bit endpoint buffer to the native ABI is invalid.
            with patch('constraint_advanced.array', side_effect=lambda _, v: array('q', v)), \
                 patch.object(helper.native_graph, 'union_roots_packed', side_effect=AssertionError('wrong ABI')), \
                 patch.object(helper.native_graph, 'cyclic_edge_blocks_packed', side_effect=AssertionError('wrong ABI')):
                getattr(helper, method)(actual)
            self.assertEqual(actual, expected)
            self.assertTrue(all(d & (1 << c) for d, c in zip(actual, known)))

    def test_cancelled_flat_global_graph_does_not_publish_partial_support(self):
        from constraint_advanced import AdvancedPruner
        data, board, known = planted_gold(9107, 1)
        problem = SearchProblem(data, board)
        domains = [sum(1 << c for c in set(row[:6]) | {known[i]})
                   for i, row in enumerate(problem.domains)]
        for method, local, cache in (
                ('prune_strands', '_local_forest', '_strand_cache'),
                ('prune_biconnected', '_local_block_certificate', '_biconnected_cache')):
            for stop_at in (1, problem.n, problem.n+1, problem.n+len(problem.edges)+1):
                helper = AdvancedPruner(problem, options={'native': 'compiled'})
                # Isolate graph construction and packing from local preparation.
                for domain in domains:
                    getattr(helper, local)(domain, lambda: None)
                calls = [0]
                def check():
                    calls[0] += 1
                    if calls[0] == stop_at:
                        raise InterruptedError
                actual = domains.copy()
                with self.assertRaises(InterruptedError):
                    getattr(helper, method)(actual, check)
                self.assertEqual(actual, domains)
                self.assertFalse(getattr(helper, cache))
                getattr(helper, method)(actual)
                expected = domains.copy()
                getattr(AdvancedPruner(problem, options={'native': 'python'}), method)(expected)
                self.assertEqual(actual, expected)

    def test_parallel_edges_self_edges_deep_graph_and_repeated_widening(self):
        from constraint_advanced import cyclic_edge_blocks
        rng=random.Random(63471)
        for n in (0,1,2,6,33,160,5280):
            edges=[] if n<2 else [(i,(i+1)%n) for i in range(n)]
            if n>1:edges.extend([edges[0],(0,0)])
            for subset in (edges,edges[:len(edges)//2],edges,[]):
                self.assertEqual(self.engine.cyclic_edge_blocks(n,subset),cyclic_edge_blocks(n,subset))
        for _ in range(30):
            edges=[(rng.randrange(33),rng.randrange(33)) for i in range(rng.randrange(180))]
            self.assertEqual(self.engine.cyclic_edge_blocks(33,edges),cyclic_edge_blocks(33,edges))
    def test_cancel_during_native_graph_traversal_discards_certificate(self):
        from constraint_advanced import cyclic_edge_blocks
        n=5280;edges=[(i,(i+1)%n) for i in range(n)];calls=[0]
        # Packing checks, C entry check, C edge-build checks, then C DFS poll.
        fail_at=1+((len(edges)+255)//256)*2+2
        class Cancel(Exception):pass
        def check():
            calls[0]+=1
            if calls[0]>=fail_at:raise Cancel
        with self.assertRaises(Cancel):self.engine.cyclic_edge_blocks(n,edges,check)
        self.assertEqual(calls[0],fail_at)
        self.assertEqual(self.engine.cyclic_edge_blocks(n,edges),cyclic_edge_blocks(n,edges))
    def test_graph_bounds_and_types_rejected(self):
        for n,edges in ((-1,[]),(5281,[]),(True,[]),(0,[(0,0)]),(2,[(0,2)]),(2,[(False,1)]),(2,[(0,1,2)])):
            with self.subTest(n=n,edges=edges),self.assertRaises(ValueError):self.engine.cyclic_edge_blocks(n,edges)
    def test_advanced_pruner_native_and_python_reductions_agree(self):
        from constraint_advanced import AdvancedPruner
        data,board,known=planted_gold(871,1);problem=SearchProblem(data,board);rng=random.Random(703)
        python=AdvancedPruner(problem,options={'native':'python'})
        compiled=AdvancedPruner(problem,options={'native':'compiled'})
        self.assertIsNone(python.native_graph);self.assertIsNotNone(compiled.native_graph)
        for amount in (30,3,10,30,2):
            domains=[sum(1<<code for code in set(rng.sample(row,min(amount,len(row))))|{known[i]}) for i,row in enumerate(problem.domains)]
            a=domains.copy();b=domains.copy()
            self.assertEqual(python.prune_biconnected(a),compiled.prune_biconnected(b))
            self.assertEqual(a,b);self.assertTrue(all(row&(1<<code) for row,code in zip(b,known)))



class NativeHallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:cls.engine=native.NativeHall()
        except (OSError,ValueError,RuntimeError) as exc:
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':raise
            raise unittest.SkipTest('Optional native Hall library unavailable: '+str(exc))
    def test_exhaustive_three_by_three_graphs_match_permutation_oracle(self):
        from itertools import product,permutations
        for rows in product(range(8),repeat=3):
            permutations_ok=[p for p in permutations(range(3)) if all(rows[i]>>p[i]&1 for i in range(3))]
            allowed,matching,failed=self.engine.filter(rows)
            if not permutations_ok:self.assertIsNone(allowed);self.assertIn(failed,range(3))
            else:
                expected=tuple(sum(1<<tile for tile in {p[i] for p in permutations_ok}) for i in range(3))
                self.assertEqual(allowed,expected,rows);self.assertIn(matching,permutations_ok)
    def test_widening_stale_matching_and_random_orientation_pruner_equivalence(self):
        from constraint_pruning import Pruner
        from test_constraint_pruning import blank_problem
        rng=random.Random(4623)
        for n in (4,10,64,160):
            problem=blank_problem(n);python=Pruner(problem,parity=False,native='python')
            compiled=Pruner(problem,parity=False,native='compiled')
            for case in range(15):
                rows=[sum(1<<code for code in range(3*n) if rng.random()<(0.03 if case%2 else 0.8)) for i in range(n)]
                if case%3==0:rows=[v|(1<<(3*i)) for i,v in enumerate(rows)]
                a=rows.copy();b=rows.copy();python.prune_hall(a);compiled.prune_hall(b)
                self.assertEqual(0 in a,0 in b,(n,case))
                if 0 not in a:self.assertEqual(a,b)
            all_tiles=(1<<n)-1
            allowed,matching,_=self.engine.filter([all_tiles]*n,[0]*n)
            self.assertEqual(allowed,(all_tiles,)*n);self.assertEqual(len(set(matching)),n)
    def test_cancelled_hall_result_is_not_applied_and_retry_succeeds(self):
        rows=[(1<<160)-1]*160;original=rows.copy();calls=[0]
        class Cancel(Exception):pass
        def check():
            calls[0]+=1
            if calls[0]==5:raise Cancel
        with self.assertRaises(Cancel):self.engine.filter(rows,check=check)
        self.assertEqual(rows,original)
        self.assertEqual(self.engine.filter(rows)[0],tuple(original))
    def test_hall_controls_and_fallback(self):
        for rows in ([],[0]*161,[-1],[2],[True]):
            with self.assertRaises(ValueError):self.engine.filter(rows)
        with self.assertRaises(ValueError):self.engine.filter([3,3],[0])
        with patch.object(native,'load_library',side_effect=OSError('missing')):
            self.assertIsNone(native.select_hall('auto')[0])
            with self.assertRaisesRegex(RuntimeError,'build_native.py'):native.select_hall('compiled')
        with patch.object(native,'load_library',side_effect=AssertionError('Python fallback')):
            self.assertIsNone(native.select_hall('python')[0])

if __name__=='__main__':unittest.main()
