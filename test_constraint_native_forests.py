"""Independent finite local connectivity checks for the optional native helper."""
import ctypes as C
import os,random,unittest
from unittest.mock import patch
import constraint_native as native


def reference(segments,domain):
    edges=[pair for code,pairs in enumerate(segments) if domain>>code&1 for pair in pairs]
    vertices={p for pair in edges for p in pair};adjacency={p:set() for p in vertices}
    for a,b in edges:adjacency[a].add(b);adjacency[b].add(a)
    components=set();remaining=set(vertices)
    while remaining:
        todo=[remaining.pop()];group=set(todo)
        while todo:
            for vertex in adjacency[todo.pop()]-group:group.add(vertex);todo.append(vertex);remaining.discard(vertex)
        components.add(frozenset(group))
    return components,sum(1<<p for p in vertices)


def forest_components(forest,used):
    edges=list(forest)+[(p,p) for p in range(33) if used>>p&1]
    return reference([edges],1)


class NativeLocalForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:cls.library=native.load_library()
        except (OSError,ValueError,RuntimeError) as exc:
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':raise
            raise unittest.SkipTest('Optional native library not built: '+str(exc))

    def helper(self,segments):
        helper=native.NativeGraphHelpers(self.library).prepare_local_forests(segments)
        self.addCleanup(helper.close);return helper

    def test_every_subset_matches_independent_graph_components(self):
        segments=[[],[(0,1)],[(1,2)],[(2,0)],[(0,0)],[(4,5),(5,6)],[(0,4)],[(32,32),(32,31),(31,32)]]
        helper=self.helper(segments)
        for domain in range(1<<len(segments)):
            actual=helper.calculate(domain)
            self.assertEqual(forest_components(*actual),reference(segments,domain),domain)

    def test_random_wide_masks_narrowing_and_sibling_widening(self):
        rng=random.Random(20261016)
        segments=[[(rng.randrange(33),rng.randrange(33)) for _ in range(rng.randrange(0,17))] for code in range(480)]
        helper=self.helper(segments);domains=[0,(1<<480)-1]
        domains.extend(1<<code for code in (0,31,32,63,64,127,128,191,192,255,256,319,320,383,384,447,448,479))
        domains.extend(sum(1<<c for c in range(480) if rng.random()<density) for density in (.01,.1,.5) for _ in range(70))
        for domain in domains+list(reversed(domains)):
            self.assertEqual(forest_components(*helper.calculate(domain)),reference(segments,domain),domain)

    def test_exact_ordered_forests_match_python_reference(self):
        from constraint_advanced import AdvancedPruner
        from search_problem import SearchProblem
        from search_fixtures import planted_gold
        for seed in range(3):
            data,board,_=planted_gold(seed+761,2,extra_edges=2)
            pruner=AdvancedPruner(SearchProblem(data,board),'gold',{'native':'python'});helper=self.helper(pruner.segments)
            rng=random.Random(seed)
            for _ in range(60):
                domain=sum(1<<c for c in range(len(pruner.segments)) if rng.random()<.25)
                self.assertEqual(helper.calculate(domain),pruner._local_forest(domain,lambda:None))

    def test_certificates_preserve_every_single_port_deletion(self):
        segments=[[],[(0,1)],[(1,2)],[(2,0)],[(2,3)],[(3,4),(4,5),(5,3)],[(32,32)],[(0,1),(4,0)]]
        helper=self.helper(segments)
        def components(vertices,edges,deleted):
            vertices=set(vertices)-{deleted};adj={v:set() for v in vertices}
            for a,b in edges:
                if a in vertices and b in vertices:adj[a].add(b);adj[b].add(a)
            result=set()
            while vertices:
                todo=[vertices.pop()];seen=set(todo)
                while todo:
                    for vertex in adj[todo.pop()]-seen:seen.add(vertex);todo.append(vertex);vertices.discard(vertex)
                result.add(frozenset(seen))
            return result
        for domain in range(1<<len(segments)):
            edges=[pair for code,pairs in enumerate(segments) if domain>>code&1 for pair in pairs]
            vertices={p for pair in edges for p in pair};certificate,used=helper.certificate(domain)
            self.assertEqual(used,sum(1<<p for p in vertices))
            for removed in {-1,*vertices}:
                self.assertEqual(components(vertices,edges,removed),components(vertices,certificate,removed),(domain,removed))

    def test_exact_certificates_match_python_reference(self):
        from constraint_advanced import AdvancedPruner
        from search_problem import SearchProblem
        from search_fixtures import planted_gold
        for seed in range(3):
            data,board,_=planted_gold(seed+761,2,extra_edges=2)
            pruner=AdvancedPruner(SearchProblem(data,board),'gold',{'native':'python'});helper=self.helper(pruner.segments)
            supports={}
            for code,pairs in enumerate(pruner.segments):
                for pair in pairs:
                    pair=tuple(sorted(pair));supports[pair]=supports.get(pair,0)|(1<<code)
            pruner._segment_supports=tuple(supports.items());rng=random.Random(seed)
            domains=[0,(1<<len(pruner.segments))-1]
            domains.extend(sum(1<<c for c in range(len(pruner.segments)) if rng.random()<density) for density in (.01,.1,.5) for _ in range(35))
            for domain in domains+list(reversed(domains)):
                self.assertEqual(helper.certificate(domain),pruner._local_block_certificate(domain,lambda:None))

    def test_certificate_cancellation_validation_and_closed_workspace(self):
        helper=self.helper([[(0,1)],[(1,2)],[(2,0)]])
        class Cancel(Exception):pass
        for point in (1,2):
            calls=[0]
            def check():
                calls[0]+=1
                if calls[0]==point:raise Cancel()
            with self.assertRaises(Cancel):helper.certificate(7,check)
            self.assertEqual(helper.certificate(7),(((0,1),(1,2),(2,0)),7))
        for bad in (-1,8,True):
            with self.assertRaises(ValueError):helper.certificate(bad)
        helper.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):helper.certificate(7)

    def test_prepared_tables_are_private_and_output_immutable(self):
        segments=[[[0,1]],[[2,3]]];helper=self.helper(segments);before=helper.calculate(3)
        segments[0][0][:]=[20,21];segments.append([(31,32)])
        self.assertEqual(helper.calculate(3),before)
        self.assertIsInstance(before[0],tuple);self.assertTrue(all(isinstance(pair,tuple) for pair in before[0]))

    def test_malformed_inputs_and_lifecycle(self):
        for segments in (None,[],[[]]*481,[[(-1,0)]],[[(0,33)]],[[(False,0)]],[[(0,)]],[[0]],[(0,1)],[[(0,1)]]*32769):
            with self.subTest(segments=repr(segments)[:70]),self.assertRaises(ValueError):self.helper(segments)
        with self.assertRaises(ValueError):self.helper([[(0,1)]*32769])
        helper=self.helper([[(0,1)],[]])
        for domain in (-1,4,True,1.0,None):
            with self.assertRaises(ValueError):helper.calculate(domain)
        helper.close();helper.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):helper.calculate(1)

    def test_cancellation_and_nonreentrancy_never_publish_partial_result(self):
        helper=self.helper([[(0,1)],[(1,2)]])
        class Cancel(Exception):pass
        for point in (1,2):
            calls=[0]
            def check():
                calls[0]+=1
                if calls[0]==point:raise Cancel()
            with self.assertRaises(Cancel):helper.calculate(3,check)
            self.assertEqual(forest_components(*helper.calculate(3)),reference([[(0,1)],[(1,2)]],3))
        def nested():
            with self.assertRaisesRegex(RuntimeError,'already active'):helper.calculate(1)
            with self.assertRaisesRegex(RuntimeError,'active'):helper.close()
        helper.calculate(3,nested)

    def test_direct_c_validation_prevents_bad_table_or_mask_access(self):
        lib=self.library;offsets=(C.c_int32*2)(0,1);ends=(C.c_uint8*2)(0,33)
        self.assertFalse(lib.dd_local_forests_create(1,1,offsets,ends))
        for values in ((1,1),(0,2),(0,-1)):
            self.assertFalse(lib.dd_local_forests_create(1,1,(C.c_int32*2)(*values),(C.c_uint8*2)(0,1)))
        self.assertFalse(lib.dd_local_forests_create(0,0,None,None))
        helper=self.helper([[(0,1)]])
        self.assertEqual(lib.dd_local_forests_calculate(helper._handle,(C.c_uint64*1)(2),(C.c_int32*33)(),C.byref(C.c_uint64())),-2)
        self.assertEqual(helper.calculate(1),(((1,0),),3))

    def test_global_ordered_union_matches_components_and_roots(self):
        from array import array
        from itertools import chain
        from constraint_advanced import _DSU
        helper=native.NativeGraphHelpers(self.library);rng=random.Random(8311)
        cases=[]
        for n in (0,1,2,4,33,160,5280):
            edges=[] if not n else [(rng.randrange(n),rng.randrange(n)) for _ in range(3*n)]
            for row in (edges,edges[:len(edges)//2],list(reversed(edges)),[]):cases.append((n,row))
        small=[(a,b) for a in range(4) for b in range(a+1,4)]
        cases.extend((4,[edge for i,edge in enumerate(small) if mask>>i&1]) for mask in range(1<<len(small)))
        for n,edges in cases:
            dsu=_DSU(n)
            for a,b in edges:dsu.join(a,b)
            roots=helper.union_roots_packed(n,array('i',chain.from_iterable(edges)).tobytes())
            self.assertEqual(roots,tuple(dsu.find(v) for v in range(n)),(n,edges[:10]))
            groups={}
            for v,root in enumerate(roots):groups.setdefault(root,set()).add(v)
            expected,_=reference([edges+[(v,v) for v in range(n)]],1)
            self.assertEqual({frozenset(group) for group in groups.values()},expected)

    def test_global_union_validation_and_stop_discard_partial_outputs(self):
        from array import array
        from itertools import chain
        helper=native.NativeGraphHelpers(self.library)
        for n,packed in ((-1,b''),(5281,b''),(True,b''),(0,b'12345678'),(1,b'x'),(1,bytearray(8)),(1,bytes(8*262145)),
                         (1,array('i',[-1,0]).tobytes()),(1,array('i',[0,1]).tobytes())):
            with self.assertRaises(ValueError):helper.union_roots_packed(n,packed)
        n=5280;packed=array('i',chain.from_iterable((v,(v+1)%n) for v in range(n))).tobytes()
        class Cancel(Exception):pass
        for point in (1,3,25,46,64):
            calls=[0]
            def check():
                calls[0]+=1
                if calls[0]==point:raise Cancel()
            with self.assertRaises(Cancel):helper.union_roots_packed(n,packed,check)
            self.assertEqual(calls[0],point)
            self.assertEqual(helper.union_roots_packed(2,array('i',[0,1]).tobytes()),(0,0))
        output=(C.c_int32*3)(991,992,993)
        self.assertEqual(self.library.dd_graph_union_roots(2,1,(C.c_int32*2)(0,2),output,native.POLL_CALLBACK(lambda:0)),-2)
        self.assertEqual(tuple(output),(991,992,993))

    def test_bound_allows_empty_and_maximum_prepared_graph(self):
        helper=self.helper([[]]*480);self.assertEqual(helper.calculate((1<<480)-1),((),0))
        helper=self.helper([[(0,1)]*32768]);self.assertEqual(helper.calculate(1),(((1,0),),3))


class NativeLocalForestAvailabilityTests(unittest.TestCase):
    def test_big_endian_rejected_before_loading_library(self):
        with patch.object(native.sys,'byteorder','big'),patch.object(native,'load_library',side_effect=AssertionError('should not load')):
            with self.assertRaisesRegex(ValueError,'little-endian'):native.NativeLocalForests([[]])
    def test_missing_optional_helper_is_truthful(self):
        with self.assertRaisesRegex(RuntimeError,'helper missing'):native.NativeLocalForests([[]],object())




def strand_reference(n,segments,edges,domains):
    """Independent full union graph using adjacency sets and breadth-first labels."""
    adjacency=[set() for _ in range(33*n)];used=[]
    for cell,domain in enumerate(domains):
        ports=set()
        for code,pairs in enumerate(segments):
            if domain>>code&1:
                for a,b in pairs:
                    ports.update((a,b));a+=33*cell;b+=33*cell
                    adjacency[a].add(b);adjacency[b].add(a)
        used.append(ports)
    for a,sa,b,sb in edges:
        for p in range(11):
            u,v=11*sa+p,11*sb+10-p
            if u in used[a] and v in used[b]:
                u+=33*a;v+=33*b;adjacency[u].add(v);adjacency[v].add(u)
    roots={}
    for vertex,row in enumerate(adjacency):
        if vertex in roots:continue
        todo=[vertex];roots[vertex]=vertex
        while todo:
            for other in adjacency[todo.pop()]:
                if other not in roots:roots[other]=vertex;todo.append(other)
    choices=[];common=None
    for cell,domain in enumerate(domains):
        mapping={};all_roots={roots[33*cell+p] for p in used[cell]}
        for code,pairs in enumerate(segments):
            if not(domain>>code&1):continue
            hosting=all_roots if len(all_roots)==1 else {roots[33*cell+a] for a,b in pairs}
            if len(hosting)==1:
                host=next(iter(hosting));mapping[host]=mapping.get(host,0)|(1<<code)
        choices.append(mapping);common=set(mapping) if common is None else common&set(mapping)
        if not common:return ((cell,0),)
    return tuple((cell,sum(mapping[root] for root in common)) for cell,mapping in enumerate(choices))


class NativeStrandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:cls.library=native.load_library()
        except (OSError,ValueError,RuntimeError) as exc:
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':raise
            raise unittest.SkipTest('Optional native library not built: '+str(exc))
        if not hasattr(cls.library,'dd_strands_create'):
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':
                raise RuntimeError('Required full native strand export is absent.')
            raise unittest.SkipTest('Optional full native strand export is absent.')

    def helper(self,n,segments,edges):
        helper=native.NativeStrandPruner(n,segments,edges,self.library)
        self.addCleanup(helper.close);return helper

    def test_all_small_domains_match_independent_full_graph(self):
        from itertools import product
        segments=[[(0,10)],[(11,21)],[(22,32)],[(0,11),(21,10)],[(0,0)],[]]
        for edges in ([],[(0,0,1,0)],[(0,0,1,0),(0,0,1,0)],[(0,s,1,s) for s in range(3)]):
            helper=self.helper(2,segments,edges)
            for domains in product(range(1,64),repeat=2):
                self.assertEqual(helper.allowed(domains),strand_reference(2,segments,edges,domains),(edges,domains))

    def test_random_domains_and_narrowing_widening_match_independent_full_graph(self):
        rng=random.Random(142031)
        for n in (1,2,4,22,40,160):
            segments=[[(rng.randrange(33),rng.randrange(33)) for _ in range(rng.randrange(0,5))] for code in range(3*n)]
            edges=[(i,0,(i+1)%n,1) for i in range(n)]
            edges.extend(edges[:min(4,n)])
            helper=self.helper(n,segments,edges)
            for density in (.01,.2,1):
                for trial in range(7):
                    domains=[sum(1<<c for c in range(3*n) if rng.random()<density) for _ in range(n)]
                    for current in (domains,[v&-v for v in domains],domains):
                        self.assertEqual(helper.allowed(current),strand_reference(n,segments,edges,current))

    def test_cache_collisions_eviction_and_restoration_are_exact(self):
        n=4;segments=[[(i%33,(i*3+1)%33)] for i in range(12)];edges=[(0,0,1,1),(2,1,3,2)]
        helper=self.helper(n,segments,edges)
        for domain in list(range(1,1300))+list(range(1299,0,-13)):
            domains=[domain,(domain*3)&4095,domain,4095]
            self.assertEqual(helper.allowed(domains),strand_reference(n,segments,edges,domains))

    def test_every_poll_cancellation_discards_output_and_reuses_workspace(self):
        n=4;segments=[[(0,10),(11,21),(22,32)]]*12;edges=[(0,s,1,s) for s in range(3)]+[(2,s,3,s) for s in range(3)]
        successful_segments=[[(p,p+1) for p in range(32)]]*12
        successful_edges=[(0,0,1,0),(1,1,2,0),(2,1,3,0)]
        for pairs,seams,successful in ((segments,edges,False),(successful_segments,successful_edges,True)):
            helper=self.helper(n,pairs,seams);domains=[4095]*n;expected=helper.allowed(domains);calls=[0]
            self.assertEqual(len(expected)==n,successful)
            def count():calls[0]+=1
            helper.allowed(domains,count)
            for at in range(1,calls[0]+1):
                polls=[0]
                def check():
                    polls[0]+=1
                    if polls[0]==at:raise InterruptedError
                before=domains.copy()
                with self.assertRaises(InterruptedError):helper.allowed(domains,check)
                self.assertEqual(domains,before);self.assertEqual(helper.allowed(domains),expected)

    def test_inputs_outputs_and_lifecycle_are_privately_owned(self):
        segments=[[(0,10)]]*6;edges=[(0,0,1,0)];helper=self.helper(2,segments,edges);domains=[63,63]
        expected=helper.allowed(domains)
        segments[0].append((1,32));edges.clear()
        self.assertEqual(helper.allowed(domains),expected)
        def mutate():domains[:]=[1,1]
        self.assertEqual(helper.allowed(domains,mutate),expected)
        self.assertEqual(domains,[1,1])
        def reenter():helper.allowed([63,63])
        with self.assertRaisesRegex(RuntimeError,'active'):helper.allowed([63,63],reenter)
        with self.assertRaisesRegex(RuntimeError,'active'):helper.allowed([63,63],helper.close)
        helper.close();helper.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):helper.allowed([63,63])

    def test_argument_bounds_and_native_endpoint_validation(self):
        helper=self.helper(2,[[(0,10)]]*6,[(0,0,1,0)])
        for domains in ([1],[-1,1],[True,1],[64,1],[1.0,1]):
            with self.subTest(domains=domains),self.assertRaises(ValueError):helper.allowed(domains)
        for n,segments,edges in ((0,[],[]),(161,[[]]*483,[]),(2,[[]]*3,[]),(2,[[]]*6,[(0,0,2,0)]),(2,[[]]*6,[(0,3,1,0)]),(2,[[]]*6,[(True,0,1,0)]),(2,[[]]*6,[(0,0,1,0)]*7)):
            with self.subTest(n=n,edges=edges),self.assertRaises(ValueError):native.NativeStrandPruner(n,segments,edges,self.library)
        invalid=(C.c_int32*4)(0,0,2,0)
        self.assertFalse(self.library.dd_strands_create(helper._forests._handle,2,1,invalid))
        domains=(C.c_uint64*2)(1<<6,1);output=(C.c_uint64*2)();failed=C.c_int32()
        self.assertEqual(self.library.dd_strands_allowed(helper._handle,domains,output,C.byref(failed),native.POLL_CALLBACK(lambda:0)),-2)

    def test_advanced_complete_supports_counters_cache_and_gold_witness(self):
        from constraint_advanced import AdvancedPruner
        from search_problem import SearchProblem
        from search_fixtures import planted_gold
        rng=random.Random(14204)
        for seed,frequency in ((541,1),(542,2)):
            tiles,board,known=planted_gold(seed,frequency,extra_edges=2);problem=SearchProblem(tiles,board)
            compiled=AdvancedPruner(problem,options={'native':'compiled'});python=AdvancedPruner(problem,options={'native':'python'})
            for size in (1,4,12,3*problem.n):
                domains=[sum(1<<c for c in set(rng.sample(range(3*problem.n),min(size,3*problem.n)))|{known[cell]}) for cell in range(problem.n)]
                for trial in range(2):
                    a,b=domains.copy(),domains.copy()
                    self.assertEqual(compiled.prune_strands(a),python.prune_strands(b));self.assertEqual(a,b);self.assertEqual(compiled.stats,python.stats)
                    self.assertTrue(all(v&(1<<c) for v,c in zip(a,known)))
                    self.assertEqual(compiled._strand_cache[tuple(domains)],python._strand_cache[tuple(domains)])
            self.assertIsNotNone(compiled._native_strands)

    def test_advanced_interruption_never_caches_partial_support(self):
        from constraint_advanced import AdvancedPruner
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        tiles,geometry,known=planted_gold(8194,1);problem=SearchProblem(tiles,geometry)
        helper=AdvancedPruner(problem,options={'native':'compiled'})
        helper._native_strands=helper.native_graph.prepare_strands(helper.n,helper.segments,problem.edges)
        original=[(1<<(3*problem.n))-1]*problem.n
        for at in (1,2,3,4,5,6,7,8):
            helper._strand_cache.clear();domains=original.copy();calls=[0]
            def check():
                calls[0]+=1
                if calls[0]==at:raise InterruptedError
            with self.assertRaises(InterruptedError):helper.prune_strands(domains,check)
            self.assertEqual(domains,original);self.assertFalse(helper._strand_cache)
        helper.prune_strands(original.copy());self.assertTrue(helper._strand_cache)

    def test_tiny_board_uses_existing_assembly_without_large_workspace(self):
        from constraint_advanced import AdvancedPruner
        from test_constraint_dfs import simple_tiles,double_face,tetrahedron_tiles
        from search_problem import SearchProblem
        tetra,geometry=tetrahedron_tiles()
        for problem in (SearchProblem(simple_tiles(),double_face()),SearchProblem(tetra,geometry)):
            helper=AdvancedPruner(problem,options={'native':'compiled'});domains=[(1<<(3*problem.n))-1]*problem.n
            helper.prune_strands(domains);self.assertIsNone(helper._native_strands)

    def test_old_exports_keep_python_orchestrated_native_fallback(self):
        from constraint_advanced import AdvancedPruner
        from search_fixtures import planted_gold
        from search_problem import SearchProblem
        library=self.library
        class OldExports:
            def __getattr__(self,name):
                if name.startswith('dd_strands_'):raise AttributeError(name)
                return getattr(library,name)
        graph=native.NativeGraphHelpers(OldExports());self.assertFalse(graph.supports_strands)
        tiles,geometry,known=planted_gold(8421,1);problem=SearchProblem(tiles,geometry)
        self.assertEqual(problem.n,10)  # Exercise the capability gate, not the tiny-board gate.
        helper=AdvancedPruner(problem,options={'native':'python'});helper.native_graph=graph
        domains=[(1<<(3*problem.n))-1]*problem.n;actual=domains.copy();helper.prune_strands(actual)
        self.assertEqual(tuple(helper._strand_cache[tuple(domains)]),strand_reference(problem.n,helper.segments,problem.edges,domains))
        self.assertTrue(all(value&(1<<code) for value,code in zip(actual,known)))
        self.assertIsNone(helper._native_strands)


if __name__=='__main__':unittest.main()
