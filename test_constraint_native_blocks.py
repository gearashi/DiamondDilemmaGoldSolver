"""Independent full-graph support checks for the optional native block filter."""
from itertools import product
import ctypes as C
import os,random,unittest
import constraint_native as native


def cycle_blocks(n,edges):
    """Tiny independent oracle: union edge sets of enumerated simple cycles."""
    adjacent=[[] for _ in range(n)];parents=list(range(len(edges)));cyclic=set()
    def root(x):
        while parents[x]!=x:x=parents[x]
        return x
    for eid,(a,b) in enumerate(edges):adjacent[a].append((b,eid));adjacent[b].append((a,eid))
    def visit(start,node,vertices,path):
        for other,eid in adjacent[node]:
            if eid in path:continue
            if other==start and len(vertices)>=2:
                ids=path+(eid,);cyclic.update(ids);base=root(ids[0])
                for edge in ids[1:]:parents[root(edge)]=base
            elif other>start and other not in vertices:visit(start,other,vertices|{other},path+(eid,))
    for start,row in enumerate(adjacent):
        if row:visit(start,start,{start},())
    return tuple(root(eid) if eid in cyclic else -1 for eid in range(len(edges)))


def full_reference(n,segments,edges,domains,block_function=cycle_blocks):
    graph=[];used=[];seams={}
    for cell,domain in enumerate(domains):
        pairs=sorted({tuple(sorted(pair)) for code,row in enumerate(segments) if domain>>code&1 for pair in row})
        used.append({p for pair in pairs for p in pair});graph.extend((33*cell+a,33*cell+b) for a,b in pairs)
    for a,sa,b,sb in edges:
        for p in range(11):
            u,v=11*sa+p,11*sb+10-p
            if u in used[a] and v in used[b]:
                seams[33*a+u]=len(graph);seams[33*b+v]=len(graph);graph.append((33*a+u,33*b+v))
    blocks=block_function(33*n,graph)
    def block(cell,port):
        eid=seams.get(33*cell+port)
        return -1 if eid is None else blocks[eid]
    choices=[];common=None
    for cell,domain in enumerate(domains):
        mapping={};all_blocks={block(cell,p) for p in used[cell]}
        for code,pairs in enumerate(segments):
            if not(domain>>code&1):continue
            hosting=all_blocks if len(all_blocks)==1 and -1 not in all_blocks else {block(cell,p) for pair in pairs for p in pair}
            if len(hosting)==1 and -1 not in hosting:
                host=next(iter(hosting));mapping[host]=mapping.get(host,0)|(1<<code)
        choices.append(mapping);common=set(mapping) if common is None else common&set(mapping)
        if not common:return ((cell,0),)
    return tuple((cell,sum(mapping[host] for host in common)) for cell,mapping in enumerate(choices))


class NativeBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:cls.library=native.load_library()
        except (OSError,ValueError,RuntimeError) as exc:
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':raise
            raise unittest.SkipTest('Optional native library not built: '+str(exc))
        if not hasattr(cls.library,'dd_biconnected_create'):
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':raise RuntimeError('Required native block export is absent.')
            raise unittest.SkipTest('Optional full native block export is absent.')

    def helper(self,n,segments,edges):
        helper=native.NativeBiconnectedPruner(n,segments,edges,self.library);self.addCleanup(helper.close);return helper

    def test_all_small_domains_match_independent_cycle_enumeration(self):
        segments=[[(0,10)],[(11,21)],[(22,32)],[(0,11),(10,21)],[(0,32),(10,22)],[]]
        for edges in ([],[(0,0,1,0)],[(0,0,1,0)]*2,[(0,s,1,s) for s in range(3)]):
            helper=self.helper(2,segments,edges)
            for domains in product(range(1,32),repeat=2):
                self.assertEqual(helper.allowed(domains),full_reference(2,segments,edges,domains),(edges,domains))
            for domains in ((0,0),(32,32),(63,63),(33,63),(63,0),(0,63)):
                self.assertEqual(helper.allowed(domains),full_reference(2,segments,edges,domains))

    def test_multiple_optional_hosts_and_missing_seams_preserve_exact_supports(self):
        segments=[[(0,10)],[(11,21)],[(22,32)]]*2;edges=[(0,s,1,s) for s in range(3)]
        helper=self.helper(2,segments,edges)
        self.assertEqual(helper.allowed([63,63]),((0,63),(1,63)))
        for domains in ([1,8],[1,16],[63,9],[5,40],[63,63]):
            self.assertEqual(helper.allowed(domains),full_reference(2,segments,edges,domains))
        for seams in (edges[:1],edges[:2],edges+edges[:1]):
            candidate=self.helper(2,segments,seams)
            self.assertEqual(candidate.allowed([63,63]),full_reference(2,segments,seams,[63,63]))

    def test_thirty_three_distinct_optional_hosts_fit_the_private_bound(self):
        # Twelve cells form an even seam ring for each of 33 ports. The local
        # self-pairs occupy the ports without merging their cyclic blocks.
        # Every distinct first-cell host needs a distinct occupied port.
        n=12;segments=[[(code%33,code%33)] for code in range(3*n)]
        edges=[(cell,side,(cell+1)%n,side) for cell in range(n) for side in range(3)]
        helper=self.helper(n,segments,edges);domains=[(1<<(3*n))-1]*n
        expected=tuple(enumerate(domains))
        self.assertEqual(full_reference(n,segments,edges,domains),expected)
        self.assertEqual(helper.allowed(domains),expected)
        domains[-1]=1<<13
        self.assertEqual(helper.allowed(domains),full_reference(n,segments,edges,domains))

    def test_wide_random_domains_narrowing_and_widening_equal_full_graph(self):
        from constraint_advanced import cyclic_edge_blocks
        rng=random.Random(150224)
        for n in (1,2,4,22,40,160):
            segments=[[(rng.randrange(33),rng.randrange(33)) for _ in range(rng.randrange(5))] for _ in range(3*n)]
            edges=[(i,0,(i+1)%n,1) for i in range(n)];edges.extend(edges[:min(n,3)])
            helper=self.helper(n,segments,edges)
            for density in (.01,.15,1):
                domains=[sum(1<<c for c in range(3*n) if rng.random()<density) for _ in range(n)]
                for current in (domains,[v&-v for v in domains],domains):
                    self.assertEqual(helper.allowed(current),full_reference(n,segments,edges,current,cyclic_edge_blocks))

    def test_certificate_cache_eviction_and_high_bit_collisions_are_exact(self):
        from constraint_advanced import cyclic_edge_blocks
        n=4;segments=[[(i%33,(i*7+1)%33)] for i in range(12)];edges=[(0,0,1,0),(2,1,3,1)]
        helper=self.helper(n,segments,edges)
        for domain in list(range(1,1300))+list(range(1299,0,-11)):
            domains=[domain,domain*3&4095,domain,4095]
            self.assertEqual(helper.allowed(domains),full_reference(n,segments,edges,domains,cyclic_edge_blocks))
        n=160;segments=[[(c%33,(c*7+1)%33)] for c in range(480)];helper=self.helper(n,segments,[])
        for bit in (10,31,32,63,64,127,128,255,256,383,384,479):
            domains=[1|(1<<bit)]*n
            self.assertEqual(helper.allowed(domains),((0,0),))

    def test_every_poll_discards_partial_success_and_failure_then_recovers(self):
        for edges,success in (([(0,0,1,0)],True),([],False)):
            helper=self.helper(2,[[(0,10)]]*6,edges);domains=[63,63];expected=helper.allowed(domains);calls=[0]
            self.assertEqual(len(expected)==2,success)
            def count():calls[0]+=1
            helper.allowed(domains,count)
            for at in range(1,calls[0]+1):
                polls=[0]
                def check():
                    polls[0]+=1
                    if polls[0]==at:raise InterruptedError
                with self.assertRaises(InterruptedError):helper.allowed(domains,check)
                self.assertEqual(domains,[63,63]);self.assertEqual(helper.allowed(domains),expected)

    def test_private_input_output_lifetime_reentry_and_validation(self):
        segments=[[(0,10)]]*6;edges=[(0,0,1,0)];helper=self.helper(2,segments,edges);expected=helper.allowed([63,63])
        segments[0].append((1,32));edges.clear();self.assertEqual(helper.allowed([63,63]),expected)
        domains=[63,63]
        def mutate():domains[:]=[1,1]
        self.assertEqual(helper.allowed(domains,mutate),expected)
        with self.assertRaisesRegex(RuntimeError,'active'):helper.allowed([63,63],lambda:helper.allowed([63,63]))
        with self.assertRaisesRegex(RuntimeError,'active'):helper.allowed([63,63],helper.close)
        for bad in ([1],[-1,1],[64,1],[True,1],[1.,1]):
            with self.assertRaises(ValueError):helper.allowed(bad)
        invalid=(C.c_int32*4)(0,0,2,0);self.assertFalse(self.library.dd_biconnected_create(helper._forests._handle,2,1,invalid))
        domain=(C.c_uint64*2)(64,1);output=(C.c_uint64*2)();failed=C.c_int32()
        self.assertEqual(self.library.dd_biconnected_allowed(helper._handle,domain,output,C.byref(failed),native.POLL_CALLBACK(lambda:0)),-2)
        helper.close();helper.close()
        with self.assertRaisesRegex(RuntimeError,'closed'):helper.allowed([63,63])

    def test_constructor_validation_and_cancellation_clean_up_private_workspace(self):
        for n,segments,edges in ((0,[],[]),(161,[[]]*483,[]),(2,None,[]),(2,[[]]*3,[]),
                                (2,[[]]*6,[(0,3,1,0)]),(2,[[]]*6,[(0,0,1,0)]*7),
                                (2,[[(0,33)]]*6,[]),(2,[[]]*6,[(True,0,1,0)])):
            with self.subTest(n=n,edges=edges),self.assertRaises(ValueError):
                native.NativeBiconnectedPruner(n,segments,edges,self.library)
        calls=[0]
        def count():calls[0]+=1
        helper=native.NativeBiconnectedPruner(2,[[(0,10)]]*6,[(0,0,1,0)],self.library,count);helper.close()
        for at in range(1,calls[0]+1):
            polls=[0]
            def check():
                polls[0]+=1
                if polls[0]==at:raise InterruptedError
            with self.assertRaises(InterruptedError):
                native.NativeBiconnectedPruner(2,[[(0,10)]]*6,[(0,0,1,0)],self.library,check)
        self.assertEqual(self.helper(2,[[(0,10)]]*6,[(0,0,1,0)]).allowed([63,63]),((0,63),(1,63)))

    def test_direct_native_filter_preserves_every_finite_gold_witness(self):
        from constraint_advanced import AdvancedPruner
        from test_constraint_dfs import tetrahedron_tiles,brute
        from search_problem import SearchProblem
        data,geometry=tetrahedron_tiles();problem=SearchProblem(data,geometry);solutions=brute(problem,'gold')
        self.assertTrue(solutions);segments=AdvancedPruner(problem,options={'native':'python'}).segments
        helper=self.helper(problem.n,segments,problem.edges);rng=random.Random(150229)
        for trial in range(48):
            domains=[rng.randrange(1,1<<12) for _ in range(4)]
            expected=[row for row in solutions if all(d&(1<<c) for d,c in zip(domains,row))]
            for cell,support in helper.allowed(domains):domains[cell]&=support
            for row in expected:self.assertTrue(all(d&(1<<c) for d,c in zip(domains,row)))

    def test_full_supports_counters_witnesses_and_first_failure_equal_python(self):
        from constraint_advanced import AdvancedPruner
        from search_problem import SearchProblem
        from search_fixtures import planted_gold
        rng=random.Random(150226)
        for size in (1,2,4):
            data,board,known=planted_gold(150225,size,extra_edges=2);problem=SearchProblem(data,board)
            self.assertTrue(problem.validate(known)['valid'])
            compiled=AdvancedPruner(problem,options={'native':'compiled'});python=AdvancedPruner(problem,options={'native':'python'})
            for count in (1,3,9,3*problem.n):
                domains=[sum(1<<c for c in {known[cell],*rng.sample(range(3*problem.n),count)}) for cell in range(problem.n)]
                for trial in range(2):
                    a,b=domains.copy(),domains.copy()
                    self.assertEqual(compiled.prune_biconnected(a),python.prune_biconnected(b));self.assertEqual(a,b);self.assertEqual(compiled.stats,python.stats)
                    self.assertEqual(compiled._biconnected_cache[tuple(domains)],python._biconnected_cache[tuple(domains)])
                    self.assertTrue(all(d&(1<<c) for d,c in zip(a,known)))
            self.assertIsNotNone(compiled._native_biconnected)

    def test_advanced_stop_never_publishes_incomplete_support(self):
        from constraint_advanced import AdvancedPruner
        from search_problem import SearchProblem
        from search_fixtures import planted_gold
        data,board,known=planted_gold(150227,1);problem=SearchProblem(data,board);helper=AdvancedPruner(problem,options={'native':'compiled'})
        helper._native_biconnected=helper.native_graph.prepare_biconnected(helper.n,helper.segments,problem.edges)
        original=[(1<<(3*problem.n))-1]*problem.n
        calls=[0]
        def count():calls[0]+=1
        helper._native_biconnected.allowed(original,count)
        for at in sorted({1,2,3,calls[0]//3,calls[0]//2,calls[0]-1,calls[0]}):
            helper._biconnected_cache.clear();domains=original.copy();calls=[0]
            def check():
                calls[0]+=1
                if calls[0]==at:raise InterruptedError
            with self.assertRaises(InterruptedError):helper.prune_biconnected(domains,check)
            self.assertEqual(domains,original);self.assertFalse(helper._biconnected_cache)
        helper.prune_biconnected(original.copy());self.assertTrue(helper._biconnected_cache)

    def test_long_segment_mapping_polls_work_and_discards_cancelled_output(self):
        # Two disconnected cyclic hosts force segment inspection, rather than
        # the all-ports-in-one-host fast path. The long row is below the 32768
        # immutable-table limit and must be interruptible within that row.
        edges=[(0,0,1,0),(0,1,1,1)]
        for cls in (native.NativeStrandPruner,native.NativeBiconnectedPruner):
            totals=[]
            for length in (1,16000):
                segments=[[(0,10)]]*6;segments[0]=[(0,10)]*length;segments[1]=[(11,21)]
                helper=cls(2,segments,edges,self.library);self.addCleanup(helper.close)
                expected=((0,3),(1,3));self.assertEqual(helper.allowed([3,3]),expected)
                calls=[0]
                def count():calls[0]+=1
                self.assertEqual(helper.allowed([3,3],count),expected);totals.append(calls[0])
                if length==16000:
                    for at in range(1,calls[0]+1):
                        polls=[0]
                        def check():
                            polls[0]+=1
                            if polls[0]==at:raise InterruptedError
                        with self.assertRaises(InterruptedError):helper.allowed([3,3],check)
                    self.assertEqual(helper.allowed([3,3]),expected)
            # At least one work poll per 256 segments for each of two cells;
            # blocks additionally count both endpoints. Local graph polls do
            # not depend on repeated identical segments.
            self.assertGreaterEqual(totals[1]-totals[0],2*(15999//256))

    def test_old_exports_and_tiny_board_retain_existing_filter(self):
        from constraint_advanced import AdvancedPruner
        from search_problem import SearchProblem
        from search_fixtures import planted_gold
        from test_constraint_dfs import simple_tiles,double_face
        library=self.library
        class OldExports:
            def __getattr__(self,name):
                if name.startswith('dd_biconnected_'):raise AttributeError(name)
                return getattr(library,name)
        data,board,known=planted_gold(150228,1);problem=SearchProblem(data,board)
        helper=AdvancedPruner(problem,options={'native':'python'});helper.native_graph=native.NativeGraphHelpers(OldExports())
        self.assertEqual(problem.n,10);self.assertFalse(helper.native_graph.supports_biconnected)
        actual=[(1<<30)-1]*10;expected=actual.copy();helper.prune_biconnected(actual)
        AdvancedPruner(problem,options={'native':'python'}).prune_biconnected(expected);self.assertEqual(actual,expected);self.assertIsNone(helper._native_biconnected)
        tiny=AdvancedPruner(SearchProblem(simple_tiles(),double_face()),options={'native':'compiled'});tiny.prune_biconnected([63,63]);self.assertIsNone(tiny._native_biconnected)

if __name__=='__main__':unittest.main()
