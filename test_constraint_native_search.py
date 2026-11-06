"""Finite independent search oracles, every-boundary resume, and native contracts."""
import itertools,json,os,random,unittest
from types import SimpleNamespace
from unittest.mock import patch
import constraint_native_search as native

def frame(domains):return {'domains':list(domains),'cell':-1,'choices':[],'next':0}
def values(mask):return [i for i in range(mask.bit_length()) if mask>>i&1]

def callback(*,leaves=None,grouped=False,distinct=True,predicate=lambda values:True):
    def process(domains,decisions):
        if distinct:
            while True:
                fixed=[d for d in domains if d.bit_count()==1]
                if len(fixed)!=len(set(fixed)):return {'action':'reject'}
                used=sum(fixed);new=[d if d.bit_count()==1 else d&~used for d in domains]
                if not all(new):return {'action':'reject'}
                if new==domains:break
                domains[:]=new
        if all(d.bit_count()==1 for d in domains):
            codes=tuple(d.bit_length()-1 for d in domains)
            if predicate(codes) and leaves is not None:leaves.append(codes)
            return {'action':'reject'}
        cell=next(i for i,d in enumerate(domains) if d.bit_count()>1)
        choices=[1<<code for code in values(domains[cell])]
        if grouped and len(choices)>2:
            middle=len(choices)//2;choices=[sum(choices[:middle]),sum(choices[middle:])]
        return {'domains':domains,'cell':cell,'choices':choices}
    return process

class NativeSearchAvailabilityTests(unittest.TestCase):
    def test_explicit_backend_and_fallback_are_honest(self):
        with patch.object(native,'load_library',side_effect=OSError('test unavailable')):
            engine,reason=native.select_native_search(3)
            self.assertIsNone(engine);self.assertIn('test unavailable',reason)
            with self.assertRaisesRegex(RuntimeError,'build_native.py'):native.select_native_search(3,mode='compiled')
        with patch.object(native,'load_library',side_effect=AssertionError('must not load')):
            self.assertIsNone(native.select_native_search(3,mode='python')[0])
    def test_manifest_escape_and_stale_source(self):
        metadata={'abi':1,'platform':native.sys.platform,'machine':native.platform.machine(),'library':'../bad.dll'}
        with patch.object(native.Path,'read_text',return_value=json.dumps(metadata)):
            with self.assertRaisesRegex(RuntimeError,'malformed'):native.load_library()
        metadata['library']='safe.dll';metadata['source_sha256']='0'*64
        with patch.object(native.Path,'read_text',return_value=json.dumps(metadata)):
            with self.assertRaisesRegex(RuntimeError,'source changed'):native.load_library()
    def test_configuration_rejects_unsupported_bounds(self):
        for args in ((0,2),(161,2),(1,0),(1,481),(True,3)):
            with self.assertRaises(ValueError):native.NativeSearch(*args)
        with patch.object(native.sys,'byteorder','big'):
            with self.assertRaisesRegex(ValueError,'little-endian'):native.NativeSearch(2)

class NativeSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:native.load_library()
        except (OSError,ValueError,RuntimeError) as exc:
            if os.environ.get('DIAMOND_REQUIRE_NATIVE')=='1':raise
            raise unittest.SkipTest(str(exc))
    def engine(self,n,count=None,**kwargs):
        engine=native.NativeSearch(n,count,**kwargs);self.addCleanup(engine.close);return engine
    def test_independent_enumerator_random_domains_and_grouped_splits(self):
        rng=random.Random(8319)
        for grouped in (False,True):
            for _ in range(30):
                domains=[rng.randrange(1,16) for _ in range(3)];leaves=[]
                predicate=lambda codes:sum(codes)%2==1
                expected={c for c in itertools.product(*(values(d) for d in domains)) if len(set(c))==3 and predicate(c)}
                result=self.engine(3,4).run([frame(domains)],callback(leaves=leaves,grouped=grouped,predicate=predicate))
                self.assertEqual(result['status'],'exhausted');self.assertEqual(result['stack'],[])
                self.assertEqual(set(leaves),expected);self.assertEqual(len(leaves),len(expected))
    def test_every_transition_resume_has_no_duplicates_or_omissions(self):
        expected=list(itertools.product(range(3),repeat=3));leaves=[];stack=[frame([7]*3)]
        transitions=0
        while stack:
            engine=self.engine(3,3)
            result=engine.run(stack,callback(leaves=leaves,distinct=False,grouped=True),max_steps=1)
            stack=result['stack'];transitions+=1
            self.assertLess(transitions,300)
        self.assertEqual(leaves,expected);self.assertEqual(len(leaves),len(set(leaves)))
    def test_interrupted_propagation_discards_uncommitted_changes(self):
        engine=self.engine(2,3);initial=[frame([7,7])]
        def interrupted(domains,decisions):domains[0]=1;raise InterruptedError('stop during propagation')
        with self.assertRaises(InterruptedError) as caught:engine.run(initial,interrupted)
        self.assertEqual(caught.exception.native_search_state['stack'],initial)
        self.assertEqual(engine.snapshot(),initial)
        leaves=[];result=engine.run(None,callback(leaves=leaves,distinct=False))
        self.assertEqual(result['status'],'exhausted');self.assertEqual(len(leaves),9)
    def test_leaf_consumed_before_event_exception(self):
        engine=self.engine(2,2);leaves=[]
        def event(name,result):
            if name=='reject':raise InterruptedError('stop after leaf')
        with self.assertRaises(InterruptedError):engine.run([frame([3,3])],callback(leaves=leaves,distinct=False),on_event=event)
        self.assertEqual(leaves,[(0,0)])
        saved=engine.last_result['stack'];engine.run(saved,callback(leaves=leaves,distinct=False))
        self.assertEqual(leaves,list(itertools.product(range(2),repeat=2)))
    def test_final_consumption_is_conclusive_when_notification_stops(self):
        engine=self.engine(1,1)
        def event(*args):raise InterruptedError('late stop')
        with self.assertRaises(InterruptedError) as caught:engine.run([frame([1])],lambda d,path:{'action':'solved','value':[0]},on_event=event)
        self.assertEqual(caught.exception.native_search_state['status'],'solved')
        self.assertEqual(caught.exception.native_search_state['value'],[0])
        self.assertEqual(engine.snapshot(),[])
    def test_repeated_terminal_call_preserves_solution_value(self):
        engine=self.engine(1,1)
        result=engine.run([frame([1])],lambda d,path:{'action':'solved','value':[0]})
        repeated=engine.run(None,lambda *args:self.fail('completed work replayed'))
        self.assertEqual(repeated,result)

    def test_poll_exception_at_all_boundaries_preserves_remaining_coverage(self):
        for stop_at in range(1,25):
            engine=self.engine(2,3);calls=[0];leaves=[]
            def check():
                calls[0]+=1
                if calls[0]==stop_at:raise InterruptedError('bounded stop')
            try:engine.run([frame([7,7])],callback(leaves=leaves,distinct=False),check)
            except InterruptedError:engine.run(engine.snapshot(),callback(leaves=leaves,distinct=False))
            self.assertEqual(leaves,list(itertools.product(range(3),repeat=2)))
    def test_deep_trail_rollback_restores_all_words(self):
        engine=self.engine(160,480);domains=[(1<<0)|(1<<479)]*160;leaves=[]
        def process(d,path):
            cell=next((i for i,v in enumerate(d) if v.bit_count()>1),None)
            if cell is None:
                leaves.append(tuple(v.bit_length()-1 for v in d));return {'action':'reject'}
            return {'domains':d,'cell':cell,'choices':[1,1<<479]}
        def event(name,result):
            if len(leaves)==3:raise InterruptedError('three leaves only')
        with self.assertRaises(InterruptedError):engine.run([frame(domains)],process,on_event=event)
        self.assertEqual(leaves[0],(0,)*160)
        self.assertEqual(leaves[1],(0,)*159+(479,))
        self.assertEqual(leaves[2],(0,)*158+(479,0))
        self.assertGreater(engine.stats()['native_search_trail_undos'],0)
        resumed=self.engine(160,480);resumed.load(engine.snapshot());self.assertEqual(resumed.snapshot(),engine.snapshot())
    def test_backjump_discards_only_requested_ancestor_subtree(self):
        engine=self.engine(3,2);leaves=[];base=callback(leaves=leaves,distinct=False)
        def process(d,path):
            # Independently known rule: first cell must be one. Once the root
            # zero branch is entered, all its descendants may be discarded.
            if path and path[0]==(0,1):return {'action':'reject','backjump_depth':1}
            return base(d,path)
        result=engine.run([frame([3]*3)],process)
        self.assertEqual(result['status'],'exhausted')
        self.assertEqual(leaves,[(1,*tail) for tail in itertools.product(range(2),repeat=2)])
    def test_resource_limit_retains_exact_pending_branch(self):
        engine=self.engine(3,2,max_depth=2);leaves=[]
        result=engine.run([frame([3]*3)],callback(leaves=leaves,distinct=False))
        self.assertEqual(result['status'],'limit');self.assertFalse(leaves)
        self.engine(3,2).run(result['stack'],callback(leaves=leaves,distinct=False))
        self.assertEqual(leaves,list(itertools.product(range(2),repeat=3)))
    def test_snapshots_at_callbacks_are_exact_and_nonreentrant(self):
        engine=self.engine(2,2);snapshots=[]
        def event(name,result):
            snapshots.append(engine.snapshot())
            with self.assertRaisesRegex(RuntimeError,'already active'):engine.run([],lambda *args:None)
            with self.assertRaisesRegex(RuntimeError,'active'):engine.close()
        engine.run([frame([3,3])],callback(distinct=False),on_event=event)
        for stack in snapshots:
            other=self.engine(2,2);other.load(stack);self.assertEqual(other.snapshot(),stack)
    def test_malformed_checkpoint_is_rejected_without_replacing_old_state(self):
        engine=self.engine(2,3);initial=[frame([7,7])];engine.load(initial)
        bad=[{'domains':[7,7],'cell':0,'choices':[1,2],'next':0},
             {'domains':[7,7],'cell':0,'choices':[3,6],'next':0},
             {'domains':[7,7],'cell':0,'choices':[True,6],'next':0},
             {'domains':[7,7],'cell':-1,'choices':[1],'next':0},
             frame([0,7]),frame([8,7])]
        for item in bad:
            with self.assertRaises(ValueError):engine.load([item])
            self.assertEqual(engine.snapshot(),initial)
        with self.assertRaises(ValueError):engine.load([{'domains':[7,7],'cell':0,'choices':[1,6],'next':1},frame([2,7])])
    def test_malformed_callback_preserves_pending_node(self):
        for bad in (None,{'action':'other'}, {'domains':[15,7],'cell':0,'choices':[1,14]},
                    {'cell':0,'choices':[1,2]}, {'action':'reject','backjump_depth':1}):
            engine=self.engine(2,3);initial=[frame([7,7])]
            with self.assertRaises(ValueError):engine.run(initial,lambda d,path:bad)
            self.assertEqual(engine.snapshot(),initial)
    def test_c_rechecks_callback_contract_before_any_commit(self):
        for kind in ('overlap','gap','widen','empty','bad_jump'):
            engine=self.engine(1,3);initial=[frame([3])];engine.load(initial)
            @native._PROCESS
            def process(_,domains,cell,choices,count,jump):
                cell[0]=0;count[0]=2;choices[0]=1;choices[1]=2;jump[0]=-1
                if kind=='overlap':choices[1]=3
                elif kind=='gap':domains[0]=7
                elif kind=='widen':domains[0]=5;choices[1]=4
                elif kind=='empty':domains[0]=0
                elif kind=='bad_jump':jump[0]=1
                return 1
            @native._POLL
            def poll(_):return 0
            @native._EVENT
            def event(_,code):self.fail('malformed node cannot emit event');return 0
            self.assertEqual(engine.lib.dd_search_run(engine._handle,process,poll,event,None,0),-3)
            self.assertEqual(engine.snapshot(),initial)

    def test_c_import_rechecks_partition_before_mutation(self):
        engine=self.engine(1,3)
        domains=(native.C.c_uint64*1)(7);choices=(native.C.c_uint64*2)(3,6)
        self.assertEqual(engine.lib.dd_search_add_frame(engine._handle,domains,0,choices,2,0),-1)
        self.assertEqual(engine.snapshot(),[])

    def test_transactional_frontier_replacement_preserves_counts_and_budget(self):
        engine=self.engine(3,2);leaves=[];replaced=[]
        def event(name,result):
            if name=='reject' and not replaced:
                replaced.append(engine.stats()['native_search_transitions'])
                # A separately established condition is first cell == one.
                return {'replace_stack':[frame([2,3,3])]}
        result=engine.run([frame([3]*3)],callback(leaves=leaves,distinct=False),on_event=event,max_steps=7)
        self.assertEqual(result['status'],'stopped')
        self.assertEqual(result['stats']['native_search_transitions'],7)
        self.assertEqual(result['stats']['native_search_nodes'],4)
        engine.run(None,callback(leaves=leaves,distinct=False))
        self.assertEqual(leaves,[(0,0,0)]+[(1,*tail) for tail in itertools.product(range(2),repeat=2)])
        self.assertGreaterEqual(engine.stats()['native_search_peak_depth'],4)

    def test_invalid_replacement_retains_committed_frontier(self):
        engine=self.engine(2,2);leaves=[];committed=[]
        def event(name,result):
            if name=='reject':
                committed.extend(engine.snapshot())
                return {'replace_stack':[frame([4,3])]}
        with self.assertRaises(ValueError) as caught:
            engine.run([frame([3,3])],callback(leaves=leaves,distinct=False),on_event=event)
        self.assertEqual(engine.snapshot(),committed)
        self.assertEqual(caught.exception.native_search_state['stack'],committed)
        engine.run(None,callback(leaves=leaves,distinct=False))
        self.assertEqual(leaves,list(itertools.product(range(2),repeat=2)))

    def test_stop_after_replacement_and_empty_replacement_are_committed(self):
        engine=self.engine(2,2);leaves=[];replaced=[False]
        def event(name,result):
            if name=='reject' and not replaced[0]:
                replaced[0]=True;return {'replace_stack':[frame([2,3])]}
        def check():
            if replaced[0]:raise InterruptedError('after frontier replacement')
        with self.assertRaises(InterruptedError):
            engine.run([frame([3,3])],callback(leaves=leaves,distinct=False),check,on_event=event)
        self.assertEqual(engine.snapshot(),[frame([2,3])])
        self.assertEqual(engine.stats()['native_search_nodes'],3)
        engine.run(None,callback(leaves=leaves,distinct=False),on_event=lambda *args:{'replace_stack':[]})
        self.assertEqual(engine.snapshot(),[])
        self.assertEqual(engine.last_result['status'],'exhausted')

    def test_replacement_reuses_verified_library_and_allocation_failure_is_atomic(self):
        engine=self.engine(2,2);initial=[frame([3,3])];engine.load(initial);leaves=[];replaced=[]
        def event(name,result):
            if name=='reject' and not replaced:
                replaced.append(True);return {'replace_stack':[frame([2,3])]}
        with patch.object(native,'load_library',side_effect=AssertionError('already verified')):
            result=engine.run(None,callback(leaves=leaves,distinct=False),on_event=event)
        self.assertEqual(result['status'],'exhausted');self.assertEqual(leaves,[(0,0),(1,0),(1,1)])
        for function,value in (('dd_search_create',None),('dd_search_add_frame',-2)):
            engine=self.engine(2,2);engine.load(initial);committed=[];leaves=[]
            def event(name,result):
                if name=='reject':
                    committed.extend(engine.snapshot());return {'replace_stack':[frame([2,3])]}
            with patch.object(engine.lib,function,return_value=value):
                with self.assertRaises(RuntimeError) as caught:
                    engine.run(None,callback(leaves=leaves,distinct=False),on_event=event)
            self.assertEqual(engine.snapshot(),committed)
            self.assertEqual(caught.exception.native_search_state['stack'],committed)
            engine.run(None,callback(leaves=leaves,distinct=False))
            self.assertEqual(leaves,list(itertools.product(range(2),repeat=2)))


def puzzle_tables(n,patterns,edges):
    count=3*n
    compatible=tuple(tuple(tuple(sum(1<<b for b in range(count)
        if a//3!=b//3 and patterns[a][sa]==patterns[b][sb])
        for a in range(count)) for sb in range(3)) for sa in range(3))
    return SimpleNamespace(compatible=compatible,arcs=tuple(edges)+tuple((b,sb,a,sa) for a,sa,b,sb in edges))

def python_basic(domains,tables):
    """Independent, deliberately simple finite reference fixpoint."""
    n=len(domains)
    while True:
        old=list(domains)
        for a,sa,b,sb in tables.arcs:
            domains[a]=sum(1<<code for code in values(domains[a]) if tables.compatible[sa][sb][code]&domains[b])
            if not domains[a]:return False
        owners={};occurrences=[[] for _ in range(n)]
        for cell,mask in enumerate(domains):
            tiles={code//3 for code in values(mask)}
            if not tiles:return False
            if len(tiles)==1:
                tile=next(iter(tiles))
                if tile in owners:return False
                owners[tile]=cell
            for tile in tiles:occurrences[tile].append(cell)
        for tile,cells in enumerate(occurrences):
            if not cells:return False
            if tile in owners:
                for cell in cells:
                    if cell!=owners[tile]:domains[cell]&=~(7<<(tile*3))
            if len(cells)==1:domains[cells[0]]&=7<<(tile*3)
        if any(not d for d in domains):return False
        if old==domains:return True

class NativeFastPropagationTests(unittest.TestCase):
    setUpClass=classmethod(NativeSearchTests.setUpClass.__func__)
    engine=NativeSearchTests.engine
    # Keep the new class focused: inherited engine helper and availability setup.
    def test_random_exact_fixpoint_and_independent_solution_sets(self):
        rng=random.Random(71022)
        for trial in range(45):
            n=3;patterns=[[rng.randrange(3) for _ in range(3)] for _ in range(3*n)]
            edges=[(0,0,1,1),(1,2,2,0)];tables=puzzle_tables(n,patterns,edges)
            domains=[rng.randrange(1,1<<(3*n)) for _ in range(n)]
            expected={codes for codes in itertools.product(*(values(d) for d in domains))
                      if len({c//3 for c in codes})==n and all(patterns[codes[a]][sa]==patterns[codes[b]][sb] for a,sa,b,sb in edges)}
            engine=self.engine(n);engine.configure_propagation(tables);leaves=[]
            result=engine.run([frame(domains)],callback(leaves=leaves,distinct=False))
            self.assertEqual(result['status'],'exhausted')
            self.assertEqual(set(leaves),expected);self.assertEqual(len(leaves),len(expected))
            direct=self.engine(n);direct.configure_propagation(tables);reference=domains.copy();valid=python_basic(reference,tables);seen=[]
            def capture(d,path):seen.append(d);return {'action':'reject'}
            direct.run([frame(domains)],capture)
            self.assertEqual(seen,[reference] if valid else [])
    def test_fast_reject_event_is_committed_before_late_stop(self):
        engine=self.engine(2);engine.configure_propagation(puzzle_tables(2,[[0]*3]*6,[]))
        events=[]
        def event(name,result):events.append((name,result));raise InterruptedError('after fast rejection')
        with self.assertRaises(InterruptedError):engine.run([frame([3,6])],lambda *args:self.fail('callback avoided'),on_event=event)
        self.assertEqual(engine.snapshot(),[]);self.assertEqual(engine.last_result['status'],'exhausted')
        self.assertEqual(events,[('fast_reject',{'action':'reject','native_fast_reject':True})])
        self.assertEqual(engine.stats()['native_search_nodes'],1);self.assertEqual(engine.stats()['native_fast_rejects'],1)
    def test_cancel_during_fast_fixpoint_keeps_unexpanded_node(self):
        engine=self.engine(3);tables=puzzle_tables(3,[[0]*3]*9,[(0,0,1,0),(1,0,2,0)])
        engine.configure_propagation(tables);initial=[frame([7,511,511])];polls=[0]
        def check():
            polls[0]+=1
            if polls[0]==3:raise InterruptedError('inside native propagation')
        with self.assertRaises(InterruptedError):engine.run(initial,lambda *args:self.fail('not reached'),check)
        self.assertEqual(engine.snapshot(),initial);self.assertEqual(engine.stats()['native_search_nodes'],0)
        leaves=[];engine.run(None,callback(leaves=leaves,distinct=False))
        expected={codes for codes in itertools.product(range(3),range(9),range(9)) if len({c//3 for c in codes})==3}
        self.assertEqual(set(leaves),expected)
    def test_every_transition_with_residues_resumes_exactly_once(self):
        n=3;tables=puzzle_tables(n,[[c%3]*3 for c in range(9)],[(0,0,1,0),(1,0,2,0)])
        engine=self.engine(n);engine.configure_propagation(tables);leaves=[];stack=[frame([511]*3)];steps=0
        while stack:
            result=engine.run(stack,callback(leaves=leaves,distinct=False,grouped=True),max_steps=1)
            stack=result['stack'];steps+=1;self.assertLess(steps,300)
        expected={codes for codes in itertools.product(range(9),repeat=3) if len({c//3 for c in codes})==3 and len({c%3 for c in codes})==1}
        self.assertEqual(set(leaves),expected);self.assertEqual(len(leaves),len(expected))
        # Reuse the live workspace without load resets: residues survive sibling undo.
        engine=self.engine(n);engine.configure_propagation(tables);engine.run([frame([511]*3)],callback(distinct=False))
        self.assertGreater(engine.stats()['native_fast_support_hits'],0)
    def test_inventory_and_seam_supports_cross_machine_words(self):
        n=23;tables=puzzle_tables(n,[[0]*3]*(3*n),[(20,2,22,1)])
        domains=[1<<(3*tile) for tile in range(n)]
        domains[20]=domains[22]=(7<<60)|(7<<66)
        engine=self.engine(n);engine.configure_propagation(tables);leaves=[]
        engine.run([frame(domains)],callback(leaves=leaves,distinct=False))
        self.assertEqual(len(leaves),18)
        self.assertTrue(all(len({code//3 for code in codes})==n for codes in leaves))
        self.assertEqual(len(set(leaves)),18)

    def test_original_statistics_abi_writes_only_eight_words(self):
        engine=self.engine(2);engine.configure_propagation(puzzle_tables(2,[[0]*3]*6,[]))
        engine.run([frame([3,6])],lambda *args:self.fail('early contradiction'))
        sentinel=0xD1A0D1A0D1A0D1A0
        buffer=(native.C.c_uint64*10)(*([0]*8+[sentinel]*2))
        self.assertEqual(engine.lib.dd_search_stats(engine._handle,buffer),0)
        self.assertEqual(list(buffer[8:]),[sentinel]*2)
        self.assertEqual(buffer[0],1)
        merged=engine.stats()
        self.assertEqual(set(merged),set(native.STAT_KEYS))
        self.assertEqual(merged['native_search_nodes'],1)
        self.assertEqual(merged['native_fast_nodes'],1)
        self.assertEqual(merged['native_fast_rejects'],1)

    def test_configuration_contract_and_copy_isolation(self):
        tables=puzzle_tables(2,[[0]*3]*6,[]);engine=self.engine(2)
        for bad in (SimpleNamespace(compatible=(),arcs=[]),SimpleNamespace(compatible=tables.compatible,arcs=[(0,3,1,0)]),
                    SimpleNamespace(compatible=[[[True]*6]*3]*3,arcs=[])):
            with self.assertRaises(ValueError):engine.configure_propagation(bad)
        engine.configure_propagation(tables);tables.arcs=((0,8,1,0),)
        leaves=[];engine.run([frame([63]*2)],callback(leaves=leaves,distinct=False))
        self.assertEqual(len(leaves),18)
        with self.assertRaises(ValueError):self.engine(2,5).configure_propagation(puzzle_tables(2,[[0]*3]*6,[]))

    def test_queue_bound_and_legacy_fast_stats_abi(self):
        tables=puzzle_tables(2,[[0]*3]*6,[(0,0,1,0)]*480)
        engine=self.engine(2);engine.configure_propagation(tables);seen=[]
        engine.run([frame([63]*2)],lambda d,p:seen.append(d) or {'action':'reject'})
        self.assertEqual(seen,[[63,63]])
        self.assertEqual(engine.stats()['native_fast_arcs'],960)
        self.assertEqual(engine.stats()['native_fast_enqueues'],960)
        sentinel=0xD1A0D1A0D1A0D1A0
        buffer=(native.C.c_uint64*9)(*([0]*7+[sentinel]*2))
        self.assertEqual(engine.lib.dd_search_fast_stats(engine._handle,buffer),0)
        self.assertEqual(list(buffer[7:]),[sentinel]*2)

    def test_live_child_certificate_accounts_for_python_narrowing(self):
        n=4;patterns=[[c%3]*3 for c in range(12)]
        edges=[(0,0,1,0),(2,0,3,0)];tables=puzzle_tables(n,patterns,edges)
        engine=self.engine(n);engine.configure_propagation(tables);leaves=[];base=callback(leaves=leaves,distinct=False)
        self.assertFalse(engine.basic_propagated)
        def process(d,path):
            self.assertTrue(engine.basic_propagated)
            reference=d.copy();self.assertTrue(python_basic(reference,tables));self.assertEqual(d,reference)
            if not path:d[2]&=sum(1<<c for c in range(12) if c%3==0)
            return base(d,path)
        engine.run([frame([4095]*4)],process)
        expected={c for c in itertools.product(range(12),repeat=4)
                  if len({x//3 for x in c})==4 and c[0]%3==c[1]%3 and c[2]%3==c[3]%3==0}
        self.assertEqual(set(leaves),expected);self.assertEqual(len(leaves),len(expected))
        self.assertGreater(engine.stats()['native_fast_seeded_nodes'],0)
        self.assertFalse(engine.basic_propagated)

    def test_configuration_and_import_invalidate_private_certificates(self):
        n=3;unrestricted=puzzle_tables(n,[[0]*3]*9,[(0,0,1,0),(1,0,2,0)])
        restricted=puzzle_tables(n,[[c%3]*3 for c in range(9)],[(0,0,1,0),(1,0,2,0)])
        engine=self.engine(n);engine.configure_propagation(unrestricted)
        saved=engine.run([frame([511]*3)],callback(distinct=False),max_steps=1)['stack']
        engine.configure_propagation(restricted);leaves=[]
        engine.run(None,callback(leaves=leaves,distinct=False))
        expected={c for c in itertools.product(range(9),repeat=3) if len({x//3 for x in c})==3 and len({x%3 for x in c})==1}
        self.assertEqual(set(leaves),expected);self.assertEqual(len(leaves),len(expected))
        imported=self.engine(n);imported.configure_propagation(restricted)
        for node in saved:node['seeded']=True;node['dirty']=[]
        imported.run(saved,callback(distinct=False),max_steps=2)
        self.assertEqual(imported.stats()['native_fast_seeded_nodes'],0)

    def test_live_seeded_stop_at_every_poll_preserves_coverage(self):
        n=3;tables=puzzle_tables(n,[[c%3]*3 for c in range(9)],[(0,0,1,0),(1,0,2,0)])
        expected={c for c in itertools.product(range(9),repeat=3) if len({x//3 for x in c})==3 and len({x%3 for x in c})==1}
        for stop_at in range(1,65):
            engine=self.engine(n);engine.configure_propagation(tables);leaves=[];calls=[0]
            def check():
                calls[0]+=1
                if calls[0]==stop_at:raise InterruptedError('seeded propagation stop')
            try:engine.run([frame([511]*3)],callback(leaves=leaves,distinct=False,grouped=True),check)
            except InterruptedError:engine.run(None,callback(leaves=leaves,distinct=False,grouped=True))
            self.assertEqual(set(leaves),expected);self.assertEqual(len(leaves),len(expected))

    def test_sparse_inventory_all_word_boundaries_matches_reference(self):
        rng=random.Random(71103);n=160;tables=puzzle_tables(n,[[0]*3]*(3*n),[])
        engine=self.engine(n);engine.configure_propagation(tables)
        for density in (.01,.04,.25,1):
            for trial in range(4):
                # Include a planted permutation so all inventory states are feasible.
                domains=[(1<<(3*cell))|sum(1<<code for code in range(3*n) if rng.random()<density) for cell in range(n)]
                expected=domains.copy();self.assertTrue(python_basic(expected,tables));seen=[]
                engine.run([frame(domains)],lambda d,p:seen.append(d) or {'action':'reject'})
                self.assertEqual(seen,[expected])

if __name__=='__main__':unittest.main()
