import copy,json,tempfile,unittest
from constraint_probe_plans_round26 import ProbePlans
from constraint_lookahead import CompletedProbeCache
from constraint_cpu_archive_round26 import CPUReasoningArchive

class ProbePlanTests(unittest.TestCase):
    def test_constructive_every_boundary_roundtrip_keeps_complete_union_without_reprobing(self):
        solutions=((0,0,0),(1,1,1));initial=[7]*3
        for grouped in (False,True):
            groups=(3,4) if grouped else (1,2,4)
            for cutoff in range(1,13):
                plans=ProbePlans();current=initial.copy();calls=0;probes=[]
                def check():
                    nonlocal calls
                    calls+=1
                    if calls==cutoff:raise InterruptedError
                def probe(trial,cell):
                    probes.append(tuple(trial));rows=[r for r in solutions if all(m>>v&1 for m,v in zip(trial,r))]
                    if not rows:return False
                    trial[:]=[sum({1<<r[c] for r in rows}) for c in range(3)];return True
                try:plans.constructive(('root',),current,0,groups,probe,check,lambda value:None,{},grouped=grouped)
                except InterruptedError:pass
                restored=ProbePlans.restore(json.loads(json.dumps(plans.snapshot())))
                # Recompute the current complete partition after any retained
                # failed group narrowed the parent; a covering plan remains valid.
                remaining=tuple(m&current[0] for m in groups if m&current[0])
                restored.constructive(('root',),current,0,remaining,probe,lambda:None,lambda value:None,{},grouped=grouped)
                self.assertEqual(current,[3]*3);self.assertEqual(len(probes),len(groups))
                remaining=tuple(m&current[0] for m in groups if m&current[0])
                restored.constructive(('root',),current,0,remaining,probe,lambda:None,lambda value:None,{},grouped=grouped)
                self.assertEqual(len(probes),len(groups))

    def test_pair_row_resumes_only_complete_failures_and_callbacks(self):
        initial=[7]*3;plans=ProbePlans();current=initial.copy();pairs=[];notifications=[]
        def singleton(trial,cell):return True
        def pair(trial,cell):
            first=trial[0].bit_length()-1;second=trial[1].bit_length()-1;pairs.append((first,second));return first<2 and first==second
        for _ in range(5):
            choices=[c for c in range(3) if current[0]>>c&1]
            plans.pairs(('scope',),current,0,choices,singleton,pair,lambda:None,lambda *row:notifications.append(row),lambda code:None,{},neighbors=(1,),max_pairs=2)
            plans=ProbePlans.restore(json.loads(json.dumps(plans.snapshot())))
        self.assertEqual(current,[3,7,7]);self.assertEqual(len(pairs),9);self.assertEqual(len(set(pairs)),9)
        self.assertEqual(len(notifications),7)
        before=len(pairs);plans.pairs(('scope',),current,0,(0,1),singleton,pair,lambda:None,lambda *row:None,lambda code:None,{},neighbors=(1,),max_pairs=2)
        self.assertEqual(len(pairs),before)

    def test_partial_private_domains_and_plans_survive_application_archive(self):
        cache=CompletedProbeCache();context=('p','edges',(7,7));current=[1,7];changes=[]
        def interrupted(trial,cell):
            trial[1]&=3;changes.append(1);raise InterruptedError('stop')
        with self.assertRaises(InterruptedError):cache.run(context,current,0,interrupted,lambda:None,{},resume_partial=True,recoverable=(InterruptedError,))
        with tempfile.TemporaryDirectory() as directory:
            archive=CPUReasoningArchive(directory,'p','edges',(7,7));archive.save_probes(cache);restored=archive.load_probes()
            self.assertIsNotNone(restored);again=[1,7]
            def finish(trial,cell):
                self.assertEqual(trial,[1,3]);self.assertIsNone(cell);trial[1]=1;return True
            self.assertTrue(restored.run(context,again,0,finish,lambda:None,{},resume_partial=True,recoverable=(InterruptedError,)))
            self.assertEqual(again,[1,1]);self.assertEqual(changes,[1]);self.assertFalse(restored.partial)
            self.assertEqual(restored.plans.stats['probe_partial_resumes'],1)

    def test_incomplete_supports_forged_completion_scope_and_eviction(self):
        plans=ProbePlans(limit=2);domains=[3,3];calls=0
        def stop():
            nonlocal calls
            calls+=1
            if calls==3:raise InterruptedError
        def probe(trial,cell):trial[1]=trial[0];return True
        with self.assertRaises(InterruptedError):plans.constructive(('s',),domains,0,(1,2),probe,stop,lambda value:None,{})
        self.assertEqual(domains,[3,3]);payload=json.loads(json.dumps(plans.snapshot()))
        bad=copy.deepcopy(payload);bad['records'][0]['done']=True
        with self.assertRaises(ValueError):ProbePlans.restore(bad)
        bad=copy.deepcopy(payload);bad['records'][0]['results'][0]['domains'][1]=4
        with self.assertRaises(ValueError):ProbePlans.restore(bad)
        bad=copy.deepcopy(payload);bad['records'].append(copy.deepcopy(bad['records'][0]))
        with self.assertRaises(ValueError):ProbePlans.restore(bad)
        for scope in ('a','b','c'):plans.constructive((scope,),[3,3],0,(1,2),probe,lambda:None,lambda value:None,{})
        self.assertEqual(len(plans.records),2);self.assertGreater(plans.stats['probe_plan_evictions'],0)

if __name__=='__main__':unittest.main()
