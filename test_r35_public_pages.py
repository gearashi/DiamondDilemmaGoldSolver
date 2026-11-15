"""Public Stop, duration and simultaneous Gold at native dispatch page boundaries."""
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from dfs_gpu import DFS
from test_gpu_gold import gold_fixture, validate_arrangement
from test_systematic_runner import Scenario, read, scratch


class PagedScenario(Scenario):
    """Use DFS.step unchanged, a host kernel double, and real runner persistence."""
    def __init__(self, output, source, event):
        super().__init__(output, source)
        self.first_event=event
        self.pages=[]
        self.advanced=[]
        self.pause_seconds=.25

    def gpu_class(self):
        scenario=self
        base=super().gpu_class()
        class Event:
            def record(self): pass
            def synchronize(self): pass
        class Paged(base):
            _page_stop=None
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                self.cp=SimpleNamespace(RawKernel=object,cuda=SimpleNamespace(
                    Event=Event,get_elapsed_time=lambda *_:0,
                    get_current_stream=lambda:SimpleNamespace(synchronize=lambda:None)))
                self.tables=()
                self.colors=self.gold_segments=self.feature_bits=0
                self.device_order=self.order
                for name in ('single_offsets','pair_offsets','pool','gold_partners','gold_ranks',
                             'gold_checked','device_policy','device_gold_blocked'):
                    setattr(self,name,None)
                self.kernel=self.on_page

            def _pause_cpu_owned(self): return ()
            def _restore_cpu_owned(self, previous): pass

            def step(self,budget):
                scenario.steps_this_run+=1
                return DFS.step(self,budget)

            def on_page(self,grid,threads,args):
                first=int(args[-1])
                stop=min(self.n,first+grid[0]*threads[0])
                scenario.pages.append((first,stop))
                scenario.clock.value+=scenario.pause_seconds
                for i in range(first,stop):
                    if self.states.values[i]!=0: continue
                    job=tuple(map(int,self.prefix_codes[i,:self.floors.values[i]]))
                    previous=int(self.work_cursor[i])
                    units=3 if scenario.first_event=='duration-retire' else 1
                    for unit in range(units): scenario.advanced.append((job,previous+unit))
                    self.work_cursor[i]+=units
                    self.counters.values[0,i]+=units
                    if self.work_cursor[i]==3:
                        self.states.values[i]=2
                        self.depths.values[i]=0
                        self.boards.values[:,i]=-1
                if len(scenario.pages)==1:
                    if scenario.first_event in ('stop','gold-and-stop','invalid-and-stop'):
                        (scenario.output/'stop.request').write_text('stop',encoding='utf8')
                    if scenario.first_event=='invalid-and-stop':
                        codes=np.arange(160,dtype=np.int16)*3
                        codes[1]=codes[0]
                        self.boards.values[:,0]=codes
                        self.depths.values[0]=self.maxdepths.values[0]=160
                        self.states.values[0]=1
                    if scenario.first_event=='gold-and-stop':
                        # Both owned jobs reach the edge-perfect state on one
                        # page. The first has three loops and the second one.
                        for lane in (0,1):
                            codes=np.arange(160,dtype=np.int16)*3
                            codes[0]+=lane
                            if lane==0: codes[23]+=1
                            assert int(self.order[0])==0 and self.prefix_codes[lane,0]==codes[0]
                            self.boards.values[:,lane]=codes
                            self.depths.values[lane]=self.maxdepths.values[lane]=160
                            self.states.values[lane]=1

        return Paged

    def run_pages(self, *, resume=False, width=256, seconds=1000, target='edges'):
        def pages(gpu,budget):
            return [(i,min(128,gpu.n-i)) for i in range(0,gpu.n,128)]
        with patch('dfs_gpu.dispatch_pages',side_effect=pages):
            self.execute(resume=resume,replicas=width,seconds=seconds,target=target)


def write_gold_input(source):
    data,_=gold_fixture()
    for i,t in enumerate(data['tiles']):
        t.update(number=i+1,group='silver')
    source.write_text(json.dumps(data),encoding='utf8')
    return data


class PublicPageTests(unittest.TestCase):
    def assert_first_page_saved(self, scenario):
        checkpoint=scenario.output/'systematic'/'checkpoint.npz'
        self.assertEqual(scenario.pages,[(0,128)])
        self.assertEqual(read(scenario.output/'status.json')['state'],'stopped')
        with np.load(checkpoint,allow_pickle=False) as saved:
            self.assertEqual(int(saved['exact_generation']),1)
            np.testing.assert_array_equal(saved['exact_job_ids'],np.arange(256))
            np.testing.assert_array_equal(saved['cursors'][159],np.r_[np.ones(128),np.zeros(128)])
            np.testing.assert_array_equal(saved['counters'][0],saved['cursors'][159])
            self.assertTrue(np.all(saved['states']==0))
            self.assertEqual(int(saved['exact_completed']),0)
        return checkpoint

    def test_stop_after_first_page_resume_shrinks_then_exhausts_without_replay(self):
        with scratch() as (root,source):
            s=PagedScenario(root/'run',source,'stop')
            s.run_pages()
            checkpoint=self.assert_first_page_saved(s)
            s.run_pages(resume=True,width=96)
            self.assertEqual(read(s.output/'status.json')['state'],'exhausted')
            expected={((code,),cursor) for code in range(480) for cursor in range(3)}
            self.assertEqual(set(s.advanced),expected)
            self.assertEqual(len(s.advanced),len(expected))
            with np.load(checkpoint,allow_pickle=False) as saved:
                self.assertEqual(saved['states'].shape,(96,))
                self.assertEqual(int(saved['exact_completed']),480)

    def test_duration_expires_inside_first_native_page_without_stop_file(self):
        with scratch() as (root,source):
            s=PagedScenario(root/'run',source,'duration')
            s.run_pages(seconds=.125)
            self.assert_first_page_saved(s)
            self.assertFalse((s.output/'stop.request').exists())

    def test_deadline_after_retirement_leaves_unassigned_queue_uninstalled(self):
        with scratch() as (root,source):
            s=PagedScenario(root/'run',source,'duration-retire')
            s.run_pages(seconds=.125)
            self.assertEqual(s.pages,[(0,128)])
            self.assertEqual(read(s.output/'status.json')['state'],'stopped')
            with np.load(s.output/'systematic'/'checkpoint.npz',allow_pickle=False) as saved:
                self.assertEqual(int(saved['exact_completed']),128)
                self.assertEqual(int(saved['exact_cursor']),256)
                self.assertTrue(np.all(saved['exact_job_ids'][:128]==-1))
                np.testing.assert_array_equal(saved['exact_job_ids'][128:],np.arange(128,256))

    def test_multiloop_and_gold_same_page_with_stop_persist_valid_gold(self):
        with scratch() as (root,source):
            data=write_gold_input(source)
            s=PagedScenario(root/'run',source,'gold-and-stop')
            s.run_pages(target='gold')
            self.assertEqual(s.pages,[(0,128)])
            self.assertEqual(read(s.output/'status.json')['state'],'solved')
            record=read(s.output/'solution.json')
            self.assertEqual(record['job_id'],1)
            self.assertTrue(validate_arrangement(data,record['codes'])['valid'])
            self.assertEqual(record['validation']['line_components'],1)
            for job,loops in ((0,3),(1,1)):
                path=s.output/'systematic'/f'candidate-job-{job:09d}.json'
                prior=read(path)
                self.assertEqual(prior['validation']['matched_edges'],240)
                self.assertEqual(prior['validation']['line_components'],loops)
            checkpoint=s.output/'systematic'/'checkpoint.npz'
            with np.load(checkpoint,allow_pickle=False) as saved:
                self.assertEqual(str(saved['exact_terminal']),'solved')
                self.assertEqual(int(saved['exact_candidates']),2)
                self.assertEqual(int(saved['states'][1]),1)
                self.assertTrue(np.all(saved['counters'][0,128:]==0))
                self.assertTrue(np.all(saved['cursors'][159,128:]==0))
            (s.output/'solution.json').unlink()
            s.run_pages(resume=True,width=1,target='gold')
            saved=read(s.output/'solution.json')
            self.assertTrue(saved['revalidated_previous_record'])
            self.assertTrue(validate_arrangement(data,saved['codes'])['valid'])
            # Restore from the terminal checkpoint itself, including when a
            # one-lane pool parks the earlier resident that actually had Gold.
            (s.output/'best.json').unlink()
            (s.output/'solution.json').unlink()
            s.run_pages(resume=True,width=1,target='gold')
            self.assertEqual(s.pages,[(0,128)])
            self.assertEqual(read(s.output/'status.json')['state'],'solved')

            recovered=read(s.output/'solution.json')
            self.assertEqual(recovered['job_id'],1)
            self.assertIsNone(recovered['replica'])
            self.assertTrue(validate_arrangement(data,recovered['codes'])['valid'])

    def test_edge_perfect_terminal_upgraded_to_gold_checks_parked_candidates(self):
        with scratch() as (root,source):
            data=write_gold_input(source)
            s=PagedScenario(root/'run',source,'gold-and-stop')
            s.run_pages(target='edges')
            self.assertEqual(read(s.output/'status.json')['state'],'edge_perfect')
            self.assertEqual(read(s.output/'best.json')['validation']['line_components'],3)
            self.assertFalse((s.output/'solution.json').exists())
            s.run_pages(resume=True,width=1,target='gold')
            self.assertEqual(s.pages,[(0,128)])
            self.assertEqual(read(s.output/'status.json')['state'],'solved')
            saved=read(s.output/'solution.json')
            self.assertEqual(saved['job_id'],1)
            self.assertIsNone(saved['replica'])
            self.assertTrue(validate_arrangement(data,saved['codes'])['valid'])

    def test_failed_republish_does_not_replace_durable_terminal(self):
        with scratch() as (root,source):
            write_gold_input(source)
            s=PagedScenario(root/'run',source,'gold-and-stop')
            s.run_pages(target='gold')
            checkpoint=s.output/'systematic'/'checkpoint.npz'
            before=checkpoint.read_bytes()
            (s.output/'best.json').unlink()
            (s.output/'solution.json').unlink()
            with patch('systematic_search.render',side_effect=OSError('injected render failure')):
                with self.assertRaisesRegex(SystemExit,'injected render failure'):
                    s.run_pages(resume=True,width=1,target='gold')
            self.assertEqual(checkpoint.read_bytes(),before)
            self.assertEqual(s.pages,[(0,128)])
            s.run_pages(resume=True,width=1,target='gold')
            self.assertEqual(read(s.output/'status.json')['state'],'solved')
            self.assertTrue(read(s.output/'solution.json')['validation']['valid'])

    def test_invalid_pending_page_preserves_previous_trusted_checkpoint(self):
        with scratch() as (root,source):
            s=PagedScenario(root/'run',source,'invalid-and-stop')
            with self.assertRaisesRegex(SystemExit,'candidate failed independent validation'):
                s.run_pages(target='gold')
            self.assertEqual(read(s.output/'status.json')['state'],'error')
            self.assertFalse((s.output/'solution.json').exists())
            with np.load(s.output/'systematic'/'checkpoint.npz',allow_pickle=False) as saved:
                self.assertEqual(int(saved['exact_generation']),0)
                self.assertFalse(np.any(saved['states']==1))
                self.assertFalse(np.any(saved['counters']))

    def test_generic_backend_needs_no_native_callback_attribute(self):
        with scratch() as (root,source):
            s=Scenario(root/'run',source)
            base=s.gpu_class()
            class Generic(base):
                # A library handle alone does not promise paged-dispatch support.
                cp=SimpleNamespace(RawKernel=object)
                def __setattr__(self,key,value):
                    if key=='_page_stop': raise AssertionError('Unexpected native callback on generic backend.')
                    super().__setattr__(key,value)
            s.stop_after=1
            with patch.object(s,'gpu_class',return_value=Generic):
                s.execute(replicas=2)
            self.assertEqual(read(s.output/'status.json')['state'],'stopped')


if __name__=='__main__':
    unittest.main()
