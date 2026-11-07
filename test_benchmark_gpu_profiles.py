"""GPU benchmark warmup contracts using no GPU execution."""
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from benchmark_gpu_profiles import warmup_domain,GPU_PROFILES
from search_options import normalize_options,validate_algorithm_options
from gpu_domain_runner import BASE_COUNTERS


class GPUWarmupTests(unittest.TestCase):
    def device(self,*,synchronize=lambda:None):
        runtime=SimpleNamespace(getDeviceProperties=lambda _:dict(name=b'MockGPU',major=7,minor=5,totalGlobalMem=6000000000,multiProcessorCount=30),
                                driverGetVersion=lambda:12090,runtimeGetVersion=lambda:12080)
        return SimpleNamespace(cuda=SimpleNamespace(runtime=runtime,Device=lambda:SimpleNamespace(id=0),get_current_stream=lambda:SimpleNamespace(synchronize=synchronize)))
    def test_profiles_are_gpu_scoped_and_record_exact_compile_policy(self):
        for profile in GPU_PROFILES.values():validate_algorithm_options('gpu-domains',normalize_options(profile))
        calls=[];closed=[]
        def factory(problem,**kwargs):
            calls.append((problem,kwargs))
            return SimpleNamespace(stats=lambda:dict.fromkeys(BASE_COUNTERS,0),close=lambda:closed.append(True),lanes=1,schema='mock-native',state_bytes=100)
        problem=SimpleNamespace(n=160,total_segments=365)
        with patch('gpu_domain_round24.DomainCUDA',side_effect=factory),patch.dict(sys.modules,{'cupy':self.device()}):
            report=warmup_domain(problem,normalize_options(GPU_PROFILES['combined']))
        self.assertEqual(len(calls),1);self.assertIs(calls[0][0],problem)
        self.assertEqual({k:v for k,v in calls[0][1].items() if k not in ('target','max_memory_bytes')},dict(joint_support=True,event_propagation=True,precise_reasons=True,support_reuse=False,watched_clauses=False,global_matching=False,cooperative=False,proof_transport=False))
        self.assertEqual(closed,[True]);self.assertEqual(report['search_steps'],0)
        self.assertEqual((report['cells'],report['total_segments'],report['lanes']),(160,365,1))
        self.assertEqual(report['hardware']['cuda_runtime_version'],12080)
        self.assertGreaterEqual(report['wall_seconds'],0)
    def test_accidental_search_or_synchronization_failure_closes_warmup(self):
        for mode in ('searched','sync_error','missing_counters'):
            closed=[]
            engine=SimpleNamespace(stats=lambda:({'decisions':0} if mode=='missing_counters' else {**dict.fromkeys(BASE_COUNTERS,0),'decisions':1 if mode=='searched' else 0}),close=lambda:closed.append(True))
            def sync():
                if mode=='sync_error':raise RuntimeError('sync failed')
            with patch('gpu_domain_round24.DomainCUDA',return_value=engine),patch.dict(sys.modules,{'cupy':self.device(synchronize=sync)}):
                with self.assertRaisesRegex(RuntimeError,'unexpectedly|sync failed|counters are incomplete'):warmup_domain(SimpleNamespace(n=160,total_segments=365),{})
            self.assertEqual(closed,[True])


if __name__=='__main__':unittest.main()
