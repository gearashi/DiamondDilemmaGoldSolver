"""GPU-domain benchmark policies and initialization-only warmup.

Importing this module does not import a GPU runtime or start a search.
"""
from __future__ import annotations
import time

GPU_PROFILES={
    'baseline':{},
    'joint':{'gpu_joint_support':True},
    'reuse':{'gpu_propagated_forks':True,'gpu_event_propagation':True,'gpu_lane_balance':True,'gpu_incremental_checkpoint':True,'shared_coverage':True},
    'learning':{'gpu_precise_reasons':True,'gpu_shared_clauses':True},
}
GPU_PROFILES['combined']={**GPU_PROFILES['joint'],**GPU_PROFILES['reuse'],**GPU_PROFILES['learning']}


def warmup_domain(problem,options):
    """Compile this exact specialization without retaining any search state."""
    from benchmark_provenance import require_assertions
    require_assertions()
    began=time.monotonic()
    from gpu_domain_round24 import DomainCUDA
    from gpu_domain_runner import BASE_COUNTERS,MAX_BYTES
    import cupy as cp
    switches={name:bool(options.get('gpu_'+name,False)) for name in ('joint_support','event_propagation','precise_reasons')}
    engine=None
    try:
        engine=DomainCUDA(problem,target='gold',max_memory_bytes=MAX_BYTES,**switches)
        cp.cuda.get_current_stream().synchronize()
        counters=engine.stats()
        if not set(BASE_COUNTERS)<=set(counters):
            raise RuntimeError('Warmup search counters are incomplete; zero search work cannot be checked.')
        if any(counters[name] for name in BASE_COUNTERS):
            raise RuntimeError('Initialization-only warmup unexpectedly performed search work.')
        device_id=cp.cuda.Device().id
        props=cp.cuda.runtime.getDeviceProperties(device_id)
        name=props['name']
        if isinstance(name,bytes):name=name.decode('utf-8',errors='replace')
        hardware={'name':str(name),'device_index':int(device_id),
                  'compute_capability':[int(props['major']),int(props['minor'])],
                  'total_vram_bytes':int(props['totalGlobalMem']),
                  'multiprocessors':int(props['multiProcessorCount']),
                  'cuda_driver_api_version':int(cp.cuda.runtime.driverGetVersion()),
                  'cuda_runtime_version':int(cp.cuda.runtime.runtimeGetVersion())}
        result={'initialization_only':True,'search_steps':0,'lanes':engine.lanes,
                'cells':problem.n,'total_segments':problem.total_segments,
                'kernel_options':switches,'schema':engine.schema,
                'state_bytes':engine.state_bytes,'hardware':hardware}
    finally:
        if engine is not None:engine.close()
    result['wall_seconds']=time.monotonic()-began
    return result
