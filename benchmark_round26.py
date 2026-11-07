"""Round26 profiles using the existing serial, source-bound benchmark harness.

Completed finite residual decisions and repeated work support retention. A
bounded original run is incomplete evidence, never a claim of optimal settings.
The inherited watchdog stops the matrix on failure and preserves its artifacts.
"""
from pathlib import Path
import benchmark_round25 as harness

CPU_PROFILES={
 'round26-reuse':{'durable_reasoning':True,'contextual_propagation':True},
 'round26-shared':{'durable_reasoning':True,'shared_sat_service':True,'semantic_witnesses':True,'sat_service_instances':1},
 'round26-deductions':{'durable_reasoning':True,'lcg':True,'proof_indexing':True,
   'relational_consistency':True,'relation_certificates':True,'relation_partial':True,
   'lattice_tree':True,'pb_row_queue':True,'regional_compiler':True,'recursive_regions':True,
   'regional_proof_reuse':True,'connectivity_obligations':True,'integer_regional_master':True},
}
GPU_PROFILES={
 'round26-reuse':{**harness.GPU_PROFILES['combined'],'gpu_support_reuse':True,'gpu_watched_clauses':True},
 'round26-matching':{**harness.GPU_PROFILES['combined'],'gpu_support_reuse':True,'gpu_watched_clauses':True,'gpu_global_matching':True,'gpu_proof_transport':True},
}
GPU_PROFILES['round26-cooperative']={**GPU_PROFILES['round26-matching'],'gpu_cooperative':True}
HYBRID_PROFILES={
 'round26-sampling':{**harness.HYBRID_PROFILES['round25-sampling'],'incremental_sampling':True},
 'round26-combined':{**harness.HYBRID_PROFILES['round25-combined'],'incremental_sampling':True,'optimal_crossover':True,'retained_repair':True,'durable_reasoning':True},
}

def main():
    harness.PROFILES.update(CPU_PROFILES);harness.GPU_PROFILES.update(GPU_PROFILES);harness.HYBRID_PROFILES.update(HYBRID_PROFILES)
    harness.SAT_PROFILES.update({'round26-guards':{'sat_guard_reuse':True},'round26-simplified':{'sat_guard_reuse':True,'sat_simplification':True},'round26-theory':{'sat_guard_reuse':True,'persistent_theory':True}})
    return harness.main(entry_path=Path(__file__).resolve())

if __name__=='__main__':main()
