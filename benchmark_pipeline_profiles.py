"""Comparable opt-in hybrid policies; every profile retains durable phase budgets.

GPU proposals, bounded local repair, and exact DFS decisions have different
units. These profiles never imply enumeration of the unsymmetrized root.
"""
HYBRID_PROFILES = {
    'baseline': {'durable_hybrid': True},
    'legacy-sampling': {'durable_hybrid': True, 'ejection_chains': True,
                        'move_tabu': True, 'replica_exchange': True},
    'repair': {'durable_hybrid': True, 'core_repair': True, 'loop_trades': True,
               'fragment_crossover': True, 'repair_learning': True, 'dependency_repair': True},
}
HYBRID_PROFILES['guided'] = {'durable_hybrid': True, 'ejection_chains': True, 'obstruction_chains': True}
HYBRID_PROFILES['combined'] = {**HYBRID_PROFILES['legacy-sampling'], **HYBRID_PROFILES['repair']}
HYBRID_PROFILES['cpu-reuse'] = {'durable_hybrid': True, 'resumable_reasoning': True,
    'incremental_hall': True, 'hall_rematch': True, 'reversible_gold_graph': True}
HYBRID_PROFILES['combined-cpu'] = {**HYBRID_PROFILES['combined'], **HYBRID_PROFILES['cpu-reuse']}
DEFAULT_HYBRID_PROFILES = ('baseline', 'legacy-sampling', 'repair', 'combined')
HYBRID_SLICE_SECONDS = 30
