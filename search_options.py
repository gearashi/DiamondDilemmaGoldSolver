"""Validated controls for exact-search optimizations; never puzzle assumptions."""
from __future__ import annotations

BOOLEAN_OPTIONS = ('failure_guided', 'lookahead', 'pair_lookahead', 'strand_connectivity', 'multi_seam_cuts', 'regional_cache',
                   'biconnected', 'separator_regions', 'backjumping', 'incremental', 'adaptive', 'persistent_learning', 'reason_tracking', 'sat_assist', 'hierarchical_regions', 'value_ordering', 'shared_conflicts', 'gold_oracle', 'connection_branching', 'lcg', 'native_search')
EXPERIMENTAL_OPTIONS = ('smart_learning','group_lookahead','deep_paths','conservation','signature_branching','impact_branching','objective_ordering','frontier_restarts','worker_portfolio','limited_discrepancy','compact_explanations','conditional_hall','wide_cuts','piece_symmetry','proof_certificates','rotated_learning','extended_regions','gpu_gold_sampling','dependency_repair','dirty_neighborhood','parity_root_reuse')
ROUND24_OPTIONS = ('resumable_reasoning','queue_pruning','shared_coverage','portable_frontier','proof_transport','persistent_sat','incremental_hall','witness_pool','support_cover','retained_relations','joint_conservation','affine_reasoning','pb_learning','sat_theory','interface_formulas','obstruction_rules','symbolic_regions','symbolic_inventory','elimination','mini_buckets','relaxed_diagrams','representative_connectivity','subset_joins','lazy_regions','certified_relaxation','labelled_routing','residual_reuse','dual_piece_branching','strand_branching','completion_ordering','learned_ordering','core_repair','loop_trades','fragment_crossover','repair_learning','ejection_chains','move_tabu','replica_exchange')
EXPERIMENTAL_OPTIONS += ROUND24_OPTIONS
BOOLEAN_OPTIONS += EXPERIMENTAL_OPTIONS
DEFAULT_OPTIONS = {**dict.fromkeys(BOOLEAN_OPTIONS, True), 'native': 'auto', 'lookahead_ms': 100, 'reason_tracking': False, 'sat_assist': False, 'hierarchical_regions': False, 'value_ordering': False, 'shared_conflicts': False, 'gold_oracle': False, 'connection_branching': False, 'lcg': False, 'native_search': False, **dict.fromkeys(EXPERIMENTAL_OPTIONS,False), 'ordering_goal': 'auto'}
EXACT_ALGORITHMS = ('dfs', 'hybrid', 'parallel-dfs')


def normalize_options(options=None):
    if options is None:
        return DEFAULT_OPTIONS.copy()
    if not isinstance(options, dict):
        raise ValueError('Search options must be an object.')
    unknown = set(options) - set(DEFAULT_OPTIONS)
    if unknown:
        raise ValueError('Unknown search options: ' + ', '.join(sorted(map(str, unknown))))
    result = {**DEFAULT_OPTIONS, **options}
    for key in BOOLEAN_OPTIONS:
        if type(result[key]) is not bool:
            raise ValueError(key + ' must be true or false.')
    if type(result['lookahead_ms']) is not int or not 1 <= result['lookahead_ms'] <= 1000:
        raise ValueError('lookahead_ms must be an integer from 1 to 1000.')
    if result['native'] not in ('auto', 'python', 'compiled'):
        raise ValueError('CPU core must be auto, python, or compiled.')
    if result['ordering_goal'] not in ('auto','solution','conflict'):
        raise ValueError('Ordering goal must be auto, solution or conflict.')
    if result['witness_pool'] or result['support_cover']:result['sat_assist']=True
    if result['proof_transport']:result['smart_learning']=True
    return result


def validate_workers(workers):
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('CPU workers must be an integer from 1 to 64.')
    return workers
