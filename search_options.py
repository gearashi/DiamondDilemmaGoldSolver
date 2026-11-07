"""Validated controls for exact-search optimizations; never puzzle assumptions."""
from __future__ import annotations

BOOLEAN_OPTIONS = ('failure_guided', 'lookahead', 'pair_lookahead', 'strand_connectivity', 'multi_seam_cuts', 'regional_cache',
                   'biconnected', 'separator_regions', 'backjumping', 'incremental', 'adaptive', 'persistent_learning', 'reason_tracking', 'sat_assist', 'hierarchical_regions', 'value_ordering', 'shared_conflicts', 'gold_oracle', 'connection_branching', 'lcg', 'native_search')
EXPERIMENTAL_OPTIONS = ('smart_learning','group_lookahead','deep_paths','conservation','signature_branching','impact_branching','objective_ordering','frontier_restarts','worker_portfolio','limited_discrepancy','compact_explanations','conditional_hall','wide_cuts','piece_symmetry','proof_certificates','rotated_learning','extended_regions','gpu_gold_sampling','dependency_repair','dirty_neighborhood','parity_root_reuse')
ROUND24_OPTIONS = ('resumable_reasoning','queue_pruning','shared_coverage','portable_frontier','proof_transport','persistent_sat','incremental_hall','witness_pool','support_cover','retained_relations','joint_conservation','affine_reasoning','pb_learning','sat_theory','interface_formulas','obstruction_rules','symbolic_regions','symbolic_inventory','elimination','mini_buckets','relaxed_diagrams','representative_connectivity','subset_joins','lazy_regions','certified_relaxation','labelled_routing','residual_reuse','dual_piece_branching','strand_branching','completion_ordering','learned_ordering','core_repair','loop_trades','fragment_crossover','repair_learning','ejection_chains','move_tabu','replica_exchange')
ROUND25_OPTIONS = ('sat_core_lifting','shared_sat_assistant','resumable_proof_checking','native_explanations','symbolic_coverage','mixed_proof_dag','higher_order_relations','incremental_affine','native_sat_persistence','conflict_pb','integer_lattice','hall_rematch','reversible_gold_graph','incremental_regions','durable_regions','compact_factors','gold_elimination','separator_search','selective_refinement','joint_symbolic_regions','algebraic_connectivity','residual_isomorphism','gold_pricing','multichain_routing','region_certificates','gpu_joint_support','gpu_propagated_forks','gpu_event_propagation','gpu_precise_reasons','gpu_shared_clauses','gpu_lane_balance','gpu_incremental_checkpoint','durable_hybrid','obstruction_chains','cycle_crossover','lazy_gold_scoring','adaptive_temperatures')
ROUND26_OPTIONS = ('symbolic_frontier', 'adaptive_allocation', 'contextual_propagation', 'coordinated_sat', 'coordinated_gpu', 'sat_guard_reuse', 'persistent_theory', 'sat_simplification', 'proof_indexing', 'durable_reasoning', 'isolated_proof_producer', 'shared_sat_service', 'semantic_witnesses', 'relational_consistency', 'relation_certificates', 'relation_partial', 'lattice_tree', 'pb_row_queue', 'regional_compiler', 'recursive_regions', 'regional_proof_reuse', 'connectivity_obligations', 'integer_regional_master', 'gpu_support_reuse', 'gpu_watched_clauses', 'gpu_global_matching', 'gpu_cooperative', 'gpu_proof_transport', 'incremental_sampling', 'optimal_crossover', 'retained_repair')
EXPERIMENTAL_OPTIONS += ROUND24_OPTIONS + ROUND25_OPTIONS + ROUND26_OPTIONS
BOOLEAN_OPTIONS += EXPERIMENTAL_OPTIONS
DEFAULT_OPTIONS = {**dict.fromkeys(BOOLEAN_OPTIONS, True), 'native': 'auto', 'lookahead_ms': 100, 'reason_tracking': False, 'sat_assist': False, 'hierarchical_regions': False, 'value_ordering': False, 'shared_conflicts': False, 'gold_oracle': False, 'connection_branching': False, 'lcg': False, 'native_search': False, **dict.fromkeys(EXPERIMENTAL_OPTIONS,False), 'ordering_goal': 'auto', 'sat_service_instances': 1}
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
    if type(result['sat_service_instances']) is not int or not 1<=result['sat_service_instances']<=8:
        raise ValueError('SAT service instances must be an integer from 1 to 8.')
    if result['native'] not in ('auto', 'python', 'compiled'):
        raise ValueError('CPU core must be auto, python, or compiled.')
    if result['ordering_goal'] not in ('auto','solution','conflict'):
        raise ValueError('Ordering goal must be auto, solution or conflict.')
    if result['witness_pool'] or result['support_cover']:result['sat_assist']=True
    if result['isolated_proof_producer']:result['proof_indexing']=True;result['proof_transport']=True
    if result['shared_sat_service'] or result['semantic_witnesses']:result['sat_assist']=True;result['witness_pool']=True
    if result['durable_reasoning']:
        result['resumable_reasoning']=True
        if result['proof_transport'] or result['mixed_proof_dag'] or result['resumable_proof_checking']:result['proof_indexing']=True
    if result['mixed_proof_dag']:result['resumable_proof_checking']=True
    if result['resumable_proof_checking']:result['proof_transport']=True
    if result['proof_transport']:result['smart_learning']=True
    if result['persistent_theory']:result['persistent_sat']=True;result['sat_theory']=True
    if result['sat_core_lifting'] or result['native_sat_persistence'] or result['sat_guard_reuse'] or result['sat_simplification']:result['persistent_sat']=True
    if result['proof_indexing']:result['resumable_proof_checking']=True;result['proof_transport']=True;result['smart_learning']=True
    if result['adaptive_allocation'] or result['coordinated_sat'] or result['coordinated_gpu']:result['symbolic_frontier']=True
    if result['gpu_proof_transport']:result['gpu_shared_clauses']=True
    if result['retained_repair']:result['durable_hybrid']=True
    if result['incremental_sampling']:result['ejection_chains']=True
    if result['optimal_crossover']:result['cycle_crossover']=True
    if result['relational_consistency'] or result['relation_certificates'] or result['relation_partial']:result['higher_order_relations']=True
    if result['shared_sat_assistant'] or result['higher_order_relations']:result['sat_assist']=True
    if result['native_explanations']:result['native_search']=True;result['lcg']=True
    if result['obstruction_chains']:result['ejection_chains']=True
    if result['lazy_gold_scoring']:result['gpu_gold_sampling']=True;result['ejection_chains']=True
    if result['adaptive_temperatures']:result['replica_exchange']=True
    if result['symbolic_coverage']:result['shared_coverage']=True
    if result['pb_row_queue']:result['pb_learning']=True
    if result['lattice_tree']:result['integer_lattice']=True
    if result['incremental_affine']:result['affine_reasoning']=True
    if result['conflict_pb']:result['pb_learning']=True
    if result['gold_elimination']:result['compact_factors']=True
    if result['durable_regions']:result['incremental_regions']=True
    if result['hall_rematch']:result['incremental_hall']=True
    if result['reversible_gold_graph']:result['strand_connectivity']=True;result['biconnected']=True
    return result



def option_engines():
    """Advertise the same per-engine controls that server validation enforces."""
    allowed={key:set(EXACT_ALGORITHMS) for key in ROUND25_OPTIONS + ROUND26_OPTIONS}
    for key in ('symbolic_frontier','adaptive_allocation','coordinated_sat','coordinated_gpu'):allowed[key]={'dfs','parallel-dfs','sat','gpu-domains'}
    allowed['sat_guard_reuse']={'sat'}
    allowed['persistent_theory']={'sat'}
    allowed['sat_simplification']={'sat'}
    allowed['regional_compiler']=set(EXACT_ALGORITHMS)|{'sat','sat-propagator','sat-channel','cp-sat','cp-sat-regions','cp-sat-circuit'}
    for key in ('gpu_support_reuse','gpu_watched_clauses','gpu_global_matching','gpu_cooperative','gpu_proof_transport'):allowed[key]={'gpu-domains'}
    for key in ('incremental_sampling','optimal_crossover','retained_repair'):allowed[key]={'hybrid'}
    for key in ('gpu_joint_support','gpu_propagated_forks','gpu_event_propagation','gpu_precise_reasons','gpu_shared_clauses','gpu_lane_balance','gpu_incremental_checkpoint'):allowed[key]={'gpu-domains'}
    allowed['durable_hybrid']={'hybrid'}
    allowed['obstruction_chains']={'hybrid'}
    allowed['cycle_crossover']={'hybrid'}
    allowed['lazy_gold_scoring']={'hybrid'}
    allowed['adaptive_temperatures']={'hybrid'}
    allowed['sat_core_lifting']={'sat'}
    allowed['native_sat_persistence']={'sat'}
    allowed['symbolic_coverage']={'dfs','parallel-dfs','sat'}
    return {key:sorted(value) for key,value in allowed.items()}


def validate_algorithm_options(algorithm, options):
    """Reject controls that no selected engine can execute."""
    allowed={key:set(value) for key,value in option_engines().items()}
    effective={algorithm}
    if options.get('symbolic_frontier'):
        if algorithm in ('dfs','parallel-dfs') or options.get('adaptive_allocation'):effective.add('dfs')
        if options.get('coordinated_sat'):effective.add('sat')
        if options.get('coordinated_gpu'):effective.add('gpu-domains')
    for key,engines in allowed.items():
        if options.get(key) and not effective&engines:
            raise ValueError(key+' is unavailable for '+algorithm+'.')
    if options.get('native_explanations') and options.get('native')=='python':
        raise ValueError('native_explanations requires the compiled CPU search core.')
    if 'sat' in effective:
        if options.get('symbolic_coverage') and not options.get('persistent_sat'):
            raise ValueError('symbolic_coverage requires persistent_sat for SAT search.')
        if options.get('persistent_theory') and options.get('native_sat_persistence'):raise ValueError('Persistent theory cannot use Glucose native proof logging.')
        if options.get('persistent_sat') and options.get('sat_theory') and not options.get('persistent_theory'):
            raise ValueError('Persistent SAT and SAT theory cannot be combined.')


def validate_workers(workers):
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('CPU workers must be an integer from 1 to 64.')
    return workers
