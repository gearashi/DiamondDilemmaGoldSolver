"""Validated controls for exact-search optimizations; never puzzle assumptions."""
from __future__ import annotations

BOOLEAN_OPTIONS = ('failure_guided', 'lookahead', 'pair_lookahead', 'strand_connectivity', 'multi_seam_cuts', 'regional_cache',
                   'biconnected', 'separator_regions', 'backjumping', 'incremental', 'adaptive', 'persistent_learning', 'reason_tracking', 'sat_assist', 'hierarchical_regions', 'value_ordering', 'shared_conflicts', 'gold_oracle', 'connection_branching', 'lcg', 'native_search')
EXPERIMENTAL_OPTIONS = ('smart_learning','group_lookahead','deep_paths','conservation','signature_branching','impact_branching','objective_ordering','frontier_restarts','worker_portfolio','limited_discrepancy','compact_explanations','conditional_hall','wide_cuts','piece_symmetry','proof_certificates','rotated_learning','extended_regions','gpu_gold_sampling','dependency_repair','dirty_neighborhood','parity_root_reuse')
ROUND24_OPTIONS = ('resumable_reasoning','queue_pruning','shared_coverage','portable_frontier','proof_transport','persistent_sat','incremental_hall','witness_pool','support_cover','retained_relations','joint_conservation','affine_reasoning','pb_learning','sat_theory','interface_formulas','obstruction_rules','symbolic_regions','symbolic_inventory','elimination','mini_buckets','relaxed_diagrams','representative_connectivity','subset_joins','lazy_regions','certified_relaxation','labelled_routing','residual_reuse','dual_piece_branching','strand_branching','completion_ordering','learned_ordering','core_repair','loop_trades','fragment_crossover','repair_learning','ejection_chains','move_tabu','replica_exchange')
ROUND25_OPTIONS = ('sat_core_lifting','shared_sat_assistant','resumable_proof_checking','native_explanations','symbolic_coverage','mixed_proof_dag','higher_order_relations','incremental_affine','native_sat_persistence','conflict_pb','integer_lattice','hall_rematch','reversible_gold_graph','incremental_regions','durable_regions')
EXPERIMENTAL_OPTIONS += ROUND24_OPTIONS + ROUND25_OPTIONS
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
    if result['mixed_proof_dag']:result['resumable_proof_checking']=True
    if result['resumable_proof_checking']:result['proof_transport']=True
    if result['proof_transport']:result['smart_learning']=True
    if result['sat_core_lifting'] or result['native_sat_persistence']:result['persistent_sat']=True
    if result['shared_sat_assistant'] or result['higher_order_relations']:result['sat_assist']=True
    if result['native_explanations']:result['native_search']=True;result['lcg']=True
    if result['symbolic_coverage']:result['shared_coverage']=True
    if result['incremental_affine']:result['affine_reasoning']=True
    if result['conflict_pb']:result['pb_learning']=True
    if result['durable_regions']:result['incremental_regions']=True
    if result['hall_rematch']:result['incremental_hall']=True
    if result['reversible_gold_graph']:result['strand_connectivity']=True;result['biconnected']=True
    return result



def validate_algorithm_options(algorithm, options):
    """Reject new controls that the selected engine cannot execute."""
    allowed={key:set(EXACT_ALGORITHMS) for key in ROUND25_OPTIONS}
    allowed['sat_core_lifting']={'sat'}
    allowed['native_sat_persistence']={'sat'}
    allowed['symbolic_coverage']={'dfs','parallel-dfs','sat'}
    for key,engines in allowed.items():
        if options.get(key) and algorithm not in engines:
            raise ValueError(key+' is unavailable for '+algorithm+'.')
    if options.get('native_explanations') and options.get('native')=='python':
        raise ValueError('native_explanations requires the compiled CPU search core.')
    if algorithm=='sat':
        if options.get('symbolic_coverage') and not options.get('persistent_sat'):
            raise ValueError('symbolic_coverage requires persistent_sat for SAT search.')
        if options.get('persistent_sat') and options.get('sat_theory'):
            raise ValueError('Persistent SAT and SAT theory cannot be combined.')


def validate_workers(workers):
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('CPU workers must be an integer from 1 to 64.')
    return workers
