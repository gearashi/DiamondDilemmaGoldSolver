"""Exact CPU DFS with MRV, seam arc consistency, tile propagation, and loop cuts.

A checkpoint stores every unfinished sibling choice and each active domain
state. Resume may repeat propagation interrupted mid-pass, but never a completed
leaf. Timeouts and requested stops are explicitly incomplete search results.
The integrity digest detects accidental corruption; it is not a proof against
modified checkpoints whose digest has also been recomputed.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import copy
import hashlib
import json
import math
import random
import time
from validator import reverse_mask
from constraint_pruning import Pruner, STAT_KEYS as PRUNER_STAT_KEYS, _tile_choices
from constraint_symmetry import canonical_root_domains
from constraint_branching import ConnectionBranching, branch_mask, branch_masks, from_native_frame, to_native_frame, STAT_KEYS as BRANCH_STAT_KEYS
from constraint_connectivity import ConnectivityPruner, STAT_KEYS as CONNECTIVITY_STAT_KEYS
from constraint_native import NativePropagation, select_native, STAT_KEYS as NATIVE_STAT_KEYS
from search_options import normalize_options
from constraint_advanced import AdvancedPruner, STAT_KEYS as ADVANCED_STAT_KEYS
from constraint_scheduling import PropagationScheduler, STAT_KEYS as SCHEDULING_STAT_KEYS
from constraint_regions_round26 import RegionalWorkstreams, KEYS as REGION26_OPTIONS, STAT_KEYS as REGION26_STATS
from constraint_reasoning_store_round26 import STAT_KEYS as ARCHIVE_STATS
from constraint_cpu_archive_round26 import CPUReasoningArchive
from pathlib import Path
from constraint_knowledge import KnowledgeStore
from constraint_reasons import ReasonState, STAT_KEYS as REASON_STAT_KEYS
from constraint_lcg import LCGState, LCGStore, LearnedClause, STAT_KEYS as LCG_STAT_KEYS
from constraint_gold_oracle import GoldOracle, STAT_KEYS as GOLD_ORACLE_STAT_KEYS, GAUGE_KEYS as GOLD_ORACLE_GAUGE_KEYS
from constraint_native_search import select_native_search, STAT_KEYS as NATIVE_SEARCH_STAT_KEYS
from constraint_native_learning import replay_fast_failure, explain_fast_failure, STAT_KEYS as NATIVE_LEARNING_STAT_KEYS
from constraint_probe_plans_round26 import STAT_KEYS as PROBE_PLAN_STATS
from constraint_propagation_cursor_round26 import PropagationCursor, STAT_KEYS as PROPAGATION_CURSOR_STATS
from constraint_lookahead import CompletedProbeCache, paired_lookahead, CACHE_STAT_KEYS, PAIR_STAT_KEYS
from constraint_sat_assist import SatAssist, STAT_KEYS as SAT_ASSIST_STAT_KEYS
from constraint_job_sat import SharedSatAssistant, JobSatAssist
from constraint_sat_service_round26 import SATServiceClient, STAT_KEYS as SERVICE_STATS
from constraint_witness_round26 import STAT_KEYS as WITNESS_STATS
from constraint_relational_round26 import STAT_KEYS as RELATIONAL_STATS
from constraint_projection_certificate_round26 import STAT_KEYS as RELATION_CERTIFICATE_STATS
from constraint_service_lifecycle_round26 import view as service_view
from constraint_higher_relations import ProjectedRelations, STAT_KEYS as HIGHER_STAT_KEYS
from constraint_ordering import SupportOrdering, STAT_KEYS as ORDERING_STAT_KEYS
from constraint_deep import DeepPruner, CertificateStore, rotate_gold_clause, minimize_failure_scope, STAT_KEYS as DEEP_STAT_KEYS
from constraint_policies import ImpactPolicy, PieceSymmetry, STAT_KEYS as POLICY_STAT_KEYS
from constraint_mask_conflicts import VerifiedMaskConflictStore, STAT_KEYS as MASK_STAT_KEYS, GAUGE_KEYS as MASK_GAUGES
from constraint_lookahead import grouped_constructive_lookahead, GROUP_STAT_KEYS
from constraint_shared_conflicts import VerifiedConflictStore, STAT_KEYS as SHARED_STAT_KEYS, GAUGE_KEYS as SHARED_GAUGES

from constraint_deduction_chain_round26 import CHAIN_STAT_KEYS
from constraint_round24 import ResearchPruner, KEYS as RESEARCH_KEYS, STAT_KEYS as RESEARCH_STAT_KEYS, MAX_GAUGE_KEYS as RESEARCH_MAX_GAUGES
from constraint_region_oracles import ResidualProofCache, RESIDUAL_STAT_KEYS
from constraint_branching_round24 import RoutingBranching, CompletionOrdering, LearnedOrdering, LEARNED_ORDERING_WEIGHTS, STAT_KEYS as ROUTING_STAT_KEYS

VERSION = 2
GUIDANCE_STAT_KEYS = ('failure_guided_choices','failure_weight_updates','lookahead_probes','lookahead_prunes','lookahead_failures','lookahead_aborted',
                      'lookahead_complete_passes','lookahead_common_prunes','regional_solutions_validated',
                      'lookahead_probe_propagations','lookahead_probe_arc_revisions') + CACHE_STAT_KEYS + PAIR_STAT_KEYS + PROBE_PLAN_STATS
KNOWLEDGE_STAT_KEYS = ('loop_witnesses',)
EXPLANATION_STAT_KEYS = ('explanation_probes', 'explanation_literals_removed', 'explanation_aborted', 'explanation_skipped', 'explanation_time_ns', 'backjumps', 'backjump_levels', 'native_basic_passes_skipped')
BASE_STAT_KEYS = ('propagations', 'arc_revisions', 'tile_prunes', 'loop_prunes', 'leaves', 'backtracks') + CHAIN_STAT_KEYS + PROPAGATION_CURSOR_STATS + RELATION_CERTIFICATE_STATS
CONFLICT_STAT_KEYS = ('conflicts_learned', 'conflicts_subsumed', 'conflict_evictions',
                      'conflict_checks', 'conflict_prunes', 'conflict_hits', 'conflict_cache_entries')


def _constructive_lookahead(domains, cell, choices, probe, check, on_failure, stats):
    """Union a COMPLETE disjunction; partial probes never certify shared cuts.

    ``probe`` narrows a private copy and returns whether it may still contain a
    solution. Its false result proves that alternative impossible. All singleton
    alternatives of the input cell must be present. Individually proved failures
    commit immediately; a cancelled call discards the unfinished support union.
    The caller owns any time budget and records conservative proof dependencies.
    """
    before = domains.copy()
    if (len(set(choices)) != len(choices) or
            sum(1 << code for code in choices) != before[cell]):
        raise ValueError('Constructive look-ahead requires a complete choice partition.')
    support = [0]*len(domains)
    changed = False
    for code in choices:
        check();stats['lookahead_probes'] += 1
        trial = before.copy();trial[cell] = 1 << code
        possible = probe(trial, cell)
        check()
        if not possible:
            domains[cell] &= ~(1 << code)
            stats['lookahead_failures'] += 1;stats['lookahead_prunes'] += 1
            changed = True
            on_failure(code)
        else:
            if (len(trial) != len(before) or not all(trial) or
                    any(value & ~old for value, old in zip(trial, before)) or
                    trial[cell] != 1 << code):
                raise ValueError('Look-ahead propagation returned invalid domains.')
            for other, value in enumerate(trial):support[other] |= value
    # No callbacks between completion and publication: shared deductions are
    # committed together, and only after every alternative has been accounted.
    stats['lookahead_complete_passes'] += 1
    if all(domains):
        narrowed = [old & allowed for old, allowed in zip(domains, support)]
        removed = sum((old ^ new).bit_count() for old, new in zip(domains, narrowed))
        if removed:
            domains[:] = narrowed
            stats['lookahead_common_prunes'] += removed;stats['lookahead_prunes'] += removed
            changed = True
    return changed


class _NogoodStore:
    """Bounded, process-local clauses proved by this DFS, never read from JSON.

    Each clause forbids a conjunction of (cell, orientation-code) equalities.
    A smaller conjunction subsumes any stored superset. Evicting a clause only
    loses an optimization: the ordinary complete search remains authoritative.
    """
    def __init__(self, fingerprint, target, limit=512, initial_domains=None):
        self.fingerprint, self.target, self.limit = fingerprint, target, limit
        self.initial_domains = None if initial_domains is None else tuple(initial_domains)
        self.clauses = []

    def learn(self, placements, stats):
        clause = frozenset(placements)
        # Contradictory equalities already cannot co-occur; storing them adds no
        # information. Caller supplies only proven conflicts, not search hints.
        if len({cell for cell, _ in clause}) != len(clause):
            return
        if any(old <= clause for old in self.clauses):
            stats['conflicts_subsumed'] += 1
            return
        retained = [old for old in self.clauses if not clause < old]
        stats['conflicts_subsumed'] += len(self.clauses) - len(retained)
        retained.append(clause)
        stats['conflicts_learned'] += 1
        if len(retained) > self.limit:
            # Prefer short explanations; ties discard the oldest clause.
            victim = max(range(len(retained)), key=lambda i: (len(retained[i]), -i))
            del retained[victim]
            stats['conflict_evictions'] += 1
        self.clauses = retained

    def prune(self, domains, stats, check, reason=None):
        changed = False
        for index, clause in enumerate(self.clauses):
            if index % 8 == 0:
                check()
            stats['conflict_checks'] += 1
            undecided = None
            active = True
            for offset, (cell, code) in enumerate(clause):
                if offset % 32 == 0:
                    check()
                bit = 1 << code
                if not domains[cell] & bit:
                    active = False
                    break
                if domains[cell] != bit:
                    if undecided is not None:
                        active = False  # At least two literals are still open.
                        break
                    undecided = (cell, bit)
            if not active:
                continue
            if undecided is None:
                stats['conflict_hits'] += 1
                domains[0] = 0
                if reason is not None:reason(0,0,tuple(cell for cell,_ in clause))
                return True
            cell, bit = undecided
            domains[cell] &= ~bit
            if reason is not None:reason(cell,domains[cell],tuple(c for c,_ in clause if c!=cell))
            stats['conflict_prunes'] += 1
            changed = True
        return changed




class _SearchRuntime:
    """Trusted process-local guidance/helper reuse; never deserialized."""
    def __init__(self,fingerprint,target,initial_domains,n):
        self.fingerprint,self.target,self.initial_domains=fingerprint,target,tuple(initial_domains)
        self.failure_weights=[1]*n
        self.advanced,self.advanced_options=None,None
        self.scheduler,self.scheduler_options=None,None
        self.explanation_misses,self.explanation_cooldown=0,0
        self.sat=None;self.higher=None;self.higher_directory=None;self.higher_sat=None;self.service_view=None;self.semantic_pool=None;self.semantic_path=None;self.semantic_saved=None
        self.ordering=None
        self.pruner_tables=None
        self.incremental_hall=None
        self.research=None;self.routing=None;self.completion=None;self.learned=None
        self.residual=None
        self.regional26=None;self.regional26_options=None
        self.reasoning_archive=None;self.reasoning_directory=None
        self.lcg=None
        self.gold_oracle=None
        self.probes=CompletedProbeCache();self.propagation_cursors={}
        self.deep=None;self.impact=ImpactPolicy(n);self.piece_symmetry=None
        self.certificates=None;self.mask_conflicts=None;self.local_mask_conflicts=None

    def close(self):
        # Retire projection guards before closing their local SAT helper.
        seen=set()
        for name in ('higher','sat','higher_sat','gold_oracle','mask_conflicts','local_mask_conflicts'):
            value=getattr(self,name,None)
            if value is None or id(value) in seen:continue
            seen.add(id(value));close=getattr(value,'close',None)
            if callable(close):close()

    def valid(self,fingerprint,target,initial_domains,n):
        return (self.fingerprint==fingerprint and self.target==target
                and self.initial_domains==tuple(initial_domains)
                and isinstance(self.failure_weights,list) and len(self.failure_weights)==n
                and all(type(value)is int and 1<=value<=1000000 for value in self.failure_weights)
                and isinstance(self.probes,CompletedProbeCache))


@dataclass(frozen=True)
class _PropagationTables:
    """Immutable seam supports reusable across slices of the same problem."""
    fingerprint: str
    target: str
    initial_domains: tuple
    tile_bits: tuple
    compatible: tuple
    groups: tuple
    arcs: tuple
    incoming: tuple


def _build_support_tables(problem, initial, target, check):
    n, count = problem.n, 3*problem.n
    tile_bits = tuple(7 << (3*tile) for tile in range(n))
    by_side = [{} for _ in range(3)]
    for code, masks in enumerate(problem.masks):
        if code % 32 == 0:
            check()
        for side, mask in enumerate(masks):
            by_side[side][int(mask)] = by_side[side].get(int(mask), 0) | (1 << code)
    reversed_masks = {mask: reverse_mask(mask) for row in by_side for mask in row}
    compatible, groups = [], []
    for side in range(3):
        compatible_row, groups_row = [], []
        for other in range(3):
            check()
            compatible_row.append(tuple(by_side[other].get(reversed_masks[int(problem.masks[code, side])], 0)
                                        & ~tile_bits[code//3] for code in range(count)))
            groups_row.append(tuple((source, by_side[other].get(reversed_masks[mask], 0))
                                    for mask, source in by_side[side].items()))
        compatible.append(tuple(compatible_row))
        groups.append(tuple(groups_row))
    arcs = tuple(problem.edges) + tuple((b, sb, a, sa) for a, sa, b, sb in problem.edges)
    incoming = [[] for _ in range(n)]
    for index, arc in enumerate(arcs):
        incoming[arc[2]].append(index)
    return _PropagationTables(problem.fingerprint, target, tuple(initial), tile_bits,
                              tuple(compatible), tuple(groups), arcs, tuple(map(tuple, incoming)))


def _filter_arc(domain, neighbor, groups, compatible, tile_bits):
    """Same seam/tile inequality as per-code scanning, grouped for dense rows.

    Several supporting rotations of one physical tile still support no code of
    that same tile. Two distinct supporting tile IDs support every source tile.
    Sparse domains use the original per-code table to avoid scanning all masks.
    """
    keep = 0
    if domain.bit_count()*2 <= len(groups):
        for code in _values(domain):
            if compatible[code] & neighbor:
                keep |= 1 << code
        return keep
    for source, support in groups:
        candidates = source & domain
        if not candidates:
            continue
        present = support & neighbor
        if not present:
            continue
        tile = ((present & -present).bit_length()-1)//3
        owned = tile_bits[tile]
        if not present & ~owned:
            candidates &= ~owned
        keep |= candidates
    return keep


class _ArcFilter:
    """Bounded exact memoization; keys include both complete input domains.

    Each object belongs to one solve invocation and one immutable support table.
    Revisited siblings need no monotonicity assumption, and caches are never
    trusted across puzzle changes, serialized checkpoints, or process restarts.
    """
    def __init__(self, tables, limit=512):
        self.tables, self.limit = tables, limit
        self.results = tuple(tuple({} for _ in range(3)) for _ in range(3))

    def __call__(self, domain, neighbor, side, other):
        cache = self.results[side][other]
        key = domain, neighbor
        keep = cache.get(key)
        if keep is None:
            tables = self.tables
            keep = _filter_arc(domain, neighbor, tables.groups[side][other],
                               tables.compatible[side][other], tables.tile_bits)
            if len(cache) >= self.limit:
                cache.clear()  # Eviction changes only speed, never propagation.
            cache[key] = keep
        return keep


def _jump_conflict(stack, decisions, clause):
    """Discard only subtrees whose ancestor decisions imply a proved conflict.

    The ordinary failing leaf/frame has already been consumed. Each retained
    frame keeps its original complete sibling list and next-choice cursor.
    """
    positions = {literal: index for index, literal in enumerate(decisions)}
    if any(literal not in positions for literal in clause):
        return 0
    keep = max((positions[literal]+1 for literal in clause), default=0)
    removed = max(0, len(stack)-keep)
    if removed:
        del stack[keep:]
    return removed


def _minimize_conflict(decisions, prove, *, attempts=4):
    """Delete assumptions only after a completed independent contradiction.

    ``prove`` must start from the immutable effective initial domains, not the
    failed frame. An inconclusive/budget-limited probe must return False or
    raise; no speculative deletion is ever yielded.
    """
    clause = list(decisions)
    for literal in list(reversed(decisions))[:attempts]:
        trial = [item for item in clause if item != literal]
        if prove(trial):
            clause = trial
        yield tuple(clause)


def _starting_arcs(count, incoming, changed_cell):
    # Only in-memory children of a completely propagated parent may use this
    # shortcut. Resumed/partially propagated frames rebuild all arcs; this
    # scheduling hint is deliberately absent from every saved checkpoint.
    return range(count) if changed_cell is None else incoming[changed_cell]


def _checkpoint_digest(payload):
    body={key:value for key,value in payload.items() if key!='digest'}
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',', ':'),allow_nan=False).encode()).hexdigest()


def _values(bits):
    while bits:
        low = bits & -bits
        yield low.bit_length() - 1
        bits ^= low


def _solve_public(problem, *, seconds=10, seed=0, target='gold', hints=None, stop=None, progress=None, resume=None, learning_state=None, options=None):
    """Apply verified fresh-root symmetry, then run the exact resumable search.

    Metadata-free v2 checkpoints retain their original search space. A pending
    record contains only an untouched root; no cache or search work is committed
    until symmetry setup completes. Ready records are independently recomputed.
    """
    began = time.monotonic()
    options = normalize_options(resume.get('search_options') if options is None and isinstance(resume,dict) else options)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative; zero gives an immediate timeout.')
    if target not in ('gold', 'edges'):
        raise ValueError('target must be gold or edges.')
    if type(seed) is not int:
        raise ValueError('seed must be an integer.')
    source_domains = [sum(1 << code for code in domain) for domain in problem.domains]
    source_hash = hashlib.sha256(json.dumps(source_domains, separators=(',', ':')).encode()).hexdigest()
    if learning_state is not None:
        if not isinstance(learning_state, dict):
            raise ValueError('learning_state must be a trusted in-memory dict, not a serialized checkpoint.')
        if learning_state:
            store = learning_state.get('_nogoods')
            if (not set(learning_state) <= {'_nogoods', '_support', '_native', '_runtime', '_knowledge', '_shared_conflicts', '_mask_conflicts', '_certificates', '_regional_cache_dir', '_shared_sat', '_reasoning_cache_dir', '_sat_service'}
                    or ('_nogoods' not in learning_state and not set(learning_state)<={'_knowledge','_shared_conflicts','_mask_conflicts','_certificates','_regional_cache_dir', '_shared_sat', '_reasoning_cache_dir', '_sat_service'})
                    or (store is not None and (not isinstance(store, _NogoodStore)
                        or store.fingerprint != problem.fingerprint or store.target != target))):
                raise ValueError('The in-memory learning cache has different puzzle, target, or effective domains, or is malformed.')
            if '_knowledge' in learning_state and (not isinstance(learning_state['_knowledge'],KnowledgeStore)
                    or not learning_state['_knowledge'].matches(problem,target)):
                raise ValueError('The shared Gold-loop knowledge differs from the puzzle, target, or effective domains.')
            if '_shared_conflicts' in learning_state and (not isinstance(learning_state['_shared_conflicts'],VerifiedConflictStore)
                    or not learning_state['_shared_conflicts'].matches(problem,target)):
                raise ValueError('Shared conflicts differ from the original puzzle domains or target.')
            if '_shared_sat' in learning_state and (not isinstance(learning_state['_shared_sat'],SharedSatAssistant) or not learning_state['_shared_sat'].matches(problem,target)):
                raise ValueError('The shared SAT assistant differs from the original puzzle/domains/objective.')
            if '_runtime' in learning_state and not isinstance(learning_state['_runtime'],_SearchRuntime):
                raise ValueError('The in-memory search runtime cache is malformed.')
            if '_native' in learning_state and not isinstance(learning_state['_native'], NativePropagation):
                raise ValueError('The in-memory native propagation cache is malformed.')
            if '_support' in learning_state and not isinstance(learning_state['_support'], _PropagationTables):
                raise ValueError('The in-memory seam support cache is malformed.')
    if resume is not None:
        if (not isinstance(resume, dict) or resume.get('version') != VERSION
                or resume.get('fingerprint') != problem.fingerprint or resume.get('target') != target):
            raise ValueError('Checkpoint version, puzzle, fixed placements, or target differs.')
        if not isinstance(resume.get('digest'), str) or resume['digest'] != _checkpoint_digest(resume):
            raise ValueError('Checkpoint integrity digest differs; saved progress may be damaged.')
        metadata = copy.deepcopy(resume.get('root_symmetry'))
        if 'root_symmetry' in resume:
            if (not isinstance(metadata, dict) or type(metadata.get('version')) is not int
                    or metadata['version'] != 1 or metadata.get('state') not in ('pending', 'ready')):
                raise ValueError('Unknown or malformed root symmetry metadata.')
            expected_keys = {'version', 'state', 'input_domains_sha256'}
            if metadata['state'] == 'ready':
                expected_keys.add('reduction')
            if set(metadata) != expected_keys or metadata.get('input_domains_sha256') != source_hash:
                raise ValueError('Root symmetry metadata or original cell domains differ.')
            if metadata['state'] == 'pending':
                untouched = {'domains': [hex(value) for value in source_domains], 'cell': -1, 'choices': [], 'next': 0}
                if (type(resume.get('nodes')) is not int or resume['nodes'] != 0
                        or resume.get('stack') != [untouched]
                        or not isinstance(resume.get('stats'), dict)
                        or any(type(value) is not int or value != 0 for value in resume['stats'].values())):
                    raise ValueError('Pending symmetry requires an untouched search root.')
            elif not isinstance(metadata.get('reduction'), dict):
                raise ValueError('Malformed root symmetry reduction metadata.')
    else:
        metadata = {'version': 1, 'state': 'pending', 'input_domains_sha256': source_hash}
    effective, current_resume = source_domains, resume
    deferred_local_masks = None
    interrupted = None
    class SetupInterrupted(Exception):
        pass
    def check_setup():
        nonlocal interrupted
        if stop is not None and stop():
            interrupted = 'stopped'
            raise SetupInterrupted
        if time.monotonic()-began >= seconds:
            interrupted = 'timeout'
            raise SetupInterrupted
    if metadata is not None:
        try:
            effective, expected = canonical_root_domains(problem, check=check_setup)
            if metadata['state'] == 'ready':
                if (json.dumps(metadata['reduction'], sort_keys=True, separators=(',', ':'), allow_nan=False)
                        != json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False)):
                    raise ValueError('Root symmetry reduction differs from the verified problem.')
            else:
                metadata = {**metadata, 'state': 'ready', 'reduction': expected}
                if resume is not None:
                    # Nothing was searched in a pending checkpoint. Narrow its
                    # untouched root atomically, retaining its seed/hints/time.
                    current_resume = copy.deepcopy(resume)
                    current_resume['stack'][0]['domains'] = [hex(value) for value in effective]
                    current_resume['owned_root_domains'] = [hex(value) for value in effective]
                    # Canonical preparation changes the exact verifier root.
                    # Optional prior claims must not cross that scope boundary.
                    current_resume.pop('local_mask_conflicts', None)
                    current_resume['root_symmetry'] = copy.deepcopy(metadata)
                    current_resume['digest'] = _checkpoint_digest(current_resume)
        except SetupInterrupted:
            # No cache has been touched. Existing ready metadata is retained for
            # verification on the next call; no propagation is allowed here.
            effective = source_domains
            if current_resume is not None and 'local_mask_conflicts' in current_resume:
                # A zero-budget validation cannot recompute the effective root.
                # Preserve scoped claims for a later verified resume, without
                # importing them into a temporary original-domain verifier.
                deferred_local_masks = copy.deepcopy(current_resume['local_mask_conflicts'])
                current_resume = copy.deepcopy(current_resume)
                current_resume.pop('local_mask_conflicts')
                current_resume['digest'] = _checkpoint_digest(current_resume)
    preparation = time.monotonic()-began
    def notify(event):
        if progress is not None:
            adjusted = {**event, 'elapsed_seconds':preparation+event['elapsed_seconds']}
            if metadata is not None:
                adjusted['root_symmetry'] = copy.deepcopy(metadata)
            progress(adjusted)
    result = _solve_core(problem, seconds=0 if interrupted else max(0.0, seconds-preparation),
                         seed=seed, target=target, hints=hints,
                         stop=(lambda: True) if interrupted == 'stopped' else stop,
                         progress=notify if progress is not None else None,
                         resume=current_resume, learning_state=None if interrupted else learning_state,
                         _initial_domains=effective, options=options)
    result['elapsed_seconds'] = time.monotonic()-began
    if deferred_local_masks is not None:
        result['local_mask_conflicts'] = copy.deepcopy(deferred_local_masks)
    if metadata is not None:
        result['root_symmetry'] = copy.deepcopy(metadata)
    if 'checkpoint' in result:
        checkpoint = result['checkpoint']
        checkpoint['elapsed_seconds'] += preparation
        if deferred_local_masks is not None:
            checkpoint['local_mask_conflicts'] = copy.deepcopy(deferred_local_masks)
        if metadata is not None:
            checkpoint['root_symmetry'] = copy.deepcopy(metadata)
        checkpoint['digest'] = _checkpoint_digest(checkpoint)
    return result


def _solve_core(problem, *, seconds=10, seed=0, target='gold', hints=None, stop=None, progress=None,
                resume=None, learning_state=None, _initial_domains=None, options=None):
    """Search exactly; learning_state is an optional trusted in-memory slice cache.

    Pass the same empty-at-first dict across calls to retain proved clauses. It
    is deliberately not JSON serializable and must not be loaded from disk.
    Losing it at process restart affects performance, never search coverage.
    """
    began = time.monotonic()
    options = normalize_options(resume.get('search_options') if options is None and isinstance(resume,dict) else options)
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds < 0:
        raise ValueError('seconds must be finite and nonnegative; zero gives an immediate timeout.')
    if target not in ('gold', 'edges'):
        raise ValueError('target must be gold or edges.')
    if type(seed) is not int:
        raise ValueError('seed must be an integer.')
    deadline = began + seconds
    n, count = problem.n, 3*problem.n
    initial = ([sum(1 << code for code in domain) for domain in problem.domains]
               if _initial_domains is None else list(_initial_domains))
    stats = dict.fromkeys(BASE_STAT_KEYS + PRUNER_STAT_KEYS + CONFLICT_STAT_KEYS + CONNECTIVITY_STAT_KEYS + NATIVE_STAT_KEYS + GUIDANCE_STAT_KEYS + ADVANCED_STAT_KEYS + EXPLANATION_STAT_KEYS + SCHEDULING_STAT_KEYS + KNOWLEDGE_STAT_KEYS + REASON_STAT_KEYS + SAT_ASSIST_STAT_KEYS + ORDERING_STAT_KEYS + SHARED_STAT_KEYS + BRANCH_STAT_KEYS + LCG_STAT_KEYS + GOLD_ORACLE_STAT_KEYS + NATIVE_SEARCH_STAT_KEYS + NATIVE_LEARNING_STAT_KEYS + GROUP_STAT_KEYS + DEEP_STAT_KEYS + POLICY_STAT_KEYS + MASK_STAT_KEYS + RESEARCH_STAT_KEYS + ROUTING_STAT_KEYS + RESIDUAL_STAT_KEYS + HIGHER_STAT_KEYS + REGION26_STATS + ARCHIVE_STATS + SERVICE_STATS + WITNESS_STATS + RELATIONAL_STATS + ('lcg_trace_budget_skips',), 0)
    nodes, previous_elapsed = 0, 0.0
    if resume is not None:
        if not isinstance(resume, dict) or resume.get('version') != VERSION or resume.get('fingerprint') != problem.fingerprint or resume.get('target') != target:
            raise ValueError('Checkpoint version, puzzle, fixed placements, or target differs.')
        if not isinstance(resume.get('digest'), str) or resume['digest'] != _checkpoint_digest(resume):
            raise ValueError('Checkpoint integrity digest differs; saved progress may be damaged.')
        if not {'seed', 'stats', 'nodes', 'stack'} <= resume.keys():
            raise ValueError('Checkpoint is missing search state.')
        seed = resume['seed']
        if type(seed) is not int or type(resume['nodes']) is not int or resume['nodes'] < 0:
            raise ValueError('Malformed checkpoint seed or node count.')
        if not isinstance(resume['stats'], dict) or not set(BASE_STAT_KEYS) <= set(resume['stats']) <= set(stats) or any(type(value) is not int or value < 0 for value in resume['stats'].values()):
            raise ValueError('Malformed checkpoint counters.')
        hints = resume.get('hints')
        stats.update(resume['stats'])
        nodes = resume['nodes']
        previous_elapsed = resume.get('elapsed_seconds', 0)
        if isinstance(previous_elapsed, bool) or not isinstance(previous_elapsed, (int, float)) or not math.isfinite(previous_elapsed) or previous_elapsed < 0:
            raise ValueError('Malformed checkpoint elapsed time.')
        if not isinstance(resume['stack'], list) or not resume['stack']:
            raise ValueError('An incomplete checkpoint must retain an unfinished search stack.')
        # Local nogoods and verifier assumptions belong to the job's owned
        # root, not the broader puzzle fingerprint. Preserve this scope across
        # propagation and warm slices. Old v2 snapshots conservatively own only
        # their first frame; losing a broader cache cannot lose pending work.
        owned = resume.get('owned_root_domains',resume['stack'][0].get('domains') if isinstance(resume['stack'][0],dict) else None)
        if not isinstance(owned,list) or len(owned)!=n or any(not isinstance(value,str) for value in owned):
            raise ValueError('Malformed checkpoint owned root.')
        try:owned=[int(value,16) for value in owned]
        except ValueError:raise ValueError('Malformed checkpoint owned root.') from None
        source=[sum(1<<code for code in row) for row in problem.domains]
        if any(value<0 or value&~allowed for value,allowed in zip(owned,source)):
            raise ValueError('Checkpoint owned root exceeds puzzle domains.')
        initial=[value&allowed for value,allowed in zip(owned,initial)]
        stack = []
        for saved in resume['stack']:
            if not isinstance(saved, dict) or not {'domains', 'cell', 'choices', 'next'} <= saved.keys() or not isinstance(saved['domains'], list) or any(not isinstance(value, str) for value in saved['domains']):
                raise ValueError('Malformed checkpoint search frame.')
            domains = [int(value, 16) for value in saved['domains']]
            if len(domains) != n or any(value < 0 or value & ~initial[i] for i, value in enumerate(domains)):
                raise ValueError('Checkpoint contains invalid cell domains.')
            cell, choices, offset = saved['cell'], saved['choices'], saved['next']
            if type(cell) is not int or not -1 <= cell < n or not isinstance(choices, list) or type(offset) is not int or not 0 <= offset <= len(choices):
                raise ValueError('Malformed checkpoint search frame.')
            if cell == -1 and (choices or offset):
                raise ValueError('An unexpanded checkpoint frame has choices.')
            if any(type(code) is not int or not 0 <= code < count or cell < 0 or not domains[cell] & (1 << code) for code in choices) or len(set(choices)) != len(choices):
                raise ValueError('Malformed checkpoint sibling choices.')
            masks = saved.get('choice_masks')
            if masks is not None:
                if (cell<0 or not isinstance(masks,list) or len(masks)!=len(choices)
                    or len(masks)<2 or any(not isinstance(value,str) for value in masks)):
                    raise ValueError('Malformed checkpoint connection choices.')
                masks=[int(value,16) for value in masks];union=0
                for code,mask in zip(choices,masks):
                    if mask<=0 or mask&~domains[cell] or union&mask or code!=(mask&-mask).bit_length()-1:
                        raise ValueError('Checkpoint connection choices must be disjoint nonempty domains.')
                    union |= mask
                if union!=domains[cell]:raise ValueError('Checkpoint connection choices omit remaining orientations.')
            elif cell >= 0 and (domains[cell].bit_count() <= 1 or set(choices) != set(_values(domains[cell]))):
                raise ValueError('Checkpoint does not retain all sibling choices.')
            if stack:
                parent = stack[-1]
                if parent['cell'] < 0 or parent['next'] == 0:
                    raise ValueError('Checkpoint child has no selected parent branch.')
                selected = branch_mask(parent,parent['next']-1)
                if domains[parent['cell']] & ~selected or any(value & ~old for value, old in zip(domains, parent['domains'])):
                    raise ValueError('Checkpoint child differs from its selected branch.')
            frame={'domains': domains, 'cell': cell, 'choices': choices.copy(), 'next': offset}
            if masks is not None:frame['choice_masks']=masks
            stack.append(frame)
    else:
        stack = [{'domains': initial.copy(), 'cell': -1, 'choices': [], 'next': 0}]
    if hints is None:
        normalized_hints = {}
    elif isinstance(hints, dict):
        normalized_hints = {int(cell): ([codes] if type(codes) is int else list(codes)) for cell, codes in hints.items()}
    else:
        if len(hints) != n:
            raise ValueError('Hints must have one preferred orientation code per cell.')
        normalized_hints = {cell: [code] for cell, code in enumerate(hints)}
    for cell, codes in normalized_hints.items():
        if not 0 <= cell < n or any(type(code) is not int or not -1 <= code < count for code in codes):
            raise ValueError('Invalid hint cell or orientation code.')
    randomizer = random.Random(seed)
    priority = list(range(count)); randomizer.shuffle(priority)
    rank = {code: index for index, code in enumerate(priority)}
    cell_order = list(range(n)); randomizer.shuffle(cell_order)
    cell_rank = {cell: index for index, cell in enumerate(cell_order)}
    last_progress = began
    if learning_state is None:
        learning_state = {}
    if not isinstance(learning_state, dict):
        raise ValueError('learning_state must be a trusted in-memory dict, not a serialized checkpoint.')
    if not set(learning_state) <= {'_nogoods', '_support', '_native', '_runtime', '_knowledge', '_shared_conflicts', '_mask_conflicts', '_certificates', '_regional_cache_dir', '_shared_sat', '_reasoning_cache_dir', '_sat_service'}:
        raise ValueError('The in-memory learning cache is malformed.')
    if '_nogoods' in learning_state:
        store = learning_state.get('_nogoods')
        if (not {'_nogoods'} <= set(learning_state) <= {'_nogoods', '_support', '_native', '_runtime', '_knowledge', '_shared_conflicts', '_mask_conflicts', '_certificates', '_regional_cache_dir', '_shared_sat', '_reasoning_cache_dir', '_sat_service'}
                or not isinstance(store, _NogoodStore)
                or store.fingerprint != problem.fingerprint or store.target != target
                or store.initial_domains != tuple(initial)):
            raise ValueError('The in-memory learning cache has different puzzle, target, or effective domains, or is malformed.')
    else:
        store = _NogoodStore(problem.fingerprint, target, initial_domains=initial)
        learning_state['_nogoods'] = store
    symmetry_scope=bool(options['piece_symmetry'])
    if hasattr(store,'symmetry_scope') and store.symmetry_scope!=symmetry_scope:
        raise ValueError('Clear the in-memory learning cache before changing piece symmetry assumptions.')
    store.symmetry_scope=symmetry_scope
    knowledge = learning_state.get('_knowledge')
    if knowledge is not None and (not isinstance(knowledge,KnowledgeStore) or not knowledge.matches(problem,target)):
        raise ValueError('The shared Gold-loop knowledge differs from the puzzle, target, or effective domains.')
    if knowledge is None and options.get('persistent_learning',True) and target=='gold':
        knowledge=KnowledgeStore(problem,target=target)
        learning_state['_knowledge']=knowledge
    if knowledge is not None and options.get('persistent_learning',True):
        for clause in knowledge.records():store.learn([tuple(pair) for pair in clause],stats)
    shared = learning_state.get('_shared_conflicts')
    if shared is not None and (not isinstance(shared,VerifiedConflictStore) or not shared.matches(problem,target)):
        raise ValueError('Shared conflicts differ from the original puzzle domains or target.')
    if options['shared_conflicts']:
        if shared is None:
            shared=VerifiedConflictStore(problem,target)
            learning_state['_shared_conflicts']=shared
    else:shared=None
    imported_shared=set()
    if shared is not None:
        for clause in shared.records():
            record=tuple(tuple(pair) for pair in clause)
            store.learn(record,stats);imported_shared.add(record)
    shared_baseline={key:stats[key] for key in SHARED_STAT_KEYS}
    shared_entry=dict(shared.stats) if shared is not None else dict.fromkeys(SHARED_STAT_KEYS,0)
    last_shared_work=0.0
    support_tables = learning_state.get('_support')
    if ('_support' in learning_state and
            (not isinstance(support_tables, _PropagationTables)
             or support_tables.fingerprint != problem.fingerprint
             or support_tables.target != target
             or support_tables.initial_domains != tuple(initial))):
        raise ValueError('The in-memory seam support cache has different effective problem data or is malformed.')
    if '_native' in learning_state and (not isinstance(learning_state['_native'],NativePropagation)
                                       or learning_state['_native'].tables is not support_tables):
        raise ValueError('The in-memory native propagation cache has different support tables or is malformed.')
    pruner, connectivity, advanced = None, None, None
    native, native_backend, native_reason = None, 'python', 'Propagation has not started.'
    runtime = learning_state.get('_runtime')
    if runtime is None:
        runtime = _SearchRuntime(problem.fingerprint,target,initial,n)
        learning_state['_runtime'] = runtime
    elif not isinstance(runtime,_SearchRuntime) or not runtime.valid(problem.fingerprint,target,initial,n):
        raise ValueError('The in-memory search runtime cache has different effective problem data or is malformed.')
    archive_was_loaded=False
    reasoning_path=learning_state.get('_reasoning_cache_dir')
    if reasoning_path is not None and not isinstance(reasoning_path,(str,Path)):
        raise ValueError('Reasoning cache directory must be a local path.')
    if options['durable_reasoning'] and reasoning_path is not None and seconds>0:
        directory=str(Path(reasoning_path).resolve())
        if runtime.reasoning_archive is None or runtime.reasoning_directory!=directory:
            runtime.reasoning_archive=CPUReasoningArchive(directory,problem.fingerprint,target,initial,
                original=[sum(1<<code for code in row) for row in problem.domains])
            runtime.reasoning_directory=directory
            archive_was_loaded=True
            if runtime.lcg is None:runtime.lcg=runtime.reasoning_archive.load_lcg()
            if not runtime.probes.entries:
                restored_probes=runtime.reasoning_archive.load_probes()
                if restored_probes is not None:runtime.probes=restored_probes
    archive=runtime.reasoning_archive if options['durable_reasoning'] else None
    shared_sat=learning_state.get('_shared_sat')
    if shared_sat is not None and (not isinstance(shared_sat,SharedSatAssistant) or not shared_sat.matches(problem,target)):
        raise ValueError('The shared SAT assistant differs from the original puzzle/domains/objective.')
    shared_service=learning_state.get('_sat_service')
    if options['shared_sat_service'] and shared_service is not None:
        if runtime.service_view is not shared_service or not isinstance(runtime.sat,SATServiceClient):
            if runtime.sat is not None:runtime.sat.close()
            runtime.sat=service_view(shared_service,problem,target,initial);runtime.service_view=shared_service
    elif isinstance(runtime.sat,SATServiceClient):
        runtime.sat.close();runtime.sat=None;runtime.service_view=None
    if options['shared_sat_assistant'] and not (options['shared_sat_service'] and shared_service is not None):
        if shared_sat is None:
            shared_sat=SharedSatAssistant(problem,target);learning_state['_shared_sat']=shared_sat
        if not isinstance(runtime.sat,JobSatAssist) or runtime.sat.owner is not shared_sat:
            if runtime.sat is not None:runtime.sat.close()
            runtime.sat=shared_sat.view(initial)
    elif isinstance(runtime.sat,JobSatAssist):
        runtime.sat=SatAssist(problem,initial,target,witness_pool=options['witness_pool'],support_cover=options['support_cover'])
    if runtime.sat is not None and not isinstance(runtime.sat,SATServiceClient) and (not isinstance(runtime.sat,(SatAssist,JobSatAssist)) or runtime.sat.initial!=tuple(initial) or runtime.sat.problem.fingerprint!=problem.fingerprint or runtime.sat.target!=target):
        raise ValueError('The private SAT assistance cache has different problem data.')
    if runtime.lcg is not None and (not isinstance(runtime.lcg,LCGStore) or not runtime.lcg.matches(problem.fingerprint,target,initial)):
        raise ValueError('The LCG store belongs to different effective root domains.')
    if any(options[k] for k in ('deep_paths','conservation','retained_relations','joint_conservation')):
        if runtime.deep is None or bool(runtime.deep.relations is not None)!=options['retained_relations']:
            runtime.deep=DeepPruner(problem,retain_relations=options['retained_relations'])
    if options['piece_symmetry'] and runtime.piece_symmetry is None:runtime.piece_symmetry=PieceSymmetry(problem)
    if options['proof_certificates']:
        runtime.certificates=learning_state.get('_certificates') or runtime.certificates or CertificateStore(problem,target)
        if not isinstance(runtime.certificates,CertificateStore) or runtime.certificates.problem.fingerprint!=problem.fingerprint or runtime.certificates.target!=target:
            raise ValueError('Portable certificate store has different puzzle or target.')
        learning_state['_certificates']=runtime.certificates
        if resume is not None and resume.get('certificates') is not None:runtime.certificates.load_snapshot(resume['certificates'])
    if options['smart_learning'] and all(initial) and all(problem.domains):
        runtime.mask_conflicts=learning_state.get('_mask_conflicts',runtime.mask_conflicts)
        if runtime.mask_conflicts is None:runtime.mask_conflicts=VerifiedMaskConflictStore(problem,target,proof_transport=options['proof_transport'],resumable_proof_checking=options.get('resumable_proof_checking',False),mixed_proof_dag=options.get('mixed_proof_dag',False),proof_indexing=options.get('proof_indexing',False))
        if not isinstance(runtime.mask_conflicts,VerifiedMaskConflictStore) or not runtime.mask_conflicts.matches(problem,target):
            raise ValueError('Mask conflict store differs from original puzzle scope.')
        learning_state['_mask_conflicts']=runtime.mask_conflicts
        if runtime.local_mask_conflicts is None:
            runtime.local_mask_conflicts=VerifiedMaskConflictStore(problem,target,initial=initial,proof_transport=options['proof_transport'],resumable_proof_checking=options.get('resumable_proof_checking',False),mixed_proof_dag=options.get('mixed_proof_dag',False),proof_indexing=options.get('proof_indexing',False))
            if resume is not None and resume.get('local_mask_conflicts') is not None and 'owned_root_domains' in resume:
                runtime.local_mask_conflicts.import_snapshot(resume['local_mask_conflicts'])
        elif not runtime.local_mask_conflicts.matches(problem,target,initial):
            raise ValueError('Local proof root changed.')
    for mask_store in (runtime.mask_conflicts,runtime.local_mask_conflicts):
        if mask_store is not None:
            mask_store.configure_policy(proof_transport=options['proof_transport'],
                resumable_proof_checking=options['resumable_proof_checking'],mixed_proof_dag=options['mixed_proof_dag'],proof_indexing=options['proof_indexing'])
            mask_store.configure_isolated_producer(options['isolated_proof_producer'])
            if options['durable_reasoning'] and reasoning_path is not None:mask_store.configure_local_cache(reasoning_path)
    if options['mixed_proof_dag'] and knowledge is not None:
        for mask_store in (runtime.mask_conflicts,runtime.local_mask_conflicts):
            if mask_store is not None:mask_store.add_gold_witnesses(knowledge.records())
    if options['lcg'] and runtime.lcg is None:runtime.lcg=LCGStore(problem.fingerprint,target,initial)
    lcg_store=runtime.lcg if options['lcg'] else None
    if options['gold_oracle']:
        if runtime.gold_oracle is None:runtime.gold_oracle=GoldOracle(problem,initial,target)
        if not isinstance(runtime.gold_oracle,GoldOracle) or runtime.gold_oracle.problem is not problem or runtime.gold_oracle.initial!=tuple(initial) or runtime.gold_oracle.target!=target:
            raise ValueError('The Gold oracle belongs to a different puzzle or effective root.')
        runtime.gold_oracle.begin_slice()
    gold_oracle=runtime.gold_oracle if options['gold_oracle'] else None
    if options['residual_reuse'] and all(initial):
        if runtime.residual is None:runtime.residual=ResidualProofCache(problem,target,root_scope=tuple(initial))
        if (not isinstance(runtime.residual,ResidualProofCache) or runtime.residual.problem is not problem
                or runtime.residual.target!=target or runtime.residual.root_scope!=tuple(initial)):
            raise ValueError('Residual proofs differ from the effective puzzle/root scope.')
    residual=runtime.residual if options['residual_reuse'] else None
    residual_entry=dict(residual.stats) if residual is not None else dict.fromkeys(RESIDUAL_STAT_KEYS,0)
    residual_baseline={key:stats[key] for key in RESIDUAL_STAT_KEYS}
    lcg_baseline={key:stats[key] for key in LCG_STAT_KEYS}
    lcg_entry=dict(lcg_store.stats) if lcg_store is not None else dict.fromkeys(LCG_STAT_KEYS,0)
    oracle_baseline={key:stats[key] for key in GOLD_ORACLE_STAT_KEYS}
    oracle_entry=dict(gold_oracle.stats) if gold_oracle is not None else dict.fromkeys(GOLD_ORACLE_STAT_KEYS,0)
    failure_weights = runtime.failure_weights
    probe_deadline = None
    pruner_baseline = {key: stats[key] for key in PRUNER_STAT_KEYS}
    connectivity_baseline = {key: stats[key] for key in CONNECTIVITY_STAT_KEYS}
    advanced_baseline = {key:stats[key] for key in ADVANCED_STAT_KEYS}
    advanced_entry = dict.fromkeys(ADVANCED_STAT_KEYS,0)
    sat_baseline={key:stats[key] for key in SAT_ASSIST_STAT_KEYS}
    sat_entry=dict(runtime.sat.stats) if runtime.sat is not None else dict.fromkeys(SAT_ASSIST_STAT_KEYS,0)

    higher_entry=dict(runtime.higher.stats) if runtime.higher is not None else dict.fromkeys(HIGHER_STAT_KEYS,0)
    higher_baseline={key:stats[key] for key in HIGHER_STAT_KEYS}
    deep_entry=dict(runtime.deep.stats) if runtime.deep is not None else dict.fromkeys(DEEP_STAT_KEYS,0)
    deep_baseline={key:stats[key] for key in DEEP_STAT_KEYS}
    mask_entry=dict(runtime.mask_conflicts.stats) if runtime.mask_conflicts is not None else dict.fromkeys(MASK_STAT_KEYS,0)
    mask_baseline={key:stats[key] for key in MASK_STAT_KEYS}
    research_baseline={key:stats[key] for key in RESEARCH_STAT_KEYS}
    research_entry=dict(runtime.research.stats) if runtime.research is not None else dict.fromkeys(RESEARCH_STAT_KEYS,0)
    regional26_baseline={key:stats[key] for key in REGION26_STATS}
    regional26_entry=runtime.regional26.stats if runtime.regional26 is not None else dict.fromkeys(REGION26_STATS,0)
    archive_baseline={key:stats[key] for key in ARCHIVE_STATS}
    archive_entry=dict(archive.local.stats) if archive is not None and not archive_was_loaded else dict.fromkeys(ARCHIVE_STATS,0)
    service_baseline={key:stats[key] for key in SERVICE_STATS}
    service_entry=dict(runtime.sat.service_stats) if isinstance(runtime.sat,SATServiceClient) else dict.fromkeys(SERVICE_STATS,0)
    witness_baseline={key:stats[key] for key in WITNESS_STATS}
    witness_entry=(dict(runtime.sat.witness_stats) if isinstance(runtime.sat,SATServiceClient) else
                   dict(runtime.semantic_pool.stats) if runtime.semantic_pool is not None else dict.fromkeys(WITNESS_STATS,0))
    relational_baseline={key:stats[key] for key in RELATIONAL_STATS}
    relational_entry=dict(runtime.higher.relations.stats) if runtime.higher is not None else dict.fromkeys(RELATIONAL_STATS,0)
    certificate_baseline={key:stats[key] for key in RELATION_CERTIFICATE_STATS}
    certificate_entry=(dict(runtime.higher.certificates.stats) if runtime.higher is not None and runtime.higher.certificates is not None else dict.fromkeys(RELATION_CERTIFICATE_STATS,0))
    probe_plan_baseline={key:stats[key] for key in PROBE_PLAN_STATS}
    probe_plan_entry=dict(runtime.probes.plans.stats)
    def sync_stats():
        if runtime.higher is not None and runtime.higher.certificates is not None:
            for key in RELATION_CERTIFICATE_STATS:stats[key]=certificate_baseline[key]+runtime.higher.certificates.stats[key]-certificate_entry[key]
        for key in PROBE_PLAN_STATS:stats[key]=probe_plan_baseline[key]+runtime.probes.plans.stats[key]-probe_plan_entry[key]
        if isinstance(runtime.sat,SATServiceClient):
            for key in SERVICE_STATS:stats[key]=service_baseline[key]+runtime.sat.service_stats[key]-service_entry[key]
        witness_current=(runtime.sat.witness_stats if isinstance(runtime.sat,SATServiceClient) else
                         runtime.semantic_pool.stats if runtime.semantic_pool is not None else None)
        if witness_current is not None:
            for key in WITNESS_STATS:stats[key]=witness_baseline[key]+witness_current[key]-witness_entry[key]
        if runtime.higher is not None:
            for key in RELATIONAL_STATS:stats[key]=relational_baseline[key]+runtime.higher.relations.stats[key]-relational_entry[key]
        if runtime.regional26 is not None:
            current=runtime.regional26.stats
            for key in REGION26_STATS:stats[key]=regional26_baseline[key]+current[key]-regional26_entry[key]
        if archive is not None:
            for key in ARCHIVE_STATS:stats[key]=archive_baseline[key]+archive.local.stats[key]-archive_entry[key]
        if runtime.higher is not None:
            for key in HIGHER_STAT_KEYS:stats[key]=higher_baseline[key]+runtime.higher.stats[key]-higher_entry[key]
        if residual is not None:
            for key,value in residual.stats.items():
                stats[key]=value if key=='residual_closed' else residual_baseline[key]+value-residual_entry[key]
        if runtime.research is not None:
            for key,value in runtime.research.stats.items():
                stats[key]=(max(research_baseline[key],value) if key in RESEARCH_MAX_GAUGES else research_baseline[key]+value-research_entry[key])
        if runtime.deep is not None:
            for key in ('path_checks','path_pairs_removed','path_prunes','conservation_checks','conservation_prunes','relation_records','relation_hits','relation_prunes','joint_conservation_checks','joint_conservation_prunes','capacity_edges'):stats[key]=deep_baseline[key]+runtime.deep.stats[key]-deep_entry[key]
        if runtime.mask_conflicts is not None:
            for key in MASK_STAT_KEYS:stats[key]=(runtime.mask_conflicts.stats[key] if key in MASK_GAUGES else mask_baseline[key]+runtime.mask_conflicts.stats[key]-mask_entry[key])
        if pruner is not None:
            # Completed lazy profiles remain valid even if a later propagation
            # check stopped this slice. Partial builds never replace tables.
            runtime.pruner_tables = pruner.tables
            for key in PRUNER_STAT_KEYS:
                stats[key] = (pruner.stats[key] if key in ('forced_bonds', 'forced_clusters')
                              else pruner_baseline[key] + pruner.stats[key])
        if connectivity is not None:
            for key in CONNECTIVITY_STAT_KEYS:
                stats[key] = connectivity_baseline[key] + connectivity.stats[key]
        if advanced is not None:
            for key in ADVANCED_STAT_KEYS:
                stats[key] = (advanced.stats[key] if key in ('regional_cache_entries','separator_cache_entries','hierarchical_cache_entries', 'hierarchical_subtable_entries','hierarchical_subtable_states','hierarchical_subtable_support_slots','hierarchical_cached_summaries','hierarchical_cached_support_slots') else
                              advanced_baseline[key]+advanced.stats[key]-advanced_entry[key])
        if runtime.sat is not None:
            for key in SAT_ASSIST_STAT_KEYS:
                stats[key]=(runtime.sat.stats[key] if key=='sat_assist_clauses' else sat_baseline[key]+runtime.sat.stats[key]-sat_entry[key])
        if shared is not None:
            for key in SHARED_STAT_KEYS:
                stats[key]=shared.stats[key] if key in SHARED_GAUGES else shared_baseline[key]+shared.stats[key]-shared_entry[key]
        if lcg_store is not None:
            for key in LCG_STAT_KEYS:
                if key not in ('lcg_nodes','lcg_rebases'):stats[key]=lcg_baseline[key]+lcg_store.stats[key]-lcg_entry[key]
        if gold_oracle is not None:
            for key in GOLD_ORACLE_STAT_KEYS:
                stats[key]=(gold_oracle.stats[key] if key in GOLD_ORACLE_GAUGE_KEYS else
                            oracle_baseline[key]+gold_oracle.stats[key]-oracle_entry[key])
        stats['conflict_cache_entries'] = len(store.clauses)
        stats['loop_witnesses'] = len(knowledge) if knowledge is not None else 0
        if native_engine is not None:
            for key,value in native_engine.stats().items():
                stats[key]=(max(native_search_baseline[key],value) if 'peak_' in key else native_search_baseline[key]+value)

    native_path=None
    native_engine=None
    search_loop_backend='python'
    search_loop_reason=None
    native_search_baseline={key:stats[key] for key in NATIVE_SEARCH_STAT_KEYS}
    def decision_facts():
        return (list(native_path) if native_path is not None else
                [(ancestor['cell'],branch_mask(ancestor,ancestor['next']-1)) for ancestor in stack[:-1]])

    def decision_conjunction():
        facts=decision_facts()
        # A pattern representative is not a placement assumption. Existing
        # placement-only learning is used only when the entire path is singleton.
        if any(mask.bit_count()!=1 for _,mask in facts):return None
        return [(cell,mask.bit_length()-1) for cell,mask in facts]


    def weight_failure(cells):
        if options['failure_guided']:
            for cell in set(cells):
                failure_weights[cell] = min(1000000, failure_weights[cell]+1)
                stats['failure_weight_updates'] += 1

    def learn(placements, counters):
        material=tuple(placements)
        store.learn(material,counters)
        if shared is not None:shared.offer(material)
        if options['smart_learning'] and runtime.mask_conflicts is not None:
            runtime.mask_conflicts.offer([(c,1<<code) for c,code in material])

    def advance_shared():
        nonlocal last_shared_work
        if time.monotonic()-last_shared_work<.1:return
        last_shared_work=time.monotonic()
        if options['smart_learning'] and runtime.local_mask_conflicts is not None:
            runtime.local_mask_conflicts.advance(check,seconds=.005,attempts=1)
            runtime.mask_conflicts.advance(check,seconds=.005,attempts=1)
        if shared is None:return
        if runtime.sat is not None:
            for candidate in runtime.sat.drain_candidates():shared.offer(candidate)
        shared.advance(check,seconds=.005,attempts=1)
        for clause in shared.records():
            key=tuple(tuple(pair) for pair in clause)
            if key not in imported_shared:
                store.learn(key,stats);imported_shared.add(key)

    def learn_failure(decisions, gold_assignments=None, reason=None, *, allow_jump=True):
        # The caller commits the known failed frame BEFORE optional explanation
        # probes. Stop during a probe therefore leaves an ordinary valid stack.
        if decisions is not None:learn(decisions, stats)
        if reason is not None and decisions is not None:
            if not set(reason)<=set(decisions):raise RuntimeError('Conflict reason escaped its active branch.')
            learn(reason,stats)
            stats['reason_conflicts']+=1
            stats['reason_literals_saved']+=len(decisions)-len(reason)
        if gold_assignments is not None:
            explanation = problem.gold_nogood(gold_assignments)
            learn(explanation, stats)
            if options.get('mixed_proof_dag'):
                for mask_store in (runtime.mask_conflicts,runtime.local_mask_conflicts):
                    if mask_store is not None:mask_store.add_gold_witnesses([explanation])
            if knowledge is not None and options.get('persistent_learning',True):
                try:knowledge.add(explanation)
                except ValueError:pass  # A generic fallback cut need not prove a proper loop.
            if options['rotated_learning'] and knowledge is not None:
                for rotated in rotate_gold_clause(problem,explanation,check):
                    try:
                        knowledge.add(rotated);learn(rotated,stats);stats['rotated_clauses']+=1
                    except ValueError:pass
            weight_failure(cell for cell,_ in explanation)
        elif decisions:
            weight_failure([decisions[-1][0]])
        if not allow_jump or decisions is None or native_path is not None or not options.get('backjumping', True) or not stack:
            return
        # Existing clauses are already proved in this effective job domain.
        # Prefer one whose latest responsible decision is earliest in the stack.
        positions={literal:i for i,literal in enumerate(decisions)}
        active=frozenset(decisions)
        usable=[clause for clause in store.clauses if clause<=active]
        best=tuple(sorted(min(usable,key=lambda clause:(max((positions[literal] for literal in clause),default=-1),len(clause))),key=positions.get)) if usable else tuple(decisions)
        removed=_jump_conflict(stack,decisions,best)
        if removed:
            stats['backjumps'] += 1;stats['backjump_levels'] += removed
            return
        if len(best)<2:
            return
        if options.get('adaptive',True) and runtime.explanation_cooldown:
            runtime.explanation_cooldown -= 1
            stats['explanation_skipped'] += 1
            return
        nonlocal probe_deadline
        old_deadline = probe_deadline
        probe_deadline = min(deadline, time.monotonic()+0.01)
        started=time.perf_counter_ns();original_size=len(best)
        def prove(assumptions):
            stats['explanation_probes'] += 1
            check()
            trial = initial.copy()
            for cell, code in assumptions:
                trial[cell] &= 1 << code
            failed = not propagate(trial, proof=True)
            if not failed and target=='gold':
                assigned = [value.bit_length()-1 if value.bit_count()==1 else -1 for value in trial]
                failed = problem.partial_loop_impossible(assigned)
            check()
            return failed
        try:
            for candidate in _minimize_conflict(best, prove):
                if len(candidate)<len(best):
                    stats['explanation_literals_removed'] += len(best)-len(candidate)
                    best=candidate
                    learn(best,stats)
        except (ProbeBudget,Interrupted):
            stats['explanation_aborted'] += 1
        finally:
            probe_deadline = old_deadline
            elapsed=max(0,time.perf_counter_ns()-started)
            stats['explanation_time_ns'] += elapsed
        if len(best)<original_size:
            runtime.explanation_misses=runtime.explanation_cooldown=0
        elif elapsed>=500_000:
            runtime.explanation_misses += 1
            if options.get('adaptive',True) and runtime.explanation_misses>=2:
                runtime.explanation_cooldown=min(32,1<<min(5,runtime.explanation_misses//2))
        removed = _jump_conflict(stack,decisions,best)
        if removed:
            stats['backjumps'] += 1
            stats['backjump_levels'] += removed

    class Interrupted(Exception):
        def __init__(self, status): self.status = status

    class ProbeBudget(Exception):
        pass

    def check():
        if stop is not None and stop():
            raise Interrupted('stopped')
        now = time.monotonic()
        if now >= deadline:
            raise Interrupted('timeout')
        if probe_deadline is not None and now >= probe_deadline:
            raise ProbeBudget

    def flush_reasoning():
        if options['relation_certificates'] and runtime.higher is not None:runtime.higher.flush()
        if options['semantic_witnesses'] and runtime.semantic_pool is not None and runtime.semantic_path is not None:
            saved=runtime.semantic_pool.snapshot()['digest']
            if saved!=runtime.semantic_saved:
                try:
                    runtime.semantic_path.parent.mkdir(parents=True,exist_ok=True)
                    runtime.semantic_pool.save(runtime.semantic_path);runtime.semantic_saved=saved
                except OSError:pass
        if archive is None:return
        if runtime.lcg is not None:archive.save_lcg(runtime.lcg)
        archive.save_probes(runtime.probes)
        path=[]
        for frame in stack:
            reason=frame.get('_lcg')
            if reason is not None and reason.enabled and reason.domains==frame['domains']:
                archive.save_reason((tuple(frame['domains']),tuple(path),reason.max_facts),reason)
            if frame['cell']>=0 and frame['next']>0:
                path.append((frame['cell'],branch_mask(frame,frame['next']-1)))

    def snapshot(*,persist=True):
        nonlocal stack
        if persist:flush_reasoning()
        if native_engine is not None:stack=[from_native_frame(frame) for frame in native_engine.snapshot()]
        sync_stats()
        payload = {'version': VERSION, 'fingerprint': problem.fingerprint, 'target': target,
                'owned_root_domains':[hex(value) for value in initial],
                'seed': seed, 'search_options': dict(options), 'hints': {str(cell): codes for cell, codes in normalized_hints.items()},
                'nodes': nodes, 'stats': dict(stats), 'elapsed_seconds': previous_elapsed + time.monotonic()-began,
                'stack': [{'domains': [hex(value) for value in frame['domains']], 'cell': frame['cell'],
                           'choices': frame['choices'].copy(), 'next': frame['next'],
                           **({'choice_masks':[hex(value) for value in frame['choice_masks']]} if 'choice_masks' in frame else {})} for frame in stack]}
        if options['resumable_reasoning'] and advanced is not None:payload['reasoning_continuation']=advanced.reasoning_snapshot()
        if runtime.local_mask_conflicts is not None and options['smart_learning']:payload['local_mask_conflicts']=runtime.local_mask_conflicts.snapshot()
        if runtime.certificates is not None and options['proof_certificates']:payload['certificates']=runtime.certificates.snapshot()
        payload['digest'] = _checkpoint_digest(payload)
        return payload

    def result(status, codes=None, report=None):
        flush_reasoning()
        sync_stats()
        value = {'status': status, 'codes': codes, 'elapsed_seconds': time.monotonic()-began,
                 'nodes': nodes, 'complete': status in ('solved', 'edge_perfect', 'infeasible'), 'stats': dict(stats),
                 'search_options':dict(options), 'native_backend':native_backend, 'native_fallback_reason':native_reason,
                 'search_loop_backend':search_loop_backend, 'search_loop_fallback_reason':search_loop_reason}
        if knowledge is not None and options.get('persistent_learning',True):value['knowledge']=knowledge.snapshot()
        if shared is not None:value['shared_conflicts']=shared.snapshot()
        if runtime.mask_conflicts is not None and options['smart_learning']:value['mask_conflicts']=runtime.mask_conflicts.snapshot()
        if runtime.local_mask_conflicts is not None and options['smart_learning']:value['local_mask_conflicts']=runtime.local_mask_conflicts.snapshot()
        if runtime.certificates is not None and options['proof_certificates']:value['certificates']=runtime.certificates.snapshot()
        if report is not None: value['validation'] = report
        if status in ('timeout', 'stopped'): value['checkpoint'] = snapshot(persist=False)
        return value

    def notify(event, **extra):
        sync_stats()
        if progress is not None:
            progress({'event': event, 'nodes': nodes, 'elapsed_seconds': time.monotonic()-began,
                      'stats': dict(stats), 'search_options':dict(options), 'native_backend':native_backend,
                      'native_fallback_reason':native_reason, 'search_loop_backend':search_loop_backend,
                      'search_loop_fallback_reason':search_loop_reason,
                      **({'knowledge':knowledge.snapshot()} if knowledge is not None and options.get('persistent_learning',True) else {}),
                      **({'shared_conflicts':shared.snapshot()} if shared is not None else {}),
                      **({'mask_conflicts':runtime.mask_conflicts.snapshot()} if options['smart_learning'] and runtime.mask_conflicts is not None else {}),
                      **({'certificates':runtime.certificates.snapshot()} if options['proof_certificates'] else {}), **extra})

    try:
        check()
        # Only immutable construction results survive a slice. Every callback,
        # scheduler wrapper, matching hint and mutable cache is newly owned.
        kwargs = {'tables': runtime.pruner_tables} if runtime.pruner_tables is not None else {}
        if options['incremental_hall']:kwargs['incremental_hall']=True
        if options.get('hall_rematch',False):kwargs['hall_rematch']=True
        pruner = Pruner(problem, check, native=options['native'], dirty_neighborhood=options['dirty_neighborhood'], parity_root_reuse=options['parity_root_reuse'], **kwargs)
        runtime.pruner_tables = pruner.tables
        advanced_options = tuple(sorted(options.items()))
        if runtime.advanced is not None and not isinstance(runtime.advanced,AdvancedPruner):
            raise ValueError('The in-memory advanced pruning cache is malformed.')
        if any(options.get(key,True) for key in ('strand_connectivity','multi_seam_cuts','regional_cache','biconnected','separator_regions','hierarchical_regions','extended_regions','symbolic_regions','symbolic_inventory')):
            if runtime.advanced is None or runtime.advanced_options!=advanced_options:
                runtime.advanced=AdvancedPruner(problem,target=target,options=options)
                runtime.advanced_options=advanced_options
            advanced=runtime.advanced
            advanced.regional_cache_dir=learning_state.get('_regional_cache_dir')
            if options['resumable_reasoning'] and resume is not None and resume.get('reasoning_continuation') is not None and not advanced._cut_pending:
                advanced.restore_reasoning(resume['reasoning_continuation'])
            advanced_entry=dict(advanced.stats)
        if options['incremental_hall']:
            if runtime.incremental_hall is None:runtime.incremental_hall=pruner.incremental_hall
            else:pruner.incremental_hall=runtime.incremental_hall
            pruner.incremental_hall.rematch_reuse=options.get('hall_rematch',False)
            pruner._incremental_hall_entry=dict(pruner.incremental_hall.stats)
        if target == 'gold':
            connectivity = ConnectivityPruner(problem)
        if support_tables is None:
            support_tables = _build_support_tables(problem, initial, target, check)
            learning_state['_support'] = support_tables
        if options['value_ordering'] or options['objective_ordering']:
            if runtime.ordering is None or runtime.ordering.tables is not support_tables:
                runtime.ordering=SupportOrdering(support_tables)
        if any(options[key] for key in REGION26_OPTIONS):
            signature=tuple(options[key] for key in (*REGION26_OPTIONS,'smart_learning'))
            if runtime.regional26 is None or runtime.regional26_options!=signature:
                runtime.regional26=RegionalWorkstreams(problem,target,options)
                runtime.regional26_options=signature
                regional26_entry=dict.fromkeys(REGION26_STATS,0)
            runtime.regional26.configure_cache_dir(learning_state.get('_regional_cache_dir'))
        branching=ConnectionBranching(problem,signature=options['signature_branching']) if options['connection_branching'] or options['signature_branching'] or options['group_lookahead'] else None
        if any(options[k] for k in RESEARCH_KEYS):
            if runtime.research is None or runtime.research.options!={k:options[k] for k in RESEARCH_KEYS}:
                runtime.research=ResearchPruner(problem,target,options)
                research_entry=dict.fromkeys(RESEARCH_STAT_KEYS,0)
            runtime.research.configure_cache_dir(learning_state.get('_regional_cache_dir'))
        if options['dual_piece_branching'] or options['strand_branching']:
            if runtime.routing is None:runtime.routing=RoutingBranching(problem)
        if options['completion_ordering'] and runtime.completion is None:runtime.completion=CompletionOrdering(problem)
        if options['learned_ordering'] and runtime.learned is None:runtime.learned=LearnedOrdering(problem,weights=LEARNED_ORDERING_WEIGHTS)
        tile_bits = support_tables.tile_bits
        arcs, incoming = support_tables.arcs, support_tables.incoming
        filter_arc = _ArcFilter(support_tables)
        saved_native = learning_state.get('_native')
        if options['native']!='python' and saved_native is not None and saved_native.matches(support_tables):
            native, native_reason = saved_native, None
        else:
            native, native_reason = select_native(support_tables, options['native'])
            if native is not None:learning_state['_native'] = native
        native_backend = 'compiled' if native is not None else 'python'
        if options['sat_assist'] and runtime.sat is None:
            runtime.sat=SatAssist(problem,initial,target,witness_pool=options['witness_pool'],support_cover=options['support_cover'])

        if options['semantic_witnesses'] and isinstance(runtime.sat,SatAssist):
            from constraint_witness_round26 import SemanticWitnessPool
            if runtime.semantic_pool is None:runtime.semantic_pool=SemanticWitnessPool(problem)
            identity=hashlib.sha256(json.dumps([problem.fingerprint,problem.domains],separators=(',',':')).encode()).hexdigest()
            desired=None if reasoning_path is None else Path(reasoning_path)/'semantic-witnesses'/(identity+'.json')
            if desired!=runtime.semantic_path:
                runtime.semantic_path=desired;runtime.semantic_saved=None
                if desired is not None:
                    try:runtime.semantic_pool.load(desired,check)
                    except (OSError,ValueError,TypeError):pass
            runtime.sat.semantic_witnesses=runtime.semantic_pool
        elif isinstance(runtime.sat,SatAssist):runtime.sat.semantic_witnesses=None
        if options['higher_order_relations'] and isinstance(runtime.sat,SATServiceClient) and runtime.higher_sat is None:
            runtime.higher_sat=SatAssist(problem,initial,target,witness_pool=options['witness_pool'],support_cover=options['support_cover'])
        if runtime.sat is not None:
            runtime.sat.witness_pool=options['witness_pool'];runtime.sat.support_cover=options['support_cover']
        if options['higher_order_relations'] and all(initial):
            rebuild=runtime.higher is None or (runtime.higher.certificates is not None)!=options['relation_certificates']
            if rebuild:
                if runtime.higher is not None:runtime.higher.close()
                runtime.higher=ProjectedRelations(problem,initial,target,semijoins=options['relational_consistency'],
                    guided=options['relational_consistency'],certificates=options['relation_certificates'],partial=options['relation_partial'])
                higher_entry=dict.fromkeys(HIGHER_STAT_KEYS,0);relational_entry=dict.fromkeys(RELATIONAL_STATS,0)
                certificate_entry=dict.fromkeys(RELATION_CERTIFICATE_STATS,0);runtime.higher_directory=None
            runtime.higher.semijoins=options['relational_consistency'];runtime.higher.guided=options['relational_consistency'];runtime.higher.partial=options['relation_partial']
            desired=None if reasoning_path is None else str(Path(reasoning_path).resolve())
            if runtime.higher.certificates is not None and (rebuild or desired!=runtime.higher_directory):
                runtime.higher.configure_cache_dir(desired,check);runtime.higher_directory=desired
        scheduler_options = tuple(sorted(options.items()))
        if runtime.scheduler is not None and not isinstance(runtime.scheduler,PropagationScheduler):
            raise ValueError('The in-memory propagation scheduler is malformed.')
        if runtime.scheduler is None or runtime.scheduler_options != scheduler_options:
            runtime.scheduler = PropagationScheduler(incremental=options.get('incremental',True),adaptive=options.get('adaptive',True),contextual=options.get('contextual_propagation',False))
            runtime.scheduler_options = scheduler_options
        scheduler=runtime.scheduler
        proof_context=[False]
        def schedule_method(owner,name,key,*,optional=False,stats_name=None):
            action=getattr(owner,name)
            def wrapped(domains,*args,**kwargs):
                return scheduler.run(stats_name or name,domains,lambda:action(domains,*args,**kwargs),stats,
                                     optional=optional,force=proof_context[0],signature=key(domains))
            setattr(owner,name,wrapped)
        tile_signature=lambda domains:tuple(_tile_choices(value) for value in domains)
        schedule_method(pruner,'prune_hall',tile_signature)
        cluster_bits=0
        for _,bits,_ in pruner.clusters:cluster_bits |= bits
        schedule_method(pruner,'prune_clusters',
                        lambda domains:tuple((value&cluster_bits,bool(value&~cluster_bits)) for value in domains))
        boundary_cells=tuple(sorted({cell for _,boundary,_ in pruner.regions for cell in boundary}))
        schedule_method(pruner,'prune_regions',lambda domains:(tile_signature(domains),tuple(domains[cell] for cell in boundary_cells)),optional=True)
        schedule_method(pruner,'prune_neighborhood',lambda domains:tuple(domains),optional=True)
        if pruner.parity is not None:
            def parity_signature(domains):
                if not pruner.parity.enabled:return ()
                return tuple((bool(value&odd),bool(value&~odd)) for value in domains for odd in pruner.parity.odd_codes)
            schedule_method(pruner.parity,'prune',parity_signature,stats_name='parity')
        def connectivity_signature(domains):
            return tuple((bool(value&occupied),bool(value&~even)) for value in domains
                         for occupied,even in zip(connectivity.nonempty,connectivity.even_positive))

        class Traces:
            def __init__(self,reason,lcg):self.reason,self.lcg=reason,lcg
            def record_cell(self,cell,value,deps,rule):
                deps=tuple(deps);count=0
                if self.reason is not None:count=self.reason.record_cell(cell,value,deps,rule)
                if self.lcg is not None:stats['lcg_nodes']+=self.lcg.record_cell(cell,value,deps,rule)
                return count
            def record(self,domains,deps,rule):
                deps=tuple(deps);count=0
                if self.reason is not None:count=self.reason.record(domains,deps,rule)
                if self.lcg is not None:stats['lcg_nodes']+=self.lcg.record(domains,deps,rule)
                return count
            def fail(self,cells):
                cells=tuple(cells)
                if self.reason is not None:self.reason.failure=self.reason.causes(cells)
                if self.lcg is not None:self.lcg.failure=cells

        def propagate(domains, changed_cell=None, *, proof=False, trace=None, basic_complete=False):
            stats['propagations'] += 1
            projections = {}
            queue = deque(() if basic_complete else _starting_arcs(len(arcs), incoming, changed_cell))
            pending = [False]*len(arcs)
            for index in queue:
                pending[index] = True
            arc_steps = 0
            def record(dependencies,rule):
                if trace is not None:
                    stats['reason_nodes']+=trace.record(domains,dependencies,rule)
            def direct(cell,value,dependencies,rule):
                if trace is not None:
                    count=trace.record_cell(cell,value,dependencies,rule)
                    stats['reason_nodes']+=count;stats['reason_direct_nodes']+=count
            def bulk(action,rule):
                try:return action()
                finally:record(range(n),rule)
            def fail(cells):
                if trace is not None:trace.fail(cells)
                return False
            def enqueue(cell):
                for index in incoming[cell]:
                    if not pending[index]:
                        queue.append(index)
                        pending[index] = True
            def narrow(cell, value, dependencies=None, rule='tile inventory'):
                if value == domains[cell]: return False
                domains[cell] = value
                direct(cell,value,range(n) if dependencies is None else dependencies,rule)
                enqueue(cell)
                return True
            clause_reason=(lambda cell,value,deps:direct(cell,value,deps,'learned clause')) if trace is not None else None
            def stage_clauses(values):
                bulk(lambda:store.prune(domains, stats, check,reason=clause_reason),'learned clauses')
                if options['proof_certificates']:
                    bulk(lambda:runtime.certificates.prune(domains,check),'portable certificate')
                if options['smart_learning'] and runtime.mask_conflicts is not None:
                    bulk(lambda:runtime.mask_conflicts.prune(domains,check),'verified mask conflicts')
                    bulk(lambda:runtime.local_mask_conflicts.prune(domains,check),'root-scoped verified masks')
                if lcg_store is not None and all(domains):
                    bulk(lambda:lcg_store.prune(domains,check,reason=clause_reason),'LCG clauses')
                if any(value == 0 for value in domains): return False
            def stage_inventory(values):
                proof_context[0]=proof
                pruner.on_narrow=(lambda cell,value,deps:direct(cell,value,deps,'Hall set')) if trace is not None else None
                try:bulk(lambda:pruner.prune(domains),'inventory and regions')
                finally:proof_context[0]=False;pruner.on_narrow=None
                if any(value == 0 for value in domains): return False
            def stage_connectivity(values):
                if connectivity is not None:
                    bulk(lambda:scheduler.run('connectivity',domains,lambda:connectivity.prune(domains,check),stats,optional=True,force=proof,
                                  signature=connectivity_signature(domains)),'Gold connectivity')
                    if any(value == 0 for value in domains): return False
            def stage_advanced(values):
                if advanced is not None:
                    if options['smart_learning']:
                        advanced.region_node_limit=int(scheduler.budget('advanced',4096,minimum=1024,maximum=16384))
                    budgets = sum(value for key,value in advanced.stats.items() if key.endswith('budget_skips'))
                    advanced.on_narrow=(lambda cell,value,deps:direct(cell,value,deps,'regional table')) if trace is not None else None
                    advanced.before_hierarchical=(lambda current:record(range(n),'earlier global rules')) if trace is not None else None
                    try:
                        bulk(lambda:scheduler.run('advanced',domains,lambda:advanced.prune(domains,check),stats,optional=True,force=proof,
                                      complete=lambda:budgets==sum(value for key,value in advanced.stats.items() if key.endswith('budget_skips')),proof_yield=lambda:advanced.stats.get('conditional_hall_certificates',0)+advanced.stats.get('hierarchical_cache_entries',0)),'advanced regions and topology')
                    finally:advanced.on_narrow=None;advanced.before_hierarchical=None
                    if any(value == 0 for value in domains): return False
            def stage_higher(values):
                if options['higher_order_relations'] and runtime.higher is not None and all(domains):
                    bulk(lambda:runtime.higher.prune(domains,runtime.higher_sat if isinstance(runtime.sat,SATServiceClient) else runtime.sat,check),'complete higher-order projection')
            def stage_deep(values):
                if runtime.deep is not None and all(domains):
                    if options['deep_paths'] or options['retained_relations']:bulk(lambda:runtime.deep.paths(domains,check),'three-cell path support')
                    if options['retained_relations'] and all(domains):bulk(lambda:runtime.deep.retained_relations(domains,check),'retained pair correlations')
                    if options['joint_conservation'] and all(domains):bulk(lambda:runtime.deep.joint_conservation(domains,check),'joint seam conservation')
                    if options['conservation'] and all(domains):bulk(lambda:runtime.deep.conservation(domains,check),'integer seam conservation')
            def stage_research(values):
                if runtime.research is not None and any(options[k] for k in RESEARCH_KEYS) and all(domains):
                    runtime.research.prune(domains,check,bulk)
                    if options['smart_learning'] and runtime.mask_conflicts is not None:
                        for claim in runtime.research.drain_deduction_chains(target,limit=8):
                            runtime.mask_conflicts.admit_deduction_chain(claim,check,stats=stats)
            def stage_regional(values):
                if runtime.regional26 is not None and any(options[k] for k in REGION26_OPTIONS) and all(domains):
                    bulk(lambda:runtime.regional26.prune(domains,check),'compiled regional and connectivity reasoning')
                    if options['smart_learning'] and runtime.mask_conflicts is not None:
                        for claim in runtime.regional26.drain_deduction_chains(check,limit=8):
                            runtime.mask_conflicts.admit_deduction_chain(claim,check,stats=stats)
            def stage_symmetry(values):
                if options['piece_symmetry'] and all(domains):
                    old_count=sum(v.bit_count() for v in domains)
                    bulk(lambda:runtime.piece_symmetry.prune(domains),'verified strand symmetry')
                    stats['piece_symmetry_prunes']+=old_count-sum(v.bit_count() for v in domains)
            actions=(('clauses',stage_clauses),('inventory',stage_inventory),('connectivity',stage_connectivity),('advanced',stage_advanced),('higher',stage_higher),('deep',stage_deep),('research',stage_research),('regional',stage_regional),('symmetry',stage_symmetry))
            if options['durable_reasoning'] and not proof:
                names=tuple(name for name,_ in actions)
                context=(tuple(decision_facts()),tuple(sorted(options.items())))
                key=(context,tuple(domains));cursor=runtime.propagation_cursors.pop(key,None)
                if cursor is None and archive is not None:
                    cursor=archive.load_propagation(key,arcs,incoming,tile_bits,names,check)
                if cursor is None:
                    cursor=PropagationCursor(domains,arcs,incoming,tile_bits,names,
                        starting=_starting_arcs(len(arcs),incoming,changed_cell))
                    stats['propagation_cursor_builds']+=1
                else:
                    stats['propagation_cursor_resumes']+=1
                    stats['propagation_cursor_stage_skips']+=cursor.stage if cursor.phase=='strong' else 0
                cursor.attach(domains);prior_retries=cursor.stats['propagation_cursor_opaque_retries']
                def retained_native(values,queued):
                    kwargs={'reason':lambda cell,value,neighbor:direct(cell,value,[neighbor],'native seam')} if trace is not None else {}
                    return bulk(lambda:native.propagate(values,queued,check,stats,**kwargs),'native seams')
                try:
                    possible=cursor.advance(filter_arc,actions,check,direct,stats,
                        native_basic=retained_native if native is not None else None)
                    if not possible and cursor.basic.failed is not None:fail(cursor.basic.failed)
                    return possible
                finally:
                    stats['propagation_cursor_opaque_retries']+=cursor.stats['propagation_cursor_opaque_retries']-prior_retries
                    if not cursor.done:
                        saved_key=(context,tuple(domains));runtime.propagation_cursors[saved_key]=cursor
                        while len(runtime.propagation_cursors)>16:
                            del runtime.propagation_cursors[next(iter(runtime.propagation_cursors))]
                            stats['propagation_cursor_evictions']+=1
                        if archive is not None:archive.save_propagation(saved_key,cursor)
            while True:
                check()
                if any(value == 0 for value in domains): return False
                if native is not None and queue:
                    kwargs={'reason':lambda cell,value,neighbor:direct(cell,value,[neighbor],'native seam')} if trace is not None else {}
                    if not bulk(lambda:native.propagate(domains, tuple(queue), check, stats,**kwargs),'native seams'):
                        return False
                    queue.clear()
                    pending[:] = [False]*len(arcs)
                while queue:
                    index = queue.popleft()
                    pending[index] = False
                    a, sa, b, sb = arcs[index]
                    keep = filter_arc(domains[a], domains[b], sa, sb)
                    if keep != domains[a]:
                        stats['arc_revisions'] += 1
                        narrow(a,keep,[b],'seam support')
                        if not keep:return False
                    arc_steps += 1
                    stats['python_arcs_checked'] += 1
                    if arc_steps % 32 == 0: check()
                # Only the current native callback may certify this exact
                # input. Strong deductions below invalidate it for the next
                # fixpoint iteration; checkpoints never carry the certificate.
                if basic_complete:
                    stats['native_basic_passes_skipped'] += 1
                    basic_complete = False
                else:
                    # A cell whose candidates all use one tile owns that tile even
                    # before its rotation is chosen. Also force each tile's sole cell.
                    owners, occurrences = {}, [[] for _ in range(n)]
                    for cell, domain in enumerate(domains):
                        possible = projections.get(domain)
                        if possible is None:
                            possible = tuple(_values(_tile_choices(domain)))
                            projections[domain] = possible
                        if len(possible) == 1:
                            tile = possible[0]
                            if tile in owners:return fail([cell,owners[tile]])
                            owners[tile] = cell
                        for tile in possible: occurrences[tile].append(cell)
                        if cell % 8 == 0: check()
                    for tile, cell in owners.items():
                        for other in occurrences[tile]:
                            if other != cell and narrow(other, domains[other] & ~tile_bits[tile],[cell],'tile ownership'):
                                stats['tile_prunes'] += 1
                                if not domains[other]: return False
                    for tile, cells in enumerate(occurrences):
                        if not cells:return fail(range(n))
                        if len(cells) == 1 and narrow(cells[0], domains[cells[0]] & tile_bits[tile]):
                            stats['tile_prunes'] += 1
                            if not domains[cells[0]]: return False
                if any(value == 0 for value in domains): return False
                # Strong reductions and learned unit clauses use the same
                # fixpoint as seam propagation. Only unexpanded frame domains
                # reach this function, so stored sibling choices stay complete.
                before = domains.copy()
                for _,action in actions:
                    if action(domains) is False:return False
                if not all(domains):return False
                for cell, (old, new) in enumerate(zip(before, domains)):
                    if old != new:
                        enqueue(cell)
                if domains != before and not queue:
                    # A board may have no internal seams; tile/learned
                    # propagation must still reach a fixpoint on that board.
                    continue
                if not queue: return True

        def evaluate(frame):
            """Shared strong pruning/branch policy for Python and native control loops."""
            nonlocal nodes,probe_deadline
            while True:
                check();advance_shared();nodes+=1
                domains=frame['domains'];decisions=decision_conjunction()
                residual_input=tuple(domains) if residual is not None and all(domains) else None
                reason=None;lcg=None
                if options['reason_tracking'] and decisions is not None:
                    reason=frame.get('_reasons')
                    if reason is None or reason.domains!=domains or len(reason.facts)>8192:
                        reason=ReasonState(domains,decisions,initial=initial)
                        frame['_reasons']=reason;stats['reason_rebases']+=1
                    reason.failure=None
                if lcg_store is not None:
                    lcg=frame.get('_lcg')
                    if lcg is None and archive is not None:
                        cached=archive.load_reason((tuple(domains),tuple(decision_facts()),8192))
                        if (cached is not None and cached.enabled and cached.domains==domains
                                and cached.binding==(problem.fingerprint,target,tuple(initial))
                                and cached.level==len(decision_facts())):
                            lcg=cached;frame['_lcg']=cached
                    if lcg is None or lcg.domains!=domains or not lcg.enabled:
                        lcg=LCGState.reconstruct(initial,domains,decision_facts(),fingerprint=problem.fingerprint,target=target)
                        frame['_lcg']=lcg;stats['lcg_rebases']+=1
                    lcg.failure=None
                trace=Traces(reason,lcg) if reason is not None or lcg is not None else None
                def failure(*,gold=None,cells=None,codes=None,report=None,proof_input=None):
                    if lcg is not None and gold is not None:
                        # A certified proper-loop witness usually mentions
                        # fewer cells than the full rejected arrangement.
                        # Keep only already fixed placements on this path.
                        witness=problem.gold_nogood(gold)
                        if witness and all(domains[cell]==1<<code for cell,code in witness):
                            cells=[cell for cell,_ in witness]
                    clause=lcg.first_uip(cells) if lcg is not None else None
                    if lcg is not None and not lcg.enabled:stats['lcg_trace_budget_skips']+=1
                    why=(reason.failure if reason.failure is not None else reason.contradiction()) if reason is not None else None
                    return {'action':'reject','decisions':decisions,'reason':why,'lcg_clause':clause,
                            'gold':gold,'codes':codes,'validation':report,'proof_input':proof_input,
                            '_residual_input':residual_input}
                if residual_input is not None and residual.contains(residual_input):
                    return failure(cells=range(n))
                selected=frame.pop('_changed_cell',None);before_propagation=domains.copy()
                possible=propagate(domains,selected,trace=trace,basic_complete=frame.pop('_native_basic_complete',False))
                if options['impact_branching'] and selected is not None:
                    runtime.impact.observe(selected,before_propagation,domains,failed=not possible);stats['impact_observations']+=1
                if not possible:return failure(proof_input=before_propagation)
                frame['_propagated']=True
                if target=='gold' and runtime.research is not None and any(options[k] for k in RESEARCH_KEYS):
                    candidate=runtime.research.solution
                    # Research may also run on trial or sibling domains. A
                    # cached witness is only an answer inside this owned frame.
                    if candidate is not None and len(candidate)==n and all(
                            type(code) is int and code>=0 and code<mask.bit_length()
                            and mask&(1<<code) for code,mask in zip(candidate,domains)):
                        report=problem.validate(candidate)
                        if not report['valid']:
                            raise ValueError('Regional Gold witness failed independent validation')
                        stats['regional_solutions_validated']+=1
                        return {'action':'solved','codes':list(candidate),'validation':report,'via_regional':True}
                if runtime.regional26 is not None and any(options[k] for k in REGION26_OPTIONS):
                    candidate=runtime.regional26.solution
                    if candidate is not None and len(candidate)==n and all(type(code) is int and 0<=code<3*n and domains[cell]>>code&1 for cell,code in enumerate(candidate)):
                        report=problem.validate(candidate)
                        if not (report['valid'] or target=='edges' and report['unique_tiles'] and report['matched_edges']==report['total_edges']):
                            raise ValueError('Compiled regional witness failed independent validation.')
                        stats['regional_solutions_validated']+=1
                        return {'action':'solved','codes':list(candidate),'validation':report,'via_regional':True}
                if options['sat_assist'] and (nodes==1 or nodes%8==0):
                    if target=='gold' and options.get('persistent_learning',True) and knowledge is not None:
                        runtime.sat.add_gold_witnesses(knowledge,check,max_records=8)
                    runtime.sat.prepare(check,seconds=.005)
                    if decisions is not None:
                        core=runtime.sat.query(decisions,check,seconds=.01)
                        if core is not None:
                            learn(core,stats)
                            return failure(cells=range(n))
                    # Domain restrictions include deductions and grouped-mask
                    # decisions that are absent from the placement-only query.
                    # Their consequences stay scoped to this complete input;
                    # they must never become unconditional placement clauses.
                    reduced=runtime.sat.filter_domains(domains,check,seconds=.01,
                                                       conflicts=256,max_candidates=16)
                    if options['smart_learning'] and runtime.mask_conflicts is not None:
                        for candidate in runtime.sat.drain_domain_candidates():runtime.mask_conflicts.offer(candidate)
                        runtime.mask_conflicts.advance(check,seconds=.005,attempts=1)
                        runtime.local_mask_conflicts.advance(check,seconds=.005,attempts=1)
                    if tuple(domains)!=reduced:
                        domains[:]=reduced
                        if trace is not None:
                            stats['reason_nodes']+=trace.record(domains,range(n),'SAT domain support')
                        if not all(domains) or not propagate(domains,trace=trace):
                            return failure(cells=range(n))
                if all(value.bit_count()==1 for value in domains):
                    codes=[value.bit_length()-1 for value in domains];report=problem.validate(codes)
                    accepted=report['valid'] or (target=='edges' and report['unique_tiles'] and report['matched_edges']==len(problem.edges))
                    stats['leaves']+=1
                    if accepted:return {'action':'solved','codes':codes,'validation':report}
                    return failure(gold=codes if target=='gold' else None,cells=range(n),codes=codes,report=report)
                assigned=[value.bit_length()-1 if value.bit_count()==1 else -1 for value in domains]
                if target=='gold' and problem.partial_loop_impossible(assigned):
                    stats['loop_prunes']+=1;cells=None
                    if reason is not None or lcg is not None:
                        witness=problem.gold_nogood(assigned)
                        cells=([cell for cell,code in witness] if all(domains[cell]==1<<code for cell,code in witness) else range(n))
                        if reason is not None:reason.failure=reason.causes(cells)
                    return failure(gold=assigned,cells=cells)
                if gold_oracle is not None and gold_oracle.should_query(domains,node=nodes):
                    answer=gold_oracle.query(domains,check,seconds=.5,node=nodes)
                    if answer.status=='solved':
                        return {'action':'solved','codes':list(answer.codes),'validation':answer.validation,'via_oracle':True}
                    if answer.status=='infeasible':return failure(cells=range(n))
                guided=options['failure_guided'] and any(weight>1 for weight in failure_weights)
                def cell_key(i):
                    neighbors=[int(nb) for nb in problem.neighbors[i] if nb>=0]
                    activity=failure_weights[i]+sum(failure_weights[nb] for nb in neighbors) if guided else 0
                    return (runtime.impact.key(i,domains[i].bit_count()) if options['impact_branching'] else domains[i].bit_count(),-activity,-sum(domains[nb].bit_count()==1 for nb in neighbors),cell_rank[i])
                cell=min((i for i,value in enumerate(domains) if value.bit_count()>1),key=cell_key)
                if guided:stats['failure_guided_choices']+=1
                if options['impact_branching']:stats['impact_choices']+=1
                if options['objective_ordering']:
                    runtime.ordering.goal=(('conflict' if stats['backtracks']>16 and stats['backtracks']>nodes*.7 else 'solution') if options['ordering_goal']=='auto' else options['ordering_goal'])
                hinted={code:index for index,code in enumerate(normalized_hints.get(cell,[]))}
                choices=(runtime.ordering.order(cell,domains,list(_values(domains[cell])),hinted,rank,check,stats)
                         if options['value_ordering'] or options['objective_ordering'] else sorted(_values(domains[cell]),key=lambda code:(0,hinted[code]) if code in hinted else (1,rank[code])))
                if options['completion_ordering']:choices=runtime.completion.order(cell,domains,choices,hinted,rank,check,stats)
                if options['learned_ordering']:choices=runtime.learned.order(cell,domains,choices,hinted,rank,check,stats)
                lookahead_plan=branching.choose(domains,cell,check,stats) if options['group_lookahead'] and branching is not None else None
                if options['lookahead'] and n>=4 and (2<=len(choices)<=4 or lookahead_plan is not None) and not frame.get('_lookahead_done'):
                    before_lookahead=domains.copy()
                    probe_deadline=min(deadline,time.monotonic()+(scheduler.budget('lookahead',options['lookahead_ms']/1000) if options['smart_learning'] else options['lookahead_ms']/1000))
                    context=(problem.fingerprint,target,tuple(initial),tuple(decision_facts()),tuple(sorted(options.items())))
                    pair_phase=False
                    def raw_probe(trial, selected, *, pair=False):
                        before_propagations=stats['propagations'];before_arcs=stats['arc_revisions']
                        try:
                            if not propagate(trial,selected):return False
                            if target=='gold':
                                assigned_trial=[value.bit_length()-1 if value.bit_count()==1 else -1 for value in trial]
                                if problem.partial_loop_impossible(assigned_trial):return False
                            return True
                        finally:
                            prefix='pair_lookahead' if pair else 'lookahead_probe'
                            stats[prefix+'_propagations']+=stats['propagations']-before_propagations
                            stats[prefix+'_arc_revisions']+=stats['arc_revisions']-before_arcs
                    def probe(trial,selected):
                        return runtime.probes.run(context,trial,selected,raw_probe,check,stats,resume_partial=options['durable_reasoning'],recoverable=(ProbeBudget,Interrupted))
                    def pair_probe(trial,selected):
                        return runtime.probes.run(context,trial,selected,
                            lambda values,cell:raw_probe(values,cell,pair=True),check,stats,resume_partial=options['durable_reasoning'],recoverable=(ProbeBudget,Interrupted))
                    def failed_probe(code):
                        if decisions is not None:learn([*decisions,(cell,code)],stats)
                        weight_failure([cell])
                    def failed_pair(first,a,second,b):
                        # The narrowed parent is justified only under this full
                        # singleton path; grouped paths deliberately skip pairs.
                        before_learned=stats['conflicts_learned']
                        learn([*decisions,(first,a),(second,b)],stats)
                        stats['pair_lookahead_learned']+=stats['conflicts_learned']-before_learned
                        weight_failure([first,second])
                    try:
                        if lookahead_plan is not None:
                            if options['durable_reasoning']:runtime.probes.plans.constructive(context,domains,lookahead_plan.cell,lookahead_plan.masks,probe,check,lambda mask:None,stats,grouped=True)
                            else:grouped_constructive_lookahead(domains,lookahead_plan.cell,lookahead_plan.masks,probe,check,lambda mask:None,stats)
                        else:
                            if options['durable_reasoning']:runtime.probes.plans.constructive(context,domains,cell,tuple(1<<code for code in choices),probe,check,failed_probe,stats)
                            else:_constructive_lookahead(domains,cell,choices,probe,check,failed_probe,stats)
                        if lookahead_plan is None and options['pair_lookahead'] and decisions is not None and all(domains) and domains==before_lookahead:
                            pair_phase=True
                            if options['durable_reasoning']:runtime.probes.plans.pairs(context,domains,cell,choices,probe,pair_probe,check,failed_pair,failed_probe,stats,neighbors=tuple(map(int,problem.neighbors[cell])),max_pairs=16)
                            else:paired_lookahead(domains,cell,choices,probe,pair_probe,check,failed_pair,failed_probe,stats,neighbors=problem.neighbors[cell],max_pairs=16)
                    except ProbeBudget:
                        stats['lookahead_aborted']+=1
                        if pair_phase:stats['pair_lookahead_aborted']+=1
                    except Interrupted:
                        stats['lookahead_aborted']+=1
                        if pair_phase:stats['pair_lookahead_aborted']+=1
                        raise
                    finally:
                        probe_deadline=None
                        # Shared and pair-row deductions use actual pre-probe
                        # facts; hypothetical singleton assumptions stay private.
                        if trace is not None:
                            stats['reason_nodes']+=trace.record(domains,range(n),'constructive look-ahead')
                    frame['_lookahead_done']=True
                    if domains!=before_lookahead:
                        frame['_propagated']=False
                        continue
                plan=(runtime.routing.choose(domains,cell,check,stats,dual_piece=options['dual_piece_branching'],strand=options['strand_branching']) if (options['dual_piece_branching'] or options['strand_branching']) else None)
                if plan is None:plan=branching.choose(domains,cell,check,stats) if branching is not None else None
                if plan is not None:
                    frame['choice_masks']=list(plan.masks);cell=plan.cell
                    choices=[(mask&-mask).bit_length()-1 for mask in plan.masks]
                frame.update(cell=cell,choices=choices,next=0)
                return {'action':'branch','domains':domains,'cell':cell,'choices':branch_masks(frame)}

        def consume(out):
            # Call only after consuming the leaf/failed frame. Optional conflict
            # work and user callbacks must not cause that completed leaf to replay.
            if out['action']=='reject':
                failed_input=out.get('_residual_input')
                if residual is not None and failed_input is not None:
                    residual.remember_completed(failed_input,status='unsatisfiable')
                stats['backtracks']+=1
                clause=out.get('lcg_clause');replacement=None
                if clause is not None:
                    lcg_store.learn(clause)
                    if options['smart_learning']:
                        if runtime.local_mask_conflicts is not None:runtime.local_mask_conflicts.offer(clause.literals)
                        lcg_store.resolve(check,max_pairs=32,max_new=4)
                        lcg_store.strengthen(check,max_tests=4,max_steps=512)
                if clause is not None and options.get('backjumping',True) and native_path is None:
                    replacement=lcg_store.asserting_frontier([to_native_frame(frame) for frame in stack],clause)
                # The existing placement explanation can sometimes jump when
                # a conservative mask assertion cannot; keep that fallback.
                learn_failure(out['decisions'],out.get('gold'),out.get('reason'),allow_jump=replacement is None)
                if replacement is not None:stack[:]=[from_native_frame(frame) for frame in replacement]
                before_propagation=out.get('proof_input')
                if before_propagation is not None:
                    # The rejected frame is already committed before these
                    # optional proof checks can be interrupted by Stop.
                    if options['proof_certificates']:runtime.certificates.capture(before_propagation,check)
                    if options['compact_explanations']:
                        from constraint_deep import hall_certificate
                        def prove_local(candidate):
                            check()
                            if hall_certificate(tuple(_tile_choices(v) for v in candidate),check) is not None:return False
                            partial=[v.bit_length()-1 if v.bit_count()==1 else -1 for v in candidate]
                            if target=='gold' and problem.partial_loop_impossible(partial):return False
                            probe=DeepPruner(problem);trial=list(candidate);probe.paths(trial,check)
                            if all(trial):probe.conservation(trial,check)
                            if all(trial):
                                witness=AdvancedPruner(problem,target=target,options={**options,'compact_explanations':False,'native':'python'})
                                witness.prune(trial,check)
                            return None if all(trial) else False
                        compact_cells,tests=minimize_failure_scope(before_propagation,initial,prove_local,check,max_tests=4)
                        stats['explanation_tests']+=tests
                        if compact_cells is not None:
                            stats['explanation_cells_removed']+=sum(a!=b for a,b in zip(before_propagation,initial))-len(compact_cells)
                            literals=tuple((c,before_propagation[c]) for c in compact_cells)
                            if lcg_store is not None:lcg_store.learn(LearnedClause(lcg_store.binding,literals,None,0,0,0))
                            if runtime.local_mask_conflicts is not None:runtime.local_mask_conflicts.offer(literals)

            if out.get('codes') is not None:
                notify('candidate',codes=out['codes'],validation=out['validation'])

        if options['native_search']:
            if not all(all(frame['domains']) for frame in stack):
                search_loop_reason='A saved zero-domain frame is completed through Python validation.'
            else:
                native_engine,search_loop_reason=select_native_search(n,count,mode='compiled' if options['native_explanations'] else options['native'])
                if native_engine is not None and len(stack)>native_engine.max_depth:
                    native_engine.close();native_engine=None
                    search_loop_reason='Saved stack exceeds native depth capacity; exact pending branches continued in Python.'
            if native_engine is not None:
                # C commits only exact seam/inventory consequences. Surviving
                # nodes receive fresh conservative reasons from their actual
                # path below; committed early contradictions get bounded
                # independent replay before optional learning.
                native_engine.configure_propagation(support_tables)
                if options['native_explanations']:native_engine.configure_explanations(True)
                search_loop_backend='compiled'
                def native_process(domains,decisions):
                    nonlocal native_path
                    native_path=decisions
                    return evaluate({'domains':domains,'cell':-1,'choices':[],'next':0,
                                     '_native_basic_complete':native_engine.basic_propagated,
                                     '_changed_cell':decisions[-1][0] if decisions else None})
                def native_event(name,out):
                    nonlocal last_progress,stack,nodes
                    if name=='fast_reject':
                        # This node was consumed by C before this callback. A
                        # later Stop must preserve that committed rejection.
                        nodes+=1;stats['backtracks']+=1;stats['propagations']+=1
                        if lcg_store is not None:
                            explain=explain_fast_failure if options['native_explanations'] else replay_fast_failure
                            clause=explain(out['native_failure'],initial=initial,
                                tables=support_tables,fingerprint=problem.fingerprint,target=target,
                                stats=stats,check=check)
                            if clause is not None:
                                lcg_store.learn(clause)
                                if options['smart_learning']:
                                    if runtime.local_mask_conflicts is not None:runtime.local_mask_conflicts.offer(clause.literals)
                                    lcg_store.resolve(check,max_pairs=32,max_new=4)
                                    lcg_store.strengthen(check,max_tests=4,max_steps=512)
                                if options.get('backjumping',True):
                                    replacement=lcg_store.asserting_frontier(
                                        native_engine.snapshot(),clause,require_jump=True)
                                    if replacement is not None:
                                        stats['native_learning_jumps']+=1
                                        return {'replace_stack':replacement}
                    elif name in ('reject','solved'):
                        consume(out)
                        clause=out.get('lcg_clause')
                        if name=='reject' and clause is not None and options.get('backjumping',True):
                            replacement=lcg_store.asserting_frontier(native_engine.snapshot(),clause,require_jump=True)
                            if replacement is not None:return {'replace_stack':replacement}
                    elif name=='backtrack':stats['backtracks']+=1
                    if time.monotonic()-last_progress>=1:
                        stack=[from_native_frame(frame) for frame in native_engine.snapshot()]
                        if stack:
                            values=stack[-1]['domains']
                            notify('progress',codes=[v.bit_length()-1 if v.bit_count()==1 else -1 for v in values],
                                   assigned_tiles=sum(v.bit_count()==1 for v in values),is_partial=True)
                        last_progress=time.monotonic()
                native_result=native_engine.run([to_native_frame(frame) for frame in stack],native_process,check,native_event,
                    capture_fast_reject=lcg_store is not None)
                stack=[from_native_frame(frame) for frame in native_result['stack']]
                if native_result['status']=='solved':
                    out=native_result['value']
                    return result('solved' if out['validation']['valid'] else 'edge_perfect',out['codes'],out['validation'])
                if native_result['status']=='exhausted':return result('infeasible')
                if native_result['status']=='limit':
                    sync_stats();native_engine.close();native_engine=None;native_path=None
                    search_loop_backend='python';search_loop_reason='Native trail/depth capacity reached; exact pending branches continued in Python.'
                else:return result('timeout')

        while stack:
            check();advance_shared();frame=stack[-1]
            if frame['cell']==-1:
                out=evaluate(frame)
                if out['action'] in ('reject','solved'):
                    stack.pop();consume(out)
                    if out['action']=='solved':return result('solved' if out['validation']['valid'] else 'edge_perfect',out['codes'],out['validation'])
                    continue
            if frame['next']>=len(frame['choices']):
                decisions=decision_conjunction();stack.pop();stats['backtracks']+=1
                if residual is not None and all(frame['domains']):
                    residual.remember_completed(tuple(frame['domains']),status='unsatisfiable')
                learn_failure(decisions)
                continue
            code=frame['choices'][frame['next']];selected_mask=branch_mask(frame,frame['next']);frame['next']+=1
            child=frame['domains'].copy();child[frame['cell']]=selected_mask
            next_frame={'domains':child,'cell':-1,'choices':[],'next':0,
                        '_changed_cell':frame['cell'] if frame.get('_propagated') else None}
            if options['reason_tracking'] and decision_conjunction() is not None and selected_mask.bit_count()==1:
                parent_trace=frame.get('_reasons')
                if parent_trace is None or parent_trace.domains!=frame['domains']:
                    parent_trace=ReasonState(frame['domains'],decision_conjunction(),initial=initial);frame['_reasons']=parent_trace
                next_frame['_reasons']=parent_trace.fork(frame['cell'],code);stats['reason_nodes']+=1
            if lcg_store is not None:
                parent_lcg=frame.get('_lcg')
                if parent_lcg is None or parent_lcg.domains!=frame['domains'] or not parent_lcg.enabled:
                    parent_lcg=LCGState.reconstruct(initial,frame['domains'],decision_facts(),fingerprint=problem.fingerprint,target=target)
                    frame['_lcg']=parent_lcg;stats['lcg_rebases']+=1
                next_frame['_lcg']=parent_lcg.fork(frame['cell'],selected_mask);stats['lcg_nodes']+=1
            stack.append(next_frame)
            if time.monotonic()-last_progress>=1:
                notify('progress',codes=[value.bit_length()-1 if value.bit_count()==1 else -1 for value in child],
                       assigned_tiles=sum(value.bit_count()==1 for value in child),is_partial=True)
                last_progress=time.monotonic()
        return result('infeasible')
    except Interrupted as exc:
        if native_engine is not None and native_engine.last_result is not None:
            saved=native_engine.last_result;stack=[from_native_frame(frame) for frame in saved['stack']]
            if saved['status']=='solved':
                out=saved['value'];return result('solved' if out['validation']['valid'] else 'edge_perfect',out['codes'],out['validation'])
            if saved['status']=='exhausted':return result('infeasible')
        # Optional reasoning after the final committed rejection cannot turn
        # completed exhaustion into an invalid empty resumable checkpoint.
        if not stack:return result('infeasible')
        return result(exc.status)
    finally:
        if native_engine is not None:native_engine.close()


def solve(problem, *, seconds=10, seed=0, target='gold', hints=None, stop=None, progress=None, resume=None, learning_state=None, options=None):
    from constraint_service_lifecycle_round26 import run
    normalized=normalize_options(resume.get('search_options') if options is None and isinstance(resume,dict) else options)
    return run(_solve_public,problem,dict(seconds=seconds,seed=seed,target=target,hints=hints,
        stop=stop,progress=progress,resume=resume,learning_state=learning_state,options=normalized))
