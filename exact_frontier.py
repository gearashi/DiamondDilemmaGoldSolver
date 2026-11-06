"""Deterministic disjoint prefixes covering all edge-compatible arrangements.

A prefix lists orientation codes in the supplied cell fill order. Refinement
replaces one prefix with every unused-tile child satisfying assigned seams.
Rejected children cannot have an edge-compatible completion. Cancellation and
resource caps retain the unsplit parent. No symmetry quotient is applied.

This partitions labelled-tile arrangements, including distinct rotation codes
with identical masks. It does not decide gold-loop connectivity or promise that
the resulting search can finish. The frontier byte cap conservatively charges
retained Python prefixes plus their eventual dense output; lookup-cache storage
has a separate fixed entry cap. Temporary single-expansion work is O(code count).
"""
from __future__ import annotations

from collections import OrderedDict, deque
from dataclasses import dataclass
import hashlib
import math
import operator
import struct
import sys

import numpy as np

FORMAT_VERSION = 1


@dataclass(frozen=True)
class FrontierPlan:
    prefixes: np.ndarray
    lengths: np.ndarray
    metadata: dict


def _positive_integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be an integer')
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(name + ' must be an integer') from exc
    if result < minimum:
        raise ValueError(name + ' must be at least ' + str(minimum))
    return result


def _integer_array(value, name):
    result = np.asarray(value)
    if result.dtype.kind not in 'iu':
        raise ValueError(name + ' must contain integers')
    return result


def _normalize(masks, neighbor_cells, neighbor_sides, order, rotations, mask_width):
    rotations = _positive_integer(rotations, 'rotations')
    mask_width = _positive_integer(mask_width, 'mask_width')
    if mask_width > 16:
        raise ValueError('mask_width cannot exceed 16')
    cells = _integer_array(neighbor_cells, 'neighbor_cells')
    sides = _integer_array(neighbor_sides, 'neighbor_sides')
    raw_masks = _integer_array(masks, 'masks')
    raw_order = _integer_array(order, 'order')
    if cells.ndim != 2 or not cells.shape[0] or not cells.shape[1]:
        raise ValueError('neighbor_cells must be a nonempty two-dimensional array')
    count, side_count = cells.shape
    if count > 32767 or count * rotations > 32768:
        raise ValueError('Puzzle exceeds int16 prefix representation')
    if sides.shape != cells.shape or raw_masks.shape != (count * rotations, side_count):
        raise ValueError('Mask and neighbor dimensions disagree')
    if raw_order.shape != (count,) or not np.array_equal(np.sort(raw_order), np.arange(count)):
        raise ValueError('order must contain each cell exactly once')
    if np.any(raw_masks < 0) or np.any(raw_masks >= (1 << mask_width)):
        raise ValueError('Mask exceeds the specified endpoint bit width')
    for cell in range(count):
        for side in range(side_count):
            other, other_side = int(cells[cell, side]), int(sides[cell, side])
            if other == -1 and other_side == -1:
                continue  # Explicit open boundary, useful for small exact fixtures.
            if not 0 <= other < count or not 0 <= other_side < side_count or other == cell:
                raise ValueError('Invalid neighbor cell or side')
            if int(cells[other, other_side]) != cell or int(sides[other, other_side]) != side:
                raise ValueError('Neighbor relation is not reciprocal')
    return (np.ascontiguousarray(raw_masks, dtype='<u2'),
            np.ascontiguousarray(cells, dtype='<i4'),
            np.ascontiguousarray(sides, dtype='<i4'),
            np.ascontiguousarray(raw_order, dtype='<i2'), rotations, mask_width)


def plan_frontier(masks, neighbor_cells, neighbor_sides, order, target_jobs, *,
                  max_jobs=None, max_frontier_bytes=256 * 1024 * 1024,
                  max_expansions=None, stop_requested=None, progress=None,
                  progress_every=256, candidate_cache_entries=4096,
                  rotations=3, mask_width=11):
    """Return a deterministic prefix-free cover, including at a resource stop.

    prefixes has shape (jobs, cells), dtype int16. Codes occupy the leading
    lengths[job] entries IN FILL ORDER; all later entries are -1. A zero-length
    prefix represents the entire search. Full-length prefixes are complete
    edge-compatible arrangements and still require the independent loop check.

    target_jobs is a refinement target, not a truncation count. Replacing a
    parent may overshoot it by at most rotations*cells-1. If max_jobs, byte
    budget, max_expansions, or stop_requested prevents further splitting, the
    returned cover remains complete. progress receives independent metadata
    dictionaries initially, periodically, and at completion. Callback exceptions
    propagate instead of returning a partial unmarked result.

    Raw arrangement counts are exact decimal strings for labelled tile identities
    and rotation codes, before any pattern or whole-board symmetry equivalence.
    """
    masks, neighbors, neighbor_sides, order, rotations, mask_width = _normalize(
        masks, neighbor_cells, neighbor_sides, order, rotations, mask_width)
    target_jobs = _positive_integer(target_jobs, 'target_jobs')
    count, side_count = neighbors.shape
    code_count = count * rotations
    if max_jobs is None:
        max_jobs = target_jobs + code_count - 1
    max_jobs = _positive_integer(max_jobs, 'max_jobs')
    max_frontier_bytes = _positive_integer(max_frontier_bytes, 'max_frontier_bytes')
    progress_every = _positive_integer(progress_every, 'progress_every')
    candidate_cache_entries = _positive_integer(candidate_cache_entries, 'candidate_cache_entries', 0)
    if max_expansions is not None:
        max_expansions = _positive_integer(max_expansions, 'max_expansions', 0)
    if stop_requested is not None and not callable(stop_requested):
        raise ValueError('stop_requested must be callable')
    if progress is not None and not callable(progress):
        raise ValueError('progress must be callable')

    # Canonical bytes include dimensions/conventions; input dtype/endianness do not
    # change the identity. The fill order has its own digest and combined plan hash.
    dimensions = struct.pack('<5I', count, side_count, rotations, mask_width, FORMAT_VERSION)
    puzzle_digest = hashlib.sha256(dimensions + masks.tobytes() + neighbors.tobytes()
                                  + neighbor_sides.tobytes()).hexdigest()
    order_digest = hashlib.sha256(order.tobytes()).hexdigest()
    plan_digest = hashlib.sha256(bytes.fromhex(puzzle_digest) + bytes.fromhex(order_digest)).hexdigest()
    rank = np.empty(count, np.int32)
    rank[order] = np.arange(count)
    reversed_masks = np.zeros_like(masks)
    for bit in range(mask_width):
        reversed_masks |= ((masks >> bit) & 1) << (mask_width - 1 - bit)
    all_codes = tuple(range(code_count))
    # Number of raw labelled arrangements extending a length-d legal prefix.
    weights = [math.factorial(count - depth) * rotations ** (count - depth)
               for depth in range(count + 1)]
    dense_bytes_per_job = count * np.dtype(np.int16).itemsize + np.dtype(np.int16).itemsize
    int_charge = sys.getsizeof(code_count)
    pointer_allowance = 4 * struct.calcsize('P')

    def node(prefix, used):
        # Charge repeated integer references as separate objects, conservatively.
        charge = (sys.getsizeof(prefix) + len(prefix) * int_charge
                  + sys.getsizeof(used) + sys.getsizeof((prefix, used, 0))
                  + pointer_allowance + dense_bytes_per_job)
        return prefix, used, charge

    root = node((), 0)
    if root[2] > max_frontier_bytes:
        raise ValueError('max_frontier_bytes cannot hold even the root cover')
    queue = deque([root])
    leaves = []
    retained_bytes = root[2]
    cache = OrderedDict()
    cache_hits = cache_misses = 0
    expanded = 0
    generated = 1
    pruned_children = 0
    dead_prefixes = 0
    raw_pruned = 0
    reason = 'planning'

    def metadata():
        jobs = len(queue) + len(leaves)
        return {
            'format_version': FORMAT_VERSION,
            'puzzle_sha256': puzzle_digest, 'order_sha256': order_digest,
            'plan_identity_sha256': plan_digest,
            'cells': count, 'rotations': rotations, 'mask_width': mask_width,
            'prefix_convention': 'orientation codes in fixed fill order, padded with -1',
            'cover_complete': True, 'prefix_free': True,
            'target_jobs': target_jobs, 'jobs': jobs, 'target_reached': jobs >= target_jobs,
            'max_jobs': max_jobs, 'max_frontier_bytes': max_frontier_bytes,
            'charged_frontier_bytes': retained_bytes,
            'frontier_byte_scope': 'retained nodes plus dense output; lookup cache and single-expansion temporaries separately bounded',
            'candidate_cache_entry_limit': candidate_cache_entries,
            'candidate_cache_entries': len(cache),
            'candidate_cache_hits': cache_hits, 'candidate_cache_misses': cache_misses,
            'expanded_prefixes': expanded, 'generated_prefixes': generated,
            'pruned_children': pruned_children, 'dead_prefixes': dead_prefixes,
            'complete_board_jobs': len(leaves),
            'raw_arrangements_total': str(weights[0]),
            'raw_arrangements_remaining': str(weights[0] - raw_pruned),
            'raw_arrangements_pruned': str(raw_pruned),
            'raw_count_scope': 'labelled physical tiles and rotation codes; not symmetry-quotiented',
            'exhausted': jobs == 0, 'all_jobs_complete_boards': not queue,
            'stop_reason': reason,
        }

    def report():
        if progress is not None:
            progress(metadata())

    def candidates(prefix):
        nonlocal cache_hits, cache_misses
        depth = len(prefix)
        cell = int(order[depth])
        constraints = []
        for side in range(side_count):
            other = int(neighbors[cell, side])
            if other >= 0 and rank[other] < depth:
                other_code = prefix[int(rank[other])]
                wanted = int(reversed_masks[other_code, int(neighbor_sides[cell, side])])
                constraints.append((side, wanted))
        signature = tuple(constraints)
        if signature in cache:
            cache_hits += 1
            values = cache.pop(signature)
            cache[signature] = values
            return values
        cache_misses += 1
        eligible = np.ones(code_count, dtype=bool)
        for side, wanted in constraints:
            eligible &= masks[:, side] == wanted
        values = tuple(code for code in all_codes if eligible[code])
        if candidate_cache_entries:
            cache[signature] = values
            if len(cache) > candidate_cache_entries:
                cache.popitem(last=False)
        return values

    report()
    while True:
        jobs = len(queue) + len(leaves)
        if stop_requested is not None and stop_requested():
            reason = 'cancelled'
            break
        if jobs >= target_jobs:
            reason = 'target_reached'
            break
        if not queue:
            reason = 'exhausted' if not leaves else 'fully_expanded'
            break
        if max_expansions is not None and expanded >= max_expansions:
            reason = 'expansion_limit'
            break
        prefix, used, old_charge = queue[0]
        depth = len(prefix)
        children = [node(prefix + (code,), used | (1 << (code // rotations)))
                    for code in candidates(prefix) if not (used >> (code // rotations)) & 1]
        new_jobs = jobs - 1 + len(children)
        new_bytes = retained_bytes - old_charge + sum(child[2] for child in children)
        if new_jobs > max_jobs:
            reason = 'job_limit'
            break
        if new_bytes > max_frontier_bytes:
            reason = 'memory_limit'
            break
        # This replacement is the only cover mutation. Never retain only part
        # of a parent's children, including when cancellation arrives mid-split.
        queue.popleft()
        destination = leaves if depth + 1 == count else queue
        destination.extend(children)
        expanded += 1
        generated += len(children)
        rejected = (count - depth) * rotations - len(children)
        pruned_children += rejected
        dead_prefixes += not children
        raw_pruned += rejected * weights[depth + 1]
        retained_bytes = new_bytes
        if expanded % progress_every == 0:
            report()

    ordered = sorted((*queue, *leaves), key=lambda item: (len(item[0]), item[0]))
    prefixes = np.full((len(ordered), count), -1, dtype=np.int16)
    lengths = np.empty(len(ordered), dtype=np.int16)
    for index, (prefix, _, _) in enumerate(ordered):
        lengths[index] = len(prefix)
        prefixes[index, :len(prefix)] = prefix
    result_metadata = metadata()
    result_metadata['minimum_depth'] = int(lengths.min()) if len(lengths) else None
    result_metadata['maximum_depth'] = int(lengths.max()) if len(lengths) else None
    result_metadata['prefixes_sha256'] = hashlib.sha256(
        prefixes.astype('<i2', copy=False).tobytes() + lengths.astype('<i2', copy=False).tobytes()).hexdigest()
    if progress is not None:
        progress(dict(result_metadata))
    return FrontierPlan(prefixes, lengths, result_metadata)
