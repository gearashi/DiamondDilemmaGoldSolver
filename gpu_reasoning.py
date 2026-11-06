"""Verified optional GPU reasoning policies and exact stack encodings.

All feature choices are bound to the checkpoint identity.  The two packed
fields retain the original low 12-bit candidate bucket/shift values and use
otherwise unused high nibbles for a dynamic frame's selected cell.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np

FEATURE_BITS = {'remote_check': 1, 'hall': 2, 'forced': 4, 'dynamic_mrv': 8,
                'connectivity': 16, 'symmetry': 32, 'shared_gold': 64, 'cpu_probe': 128, 'cpu_endgame': 256}
STACK_FIRST_MASK = STACK_SHIFT_MASK = 4095
SHARED_LIMIT = 32
SHARED_WIDTH = 1 + 2*160
POLICY_SIZE = 161 + SHARED_LIMIT*SHARED_WIDTH


def normalize_features(features=None):
    if features is None: features = {}
    if not isinstance(features, dict) or set(features)-set(FEATURE_BITS):
        raise ValueError('Unknown GPU reasoning feature.')
    if any(type(value) is not bool for value in features.values()):
        raise ValueError('GPU reasoning switches must be booleans.')
    return {name: features.get(name, False) for name in FEATURE_BITS}


def feature_bits(features):
    return sum(bit for name, bit in FEATURE_BITS.items() if features[name])


def pack_frame(first, shift, cell):
    if not 0 <= first < 4096 or not 0 <= shift < 4096 or not 0 <= cell < 160:
        raise ValueError('GPU frame cannot be represented exactly.')
    return first | ((cell & 15) << 12), shift | ((cell >> 4) << 12)


def frame_cells(firsts, shifts):
    return (np.asarray(firsts, np.uint16) >> 12) | ((np.asarray(shifts, np.uint16) >> 12) << 4)


def prepare_policy(engine, features, gold_data):
    engine.gpu_features = normalize_features(features)
    engine.feature_bits = feature_bits(engine.gpu_features)
    engine.dynamic_order = bool(engine.feature_bits & (4 | 8))
    engine.host_policy = np.zeros(POLICY_SIZE, np.uint32)
    engine.host_policy[:160] = 1
    engine.policy_metadata = {'version': 1, 'features': engine.gpu_features}
    engine._gold_problem = None
    engine._shared_cuts = []
    engine.cpu_assist = None
    if engine.gpu_features['symmetry']:
        if gold_data is None:
            raise ValueError('Fresh GPU symmetry requires explicit piece segment data.')
        from search_problem import SearchProblem
        from constraint_symmetry import canonical_root_domains
        problem = SearchProblem(gold_data)
        if not (np.array_equal(problem.neighbors, engine.neighbors) and
                np.array_equal(problem.sides, engine.neighbor_sides)):
            raise ValueError('GPU symmetry requires the verified canonical board.')
        domains, metadata = canonical_root_domains(problem)
        if not metadata['applied']:
            raise ValueError('GPU symmetry could not be verified: '+metadata['reason'])
        engine.host_policy[:160] = np.array([bool(mask & 7) for mask in domains], np.uint32)
        engine.policy_metadata['symmetry'] = metadata
        engine._gold_problem = problem
    if engine.feature_bits:
        identity = json.dumps(engine.policy_metadata, sort_keys=True, separators=(',', ':')).encode()
        engine.fingerprint = hashlib.sha256(bytes.fromhex(engine.fingerprint) + identity).hexdigest()
    if engine.feature_bits & 384:
        if gold_data is None:raise ValueError('CPU/GPU assistance requires explicit Gold piece data.')
        from gpu_cpu_assist import CPUAssist
        engine.cpu_assist = CPUAssist(gold_data, engine.fingerprint,
            probe=engine.gpu_features['cpu_probe'], endgame=engine.gpu_features['cpu_endgame'])
        if not (np.array_equal(engine.cpu_assist.problem.neighbors,engine.neighbors) and np.array_equal(engine.cpu_assist.problem.sides,engine.neighbor_sides)):
            raise ValueError('CPU/GPU assistance geometry differs from its independently constructed problem.')
    if gold_data is not None and (engine.gpu_features['shared_gold'] or engine.gpu_features['connectivity']):
        if engine._gold_problem is None:
            from search_problem import SearchProblem
            problem = SearchProblem(gold_data)
            if np.array_equal(problem.neighbors, engine.neighbors) and np.array_equal(problem.sides, engine.neighbor_sides):
                engine._gold_problem = problem


def verified_gold_cut(engine, literals):
    """Validate an alleged exclusion independently from explicit segment pairs.

    Only complete proper closed components are admitted.  Received clauses
    never become trusted because of a source label, score, or fingerprint.
    """
    problem = engine._gold_problem
    if not engine.gpu_features['shared_gold'] or problem is None:
        return None
    if not isinstance(literals, (list, tuple)) or not literals:
        raise ValueError('Gold exclusions need placement literals.')
    assignment = [-1]*160
    for literal in literals:
        if not isinstance(literal, (list, tuple)) or len(literal) != 2:
            raise ValueError('Malformed Gold exclusion literal.')
        cell, code = literal
        if type(cell) is not int or type(code) is not int or not 0 <= cell < 160 or not 0 <= code < 480:
            raise ValueError('Gold exclusion literal is out of range.')
        if assignment[cell] not in (-1, code):
            raise ValueError('Contradictory Gold exclusion literals.')
        assignment[cell] = code
    components = problem._closed_components(assignment)
    proper = [(size, cells) for size, cells in components if size < problem.total_segments]
    if not proper: return None
    _, cells = min(proper, key=lambda item: (len(item[1]), item[0]))
    return tuple(sorted((cell, assignment[cell]) for cell in set(cells)))


def install_gold_cut(engine, literals):
    cut = verified_gold_cut(engine, literals)
    if cut is None or cut in engine._shared_cuts:
        return False
    return append_verified_cut(engine, cut)


def append_verified_cut(engine, cut):
    if not cut or cut in engine._shared_cuts:return False
    engine._shared_cuts.append(cut)
    engine._shared_cuts[:] = engine._shared_cuts[-SHARED_LIMIT:]
    engine.host_policy[160:] = 0
    engine.host_policy[160] = len(engine._shared_cuts)
    for index, item in enumerate(engine._shared_cuts):
        offset = 161 + index*SHARED_WIDTH
        engine.host_policy[offset] = len(item)
        engine.host_policy[offset+1:offset+1+2*len(item)] = np.asarray(item, np.uint32).ravel()
    engine._upload_policy()
    return True
