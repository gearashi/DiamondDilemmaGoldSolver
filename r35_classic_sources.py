"""Reviewed dormant CPU-endgame boundary for classic GPU exhaustion evidence."""
import hashlib
import json
import re

from exact_source_evidence import _source_references


# SHA256 of the reviewed active controllers, kernels, validators, and provenance
# code in each supported text-normalized image. This module contains the pins;
# its own full hash is still compared between the saved and current source sets.
REVIEWED = frozenset({
    '9ef7d713a75de279771abb29ea7253fa659a5610b49dcc969b7e90cd86ef89c4',
    '8e687009420a495e9b9540522825e4df9a70933d380040ba8e31b7e1376f603a',
    'b111d4f02168a9a78a57df276fb4148c5a0738ace25df75392209ad8d9ce1897',
})
SEEDS = {"systematic_search.py", "dfs_webgpu.py", "constraint_native.c", "constraint_native_search.c"}
CPU_FLAGS = {"cpu_probe", "cpu_endgame", "proof_probes"}
GPU_FLAGS = {"remote_check", "hall", "forced", "dynamic_mrv", "connectivity", "symmetry", "shared_gold"}


def active_image(root, normalized, *, include_endgame=False, known=()):
    visited = {}
    pending = list(SEEDS)
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        if name not in normalized:
            return None
        visited[name] = normalized[name]
        if name.endswith('.py'):
            # The reader has already authenticated every archived/current file.
            body = (root/name).read_bytes()
            if hashlib.sha256(body.replace(b'\r\n', b'\n')).hexdigest() != normalized[name]:
                raise ValueError('Classic proof source changed during dependency inspection.')
            modules, resources = _source_references(name,body)
            pending.extend(resources)
            for module in modules:
                if not include_endgame and name == 'gpu_cpu_assist.py' and module == 'constraint_dfs':
                    continue
                child = module + '.py'
                if (child in normalized or child in known or (root/child).is_file()
                        or module.startswith(('constraint_', 'gpu_', 'domain_', 'sampling_', 'dfs_', 'r27_'))
                        or re.match(r'^r[0-9]+_',module)):
                    pending.append(child)
    return visited


def image_digest(image):
    core = {key:v for key,v in image.items() if key != 'r35_classic_sources.py'}
    body = json.dumps(core, sort_keys=True, separators=(',',':')).encode()
    return hashlib.sha256(body).hexdigest()


def compatible(old_root, old, current_root, current, policy):
    if (not isinstance(policy,dict) or not {'cpu_probe','cpu_endgame'} <= policy.keys()
            or set(policy) - (CPU_FLAGS | GPU_FLAGS) or any(type(flag) is not bool for flag in policy.values())
            or any(policy.get(flag,False) for flag in CPU_FLAGS)):
        return False
    known = old.keys() | current.keys()
    active = active_image(old_root,old,known=known)
    if active is None or image_digest(active) not in REVIEWED:
        return False
    # Both backends can own part of a resumed search. Always retain both kernels.
    if active_image(current_root,current,known=known) != active:
        return False
    # A genuinely removed dormant dependency drops out of its own full graph.
    # A missing manifest entry that another archived file still imports is damage.
    return (active_image(old_root,old,include_endgame=True,known=known) == old
            and active_image(current_root,current,include_endgame=True,known=known) == current)
