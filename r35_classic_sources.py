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
# New producer images record a source-bound union of every backend entrusted
# with the native stack. A pre-tracking marker always requires both shaders.
# v1.17.1 adds the Windows device probe and kernel installer. Comparison of
# the complete active source image still runs independently for every resume.
GUARDED = {
    ('cuda',): ('76fcae6286598558d042bbb1c4ea1e692da7df245f14dd6fd295086a1ce6d94c', '29205e050b8e7c852aa95c968c5d691bdc48780d5cd059dc796245f0104dea67', 'c1746a0532542cf4f400c191ee7371ee659dc242bf5713dc09056538f3a99574', '98938879c7c7192125d534f300aa027ab170cc55ef0bbbe847bbf5c59fc5f1a3'),
    ('webgpu',): ('447b23fdb60459bc776c7a5e7b5070ff0a20b5eafed49dba74608484d4f014e7', 'b57e4ca0ac4a17000ba114b9d51a733be8946ff6626618b5befb87cb6869eb74', '911ecd7dc8a002ecc36352a871a5f86102cba454e616e8caf09f08e0233de460', 'c0ae42829ab139d18bb59ad7130b934ae73e3da103409d166316767e2e97412f'),
    ('cuda', 'webgpu'): ('2e4da22c5f4b32326a45ab2bd075f7e4b62d2131a7c80181aeafc1c920e84c47', '887997b0d022d130c97f31b24d578487f72c51bdbce999f8b0c28505aac1e32d', '9e11dea2f65ca2e3f08350dd53a691a44fa439e6acf4ca960488e23012822c22', 'fa4b2ac57e2e7cd30c348a469eff709297f0e74e52ee8ca30ca8f17ed29b8460'),
}
BACKENDS = ('cuda','webgpu')
CUDA_RESOURCES = frozenset({
    'dfs_kernels.cu', 'dfs_checkpoint_frames.cu', 'dfs_checkpoint_packet.cu',
    'dfs_checkpoint_delta.cu', 'dfs_checkpoint_frame_defaults.cu',
    'r35_board_codec.cu', 'r35_frame_spans.cu',
})
WEBGPU_RESOURCES = frozenset({
    'dfs_kernels.wgsl', 'dfs_counter_reduce.wgsl', 'dfs_column_copies.wgsl',
    'dfs_checkpoint_packet.wgsl', 'dfs_checkpoint_delta.wgsl',
    'dfs_checkpoint_frame_defaults.wgsl', 'r35_board_codec.wgsl', 'r35_frame_spans.wgsl',
})



def history(value):
    if (type(value) is not list or not value or any(type(v) is not str or v not in BACKENDS for v in value)
            or value != sorted(set(value))):
        raise ValueError('Invalid classic backend contribution history.')
    return tuple(value)


SEEDS = {"systematic_search.py", "dfs_webgpu.py", "constraint_native.c", "constraint_native_search.c"}
CPU_FLAGS = {"cpu_probe", "cpu_endgame", "proof_probes"}
GPU_FLAGS = {"remote_check", "hall", "forced", "dynamic_mrv", "connectivity", "symmetry", "shared_gold"}


def active_image(root, normalized, *, include_endgame=False, known=(), backends=BACKENDS):
    visited = {}
    pending = list(SEEDS)
    while pending:
        name = pending.pop()
        if not include_endgame:
            if 'cuda' not in backends and name in CUDA_RESOURCES:
                continue
            if 'webgpu' not in backends and name in WEBGPU_RESOURCES:
                continue
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


def compatible(old_root, old, current_root, current, policy, *, backends=None):
    if (not isinstance(policy,dict) or not {'cpu_probe','cpu_endgame'} <= policy.keys()
            or set(policy) - (CPU_FLAGS | GPU_FLAGS) or any(type(flag) is not bool for flag in policy.values())
            or policy.get('cpu_endgame')):
        return False
    try:
        used=history(backends) if backends is not None else BACKENDS
    except ValueError:
        return False
    known=old.keys() | current.keys()
    active=active_image(old_root,old,known=known,backends=used)
    if active is None:
        return False
    digest=image_digest(active)
    guarded=digest in GUARDED.get(used,())
    if not guarded:
        # Older reviewed controllers could carry a CPU endgame owner into
        # a probe-only profile. They also cannot attest a single backend.
        if (used!=BACKENDS or policy.get('cpu_probe') or policy.get('proof_probes')
                or digest not in REVIEWED):
            return False
    if active_image(current_root,current,known=known,backends=used)!=active:
        return False
    # Independently reconstruct both complete manifests, including every
    # diagnostic byte under dormant endgame and shader edges.
    return (active_image(old_root,old,include_endgame=True,known=known)==old
            and active_image(current_root,current,include_endgame=True,known=known)==current)
