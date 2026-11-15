"""Verify source and frontier provenance for classic GPU exhaustive results."""
from functools import lru_cache
import hashlib
import io
import json
from pathlib import Path
import re
import zipfile

from exact_source_evidence import snapshot_search_sources


# Complete normalized source manifests written by v1.14.0 (8e010cc).
_V114 = {
    'cuda': '07f1f816bc4383aca96ad8445ce123931920516eeb564a18d11f215a4e30a068',
    'webgpu': 'c52111e0b179225124eff961ed4b0e684f10fede0a9ca501b5bfda23a6a4b809',
}
_SHADER_FEATURES = {'remote_check', 'hall', 'forced', 'dynamic_mrv', 'connectivity', 'symmetry', 'shared_gold'}
SCHEMA = 'diamond-classic-exhaustion-v1'


def checksum(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def profile_name(features):
    from gpu_reasoning import feature_identity
    return 'systematic-' + checksum(feature_identity(features))[:12] if any(features.values()) else 'systematic'


@lru_cache(maxsize=128)
def _terminal_header(checkpoint, file_identity):
    import numpy as np
    try:
        with np.load(checkpoint,allow_pickle=False) as archive:
            if (str(archive['exact_terminal']) == 'exhausted' and 'exact_source_origin' in archive
                    and 'exact_source_binding' in archive and str(archive['exact_source_binding'])):
                features = json.loads(str(archive['exact_gpu_features']))
                target = (str(archive['exact_source_target']) if 'exact_source_target' in archive else
                          read(run_dir(checkpoint.parent.parent,str(archive['exact_source_origin']))/'run-config.json')['target'])
                if target=='gold' and profile_name(features)==checkpoint.parent.name:
                    return tuple(sorted(key for key,value in features.items() if value))
    except (OSError, EOFError, zipfile.BadZipFile, ValueError, KeyError, TypeError, AttributeError):
        pass
    return None


def terminal_features(checkpoint):
    try:
        st = checkpoint.stat()
        return _terminal_header(checkpoint,(st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def terminal_profiles(runtime):
    profiles = []
    for folder in (runtime/'systematic', *runtime.glob('systematic-????????????')):
        result = terminal_features(folder/'checkpoint.npz')
        if result is not None:
            profiles.append(result)
    return profiles


def capture(root, evidence, data_sha):
    # Both backend sources permit CUDA <-> WebGPU checkpoint transfer.
    names = {'systematic_search.py', 'dfs_webgpu.py', 'constraint_native.c', 'constraint_native_search.c'}
    hashes = snapshot_search_sources(root, evidence, names, include_families=False)
    return {**hashes, 'tiles.json': data_sha}


def run_dir(run, run_id):
    if not isinstance(run_id, str) or re.fullmatch(r'[0-9]{8}T[0-9]{12}Z', run_id) is None:
        raise ValueError('Invalid systematic evidence run ID.')
    directory = run/'runs'/run_id
    if directory.resolve().parent != (run/'runs').resolve():
        raise ValueError('Systematic evidence directory is outside this run.')
    return directory


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def archived_sources(directory, recorded):
    normalized = {}
    for name, expected in recorded.items():
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_]+\.(?:py|c|cu|wgsl|json)', name):
            raise ValueError('Invalid name in archived systematic source list.')
        file = directory/name
        if file.resolve().parent != directory.resolve():
            raise ValueError('Archived source is outside its systematic run.')
        data = file.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Archived systematic source differs: '+name)
        if name != 'tiles.json':
            normalized[name] = hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
    return normalized


class HashedStream:
    """A write-only ZIP stream. Data descriptors keep each final byte in write order."""
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()

    def write(self, data):
        size = self.stream.write(data)
        self.hash.update(memoryview(data).cast('B')[:size])
        return size

    def read(self,*args):
        raise io.UnsupportedOperation("This archive stream is write-only.")

    def flush(self):
        self.stream.flush()


def check_receipt(run, origin, generation, checkpoint, binding, *, parked=False):
    suffix = '-parked' if parked else ''
    receipt = read(run_dir(run,origin)/f'checkpoint-{generation}{suffix}.json')
    with checkpoint.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if receipt != {'sha256':actual, 'bytes':checkpoint.stat().st_size, 'source_binding':binding}:
        raise ValueError('Native systematic checkpoint differs from the archived write receipt.')


def origin_binding(hashes, scope):
    return checksum({'schema': SCHEMA, 'source_sha256': hashes, **scope})


def verify_origin(run, origin, binding, current, scope):
    """Check native binding and original bytes; allow equivalent LF/CRLF checkouts."""
    directory = run_dir(run, origin)
    config = read(directory/'run-config.json')
    source = config['source_sha256']
    if (config.get('run_id') != origin or config.get('verified_coverage_origin') is not True
            or origin_binding(source, scope) != binding
            or source.get('tiles.json') != current.get('tiles.json') or 'tiles.json' not in source):
        raise ValueError('Systematic checkpoint source or objective binding differs.')
    normalized = archived_sources(directory, source)
    if source != current:
        current_root = Path(__file__).resolve().parent
        live = archived_sources(current_root,{k:v for k,v in current.items() if k != 'tiles.json'})
        if live != normalized:
            from r35_classic_sources import compatible
            if not compatible(directory,normalized,current_root,live,scope.get('gpu_policy')):
                raise ValueError('Systematic source has changed outside a reviewed dormant CPU endgame.')
    return source


def record(hashes, scope, origin, total, checked):
    body = {'schema': SCHEMA, 'run_id': origin, 'state': 'exhausted', 'source_sha256': hashes,
            'origin_binding':origin_binding(hashes,scope),
            **scope, 'completed_jobs': total, 'total_jobs': total, 'candidates_checked': checked}
    return {**body, 'sha256': checksum(body)}


def check_terminal_record(run, origin, source, scope, total, checked, saved_digest, marker):
    from io_utils import atomic_json
    expected = record(source, scope, origin, total, checked)
    if saved_digest != expected['sha256']:
        raise ValueError('Systematic exhaustion record does not match its checkpoint and source.')
    existing = read(marker) if marker.exists() else None
    if existing is not None and existing != expected:
        # A native historical migration can commit just before replacing its old marker.
        prior = run_dir(run,origin)/'previous-exhaustion.json'
        if not prior.is_file() or read(prior) != existing:
            raise ValueError('Systematic exhaustion record does not match its checkpoint and source.')
    if existing != expected:
        atomic_json(marker,expected)


def reopen_terminal(run, checkpoint, marker, archive, hashes, scope, jobs):
    """Read the authenticated completed ownership ledger without allocating a GPU."""
    import numpy as np
    origin = str(archive['exact_source_origin'])
    binding = str(archive['exact_source_binding'])
    try:
        check_receipt(run,origin,int(archive['exact_generation']),checkpoint,binding)
        source = verify_origin(run,origin,binding,hashes,scope)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError('Systematic exhaustion source evidence failed: '+str(exc)) from exc
    ids, states, depths = archive['exact_job_ids'], archive['states'], archive['depths']
    counters, retired = archive['counters'], archive['exact_retired_counters']
    if (int(archive['exact_completed']) != jobs or int(archive['exact_total']) != jobs
            or int(archive['exact_cursor']) != jobs or len(archive['exact_paused_job_ids'])
            or ids.ndim != 1 or not len(ids) or np.any(ids != -1) or states.shape != ids.shape
            or not np.all((states == 2) | (states == 3)) or depths.shape != ids.shape or np.any(depths != 0)
            or counters.shape != (3,len(ids)) or retired.shape != (3,)):
        raise ValueError('Checkpoint claims exhaustion while branches remain unfinished.')
    checked = int(archive['exact_candidates'])
    check_terminal_record(run,origin,source,scope,jobs,checked,str(archive['exact_exhaustion_record']),marker)
    counts = (counters.sum(axis=1,dtype=np.uint64)+retired).tolist()
    return dict(verified_coverage_origin=True, terminal_reopened_on_host=True, source_result_run_id=origin,
                replicas=len(ids), total_jobs=jobs, assigned_jobs=jobs, exhausted_jobs=jobs,
                active_jobs=0, paused_jobs=0, queued_jobs=0, unassigned_jobs=0,
                candidates_checked=checked, nodes_checked=counts[0], counters=counts,
                max_depth=int(archive['exact_max_depth']))


def legacy_origin(run, evidence, scope, policy, summary, size, generation, terminal, marker):
    """Validate the shipped default/shader-only runner and the linked session history."""
    if any(enabled and key not in _SHADER_FEATURES for key, enabled in policy.items()):
        return False
    # A failed attempt, or a different algorithm using the dashboard, can replace
    # the two global status files. Each finished historical session has its own archive.
    candidates = [evidence]
    candidates.extend(sorted((run/'runs').iterdir(),reverse=True))
    for directory in candidates:
        prefix = 'previous-' if directory == evidence else ''
        try:
            status = read(directory/(prefix+'status.json' if prefix else 'final-status.json'))
            config = read(directory/(prefix+'run-config.json'))
            if (config.get('verified_coverage_origin') is not None
                    or status.get('checkpoint',{}).get('completed_generation') != generation):
                continue
            history = _historical_chain(run, config, status, scope, policy, summary, size, generation, terminal, marker)
            if history: return history
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return False


def _historical_chain(run, config, status, scope, policy, summary, size, generation, terminal, marker):
    try:
        seen, chain = set(), []
        last = True
        while True:
            origin = config['run_id']
            if origin in seen:
                return False
            seen.add(origin)
            directory = run_dir(run, origin)
            source = config['source_sha256']
            if (config != read(directory/'run-config.json') or status != read(directory/'final-status.json')
                    or config['method'] != 'systematic' or status['method'] != 'systematic'
                    or status['run_id'] != origin or status['source_sha256'] != source
                    or config['gpu_features'] != policy or config['target'] != scope['target']
                    or config['frontier_sha256'] != scope['frontier_sha256']
                    or source['tiles.json'] != scope['input_sha256']
                    or checksum(archived_sources(directory,source)) != _V114.get(config['backend'])):
                return False
            chain.append((config,status))
            if last:
                if (status.get('state') != (terminal or 'stopped') or any(status.get(k) != v for k,v in summary.items())
                        or status['checkpoint']['file_bytes'] != size or status['checkpoint']['completed_generation'] != generation):
                    return False
                if terminal == 'exhausted':
                    prior = read(marker)
                    if prior != {'state':'exhausted', 'source_sha256':source, 'frontier_sha256':scope['frontier_sha256'],
                                 'completed_jobs':status['exhausted_jobs'], 'total_jobs':status['total_jobs'],
                                 'candidates_checked':status['candidates_checked']}:
                        return False
                last = False
            if config['resumed'] is False:
                return list(reversed(chain))
            status = read(directory/'previous-status.json')
            config = read(directory/'previous-run-config.json')
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def verify_native_history(engine, inputs, features, checkpoint, history, prefixes, lengths, stop):
    """Replay a small fixed-budget v1.14 history and compare the full native stack and ledger."""
    import numpy as np
    from systematic_jobs import JobLedger

    # A replay chain has one puzzle. Keep its first independently computed
    # candidate buckets; sizes, seeds and mutable search fields remain per engine.
    buckets = {}
    # v1.14 used mask compatibility and loop tracing before the physical mate table.
    class HistoricalDFS(engine):
        def _prepare_tables(self):
            key = self.masks.tobytes()
            if key in buckets:
                for name,value in buckets[key].items():
                    setattr(self,name,value)
                return
            super()._prepare_tables()
            names=('colors','host_faces','host_reverse','host_single_offsets','host_pair_offsets','host_pool')
            values={name:getattr(self,name) for name in names}
            for value in values.values():
                if isinstance(value,np.ndarray):
                    value.setflags(write=False)
            buckets.clear()
            buckets[key]=values

        def _prepare_gold(self, data):
            super()._prepare_gold(data)
            self.host_gold_blocked = np.zeros(1,np.uint32)
            self.host_future_poses = np.empty(0,np.uint32)
            self.host_gold_rules = self.host_gold_blocked

    cost = launches = last_generation = 0
    for config, status in history:
        generation = status['checkpoint']['completed_generation']
        budget = status['gpu_work_budget']
        count = config['replicas']
        if config['adaptive_gpu_batches'] or not 1 <= count <= 1024 or not 1 <= budget <= 512 or generation < last_generation:
            return False
        cost += (generation-last_generation)*count*budget
        launches += generation-last_generation
        last_generation = generation
    if cost > 2_000_000 or launches > 8192:
        return False

    generation, carry = 0, None
    owner=inputs.get('reuse_from')
    tables=owner.replay_tables() if owner is not None and len(history)>1 else None
    if tables is not None:
        inputs={**inputs,'replay_tables':tables}
    try:
        for config, status in history:
            gpu = HistoricalDFS(**inputs, n=config['replicas'], seed=config['seed'], prefixes=[],
                                **({'gpu_features':features} if any(features.values()) else {}))
            try:
                if carry is None:
                    ledger = JobLedger.empty(len(prefixes),gpu.n)
                    ledger.refill(gpu,prefixes,lengths)
                else:
                    ledger = JobLedger.resume(gpu,carry,prefixes,lengths)
                end = status['checkpoint']['completed_generation']
                while generation < end:
                    if stop(): return False
                    gpu.step(status['gpu_work_budget'])
                    generation += 1
                    flags = gpu.states.get()
                    if np.any(flags == 1): return False
                    ledger.retire(flags)
                    if (generation < end or ledger.cursor < status['assigned_jobs']
                            or ledger.paused_count > status['paused_jobs']):
                        if (ledger.paused_count or ledger.cursor < ledger.total) and np.any(ledger.ids < 0):
                            ledger.refill(gpu,prefixes,lengths)
                carry = {**gpu.snapshot_checkpoint(), **ledger.fields()}
            finally:
                if hasattr(gpu,'close'): gpu.close()
        with np.load(checkpoint,allow_pickle=False) as archive:
            # Check both ownership and native per-cell cursors, rotations, RNG and counters.
            return all(k in archive and np.array_equal(v,archive[k]) for k,v in carry.items())
    except (KeyError, ValueError, RuntimeError, MemoryError):
        return False
    finally:
        if tables is not None:
            tables.close()
