"""Installed-package availability, without importing solvers or probing GPU drivers.

Native explanation availability additionally checks local build metadata and hashes.
An available package/build is not a guarantee of a working device or native library.
The chosen engine still validates those when its search starts.
"""
from __future__ import annotations
import importlib.util
import hashlib,json
from pathlib import Path
from importlib.metadata import version as package_version, PackageNotFoundError
import platform
import sys

from search_options import EXACT_ALGORITHMS, validate_algorithm_options, option_engines


def _installed(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_metadata(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _native_explanation_build(root=None, *, display=False):
    """Check the required local build without loading executable native code."""
    root=Path(root) if root is not None else Path(__file__).resolve().parent
    reason='Native explanations require a matching local compiled search library. Run python build_native.py in this installation first.'
    try:
        folder=root/'.native'
        load_metadata = _build_metadata
        digest = _digest
        if display:
            from functools import partial
            from dashboard_snapshots import read_display_file
            load_metadata = partial(read_display_file, reader=_build_metadata)
            digest = partial(read_display_file, reader=_digest)
        metadata = load_metadata(folder/'search-build.json')
        name=metadata.get('library')
        if (metadata.get('abi')!=1 or metadata.get('platform')!=sys.platform
                or metadata.get('machine')!=platform.machine()
                or not isinstance(name,str) or not name or Path(name).name!=name):
            return {'available':False,'reason':reason,'detection':'native_build_metadata'}
        available=(metadata.get('source_sha256')==digest(root/'constraint_native_search.c')
                   and metadata.get('library_sha256')==digest(folder/name))
    except (OSError,ValueError,TypeError,AttributeError):available=False
    return {'available':bool(available),'reason':'' if available else reason,'detection':'native_build_metadata'}


def _tested_sat_version():
    try:
        return package_version('python-sat') == '1.9.dev15'
    except PackageNotFoundError:
        return False


def runtime_capabilities(*, display=False):
    """Describe this interpreter, never another installed Python environment."""
    def feature(available, requirement):
        return {'available': bool(available), 'reason': '' if available else requirement}

    cp = feature(_installed('ortools'), 'Requires OR-Tools in this Python environment.')
    sat = feature(_installed('pysat'), 'Requires python-sat in this Python environment.')
    proof_arch=platform.machine().lower() in ({'x86_64','amd64'} if sys.platform=='linux' else {'x86_64','amd64','arm64','aarch64'})
    sat_persistence=feature(sat['available'] and sys.platform in ('linux','darwin') and proof_arch
                            and platform.python_implementation()=='CPython' and sys.version_info[:2]==(3,12) and _tested_sat_version(),
                            'Verified native SAT clause persistence requires tested Linux x86-64 or macOS x86-64/ARM64, CPython 3.12 and python-sat 1.9.dev15; the tested Windows build is unavailable.')
    cuda = feature(sys.platform != 'darwin' and _installed('cupy'),
                   'CUDA is unavailable on macOS.' if sys.platform == 'darwin' else
                   'Requires CuPy in this Python environment. Choose WebGPU if installed.')
    webgpu = feature(_installed('wgpu'), 'Requires wgpu in this Python environment.')
    gpu = feature(cuda['available'] or webgpu['available'],
                  'Requires a GPU package in this Python environment: CuPy or wgpu.')
    return {
        'option_engines': option_engines(),
        'python': platform.python_version(), 'detection': 'installed_packages',
        'algorithms': {'dfs': feature(True, ''), 'parallel-dfs': feature(True, ''),
                       'gpu-dfs': dict(gpu), 'hybrid': dict(gpu), 'gpu-domains':dict(cuda),
                       'cp': dict(cp), 'cp-sat': dict(cp), 'cp-sat-circuit': dict(cp),
                       'sat': dict(sat), 'sat-propagator':dict(sat),
                       'sat-channel':dict(sat),'cp-sat-regions':dict(cp)},
        'backends': {'auto': gpu, 'cuda': cuda, 'webgpu': webgpu},
        'options': {'retained_repair':dict(cp),'sat_assist': dict(sat), 'gold_oracle': dict(cp), 'gold_pricing':dict(cp), 'smart_learning':dict(sat), 'shared_sat_assistant':dict(sat), 'shared_sat_service':dict(sat), 'semantic_witnesses':dict(sat), 'higher_order_relations':dict(sat), 'native_sat_persistence':sat_persistence,'gpu_shared_clauses':dict(sat),'native_explanations':_native_explanation_build(display=display)},
        'features': {'repair': dict(cp)},
    }


def validate_runtime_selection(algorithm, backend='auto', options=None, *,
                               stochastic=False, capabilities=None):
    """Reject explicitly selected missing engines before changing search files."""
    validate_algorithm_options(algorithm, options or {})
    caps = runtime_capabilities() if capabilities is None else capabilities
    selections = [('algorithms', algorithm)]
    if algorithm=='gpu-domains':
        if backend=='webgpu':raise ValueError('GPU domain search currently requires CUDA.')
        selections.append(('backends','cuda'))
    if (options or {}).get('coordinated_gpu'):
        if backend=='webgpu':raise ValueError('Coordinated exact GPU work requires CUDA.')
        selections.append(('backends','cuda'))
    if (options or {}).get('coordinated_sat'):selections.append(('algorithms','sat'))
    if stochastic:
        selections.append(('backends', 'cuda'))
    elif algorithm in ('gpu-dfs', 'hybrid'):
        selections.append(('backends', backend))
    if (algorithm=='gpu-domains' or (options or {}).get('coordinated_gpu')) and (options or {}).get('gpu_shared_clauses') and not (options or {}).get('gpu_proof_transport'):
        selections.append(('options','gpu_shared_clauses'))
    if (algorithm=='sat' or (options or {}).get('coordinated_sat')) and (options or {}).get('native_sat_persistence'):
        selections.append(('options','native_sat_persistence'))
    if algorithm in EXACT_ALGORITHMS or (options or {}).get('symbolic_frontier'):
        selections.extend(('options', key) for key, enabled in (options or {}).items()
                          if enabled and key in caps['options'] and not (key=='smart_learning' and (options or {}).get('proof_transport'))
                          and not (key=='gpu_shared_clauses' and (options or {}).get('gpu_proof_transport')))
    for category, name in selections:
        item = caps.get(category, {}).get(name)
        if item is None:
            raise ValueError('Unknown runtime selection: ' + str(name))
        if not item['available']:
            raise ValueError(name + ' is unavailable. ' + item['reason'])
