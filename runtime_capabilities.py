"""Installed-package availability, without importing solvers or probing GPU drivers.

An available package is not a guarantee of a working device or native library.
The chosen engine still validates those when its search starts.
"""
from __future__ import annotations
import importlib.util
from importlib.metadata import version as package_version, PackageNotFoundError
import platform
import sys

from search_options import EXACT_ALGORITHMS, validate_algorithm_options


def _installed(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def runtime_capabilities():
    """Describe this interpreter, never another installed Python environment."""
    def feature(available, requirement):
        return {'available': bool(available), 'reason': '' if available else requirement}

    cp = feature(_installed('ortools'), 'Requires OR-Tools in this Python environment.')
    sat = feature(_installed('pysat'), 'Requires python-sat in this Python environment.')
    try:sat_version=package_version('python-sat')
    except PackageNotFoundError:sat_version=None
    proof_arch=platform.machine().lower() in ({'x86_64','amd64'} if sys.platform=='linux' else {'x86_64','amd64','arm64','aarch64'})
    sat_persistence=feature(sat['available'] and sys.platform in ('linux','darwin') and proof_arch
                            and platform.python_implementation()=='CPython' and sys.version_info[:2]==(3,12) and sat_version=='1.9.dev15',
                            'Verified native SAT clause persistence requires tested Linux x86-64 or macOS x86-64/ARM64, CPython 3.12 and python-sat 1.9.dev15; the tested Windows build is unavailable.')
    cuda = feature(sys.platform != 'darwin' and _installed('cupy'),
                   'CUDA is unavailable on macOS.' if sys.platform == 'darwin' else
                   'Requires CuPy in this Python environment. Choose WebGPU if installed.')
    webgpu = feature(_installed('wgpu'), 'Requires wgpu in this Python environment.')
    gpu = feature(cuda['available'] or webgpu['available'],
                  'Requires a GPU package in this Python environment: CuPy or wgpu.')
    return {
        'python': platform.python_version(), 'detection': 'installed_packages',
        'algorithms': {'dfs': feature(True, ''), 'parallel-dfs': feature(True, ''),
                       'gpu-dfs': dict(gpu), 'hybrid': dict(gpu), 'gpu-domains':dict(cuda),
                       'cp': dict(cp), 'cp-sat': dict(cp), 'cp-sat-circuit': dict(cp),
                       'sat': dict(sat), 'sat-propagator':dict(sat),
                       'sat-channel':dict(sat),'cp-sat-regions':dict(cp)},
        'backends': {'auto': gpu, 'cuda': cuda, 'webgpu': webgpu},
        'options': {'sat_assist': dict(sat), 'gold_oracle': dict(cp), 'smart_learning':dict(sat), 'shared_sat_assistant':dict(sat), 'higher_order_relations':dict(sat), 'native_sat_persistence':sat_persistence},
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
    if stochastic:
        selections.append(('backends', 'cuda'))
    elif algorithm in ('gpu-dfs', 'hybrid'):
        selections.append(('backends', backend))
    if algorithm=='sat' and (options or {}).get('native_sat_persistence'):
        selections.append(('options','native_sat_persistence'))
    if algorithm in EXACT_ALGORITHMS:
        selections.extend(('options', key) for key, enabled in (options or {}).items()
                          if enabled and key in caps['options'] and not (key=='smart_learning' and (options or {}).get('proof_transport')))
    for category, name in selections:
        item = caps.get(category, {}).get(name)
        if item is None:
            raise ValueError('Unknown runtime selection: ' + str(name))
        if not item['available']:
            raise ValueError(name + ' is unavailable. ' + item['reason'])
