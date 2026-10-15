"""Installed-package availability, without importing solvers or probing GPU drivers.

An available package is not a guarantee of a working device or native library.
The chosen engine still validates those when its search starts.
"""
from __future__ import annotations
import importlib.util
import platform
import sys

from search_options import EXACT_ALGORITHMS


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
    cuda = feature(sys.platform != 'darwin' and _installed('cupy'),
                   'CUDA is unavailable on macOS.' if sys.platform == 'darwin' else
                   'Requires CuPy in this Python environment. Choose WebGPU if installed.')
    webgpu = feature(_installed('wgpu'), 'Requires wgpu in this Python environment.')
    gpu = feature(cuda['available'] or webgpu['available'],
                  'Requires a GPU package in this Python environment: CuPy or wgpu.')
    return {
        'python': platform.python_version(), 'detection': 'installed_packages',
        'algorithms': {'dfs': feature(True, ''), 'parallel-dfs': feature(True, ''),
                       'gpu-dfs': dict(gpu), 'hybrid': dict(gpu),
                       'cp': dict(cp), 'cp-sat': dict(cp), 'cp-sat-circuit': dict(cp),
                       'sat': dict(sat)},
        'backends': {'auto': gpu, 'cuda': cuda, 'webgpu': webgpu},
        'options': {'sat_assist': dict(sat), 'gold_oracle': dict(cp)},
        'features': {'repair': dict(cp)},
    }


def validate_runtime_selection(algorithm, backend='auto', options=None, *,
                               stochastic=False, capabilities=None):
    """Reject explicitly selected missing engines before changing search files."""
    caps = runtime_capabilities() if capabilities is None else capabilities
    selections = [('algorithms', algorithm)]
    if stochastic:
        selections.append(('backends', 'cuda'))
    elif algorithm in ('gpu-dfs', 'hybrid'):
        selections.append(('backends', backend))
    if algorithm in EXACT_ALGORITHMS:
        selections.extend(('options', key) for key, enabled in (options or {}).items()
                          if enabled and key in caps['options'])
    for category, name in selections:
        item = caps.get(category, {}).get(name)
        if item is None:
            raise ValueError('Unknown runtime selection: ' + str(name))
        if not item['available']:
            raise ValueError(name + ' is unavailable. ' + item['reason'])
