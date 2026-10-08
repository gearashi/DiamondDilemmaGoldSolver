"""Select a native GPU engine without importing either driver during UI startup."""
from functools import lru_cache
import importlib.util
import os
import shutil
import subprocess
import sys

BACKENDS = ('auto', 'cuda', 'webgpu')

@lru_cache(maxsize=1)
def _automatic_backend():
    if sys.platform == 'darwin':
        return 'webgpu'
    if importlib.util.find_spec('cupy') is not None:
        executable = shutil.which('nvidia-smi')
        if executable:
            try:
                result = subprocess.run(
                    [executable, '--query-gpu=name', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                if result.returncode == 0 and result.stdout.strip():
                    return 'cuda'
            except (OSError, subprocess.TimeoutExpired):
                pass
    return 'webgpu'

def choose_backend(name='auto'):
    if name not in BACKENDS:
        raise ValueError('Choose GPU backend auto, cuda, or webgpu.')
    if name == 'cuda' and sys.platform == 'darwin':
        raise ValueError('CUDA is unavailable on macOS. Choose auto or webgpu for Metal.')
    return _automatic_backend() if name == 'auto' else name
