"""Compile and execute a CUDA kernel with the installed wheels. Exit 2 means driver/device.

Setup records the result in its own venv. The dashboard can read it without
keeping a CUDA context on the display GPU. An untested install gets a fresh probe.
"""
import argparse
import ctypes
from functools import lru_cache
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time


class DriverUnavailable(Exception):
    pass


def check_driver():
    try:
        driver = ctypes.WinDLL('nvcuda.dll') if sys.platform == 'win32' else ctypes.CDLL('libcuda.so.1')
    except OSError as exc:
        raise DriverUnavailable('Missing NVIDIA display driver.') from exc
    status = driver.cuInit(0)
    if status == 100:
        raise DriverUnavailable('The NVIDIA display driver found no CUDA device.')
    if status in (35, 803):
        raise DriverUnavailable(f'The NVIDIA display driver needs updating (CUDA error {status}).')
    if status:
        raise RuntimeError(f'The NVIDIA driver could not initialize CUDA (error {status}).')


def check_kernel():
    # Use a fresh cache so a previous binary cannot conceal a missing NVRTC wheel.
    with tempfile.TemporaryDirectory(prefix='diamond-cuda-check-') as cache:
        previous_cache = os.environ.get('CUPY_CACHE_DIR')
        os.environ['CUPY_CACHE_DIR'] = cache
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() == 0:
                raise DriverUnavailable('The NVIDIA display driver found no CUDA device.')
            kernel = cp.RawKernel('extern "C" __global__ void check(int *p) { p[0] = 42; }', 'check')
            value = cp.zeros(1, dtype=cp.int32)
            kernel((1,), (1,), (value,))
            actual = int(value.get()[0])
            if actual != 42:
                raise RuntimeError(f'CUDA kernel returned {actual}, expected 42.')
            return cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode()
        finally:
            if previous_cache is None:
                os.environ.pop('CUPY_CACHE_DIR', None)
            else:
                os.environ['CUPY_CACHE_DIR'] = previous_cache


def test_cuda():
    try:
        check_driver()
        device = check_kernel()
        return 0, f'NVRTC compiled and CUDA returned 42 on {device}.'
    except DriverUnavailable as exc:
        return 2, str(exc)
    except Exception as exc:
        if getattr(exc, 'status', None) in (35, 803):
            return 2, f'The NVIDIA display driver needs updating for the installed CUDA runtime: {exc}'
        return 1, f'CUDA kernel check failed ({type(exc).__name__}): {exc}'


def record_path():
    return Path(sys.prefix) / 'diamond-cuda-check.json'


def installation_stamp():
    # Detect driver replacement and wheel installation/removal between setups.
    packages = Path(sys.prefix) / 'Lib' / 'site-packages'
    driver = Path(os.environ.get('SystemRoot', 'C:/Windows')) / 'System32' / 'nvcuda.dll'
    paths = [driver, packages / 'nvidia/cuda_nvrtc/bin', packages / 'nvidia/cuda_runtime/bin', packages / 'cupy']
    stamp = []
    for path in paths:
        try:
            stamp.append(path.stat().st_mtime_ns)
        except OSError:
            stamp.append(None)
    return tuple(stamp)


def save_result(code, reason):
    if sys.prefix == sys.base_prefix:
        raise RuntimeError('Only an isolated solver venv can record a setup check.')
    path = record_path()
    payload = dict(executable=sys.executable, installation=installation_stamp(), checked=time.time(), code=code, reason=reason)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as temp:
        json.dump(payload, temp)
    os.replace(temp.name, path)


@lru_cache(maxsize=1)
def _fresh_result(stamp):
    try:
        result = subprocess.run([sys.executable, '-B', __file__], capture_output=True, text=True,
                                errors='replace', timeout=60,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        return result.returncode, result.stdout.strip() or result.stderr.strip() or 'CUDA check failed.'
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, f'Could not finish the CUDA check: {exc}'


def installed_result(*, probe_on_miss=True):
    stamp = installation_stamp()
    try:
        data = json.loads(record_path().read_text(encoding='utf-8'))
        if (data['executable'] == sys.executable and data['installation'] == list(stamp) and type(data['code']) is int
                and data['code'] in (0, 1, 2) and isinstance(data['reason'], str)):
            return data['code'], data['reason']
    except (OSError, ValueError, TypeError, KeyError):
        pass
    if not probe_on_miss:
        return 1, 'Setup has not verified CUDA for the current driver and wheels.'
    return _fresh_result(stamp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    code, reason = test_cuda()
    if args.record:
        save_result(code, reason)
    print(reason)
    sys.exit(code)
