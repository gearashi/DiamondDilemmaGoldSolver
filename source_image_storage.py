"""Capture source bytes as independently writable files in exact-run evidence."""
from contextlib import contextmanager
from functools import cache
import errno
import os
from pathlib import Path
import stat
import sys
from tempfile import TemporaryDirectory

from io_utils import replace_with_retry


def _linux_clone(source, stage):
    import fcntl
    with source.open('rb') as original, stage.open('xb') as copy:
        fcntl.ioctl(copy.fileno(), fcntl.FICLONE, original.fileno())


@cache
def _darwin_api():
    import ctypes
    function = ctypes.CDLL(None, use_errno=True).clonefile
    function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    function.restype = ctypes.c_int
    return function


def _darwin_clone(source, stage):
    import ctypes
    if _darwin_api()(os.fsencode(source), os.fsencode(stage), 0):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def _native_clone():
    if sys.platform == 'linux':
        import fcntl
        if hasattr(fcntl, 'FICLONE'):
            return _linux_clone
    if sys.platform == 'darwin':
        try:
            _darwin_api()
        except (AttributeError, OSError):
            return None
        return _darwin_clone
    return None


@contextmanager
def source_image_writer(destination):
    """Capture raw images and remember unavailable cloning for the current run."""
    clone = _native_clone()
    with TemporaryDirectory(prefix='.source-images-', dir=destination) as staging:
        stage = Path(staging) / 'image'

        def write(source, expected):
            nonlocal clone
            target = destination / source.name
            if clone is None:
                # Fresh per-run paths need just the original data write. The
                # caller only publishes a receipt after ALL images succeeded.
                # Preserve an existing image via the staging path below.
                try:
                    stream = target.open('xb')
                except FileExistsError:
                    pass
                else:
                    with stream:
                        stream.write(expected)
                    return False
            copied = False
            if clone is not None and not source.is_symlink():
                try:
                    clone(source, stage)
                    # Source may have changed since the parser captured these
                    # bytes. Compare the private clone before publishing it.
                    info = stage.lstat()
                    copied = (stat.S_ISREG(info.st_mode) and info.st_nlink == 1
                              and stage.read_bytes() == expected)
                    if copied and not info.st_mode & stat.S_IWUSR:
                        stage.chmod(stat.S_IMODE(info.st_mode) | stat.S_IWUSR)
                except OSError as exc:
                    if exc.errno in (errno.EXDEV, errno.ENOTSUP, errno.ENOTTY, errno.ENOSYS, errno.EINVAL, errno.EBADF):
                        clone = None
            if not copied:
                stage.unlink(missing_ok=True)
                stage.write_bytes(expected)
            replace_with_retry(stage, target)
            return copied

        yield write
