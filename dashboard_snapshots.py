"""Reuse unchanged files for read-only dashboard polling."""
from threading import Lock

_lock = Lock()
_snapshots = {}


def file_version(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def read_display_file(path, reader):
    # The solver publishes snapshots by atomic replacement. A same-size newer
    # file with a preserved modification time still has its own file identity.
    key = (path, reader)
    with _lock:
        version = file_version(path)
        saved = _snapshots.get(key)
        if saved is not None and saved[0] == version:
            return saved[1]
        value = reader(path)
        try:
            stable = file_version(path) == version
        except OSError:
            stable = False
        if stable:
            if len(_snapshots) >= 16 and key not in _snapshots:
                _snapshots.pop(next(iter(_snapshots)))
            _snapshots[key] = (version, value)
        return value
