"""Bounded asynchronous checkpoint publication for comparable benchmark I/O.

Submitted payloads are immutable producer-owned values: neither the caller nor
any later solver slice may mutate them until this context has closed. The writer
does not deep-copy, deduplicate, or coalesce large checkpoints. At most capacity
queued references plus one in-flight reference are retained. Backpressure is part
of the observed solver cost, and context exit includes the final drain.

This app-like writer does not model dashboard/UI consumer latency.
This measures atomic JSON replacements, not engine chunk-store I/O, fsync-backed
durability, or allocated SSD bytes. A permanently blocked filesystem operation
cannot be killed safely inside a Python thread; close waits for its return rather
than leaving an orphan writer. The benchmark's outer process watchdog owns that
exceptional case.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import threading
import time
from io_utils import atomic_json


class CheckpointPublicationError(RuntimeError):
    """A writer failed; no successful benchmark report may ignore this error."""


class CheckpointPublisher:
    def __init__(self, path, *, queue_capacity=2):
        if type(queue_capacity) is not int or not 1 <= queue_capacity <= 16:
            raise ValueError('Checkpoint queue capacity must be an integer from1 to16.')
        self.path=Path(path)
        if self.path.suffix.lower()!='.json':
            raise ValueError('Checkpoint publication requires a fresh .json path.')
        self.queue_capacity=queue_capacity
        self._lock_path=self.path.with_name(self.path.name+'.writer.lock')
        self._condition=threading.Condition()
        self._pending=deque()
        self._thread=None
        self._entered=False
        self._closing=False
        self._closed=False
        self._owns_lock=False
        self._error=None
        self._cleanup_error=None
        self._inflight=False
        self._counts=dict(submitted_count=0,published_count=0,published_bytes=0,
                          discarded_count=0,failed_write_count=0,max_queue_depth=0,
                          final_file_bytes=0,write_seconds=0.0,producer_wait_seconds=0.0)

    def __enter__(self):
        with self._condition:
            if self._entered or self._closed:
                raise RuntimeError('Checkpoint publisher contexts cannot be reused.')
            if not self.path.parent.is_dir():
                raise ValueError('Checkpoint publication needs an existing fresh case directory.')
            if self.path.exists() or self.path.is_symlink():
                raise FileExistsError('Checkpoint publication target already exists: '+str(self.path))
            # An exclusive sidecar prevents two publishers claiming the same
            # still-absent target. Never remove a lock this instance did not own.
            with self._lock_path.open('xb'):
                self._owns_lock=True
            try:
                if self.path.exists() or self.path.is_symlink():
                    raise FileExistsError('Checkpoint publication target is no longer fresh.')
                self._entered=True
                self._thread=threading.Thread(target=self._write_loop,
                    name='benchmark-checkpoint-publisher',daemon=False)
                self._thread.start()
            except BaseException:
                self._closed=True
                self._entered=False
                self._release_lock()
                raise
        return self

    def _release_lock(self):
        if self._owns_lock:
            self._lock_path.unlink(missing_ok=True)
            self._owns_lock=False

    def _raise_error(self):
        failures=[]
        if self._error is not None:failures.append('write: '+str(self._error))
        if self._cleanup_error is not None:failures.append('cleanup: '+str(self._cleanup_error))
        if failures:
            cause=self._error if self._error is not None else self._cleanup_error
            raise CheckpointPublicationError('Asynchronous checkpoint publication failed: '+
                                             '; '.join(failures)) from cause

    def failed(self):
        """Thread-safe solver Stop predicate; a failed writer invalidates a run."""
        with self._condition:
            return self._error is not None or self._cleanup_error is not None

    def publish(self, payload):
        """Queue an immutable snapshot; block under bounded FIFO backpressure."""
        with self._condition:
            self._raise_error()
            if not self._entered or self._closing or self._closed:
                raise RuntimeError('Checkpoint publisher is not accepting snapshots.')
            if payload is None:
                return False
            if not isinstance(payload,dict):
                raise TypeError('Checkpoint payload must be an immutable-by-contract dictionary.')
            waiting=None
            try:
                while len(self._pending)>=self.queue_capacity:
                    if waiting is None:waiting=time.perf_counter()
                    self._condition.wait()
                    self._raise_error()
                    if self._closing or self._closed:
                        raise RuntimeError('Checkpoint publisher closed while waiting for queue space.')
                self._pending.append(payload)
                self._counts['submitted_count']+=1
                self._counts['max_queue_depth']=max(self._counts['max_queue_depth'],len(self._pending))
                self._condition.notify_all()
                return True
            finally:
                if waiting is not None:
                    self._counts['producer_wait_seconds']+=time.perf_counter()-waiting

    def progress(self, event):
        """Accept checkpoint progress events; ordinary progress has no disk work."""
        if not isinstance(event,dict):raise TypeError('Progress event must be a dictionary.')
        with self._condition:self._raise_error()
        payload=event.get('payload') if event.get('event')=='checkpoint' else None
        if payload is None:payload=event.get('checkpoint')
        if payload is None:return False
        return self.publish(payload)

    def _write_loop(self):
        while True:
            with self._condition:
                while not self._pending and not self._closing:
                    self._condition.wait()
                if not self._pending:
                    return
                payload=self._pending.popleft()
                self._inflight=True
                self._condition.notify_all()
            began=time.perf_counter()
            try:
                atomic_json(self.path,payload)
                size=self.path.stat().st_size
            except BaseException as error:
                with self._condition:
                    self._counts['write_seconds']+=time.perf_counter()-began
                    self._counts['failed_write_count']+=1
                    self._counts['discarded_count']+=len(self._pending)
                    self._pending.clear()
                    self._error=error
                    self._inflight=False
                    self._closing=True
                    self._condition.notify_all()
                return
            else:
                with self._condition:
                    self._counts['write_seconds']+=time.perf_counter()-began
                    self._counts['published_count']+=1
                    self._counts['published_bytes']+=size
                    self._counts['final_file_bytes']=size
                    self._inflight=False
                    self._condition.notify_all()
            finally:
                payload=None  # Do not keep the last large snapshot while idle.

    def close(self):
        """Stop admission, drain every accepted snapshot, and join the writer."""
        with self._condition:
            if not self._entered:
                if not self._closed:raise RuntimeError('Checkpoint publisher was not entered.')
                return
            self._closing=True
            self._condition.notify_all()
            thread=self._thread
        if thread is threading.current_thread():
            raise RuntimeError('The checkpoint writer cannot join itself.')
        # No timeout: returning while this owned writer could still replace a
        # file would violate the benchmark's case isolation contract.
        thread.join()
        with self._condition:
            self._closed=True
            try:self._release_lock()
            except Exception as cleanup_error:self._cleanup_error=cleanup_error
            self._raise_error()

    def __exit__(self, exc_type, error, traceback):
        try:
            self.close()
        except Exception as writer_error:
            if error is None:raise
            # A solver/body failure stays primary even if sidecar cleanup or
            # another ordinary close operation also fails after the join.
            error.add_note('Checkpoint publisher close also failed: '+str(writer_error))
        return False

    def report(self):
        with self._condition:
            return {**self._counts,'mode':'bounded_asynchronous_atomic_json',
                    'path':str(self.path.resolve()),'queue_capacity':self.queue_capacity,
                    'queued':len(self._pending),'inflight':self._inflight,
                    'closed':self._closed,'failed':self._error is not None or self._cleanup_error is not None,
                    'write_error':None if self._error is None else str(self._error),
                    'cleanup_error':None if self._cleanup_error is None else str(self._cleanup_error),
                    'writer_alive':bool(self._thread and self._thread.is_alive()),
                    'scope':'Asynchronous JSON publication only; excludes engine chunk/manifest/head writes, lock metadata and UI consumer latency. Published bytes are cumulative logical serialized JSON bytes (successful replacement file sizes), not physical SSD writes or allocated bytes. Write seconds include failed attempts; max queue excludes the in-flight write.',
                    'producer_contract':'Payloads remain immutable until context exit; no deepcopy, deduplication or coalescing.'}
