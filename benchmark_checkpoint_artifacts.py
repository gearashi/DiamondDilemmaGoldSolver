"""Bind checkpoint artifacts to their retained local dependencies."""
from __future__ import annotations
import hashlib,json
from pathlib import Path


def retained_store_snapshot(directory):
    """Hash the complete bounded store after the search and writer have stopped.

    This includes retained predecessors and orphaned interrupted writes, so it
    is deliberately a superset of the final generation's direct dependencies.
    It measures logical file contents, not allocated or physically written bytes.
    """
    from gpu_checkpoint_round25 import DEFAULT_STORE_BYTES,MAX_FILES
    directory=Path(directory)
    if not directory.exists():return None
    if directory.is_symlink() or getattr(directory,'is_junction',lambda:False)() or not directory.is_dir():raise ValueError('Checkpoint store must be a regular local directory.')
    root=directory.resolve();records={};total=0
    for path in sorted(root.rglob('*')):
        if path.is_symlink() or getattr(path,'is_junction',lambda:False)() or not path.resolve().is_relative_to(root):raise ValueError('Checkpoint dependency escapes its local store.')
        if path.is_dir():continue
        if not path.is_file():raise ValueError('Checkpoint dependency is not a regular file.')
        length=path.stat().st_size
        if len(records)>=MAX_FILES or total+length>DEFAULT_STORE_BYTES:raise ValueError('Checkpoint dependency store exceeds its declared bound.')
        hasher=hashlib.sha256();observed=0
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):
                observed+=len(block)
                if observed>length:raise ValueError('Checkpoint dependency changed while being recorded.')
                hasher.update(block)
        if observed!=length or path.stat().st_size!=length:raise ValueError('Checkpoint dependency changed while being recorded.')
        records[path.relative_to(root).as_posix()]={'bytes':length,'sha256':hasher.hexdigest()};total+=length
    return {'directory':str(root),'files':records,'logical_file_bytes':total,'file_count':len(records),
            'scope':'Complete retained store after writer drain; includes predecessors and orphaned objects. Logical contents, not physical SSD writes.'}


def reopen_gpu_checkpoint(artifact,directory,scope):
    """Recheck the saved envelope, immutable chunks and exact remaining scope.

    No native arrays are decoded and no GPU search is resumed here. Hardware
    continuation equivalence is tested separately by the checkpoint suite.
    """
    from gpu_checkpoint_round25 import resolve_checkpoint,checkpoint_frontier,is_reference
    from constraint_work import WorkScope,restore_frontier
    raw=Path(artifact['path']).read_bytes()
    if len(raw)!=artifact['bytes'] or hashlib.sha256(raw).hexdigest()!=artifact['sha256']:
        raise ValueError('Saved checkpoint artifact differs from its receipt.')
    payload=json.loads(raw)
    resolved=resolve_checkpoint(payload,directory)
    frontier=checkpoint_frontier(resolved,directory)
    if WorkScope.restore(frontier['scope'])!=scope:raise ValueError('Reopened checkpoint scope differs.')
    remaining=restore_frontier(frontier,scope)
    return {'verified':True,'referenced_store':is_reference(payload),'frontier_digest':frontier['digest'],
            'remaining_cubes':len(remaining),'native_arrays_decoded':False,'search_resumed':False,
            'scope':'Saved JSON/hash, referenced compressed chunks, and portable remaining scope checked; no native continuation or proof of exhaustion is inferred.'}
