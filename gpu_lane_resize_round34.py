"""Partition validated native DFS columns on the host without restoring old-width GPU allocations."""
from copy import copy, deepcopy

import numpy as np

from constraint_work import disjoint


def lane_cubes(archives, max_bytes):
    """Bound untrusted dimensions before inflating the first compressed state."""
    cubes=[]
    total_bytes=0
    for native in archives:
        if not isinstance(native,dict):
            raise ValueError("Malformed native lane archive")
        shape=native.get("shape")
        if (not isinstance(shape,list) or len(shape)!=2 or any(type(v) is not int or v<1 for v in shape)
                or shape[0]!=native.get("rows") or not isinstance(native.get("cubes"),list)
                or len(native["cubes"])!=shape[1]):
            raise ValueError("Malformed saved native lane dimensions")
        total_bytes+=4*shape[0]*shape[1]
        if len(cubes)+shape[1]>512 or total_bytes>max_bytes:
            raise ValueError("Saved native lane backlog exceeds the supported size")
        cubes.extend(tuple(int(v,16) for v in row) for row in native["cubes"])
    if not disjoint(cubes):
        raise ValueError("Saved native lane roots overlap")
    return cubes


class ParkedStateCache:
    """Retain a bounded prefix of already decoded, private native state until dispatch."""
    def __init__(self, limit):
        self.limit = limit
        self.bytes = 0
        self.entries = {}

    def remember(self, native, state):
        if id(native) in self.entries or state.nbytes > self.limit - self.bytes or not state.flags.owndata:
            return
        state.setflags(write=False)
        self.entries[id(native)] = (native, native["compressed_state"], state)
        self.bytes += state.nbytes

    def take(self, native, engine, unpack):
        entry = self.entries.pop(id(native), None)
        if entry is not None:
            original, encoded, state = entry
            self.bytes -= state.nbytes
            # Progress callbacks may have seen the archive. A modified frame
            # must still go through its normal decoder. A cache hit is followed
            # by the same binding, hash and mechanical checks as a cold restore.
            if (native is original and native.get("compressed_state") is encoded
                    and "factored_native_version" not in native
                    and native.get("shape") == list(state.shape) == [engine.rows,engine.lanes]):
                snapshot = dict(native)
                del snapshot["shape"], snapshot["compressed_state"]
                snapshot["state"] = state
                return snapshot
        return unpack(native, engine)


def restore_slices(engine, archives, replicas, owner_ids, counter_names, pack, unpack, *, remember=None):
    """Fill the active group and retain validated sleeping archives in lane order."""
    pending=[]
    group=[]
    group_roots=[]
    records=[]
    first=True
    packed_fields = {"schema", "binding", "cubes", "state_sha256", "delivered",
                     "rows", "shape", "compressed_state"}

    def park(view, state, native):
        counts = view.stats(snapshot=state)
        if remember is not None:
            remember(native,state)
        pending.append({"native": native, "boxes": view.remaining_domain_cubes(snapshot=state),
                        "counts": {key: counts.get(key, 0) for key in (*counter_names, "exhausted")}})

    def emit():
        nonlocal first
        view=copy(engine)
        view.lanes=len(group_roots)
        view.cubes=tuple(group_roots)
        state=np.ascontiguousarray(group[0]) if len(group)==1 else np.concatenate(group,axis=1)
        snapshot=view._checkpoint_payload(state)
        if first:
            engine.restore(snapshot)
            first=False
        else:
            park(view, state, pack(view,snapshot))
        group.clear()
        group_roots.clear()
    for native in archives:
        view=copy(engine)
        view.cubes=tuple(view._validate_cube(tuple(int(x,16) for x in row)) for row in native["cubes"])
        view.lanes=len(view.cubes)
        snapshot=unpack(native,view)
        state,delivered=view.validate_checkpoint(snapshot)
        if delivered:
            raise ValueError("Enumeration checkpoints cannot be used as infeasibility searches")
        for lane,cube in enumerate(view.cubes):
            idle=False
            if owner_ids is not None and owner_ids[len(records)] is None:
                counts=view.stats([lane],snapshot=state)
                idle=state[0,lane]==2 and not any(counts.get(key,0) for key in counter_names)
            records.append((cube,idle))
        # Only self-contained legacy archives can be carried byte-for-byte into
        # the native queue. External chunk references belong to their generation.
        if not first and not group and native.keys() == packed_fields:
            # Counters, frontier and ownership still come from the verified state.
            park(view, state, deepcopy(native))
            continue
        begin=0
        while begin<view.lanes:
            end=min(view.lanes,begin+replicas-len(group_roots))
            group_roots.extend(view.cubes[begin:end])
            group.append(state[:,begin:end].copy())
            if len(group_roots)==replicas:
                emit()
            begin=end
        if not first and group_roots:
            # Keep this leftover in source order. Subsequent intact archives can
            # wait in their existing encoding until a device batch needs them.
            emit()
    if group_roots:
        emit()
    return pending,records
