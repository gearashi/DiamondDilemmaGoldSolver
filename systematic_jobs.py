"""Persistent disjoint jobs and immutable paused DFS stacks for resized GPU pools."""
from dataclasses import dataclass, field
import numpy as np
from dfs_gpu import DFS, ensure_disjoint_prefixes

BANK_VERSION = 1
FIELD_SHAPES = {'boards': (160,), 'used': (5,), 'depths': (), 'maxdepths': (),
                'states': (), 'floors': (), 'firsts': (160,), 'lengths': (160,),
                'cursors': (160,), 'shifts': (160,), 'rng': (), 'counters': (3,)}
FIELD_DTYPES = {'boards': np.int16, 'used': np.uint32, 'depths': np.int16, 'maxdepths': np.int16,
                'states': np.uint8, 'floors': np.int16, 'firsts': np.uint16, 'lengths': np.uint16,
                'cursors': np.uint16, 'shifts': np.uint16, 'rng': np.uint32, 'counters': np.uint64}


def _slice(payload, indices):
    """Select lanes; prefix metadata uses row order, device matrices use columns."""
    out = {name: (payload[name][:, indices] if FIELD_SHAPES[name] else payload[name][indices])
           for name in DFS.DEVICE_FIELDS}
    out['prefix_codes'] = payload['prefix_codes'][indices]
    return out


def _with_metadata(gpu, payload):
    return dict(payload, roots=payload['prefix_codes'][:, 0].copy(),
                dfs_version=np.int32(gpu.CHECKPOINT_VERSION),
                fingerprint=np.array(gpu.fingerprint), order=gpu.order.copy())


def _empty_arrays():
    return dict({name: np.empty((*shape, 0), FIELD_DTYPES[name]) for name, shape in FIELD_SHAPES.items()},
                prefix_codes=np.empty((0, 160), np.int16))


class _PausedBank:
    """Backing arrays never change; consumption advances only an integer offset.

    Snapshot views remain valid for background writers even after jobs are moved
    onto the GPU. Only a later resize replaces this bank with a new instance.
    """
    def __init__(self, ids=None, arrays=None):
        ids = np.empty(0, np.int64) if ids is None else ids
        arrays = _empty_arrays() if arrays is None else arrays
        if ids.ndim != 1 or ids.dtype != np.int64 or np.any(ids < 0) or len(np.unique(ids)) != len(ids):
            raise ValueError('Malformed or duplicate paused job IDs.')
        for name, shape in FIELD_SHAPES.items():
            value = arrays[name]
            if value.shape != (*shape, len(ids)) or value.dtype != FIELD_DTYPES[name]:
                raise ValueError('Malformed paused checkpoint field: ' + name)
        if arrays['prefix_codes'].shape != (len(ids), 160) or arrays['prefix_codes'].dtype != np.int16:
            raise ValueError('Malformed paused fixed prefixes.')
        if np.any(~np.isin(arrays['states'], [0, 1])):
            raise ValueError('Paused bank may contain only unfinished jobs.')
        self.ids, self.arrays, self.offset = ids, arrays, 0
        self.owned = set(map(int, ids))
        self.max_id = int(ids.max()) if len(ids) else -1
        for value in (ids, *arrays.values()):
            value.setflags(write=False)

    @property
    def count(self):
        return len(self.ids) - self.offset

    def remaining_ids(self):
        return self.ids[self.offset:]

    def remaining_arrays(self):
        return _slice(self.arrays, slice(self.offset, None))

    def take(self, count):
        end = self.offset + count
        if not 0 <= count <= self.count:
            raise ValueError('Paused bank take exceeds available jobs.')
        return self.ids[self.offset:end], _slice(self.arrays, slice(self.offset, end))

    def consume(self, count):
        ids, _ = self.take(count)
        self.owned.difference_update(map(int, ids))
        self.offset += count


@dataclass
class JobLedger:
    total: int
    ids: np.ndarray
    cursor: int = 0
    completed: int = 0
    paused: _PausedBank = field(default_factory=_PausedBank, repr=False)
    retired_counters: np.ndarray = field(default_factory=lambda: np.zeros(3, np.uint64))

    @property
    def paused_count(self):
        return self.paused.count

    @classmethod
    def empty(cls, total, replicas):
        return cls(int(total), np.full(replicas, -1, np.int64))

    def validate(self, states):
        states = np.asarray(states)
        if self.ids.dtype != np.int64 or self.ids.shape != states.shape:
            raise ValueError('Job ledger shape or dtype differs from GPU lanes.')
        if (not 0 <= self.completed <= self.cursor <= self.total or np.any(self.ids < -1)
                or np.any(self.ids >= self.cursor) or self.paused.max_id >= self.cursor):
            raise ValueError('Job ledger cursor or ownership is invalid.')
        owned = self.ids[self.ids >= 0]
        if len(np.unique(owned)) != len(owned) or any(int(job) in self.paused.owned for job in owned):
            raise ValueError('Two active or paused lanes own the same job.')
        if self.completed != self.cursor - len(owned) - self.paused_count:
            raise ValueError('Completed-job count does not match active/paused ownership.')
        if not np.array_equal(self.ids >= 0, np.isin(states, [0, 1])):
            raise ValueError('GPU lane state differs from persistent job ownership.')
        if self.retired_counters.shape != (3,) or self.retired_counters.dtype != np.uint64:
            raise ValueError('Malformed retired counter totals.')
        return True

    def retire(self, states):
        states = np.asarray(states)
        done = np.flatnonzero((self.ids >= 0) & (states == 2))
        self.ids[done] = -1
        self.completed += len(done)
        self.validate(states)
        return done

    def refill(self, gpu, prefixes, lengths):
        states = gpu.states.get()
        self.retire(states)
        free = np.flatnonzero(self.ids < 0)
        resumed = min(len(free), self.paused_count)
        if resumed:
            lanes = free[:resumed]
            jobs, payload = self.paused.take(resumed)
            old_counts = gpu.counters[:, lanes].get().sum(axis=1, dtype=np.uint64)
            gpu.restore_lanes(_with_metadata(gpu, payload), lanes)
            self.retired_counters += old_counts
            self.ids[lanes] = jobs
            self.paused.consume(resumed)
            free = free[resumed:]
        count = min(len(free), self.total - self.cursor)
        if count:
            lanes = free[:count]
            jobs = np.arange(self.cursor, self.cursor + count, dtype=np.int64)
            gpu.load_prefixes([prefixes[j, :int(lengths[j])].tolist() for j in jobs], lanes=lanes)
            self.ids[lanes] = jobs
            self.cursor += count
        self.validate(gpu.states.get())
        return resumed + count

    @staticmethod
    def _check_prefixes(ids, rows, floors, prefixes, lengths):
        lanes = np.flatnonzero(ids >= 0)
        for start in range(0, len(lanes), 2048):
            selected = lanes[start:start + 2048]
            jobs = ids[selected]
            if (not np.array_equal(floors[selected], lengths[jobs])
                    or not np.array_equal(rows[selected], prefixes[jobs])):
                raise ValueError('GPU or paused fixed prefix does not match its persistent job ID.')

    def validate_prefixes(self, prefix_codes, floors, prefixes, lengths):
        """Bind all active and paused work to the immutable disjoint frontier."""
        rows, floors = np.asarray(prefix_codes), np.asarray(floors)
        if rows.shape != (len(self.ids), prefixes.shape[1]) or floors.shape != self.ids.shape:
            raise ValueError('GPU fixed-prefix dimensions differ from the job ledger.')
        self._check_prefixes(self.ids, rows, floors, prefixes, lengths)
        if self.paused_count:
            bank = self.paused.remaining_arrays()
            self._check_prefixes(self.paused.remaining_ids(), bank['prefix_codes'], bank['floors'], prefixes, lengths)
        return True

    def counter_totals(self, gpu):
        paused_counts = self.paused.arrays['counters'][:, self.paused.offset:].sum(axis=1, dtype=np.uint64)
        return gpu.counters.get().sum(axis=1, dtype=np.uint64) + paused_counts + self.retired_counters

    def fields(self):
        result = {'exact_job_ids': self.ids.copy(), 'exact_cursor': np.array(self.cursor, np.int64),
                  'exact_completed': np.array(self.completed, np.int64), 'exact_total': np.array(self.total, np.int64),
                  'exact_bank_version': np.array(BANK_VERSION, np.int32),
                  'exact_paused_job_ids': self.paused.remaining_ids(),
                  'exact_retired_counters': self.retired_counters.copy()}
        result.update({'exact_paused_' + name: value for name, value in self.paused.remaining_arrays().items()})
        for value in result.values():
            value.setflags(write=False)
        return result

    @classmethod
    def restore(cls, archive, states):
        has_bank = 'exact_bank_version' in archive
        if not has_bank and any(key.startswith('exact_paused_') or key == 'exact_retired_counters' for key in archive):
            raise ValueError('Paused lane metadata is missing its version; refusing to discard saved work.')
        bank, retired = _PausedBank(), np.zeros(3, np.uint64)
        if has_bank:
            if int(archive['exact_bank_version']) != BANK_VERSION:
                raise ValueError('Unsupported systematic lane-bank checkpoint version.')
            try:
                bank = _PausedBank(archive['exact_paused_job_ids'],
                                   {name: archive['exact_paused_' + name] for name in (*DFS.DEVICE_FIELDS, 'prefix_codes')})
                retired = archive['exact_retired_counters'].copy()
            except KeyError as exc:
                raise ValueError('Incomplete systematic lane-bank checkpoint.') from exc
        ledger = cls(int(archive['exact_total']), archive['exact_job_ids'].copy(),
                     int(archive['exact_cursor']), int(archive['exact_completed']), bank, retired)
        ledger.validate(states)
        return ledger

    @classmethod
    def resume(cls, gpu, archive, prefixes, lengths):
        """Rebalance saved exact states to gpu.n lanes, retaining every other job on CPU."""
        saved = gpu.read_checkpoint(archive)  # Validate the complete source before selecting lanes.
        previous = cls.restore(archive, saved['states'])
        if previous.total != len(prefixes):
            raise ValueError('Job ledger frontier size changed.')
        previous.validate_prefixes(saved['prefix_codes'], saved['floors'], prefixes, lengths)
        bank = previous.paused.remaining_arrays()
        if previous.paused_count:
            gpu.read_checkpoint(_with_metadata(gpu, bank))
        # Check overlap across GPU and paused stacks, including after an untrusted file edit.
        living = np.flatnonzero(previous.ids >= 0)
        all_prefixes = [tuple(map(int, saved['prefix_codes'][i, :saved['floors'][i]])) for i in living]
        all_prefixes += [tuple(map(int, row[:floor])) for row, floor in zip(bank['prefix_codes'], bank['floors'])]
        ensure_disjoint_prefixes(all_prefixes)
        sources = [(saved, previous.ids), (bank, previous.paused.remaining_ids())]
        selection = []
        for state in (1, 0):  # Pending certificates must be surfaced before doing more search.
            for source_index, (payload, ids) in enumerate(sources):
                selection.extend((source_index, int(i)) for i in np.flatnonzero((ids >= 0) & (payload['states'] == state)))
        active_selection, paused_selection = selection[:gpu.n], selection[gpu.n:]
        target = gpu.snapshot_checkpoint()
        if np.any(target['states'] != 3) or np.any(target['counters']):
            raise ValueError('Resize restore requires a fresh, idle GPU pool.')
        # The snapshot is read-only; allocate owned arrays for the selected active pool.
        target = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in target.items()}
        target_ids = np.full(gpu.n, -1, np.int64)
        for source_index, (payload, ids) in enumerate(sources):
            destinations = np.array([j for j, (src, _) in enumerate(active_selection) if src == source_index], np.int64)
            originals = np.array([i for src, i in active_selection if src == source_index], np.int64)
            if not len(destinations):
                continue
            for name in DFS.DEVICE_FIELDS:
                if FIELD_SHAPES[name]:
                    target[name][:, destinations] = payload[name][:, originals]
                else:
                    target[name][destinations] = payload[name][originals]
            target['prefix_codes'][destinations] = payload['prefix_codes'][originals]
            target_ids[destinations] = ids[originals]
        target['roots'] = target['prefix_codes'][:, 0].copy()
        paused_arrays = {name: np.empty((*shape, len(paused_selection)), FIELD_DTYPES[name]) for name, shape in FIELD_SHAPES.items()}
        paused_arrays['prefix_codes'] = np.empty((len(paused_selection), 160), np.int16)
        paused_ids = np.empty(len(paused_selection), np.int64)
        for source_index, (payload, ids) in enumerate(sources):
            destinations = np.array([j for j, (src, _) in enumerate(paused_selection) if src == source_index], np.int64)
            originals = np.array([i for src, i in paused_selection if src == source_index], np.int64)
            for name in DFS.DEVICE_FIELDS:
                if FIELD_SHAPES[name]:
                    paused_arrays[name][:, destinations] = payload[name][:, originals]
                else:
                    paused_arrays[name][destinations] = payload[name][originals]
            paused_arrays['prefix_codes'][destinations] = payload['prefix_codes'][originals]
            paused_ids[destinations] = ids[originals]
        retired = previous.retired_counters + saved['counters'][:, previous.ids < 0].sum(axis=1, dtype=np.uint64)
        ledger = cls(previous.total, target_ids, previous.cursor, previous.completed,
                     _PausedBank(paused_ids, paused_arrays), retired)
        ledger.validate(target['states'])
        ledger.validate_prefixes(target['prefix_codes'], target['floors'], prefixes, lengths)
        gpu.restore_checkpoint(target)
        return ledger
