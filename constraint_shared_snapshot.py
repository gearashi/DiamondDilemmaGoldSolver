"""Shared immutable JSON checkpoint objects with lazy, alias-free decoding.

Exact structural interning reduces repeated domain vectors/metadata. Checksums
are corruption checks; backend validators still validate restored search state.
No executable objects, file references, pickle payloads or floating NaNs exist.
"""
from __future__ import annotations
import json
import math
from constraint_work import digest


class SharedSnapshot:
    VERSION = 1

    def __init__(self, *, max_nodes=1000000, max_atom_bytes=268435456):
        if type(max_nodes) is not int or not 1 <= max_nodes <= 1000000:
            raise ValueError('Invalid shared snapshot node bound.')
        if type(max_atom_bytes) is not int or not 1 <= max_atom_bytes <= 536870912:
            raise ValueError('Invalid shared snapshot byte bound.')
        self.max_nodes, self.max_atom_bytes = max_nodes, max_atom_bytes
        self.nodes, self.roots, self.index = [], [], {}
        self.atom_bytes = 0
        self.reuses = 0

    def add(self, value):
        active = set()

        def intern(item, depth=0):
            if depth > 512:
                raise ValueError('Checkpoint nesting exceeds bound.')
            if type(item) in (dict, list, tuple):
                identity = id(item)
                if identity in active:
                    raise ValueError('Cyclic checkpoint value.')
                active.add(identity)
                try:
                    if type(item) is dict:
                        if any(type(k) is not str for k in item):
                            raise ValueError('Checkpoint keys must be strings.')
                        key = ('dict', tuple((k, intern(v, depth + 1)) for k, v in sorted(item.items())))
                    else:
                        key = ('list', tuple(intern(v, depth + 1) for v in item))
                finally:
                    active.remove(identity)
                size = 0
            else:
                if type(item) not in (str, bool, int, float, type(None)) or type(item) is float and not math.isfinite(item):
                    raise ValueError('Unsupported checkpoint scalar.')
                encoded = json.dumps(item, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
                key = ('atom', type(item).__name__, encoded)
                size = len(encoded)
            if key in self.index:
                self.reuses += 1
                return self.index[key]
            if len(self.nodes) >= self.max_nodes or self.atom_bytes + size > self.max_atom_bytes:
                raise ValueError('Shared checkpoint capacity exceeded.')
            index = len(self.nodes)
            if key[0] == 'dict':
                record = ['dict', [[k, ref] for k, ref in key[1]]]
            elif key[0] == 'list':
                record = ['list', list(key[1])]
            else:
                record = ['atom', item]
            self.nodes.append(record)
            self.index[key] = index
            self.atom_bytes += size
            return index

        root = intern(value)
        self.roots.append(root)
        return len(self.roots) - 1

    def get(self, root):
        if type(root) is not int or not 0 <= root < len(self.roots):
            raise ValueError('Unknown shared checkpoint root.')
        visits = 0

        def decode(index, depth=0):
            nonlocal visits
            visits += 1
            if depth > 512 or visits > self.max_nodes * 32:
                raise ValueError('Expanded checkpoint exceeds bound.')
            kind, value = self.nodes[index]
            if kind == 'atom':
                return value
            if kind == 'list':
                return [decode(ref, depth + 1) for ref in value]
            return {key: decode(ref, depth + 1) for key, ref in value}

        return decode(self.roots[root])

    def snapshot(self):
        body = {'shared_snapshot_version': self.VERSION, 'nodes': self.nodes,
                'roots': self.roots, 'max_nodes': self.max_nodes,
                'max_atom_bytes': self.max_atom_bytes}
        # Return an owned value; caller mutation cannot change the internal pool.
        body = json.loads(json.dumps(body, allow_nan=False))
        body['digest'] = digest(body)
        return body

    @classmethod
    def restore(cls, payload):
        fields = {'shared_snapshot_version', 'nodes', 'roots', 'max_nodes', 'max_atom_bytes', 'digest'}
        if not isinstance(payload, dict) or set(payload) != fields or type(payload['shared_snapshot_version']) is not int or payload['shared_snapshot_version'] != cls.VERSION:
            raise ValueError('Malformed shared snapshot.')
        if digest({k: v for k, v in payload.items() if k != 'digest'}) != payload['digest']:
            raise ValueError('Shared snapshot digest differs.')
        out = cls(max_nodes=payload['max_nodes'], max_atom_bytes=payload['max_atom_bytes'])
        if not isinstance(payload['nodes'], list) or len(payload['nodes']) > out.max_nodes:
            raise ValueError('Shared snapshot nodes exceed bound.')
        for index, row in enumerate(payload['nodes']):
            if not isinstance(row, list) or len(row) != 2 or row[0] not in ('atom', 'list', 'dict'):
                raise ValueError('Invalid shared node.')
            kind, value = row
            if kind == 'atom':
                if type(value) not in (str, bool, int, float, type(None)) or type(value) is float and not math.isfinite(value):
                    raise ValueError('Invalid shared scalar.')
                encoded = json.dumps(value, ensure_ascii=True, allow_nan=False, separators=(',', ':'))
                key = ('atom', type(value).__name__, encoded)
                out.atom_bytes += len(encoded)
                if out.atom_bytes > out.max_atom_bytes:
                    raise ValueError('Shared atoms exceed bound.')
            else:
                if not isinstance(value, list):
                    raise ValueError('Invalid shared container.')
                if kind == 'dict':
                    if any(not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not str for pair in value):
                        raise ValueError('Invalid shared mapping.')
                    if [pair[0] for pair in value] != sorted({pair[0] for pair in value}):
                        raise ValueError('Repeated or noncanonical shared mapping keys.')
                    refs = [pair[1] for pair in value]
                    key = ('dict', tuple((k, ref) for k, ref in value))
                else:
                    refs = value
                    key = ('list', tuple(value))
                if any(type(ref) is not int or not 0 <= ref < index for ref in refs):
                    raise ValueError('Cyclic or invalid shared reference.')
            if key in out.index:
                raise ValueError('Noncanonical duplicate shared node.')
            out.index[key] = index
            out.nodes.append(json.loads(json.dumps(row)))
        roots = payload['roots']
        if not isinstance(roots, list) or len(roots) > 4096 or any(type(ref) is not int or not 0 <= ref < len(out.nodes) for ref in roots):
            raise ValueError('Invalid shared roots.')
        out.roots = list(roots)
        return out