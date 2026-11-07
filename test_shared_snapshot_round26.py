import copy
import json
import math
import unittest
from constraint_shared_snapshot import SharedSnapshot
from constraint_work import digest

class SharedSnapshotTests(unittest.TestCase):
    def test_exact_values_and_no_mutable_aliases(self):
        repeated = {'domains': ['0xff'] * 160, 'choices': [1, 2, 3]}
        values = [{'a': repeated, 'b': repeated, 'typed': [True, 1, False, 0, None, 1.0, '1']} for _ in range(20)]
        store = SharedSnapshot()
        for value in values: store.add(value)
        self.assertGreater(store.reuses, 100)
        restored = SharedSnapshot.restore(json.loads(json.dumps(store.snapshot())))
        for i, expected in enumerate(values): self.assertEqual(restored.get(i), expected)
        one = restored.get(0)
        one['a']['domains'][0] = 'changed'
        self.assertEqual(one['b']['domains'][0], '0xff')
        self.assertEqual(restored.get(0), values[0])
        types = [type(x) for x in restored.get(0)['typed']]
        self.assertEqual(types, [bool, int, bool, int, type(None), float, str])

    def test_repeated_native_frames_compress(self):
        domain = [hex((1 << 480) - 1)] * 160
        values = [{'stack': [{'domains': list(domain), 'cell': -1, 'next': 0, 'choices': []}],
                   'stats': {'nodes': i}, 'root': list(domain)} for i in range(100)]
        store = SharedSnapshot()
        for value in values: store.add(value)
        packed = json.dumps(store.snapshot())
        self.assertLess(len(packed), len(json.dumps(values)) // 20)
        self.assertEqual(SharedSnapshot.restore(json.loads(packed)).get(99), values[99])

    def test_rejects_invalid_graphs_and_unsupported_values(self):
        for value in [float('nan'), float('inf'), {1: 'x'}, object()]:
            with self.assertRaises(ValueError): SharedSnapshot().add(value)
        cyclic = []; cyclic.append(cyclic)
        with self.assertRaises(ValueError): SharedSnapshot().add(cyclic)
        source = SharedSnapshot(); source.add({'x': [1]})
        for kind in ('cycle', 'bool-reference', 'duplicate-key'):
            data = copy.deepcopy(source.snapshot())
            if kind == 'cycle': data['nodes'][1] = ['list', [1]]
            if kind == 'bool-reference': data['roots'] = [True]
            if kind == 'duplicate-key': data['nodes'][-1] = ['dict', [['x', 1], ['x', 1]]]
            data['digest'] = digest({k:v for k,v in data.items() if k != 'digest'})
            with self.assertRaises(ValueError): SharedSnapshot.restore(data)

    def test_limits(self):
        with self.assertRaises(ValueError): SharedSnapshot(max_atom_bytes=2).add('long')
        with self.assertRaises(ValueError): SharedSnapshot(max_nodes=2).add([1, 2, 3])
        store = SharedSnapshot(); index = store.add([])
        self.assertEqual(SharedSnapshot.restore(store.snapshot()).get(index), [])

if __name__ == '__main__': unittest.main()