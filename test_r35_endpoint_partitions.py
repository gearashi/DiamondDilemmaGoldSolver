from importlib.util import find_spec
from itertools import product
from types import SimpleNamespace
from unittest import TestCase,skipUnless


@skipUnless(find_spec('ortools'), 'Native CP comparison')
class PhysicalCuts(TestCase):
    def test_every_phase_and_partition_with_absent_and_conditional_labels(self):
        from ortools.sat.python import cp_model
        from r35_endpoint_partitions import compile_endpoint_cut,add_compiled_cut
        data=SimpleNamespace(strands=[([(0,0),(1,1)],[(2,3)]),
                                      ([(0,1)],[(2,2)]),
                                      ([(1,2)],[(3,0)])])
        for subset in range(16):
            group={x for x in range(4) if subset>>x&1}
            compiled=compile_endpoint_cut(data,group)
            for bits in product(range(2),repeat=3):
                edges=[edge for i,phase in enumerate(bits) for edge in data.strands[i][phase]]
                vertices={v for edge in edges for v in edge}
                crosses=sum((a in group)!=(b in group) for a,b in edges)
                expected=not (vertices&group and vertices-group and crosses<2)
                m=cp_model.CpModel()
                xs=[m.new_bool_var('phase') for _ in bits]
                for x,z in zip(xs,bits):
                    m.add(x==z)
                add_compiled_cut(m,xs,compiled)
                native=cp_model.CpSolver()
                native.parameters.num_search_workers=1
                self.assertEqual(native.solve(m)==cp_model.OPTIMAL,expected)
