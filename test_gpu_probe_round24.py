"""Deterministic larger-than-triple Hall obstruction for CPU probe selection."""
import unittest
from geometry import build_board
from gpu_cpu_assist import CPUAssist

class ProbeTargetRound24Tests(unittest.TestCase):
    def test_four_cell_shortage_is_prioritized_and_independently_reproved(self):
        board=build_board();neighbors=board.neighbor_cells
        chosen=[];forbidden=set()
        for root in range(160):
            if root in forbidden:continue
            chosen.append(root);near={root}
            for _ in range(4):near|={int(other) for cell in tuple(near) for other in neighbors[cell]}
            forbidden|=near
            if len(chosen)==4:break
        self.assertEqual(len(chosen),4)
        def tile(index,a_sides):
            return {'id':index,'segments':[[[side,2 if side in a_sides else 4],[side,10 if side in a_sides else 8]] for side in range(3)]}
        tiles=[tile(i,{0,1,2}) for i in range(3)]+[tile(i,{0}) for i in range(3,15)]+[tile(i,set()) for i in range(15,160)]
        bridge=CPUAssist({'tiles':tiles},'test',probe=True,proof_targeted=True,seconds=1)
        assignment=[-1]*160;next_tile=3
        for cell in chosen:
            for side,other in enumerate(neighbors[cell]):
                other=int(other);self.assertEqual(assignment[other],-1)
                rotation=int(bridge.problem.sides[cell,side]);assignment[other]=3*next_tile+rotation;next_tile+=1
        self.assertFalse(bridge.problem.partial_loop_impossible(assignment))
        priority=bridge.obstruction_priority(assignment)
        self.assertEqual(priority[0],1)
        healthy=[-1]*160
        self.assertGreater(priority,bridge.obstruction_priority(healthy))
        result=bridge.probe(assignment,seconds=1)
        self.assertEqual(result['status'],'infeasible')
        self.assertTrue(any(value==0 for value in result['domains']))

if __name__=='__main__':unittest.main()
