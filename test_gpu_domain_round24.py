"""Finite CUDA enumeration/ownership tests for the persistent domain engine."""
import math
import os
import random
import unittest
import numpy as np
from search_problem import SearchProblem
from gpu_domain_round24 import DomainCUDA
from test_constraint_dfs import simple_tiles,multiple_loops,double_face,tetrahedron_tiles,board,brute

def enumerate_engine(engine,quantum=256,limit=5000):
    found=[]
    for _ in range(limit):
        state=engine.step(quantum)
        pending=engine.candidates()
        for item in pending:found.append(tuple(item['codes']))
        if pending:engine.acknowledge([item['lane'] for item in pending])
        if all(state==2):return found
    raise AssertionError('Finite enumeration did not complete.')

@unittest.skipUnless(os.environ.get('DIAMOND_TEST_CUDA')=='1','Optional CUDA domain engine tests')
class DomainCUDATests(unittest.TestCase):
    def test_complete_answers_match_independent_bruteforce(self):
        for tiles,geometry in [(simple_tiles(),double_face()),(multiple_loops(),double_face()),tetrahedron_tiles()]:
            problem=SearchProblem(tiles,geometry)
            for target in ('edges','gold'):
                expected=set(brute(problem,target))
                for learning in (False,True):
                    for strands in (False,True):
                        engine=DomainCUDA(problem,target=target,learning=learning,incremental_strands=strands)
                        try:
                            found=enumerate_engine(engine,17)
                            self.assertEqual(set(found),expected);self.assertEqual(len(found),len(set(found)))
                        finally:engine.close()
    def test_every_primitive_remaining_partition_and_checkpoint_resume(self):
        problem=SearchProblem(*tetrahedron_tiles());expected=set(brute(problem,'gold'))
        engine=DomainCUDA(problem);seen=set()
        try:
            for iteration in range(3000):
                state=engine.step(1)
                cubes=engine.remaining_domain_cubes()
                for i,cube in enumerate(cubes):
                    for other in cubes[:i]:self.assertTrue(any(not(a&b) for a,b in zip(cube,other)))
                remaining={answer for answer in expected if any(all(mask&(1<<code) for mask,code in zip(cube,answer)) for cube in cubes)}
                self.assertEqual(remaining,expected-seen)
                pending=engine.candidates()
                for item in pending:
                    answer=tuple(item['codes']);self.assertNotIn(answer,seen);seen.add(answer)
                if pending:engine.acknowledge([item['lane'] for item in pending])
                if iteration%31==0:
                    saved=engine.checkpoint();replacement=DomainCUDA(problem);replacement.restore(saved);engine.close();engine=replacement
                if all(state==2):break
            else:self.fail('No finite completion')
            self.assertEqual(seen,expected)
        finally:engine.close()
    def test_portable_cubes_resume_with_changed_learning_policy(self):
        problem=SearchProblem(*tetrahedron_tiles());expected=set(brute(problem,'gold'));first=DomainCUDA(problem)
        first.step(20);cubes=first.remaining_domain_cubes();first.close()
        second=DomainCUDA(problem,cubes,target='gold',learning=False)
        try:self.assertEqual(set(enumerate_engine(second)),expected)
        finally:second.close()
    def test_learning_closes_independent_hall_subtree_with_fewer_decisions(self):
        n=8;geometry=board(n,[(a,s,a+1,s) for a in range(0,n,2) for s in range(3)])
        tiles=[{'id':i,'segments':[[[s,2],[s,10]] for s in range(3)]} for i in range(n)]
        problem=SearchProblem(tiles,geometry)
        problem.domains=[[0,3],[0,3]]+[[6,9,12,15] for _ in range(5)]+[[18,21]]
        reports=[]
        for learning in (False,True):
            engine=DomainCUDA(problem,target='edges',learning=learning)
            try:self.assertEqual(enumerate_engine(engine),[]);reports.append(engine.stats())
            finally:engine.close()
        self.assertGreater(reports[1]['backjumps'],0)
        self.assertLess(reports[1]['decisions'],reports[0]['decisions'])
    def test_incremental_strands_avoid_retracing_completed_paths(self):
        problem=SearchProblem(*tetrahedron_tiles());reports=[]
        for incremental in (False,True):
            engine=DomainCUDA(problem,incremental_strands=incremental)
            try:self.assertEqual(set(enumerate_engine(engine)),set(brute(problem,'gold')));reports.append(engine.stats())
            finally:engine.close()
        self.assertLess(reports[1]['strand_joins'],reports[0]['strand_joins'])
    def test_unassigned_seam_inconsistency_is_proved_without_branching(self):
        tiles=simple_tiles();tiles[1]['segments']=[[[0,2],[1,4]]]
        engine=DomainCUDA(SearchProblem(tiles,double_face()))
        try:self.assertEqual(enumerate_engine(engine),[]);self.assertEqual(engine.stats()['decisions'],0)
        finally:engine.close()
    def test_deep160cell_frames_preserve_exact_remaining_volume(self):
        n=160;geometry=board(n,[(a,s,a+1,s) for a in range(0,n,2) for s in range(3)])
        tiles=[{'id':i,'segments':[[[s,2],[s,10]] for s in range(3)]} for i in range(n)]
        problem=SearchProblem(tiles,geometry);problem.domains=[list(range(3*c,3*c+3)) for c in range(n)]
        engine=DomainCUDA(problem,target='edges');seen=0
        try:
            for _ in range(2000):
                engine.step(4096)
                pending=engine.candidates()
                if pending:
                    self.assertEqual(len(pending),1);seen+=1;engine.acknowledge([0])
                    cubes=engine.remaining_domain_cubes()
                    self.assertEqual(sum(math.prod(mask.bit_count() for mask in cube) for cube in cubes),3**160-seen)
                    saved=engine.checkpoint();engine.restore(saved)
                    if seen==2:break
            self.assertEqual(seen,2)
        finally:engine.close()
    def test_checkpoint_mutation_and_excess_memory_are_rejected(self):
        problem=SearchProblem(simple_tiles(),double_face())
        with self.assertRaisesRegex(ValueError,'budget'):DomainCUDA(problem,max_memory_bytes=1)
        engine=DomainCUDA(problem)
        try:
            saved=engine.checkpoint();saved['state'][0,0]=2
            with self.assertRaisesRegex(ValueError,'checksum'):engine.restore(saved)
        finally:engine.close()

if __name__=='__main__':unittest.main()
