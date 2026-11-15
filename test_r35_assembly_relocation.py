"""Independent physical checks for mandatory dimer exchanges and local displacement prices."""
from collections import Counter
from itertools import combinations,permutations,product
import unittest
import numpy as np

from r27_incumbent import BoardModel
from r31_fresh import Assemblies,physical_bonds
from r35_assembly_relocation import AssemblyRelocator
from search_fixtures import planted_gold
from search_problem import SearchProblem


def conditional_assemblies(model,groups,forbidden=()):
    assembly=object.__new__(Assemblies)
    assembly.model=model
    assembly.groups=groups
    assembly.tiles=set().union(*(g for g,_ in groups)) if groups else set()
    assembly.forbidden=frozenset(forbidden)
    return assembly


class AssemblyTests(unittest.TestCase):
    def test_plant_two_mutually_blocking_gold_dimers(self):
        tiles,geo,reference=planted_gold(3538,1,5)
        model=BoardModel(SearchProblem(tiles,geo))
        target=np.array(reference)
        for e,f in combinations(model.problem.edges,2):
            a,s,b,t=e
            c,u,d,v=f
            if len({a,b,c,d})!=4:continue
            groups=[]
            old=target.copy()
            for previous,slots in (((a,s,b,t),(c,u,d,v)),((c,u,d,v),(a,s,b,t))):
                x,sx,y,sy=previous
                j,sj,k,sk=slots
                p,q=target[x],target[y]
                pj=3*(p//3)+(sj-sx+p%3)%3
                qk=3*(q//3)+(sk-sy+q%3)%3
                old[j],old[k]=pj,qk
                groups.append(({int(p//3),int(q//3)},[((x,int(p)),(y,int(q))),((j,int(pj)),(k,int(qk)))]))
            if model.matches(old).sum()<len(model.problem.edges):break
        else:self.fail('No nontrivial planted exchange.')
        assembly=conditional_assemblies(model,groups)
        loc=AssemblyRelocator(assembly)
        base=assembly.relocate(old,np.random.default_rng(3),trials=2)
        one,_=loc.relocate(old,np.random.default_rng(3),'priced',trials=2)
        pair,work=loc.relocate(old,np.random.default_rng(3),'reciprocal',trials=2)
        np.testing.assert_array_equal(base,old)
        np.testing.assert_array_equal(one,old)
        np.testing.assert_array_equal(pair,target)
        self.assertTrue(model.problem.validate(pair.tolist())['valid'])
        self.assertGreater(work['reciprocal_incremental_gain'],0)
        repeated,repeated_work=loc.relocate(old,np.random.default_rng(3),'reciprocal',trials=2)
        np.testing.assert_array_equal(repeated,target)
        self.assertEqual(work,repeated_work)
        self.assertGreater(loc.cache_work['reciprocal_families_reused'],0)

    def test_all_rotations_of_displaced_pieces_match_full_scores_and_dart_restrictions(self):
        tiles,geo,reference=planted_gold(35381,1,4)
        model=BoardModel(SearchProblem(tiles,geo))
        board=np.array(reference)
        rng=np.random.default_rng(398)
        for _ in range(20):
            p,q,x,y=map(int,rng.choice(model.n,4,replace=False))
            old={p,q}
            pose={x:int(board[p]),y:int(board[q])}
            candidate=board.copy()
            candidate[[p,q]]=board[[x,y]]
            candidate[[x,y]]=board[[p,q]]
            banned=(physical_bonds(model,candidate)-physical_bonds(model,board))
            assembly=conditional_assemblies(model,[],list(banned)[:1])
            price=AssemblyRelocator(assembly)
            scores=[]
            for donors in permutations((board[x]//3,board[y]//3)):
                for r,s in product(range(3),repeat=2):
                    row=candidate.copy()
                    row[p],row[q]=3*donors[0]+r,3*donors[1]+s
                    if physical_bonds(model,row) & assembly.forbidden:continue
                    scores.append(int(model.matches(row).sum()-model.matches(board).sum()))
            out=price.price(board,old,pose,Counter())
            self.assertEqual(out is None,not scores)
            if out is not None:
                self.assertEqual(out[0],max(scores))
                self.assertFalse(assembly.forbidden&physical_bonds(model,out[1]))

    def test_equal_exterior_mask_with_distinct_forbidden_dart_changes_key(self):
        tiles,geo,reference=planted_gold(35383,2,4)
        model=BoardModel(SearchProblem(tiles,geo))
        board=np.array(reference)
        a,s,b,t=map(int,model.problem.edges[0])
        current=board[b]
        dart=3*(current//3)+(t-current%3)%3
        inside=3*(board[a]//3)+(s-board[a]%3)%3
        assembly=conditional_assemblies(model,[],[(min(dart,inside),max(dart,inside))])
        loc=AssemblyRelocator(assembly)
        replacements=[i for i in range(3*model.n) if i//3 not in (current//3,board[a]//3)
                      and model.masks[i,t]==model.masks[current,t]]
        self.assertTrue(replacements)
        # Preserve geometry, the exterior mask and the oriented code at the affected cell.
        new=board.copy()
        other=next(i for i,x in enumerate(board) if x//3==replacements[0]//3)
        new[other],new[b]=current,replacements[0]
        self.assertNotEqual(loc.boundary(board,(a,))[2],loc.boundary(new,(a,))[2])

    def test_checked_boundary_hit_preserves_remote_board_edits(self):
        tiles,geo,reference=planted_gold(35382,2,4)
        model=BoardModel(SearchProblem(tiles,geo))
        assembly=conditional_assemblies(model,[])
        loc=AssemblyRelocator(assembly)
        board=np.array(reference)
        old={0,1}
        pose={2:int(board[0]),3:int(board[1])}
        base=loc.price(board,old,pose,Counter())
        touched={0,1,2,3}|set(model.problem.neighbors[[0,1,2,3]].ravel())
        x,y=sorted(set(range(model.n))-touched)[:2]
        new=board.copy()
        new[x],new[y]=board[y],board[x]
        cached=loc.price(new,old,pose,Counter())
        self.assertEqual(loc.cache_work['price_hits'],1)
        exact=loc.score(new,old,pose,Counter())
        self.assertEqual(cached[0],base[0])
        np.testing.assert_array_equal(exact[1],cached[1])
        self.assertEqual(cached[1][x],new[x])


if __name__=='__main__':
    unittest.main()
