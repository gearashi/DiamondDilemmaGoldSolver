from itertools import combinations, permutations
import unittest
from importlib.util import find_spec
import numpy as np
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r30_exact_routes import cycle_type
from r34_cycle_orders import CycleOrders
from r35_route_distance import DistanceModel
from r35_remote_donors import response_pool
from r35_route_shapes import ContactShapes
from search_fixtures import planted_gold
from search_problem import SearchProblem


class AdvancedRoutes(unittest.TestCase):
    def test_disconnected_shapes_keep_every_orientation_and_two_profitable_groups(self):
        tiles,geo,codes=planted_gold(68,2,4)
        m=BoardModel(SearchProblem(tiles,geo))
        b=ExchangeBound(m,codes)
        s=StarClusters(b)
        screen=ContactShapes(s)
        pool=(0,4,11,18,24,31,37)
        for family in ((2,2),(2,3),(2,2,3)):
            for cells in combinations(pool,sum(family)):
                orders=CycleOrders(family)
                original={o:s.price(cells,o) for o in orders}
                thresholds=sorted({v//2 for v in original.values()})
                for target in thresholds:
                    keep=set(screen.allowed_orders(cells,orders,target))
                    self.assertTrue(all(o in keep for o,v in original.items() if v>=2*target))
        self.assertGreater(screen.work['disconnected_shapes'],0)

    @unittest.skipUnless(find_spec('ortools'),'requires OR-Tools')
    def test_exact_general_distance_matches_every_enumerated_permutation(self):
        tiles,geo,codes=planted_gold(89,1,7)
        m=BoardModel(SearchProblem(tiles,geo))
        cells=(0,2,4,7)
        values=np.array(codes)
        values[list(cells)]=values[[2,4,0,7]]
        q=DistanceModel(m,values,cells)
        oracle=RotationOracle(m)
        best={}
        for order in permutations(range(len(cells))):
            rearranged=values.copy()
            rearranged[list(cells)]=values[[cells[j] for j in order]]
            distance=sum(k-1 for k in cycle_type(order))
            score=oracle.optimum(rearranged)[0]
            best[distance]=max(best.get(distance,-1),score)
        for limit in range(len(cells)):
            for target in sorted(set(best.values())):
                for low in (0,limit):
                    actual=q.solve(target,minimum=low,maximum=limit,optimize=True)
                    expected=min((k for k,v in best.items() if low<=k<=limit and v>=target),default=None)
                    self.assertEqual(actual.get('distance'),expected)
                    if expected is None:
                        self.assertEqual(actual['status'],'INFEASIBLE')
                    else:
                        self.assertEqual(actual['distance_lower_bound'],expected)
                        self.assertGreaterEqual(m.problem.validate(actual['codes'])['matched_edges'],target)

    @unittest.skipUnless(find_spec('ortools'),'requires OR-Tools')
    def test_distant_return_recovery_and_unavailable_return(self):
        from r31_fresh import joint_repair
        tiles,geo,codes=planted_gold(91,2,6)
        m=BoardModel(SearchProblem(tiles,geo))
        free=list(map(int,m.free))
        near={0,*map(int,m.problem.neighbors[0])}
        near.update(int(n) for c in tuple(near) if c>=0 for n in m.problem.neighbors[c])
        remote=next(c for c in free if c not in near and c>0)
        bad=np.array(codes)
        bad[[0,remote]]=bad[[remote,0]]
        avail=[0,remote]
        pool,work=response_pool(m,bad,2,np.random.default_rng(35),imports=1,anchors=[0],available=avail)
        self.assertEqual(work['imports'][0]['donor'],remote)
        recovered,_=joint_repair(m,bad,pool,contacts=-1,effort=1)
        self.assertEqual(m.problem.validate(recovered.tolist())['matched_edges'],len(m.problem.edges))
        # The defect destination can take the foreign identity, its source
        # cannot legally receive the displaced tile on the return edge.
        m.legal[remote,3*(bad[0]//3):3*(bad[0]//3)+3]=False
        _,work=response_pool(m,bad,2,np.random.default_rng(35),imports=1,anchors=[0],available=avail)
        self.assertEqual(work['usable'],0)

    @unittest.skipUnless(find_spec('ortools'),'requires OR-Tools')
    def test_strict_downhill_route_and_twelve_mover_six_swap_threshold(self):
        tiles,geo,codes=planted_gold(16,1,9)
        m=BoardModel(SearchProblem(tiles,geo))
        pool=(2,3,5,7,8)
        board=np.array(codes)
        board[list(pool)]=np.array(codes)[[pool[i] for i in (1,4,2,0,3)]]
        oracle=RotationOracle(m)
        self.assertEqual(oracle.optimum(board)[0],12)
        for a,b in combinations(pool,2):
            neighbor=board.copy();neighbor[[a,b]]=neighbor[[b,a]]
            self.assertLess(oracle.optimum(neighbor)[0],12)
        witness=DistanceModel(m,board,pool).solve(13,optimize=True)
        self.assertEqual((witness['distance'],witness['distance_lower_bound']),(3,3))
        tiles,geo,codes=planted_gold(3522,2,55)
        m=BoardModel(SearchProblem(tiles,geo))
        pool=tuple(range(12))
        board=np.array(codes)
        for i in range(0,len(pool),2):
            board[[i,i+1]]=board[[i+1,i]]
        for c in range(m.n):
            m.legal[c]=False
            for v in (codes[c],board[c]):
                m.legal[c,3*(v//3):3*(v//3)+3]=True
        query=DistanceModel(m,board,pool)
        score=len(m.problem.edges)
        self.assertEqual(query.solve(score,maximum=5)['status'],'INFEASIBLE')
        hit=query.solve(score,maximum=6,optimize=True)
        self.assertEqual((hit['distance'],hit['score']),(6,score))

    def test_two_disconnected_individually_profitable_components_can_both_be_needed(self):
        tiles,geo,codes=planted_gold(264,2,55)
        m=BoardModel(SearchProblem(tiles,geo))
        planted=np.asarray(codes)
        pair1=(0,int(m.problem.neighbors[0,0]))
        near=StarClusters(ExchangeBound(m,planted)).close
        found=False
        for pair2 in combinations(range(m.n),2):
            if not near[pair2] or any(near[x,y] for x in pair1 for y in pair2):
                continue
            bad=planted.copy()
            for a,b in (pair1,pair2):
                bad[[a,b]]=bad[[b,a]]
            stars=StarClusters(ExchangeBound(m,bad))
            first=stars.price(pair1,(1,0))
            second=stars.price(pair2,(1,0))
            both=first+second-stars.bound.total
            if both//2<=max(first,second)//2:
                continue
            shape=ContactShapes(stars)
            self.assertTrue(shape.admissible((pair1,pair2),both//2))
            self.assertEqual(shape.work['disconnected_shapes'],1)
            found=True
            break
        self.assertTrue(found)
