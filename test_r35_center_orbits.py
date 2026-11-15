from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from r27_incumbent import BoardModel
from r30_multicenter import run
from r31_routes import rank
from r34_centers import select
from r34_cycle_orders import CycleOrders
from r35_center_orbits import CenterOrbits
from search_fixtures import planted_gold
from search_problem import SearchProblem


class OrbitControls(unittest.TestCase):
    def setUp(self):
        self.tiles,self.geo,self.gold=planted_gold(181,1,4)
        self.p=SearchProblem(self.tiles,self.geo);self.model=BoardModel(self.p)
        self.orbits=CenterOrbits(self.model)
        self.family=(2,4);self.center=list(self.gold)
        order=CycleOrders(self.family)[0]
        for i in range(6):self.center[i]=self.gold[order[i]]

    def test_caps_selection_and_inverse_paths_preserve_coordinates(self):
        o=self.orbits;images=[o.forward(self.center,i) for i in range(10)]
        self.assertEqual(len({o.key([v//3 for v in b])[0] for b in images}),1)
        for i in range(10):
            self.assertEqual(o.restore(images[i],i),self.center)
            other=list(self.center);other[2],other[5]=other[5],other[2]
            rotated=o.forward(other,i)
            positions={j for j,(a,b) in enumerate(zip(rotated,images[i])) if a//3!=b//3}
            self.assertEqual(positions,{o.transforms[i].cells[2],o.transforms[i].cells[5]})
            self.assertEqual(o.restore(rotated,i),other)
        p=SearchProblem(self.tiles,self.geo,fixed={9:self.gold[9]})
        restricted=CenterOrbits(BoardModel(p))
        self.assertFalse(restricted.transforms)
        self.assertNotEqual(restricted.key([c//3 for c in images[1]])[0],restricted.key([c//3 for c in images[0]])[0])
        with TemporaryDirectory() as tmp,patch('r34_centers.budget_upper',wraps=__import__('r30_bounds').budget_upper) as price:
            path=Path(tmp)/'cap.db'
            _,_,full=select(self.p,images,path,self.family,0,10)
            self.assertEqual(full['equivalent_centers'],9);self.assertEqual(price.call_count,1)
            _,_,partial=select(self.p,images,path,self.family,0,10,distinct_orbits=False)
            self.assertEqual(partial['candidates'],10);self.assertEqual(price.call_count,1)
            self.assertEqual(partial['reused_mask_prices'],10)
            changed=images[0].copy();changed[0]=3*(changed[0]//3)+(changed[0]+1)%3
            _,_,update=select(self.p,[changed,*images],path,self.family,0,10)
            self.assertEqual((update['fresh_mask_prices'],update['reused_mask_prices']),(1,10))

    def test_public_rotated_exact_optimum_restores_the_actual_requested_support(self):
        o=self.orbits;transform=next(i for i,t in enumerate(o.transforms) if set(t.cells[:6])!=set(range(6)))
        moved=sorted(o.transforms[transform].cells[:6]);new_rank=rank(moved,self.model.free)
        opts=dict(families=(self.family,),target=len(self.p.edges),page=1,filters=())
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'optima.db'
            first=run(self.p,[self.center],path,start=0,stop=1,**opts)
            rotated=o.forward(self.center,transform)
            trial=run(self.p,[rotated],path,start=new_rank,stop=new_rank+1,orbit_scores=True,**opts)
            reference=run(self.p,[rotated],Path(tmp)/'direct.db',start=new_rank,stop=new_rank+1,**opts)
            self.assertGreater(trial['work']['rotated_witnesses'],0)
            self.assertLess(trial['work'].get('exact_solved',0),reference['work']['exact_solved'])
            self.assertEqual(first['best']['score'],trial['best']['score'])
            whole=self.p.validate(trial['best']['codes']);self.assertTrue(whole['valid'])
            self.assertEqual([i for i in range(self.p.n) if trial['best']['codes'][i]//3!=rotated[i]//3],moved)
            self.assertEqual(o.restore(trial['best']['codes'],transform),first['best']['codes'])


if __name__=='__main__':unittest.main()
