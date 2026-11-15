"""Independent score and canonical ownership controls for contacting donor cycles."""
from collections import Counter
from contextlib import closing
from itertools import combinations, product
from tempfile import TemporaryDirectory
from pathlib import Path
import sqlite3
import unittest

import numpy as np

from r27_incumbent import BoardModel
from r30_fresh import fresh_start
from r31_fresh import Engine
from r34_pair_cycles import cycle_prices, pair_repair
from r35_cycle_contacts import Prices, connected_triples, contact_graph, corrected_moves
from search_fixtures import planted_gold
from search_problem import SearchProblem


class CycleContactTests(unittest.TestCase):
    def test_every_connected_triple_and_shared_rotation_price(self):
        tiles,geo,codes = planted_gold(127,1,5)
        model = BoardModel(SearchProblem(tiles,geo))
        board = fresh_start(model,np.random.default_rng(127))
        cells = (0,1,2,3,4,5)
        # Eliminate selected alternate orientations and preserve the input.
        for i in cells:
            model.legal[i,1::3] = False
            model.legal[i,board[i]] = True
        before,rows,_,_ = cycle_prices(model,board,cells,2)
        graph = contact_graph(rows)
        triples = list(connected_triples(rows,graph))
        expected = set()
        for triple in combinations(range(len(rows)),3):
            a,b,c = (rows[i][0] for i in triple)
            if a&b or b&c or a&c:
                continue
            contact = lambda u,v: any(y in model.problem.neighbors[x] for x in u for y in v)
            if contact(a,b)+contact(a,c)+contact(b,c) >= 2:
                expected.add(triple)
        self.assertEqual({tuple(sorted(t)) for t in triples},expected)
        self.assertEqual(len(triples),len(expected))
        self.assertTrue(expected)
        work = Counter()
        prices = Prices(rows,before,graph,work)
        sets = [t for t in combinations(range(len(rows)),2) if t[1] in graph[t[0]]] + triples
        for indices in sets:
            mixed,changes = prices.price(indices)
            full,raw_changes = prices.price(indices,compress=False)
            self.assertEqual((mixed,changes),(full,raw_changes))
            # Directly score every physical board without any local gain or shared-seam algebra.
            for_rows = [rows[i] for i in indices]
            perms = np.array(list(product(*(range(len(row[2])) for row in for_rows))))
            boards = np.tile(board, (len(perms),1))
            for pos,row in enumerate(for_rows):
                boards[:,row[1]] = row[2][perms[:,pos]]
            actual = (model.masks[boards[:,model.a],model.sa] ==
                      model.reverse[model.masks[boards[:,model.b],model.sb]]).sum(1)
            self.assertEqual(mixed,int(actual.max()-before.sum()))
            changed = board.copy()
            for c,v in changes.items():
                changed[c] = v
            model.legal_board(changed)
            self.assertEqual(int(model.matches(changed).sum()),int(actual.max()))
        self.assertLess(work['priced_3_rotation_products'],work['raw_3_rotation_products'])

    def test_added_choices_preserve_shipped_packed_score(self):
        tiles,geo,codes = planted_gold(37,1,5)
        model = BoardModel(SearchProblem(tiles,geo))
        rng = np.random.default_rng(734)
        for _ in range(6):
            board = fresh_start(model,rng)
            pool = rng.choice(model.free,6,replace=False)
            released,_ = pair_repair(model,board,pool,3)
            before,rows,singles,_ = cycle_prices(model,board,pool,3)
            _,changes,_ = corrected_moves(model,rows,before,singles,1)
            compressed = board.copy()
            for item in changes:
                for cell,code in item.items():
                    compressed[cell] = code
            self.assertEqual(compressed.tolist(),released.tolist())
            independent,_ = pair_repair(model,board,pool,3,contact_group=1)
            oracle,_ = pair_repair(model,board,pool,3,interact=False)
            self.assertEqual(independent.tolist(),oracle.tolist())
            previous = int(model.matches(released).sum())
            for group in (2,3):
                better,work = pair_repair(model,board,pool,3,contact_group=group)
                checked = model.problem.validate(model.legal_board(better).tolist())
                self.assertGreaterEqual(checked['matched_edges'],previous)
                self.assertEqual(checked['matched_edges']-int(model.matches(board).sum()),
                                 work['gain_pairs' if group==2 else 'gain_triples'])
                previous = checked['matched_edges']

    def test_single_cycle_prices_equal_full_legal_rotation_tables(self):
        tiles,geo,_ = planted_gold(103,1,5)
        rng = np.random.default_rng(123)
        for repetition in range(12):
            model = BoardModel(SearchProblem(tiles,geo))
            board = fresh_start(model,rng)
            # Impose asymmetric orientation domains and reserve one cell.
            model.legal &= rng.integers(0,2,size=model.legal.shape,dtype=bool)
            model.legal[np.arange(model.n),board] = True
            model.legal[1] = False
            model.legal[1,board[1]] = True
            for size in (0,1,4,8):
                pool = list(map(int,rng.choice(model.n,size,replace=False)))
                for length in (2,3):
                    old,control = pair_repair(model,board,pool,length,interact=False)
                    actual,work = pair_repair(model,board,pool,length,contact_group=1)
                    self.assertEqual(actual.tolist(),old.tolist())
                    self.assertEqual(work.get('cycles_priced',0),control.get('cycles_priced',0))

    def test_trajectory_keys_bind_contact_groups(self):
        tiles,geo,_ = planted_gold(89,1,5)
        problem = SearchProblem(tiles,geo)
        with TemporaryDirectory() as folder:
            db = Path(folder)/'traj.sqlite'
            for size in (0,1,2,3):
                engine = Engine(problem,pair_cycles=3,contact_group=size)
                cold = engine.run(11,rounds=1,trajectory_db=db)
                warm = engine.run(11,rounds=1,trajectory_db=db)
                self.assertEqual(cold['codes'],warm['codes'])
                self.assertEqual(warm['new_repair_steps'],0)
                self.assertEqual(cold['reused_repair_steps'],0)
            with closing(sqlite3.connect(db)) as conn:
                self.assertEqual(conn.execute('SELECT count(DISTINCT scope) FROM r31_repair_prefix').fetchone()[0],4)


if __name__=='__main__':
    unittest.main()
