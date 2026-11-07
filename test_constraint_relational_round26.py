from test_constraint_sat_assist import HAS_SAT
import itertools,random,unittest
from collections import OrderedDict
from constraint_relational_round26 import SemijoinNetwork,TupleIndex,guided_cells
from constraint_higher_relations import ProjectedRelations
from constraint_sat_assist import SatAssist
from test_constraint_sat_domains import fixture,ordinary_fixpoint,admitted
from test_constraint_dfs import brute

SCOPES=((4095,2035,2039,3581),(3581,4083,1335,4019),(3455,3959,3511,3518),(3583,1907,3510,3569))
CELLS=((1,2,3),(0,2,3),(0,1,3),(0,1,2))
CURRENT=(2413,865,1334,3504)


def unary_fixpoint(problem,tables,domains):
    current=list(domains)
    while all(current):
        old=current.copy()
        for (scope,cells),rows in tables.items():
            if any(mask&~root for mask,root in zip(current,scope)):continue
            kept=[row for row in rows if all(current[c]>>v&1 for c,v in zip(cells,row))]
            for index,cell in enumerate(cells):current[cell]&=sum(1<<v for v in {row[index] for row in kept})
        current=ordinary_fixpoint(problem,current)
        if current==old:break
    return current


def native_semijoin_fixture():
    problem,_=fixture();root=[4095]*4;helper=SatAssist(problem,root,'edges');tables=OrderedDict();queries=decisions=0
    try:
        for scope,cells in zip(SCOPES,CELLS):
            compiler=ProjectedRelations(problem,root,'edges',guided=True)
            try:
                values=list(scope)
                for _ in range(10):
                    compiler.prune(values,helper,seconds=.5,max_calls=100,projection_cells=cells)
                    if compiler.tables:break
                if not compiler.tables:raise AssertionError('Native projection incomplete.')
                tables.update(compiler.tables);queries+=compiler.stats['higher_queries'];decisions+=compiler.stats['higher_decisions']
            finally:compiler.close()
        solutions=brute(problem,'edges')
        for (scope,cells),rows in tables.items():
            if set(rows)!={tuple(board[c] for c in cells) for board in admitted(solutions,scope)}:raise AssertionError('Native table differs from independent completions.')
        before=unary_fixpoint(problem,tables,CURRENT);network=SemijoinNetwork();after=before.copy();network.filter(tables,after)
        if admitted(solutions,before)!=admitted(solutions,after):raise AssertionError('Semijoin removed an independent completion.')
        return {'before':before,'after':after,'edge_completions':len(admitted(solutions,before)),
            'native_compile_queries':queries,'native_compile_decisions':decisions,'stats':network.stats}
    finally:helper.close()


class RelationalTests(unittest.TestCase):
    @unittest.skipUnless(HAS_SAT,'optional python-sat unavailable')
    def test_native_complete_puzzle_tables_add_deductions_beyond_unary_and_existing_pruning(self):
        result=native_semijoin_fixture();self.assertEqual(result['before'],list(CURRENT));self.assertEqual(result['after'],[2413,865,1334,3360])
        self.assertEqual(result['edge_completions'],15);self.assertEqual(result['stats']['relation_domain_prunes'],2)
        self.assertGreater(result['native_compile_queries'],100)

    def test_random_relations_preserve_every_joined_assignment_and_widening(self):
        rng=random.Random(26061)
        for _ in range(120):
            tables=OrderedDict()
            for cells in ((0,1,2),(1,2,3),(0,2,3)):
                rows=tuple(row for row in itertools.product(range(3),repeat=3) if rng.random()<.35)
                tables[(7,7,7,7),cells]=rows
            network=SemijoinNetwork()
            for domains in ([7]*4,[3,7,5,7],[7]*4):
                expected=[board for board in itertools.product(range(3),repeat=4) if all(mask>>v&1 for mask,v in zip(domains,board)) and all(tuple(board[c] for c in cells) in rows for (scope,cells),rows in tables.items())]
                current=domains.copy();network.filter(tables,current)
                self.assertTrue(all(all(mask>>v&1 for mask,v in zip(current,board)) for board in expected))
                cold=domains.copy();SemijoinNetwork().filter(tables,cold);self.assertEqual(current,cold)

    def test_scope_cancellation_and_partial_tables_are_not_admitted(self):
        tables=OrderedDict([(((3,3,3,3),(0,1,2)),((0,0,0),(1,1,1))),(((3,3,3,3),(0,1,3)),((0,1,0),(1,0,1)))])
        network=SemijoinNetwork();current=[7]*4;network.filter(tables,current);self.assertEqual(current,[7]*4)
        before=[3]*4
        for boundary in range(1,10):
            current=before.copy();calls=0
            def stop():
                nonlocal calls
                calls+=1
                if calls==boundary:raise InterruptedError('stop')
            try:network.filter(tables,current,stop)
            except InterruptedError:self.assertEqual(current,before)
        current=before.copy();network.filter(tables,current);self.assertFalse(all(current))

    def test_dag_projection_operator_matches_rows_and_guidance_is_only_ordering(self):
        rows=tuple(itertools.product(range(4),repeat=4));index=TupleIndex((3,1,2,0),rows)
        for cells in ((3,1),(0,3),(1,),(3,1,2,0),()):
            expected={}
            for position,row in enumerate(index.rows):
                key=tuple(row[index.cells.index(c)] for c in cells);expected[key]=expected.get(key,0)|(1<<position)
            self.assertEqual(index.projection(cells),expected)
        self.assertLess(index.dag_projection_nodes,5*len(rows))
        class Helper:
            domain_cores={}
            domain_sessions={(7,7,7,7):{'supported':[7,7,1,7]}}
        network=SemijoinNetwork();network.pressure[3]=4;stats={}
        selected=guided_cells([7]*4,Helper(),3,12,stats,network)
        self.assertEqual(selected,(3,2,0));self.assertEqual(stats['relation_missing_support_guidance'],2);self.assertEqual(stats['relation_spurious_guidance'],4)

    def test_suffix_dag_and_core_guidance_are_bounded_exact(self):
        rows=tuple(itertools.product(range(4),repeat=4));index=TupleIndex((0,1,2,3),rows)
        self.assertLess(len(index.nodes),len(rows));self.assertEqual(index.full.bit_count(),256)
        class Helper:domain_cores=OrderedDict([(((1,3),(3,3)),None)])
        stats={'relation_guided_selections':0};selected=guided_cells([7]*4,Helper(),3,12,stats)
        self.assertEqual(selected,(1,3,0));self.assertEqual(stats['relation_guided_selections'],1)

if __name__=='__main__':unittest.main()
