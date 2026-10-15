"""Independent partition, checkpoint and physical seam-pattern regressions."""
import copy,itertools,random,unittest
from constraint_branching import ConnectionBranching,STAT_KEYS,to_native_frame,from_native_frame
from constraint_dfs import solve,_checkpoint_digest
from constraint_parallel import partition_checkpoint
from search_problem import SearchProblem
from test_constraint_dfs import simple_tiles,double_face,tetrahedron_tiles

class ConnectionTests(unittest.TestCase):
    def test_patterns_partition_all_orientations_and_match_actual_masks(self):
        tiles,board=tetrahedron_tiles();problem=SearchProblem(tiles,board);helper=ConnectionBranching(problem)
        rng=random.Random(17);found=0
        for _ in range(100):
            domains=[sum(1<<c for c in row if rng.random()<.8) for row in problem.domains]
            stats=dict.fromkeys(STAT_KEYS,0);plan=helper.choose(domains,0,lambda:None,stats)
            if plan is None:continue
            found+=1;seen=set()
            for pattern,mask in zip(plan.patterns,plan.masks):
                members={c for c in range(3*problem.n) if mask>>c&1}
                self.assertTrue(members);self.assertFalse(seen&members);seen|=members
                self.assertTrue(all(int(problem.masks[c,plan.side])==pattern for c in members))
            self.assertEqual(seen,{c for c in range(3*problem.n) if domains[plan.cell]>>c&1})
        self.assertGreater(found,0)

    def checkpoint(self):
        problem=SearchProblem(simple_tiles(),double_face());saved=solve(problem,seconds=0)['checkpoint']
        saved.pop('root_symmetry',None)
        saved['stack']=[{'domains':['0x3f','0x3f'],'cell':0,'choices':[0,3],'choice_masks':['0x7','0x38'],'next':1},
            {'domains':['0x7','0x3f'],'cell':1,'choices':[0,2],'choice_masks':['0x3','0x3c'],'next':2},
            {'domains':['0x7','0x3c'],'cell':-1,'choices':[],'next':0}]
        saved['digest']=_checkpoint_digest(saved)
        return problem,saved

    def test_partition_pending_mask_branches_exactly_once(self):
        problem,saved=self.checkpoint()
        roots=partition_checkpoint(problem,saved,parts=7)
        expected=set(itertools.product(range(3,6),range(6)))|set(itertools.product(range(3),range(2,6)))
        seen=set()
        for root in roots:
            domains=[int(v,16) for v in root['stack'][0]['domains']]
            values=[tuple(c for c in range(6) if mask>>c&1) for mask in domains]
            actual=set(itertools.product(*values));self.assertFalse(seen&actual);seen|=actual
        self.assertEqual(seen,expected)

    def test_resume_keeps_grouped_choices_when_option_is_disabled(self):
        problem,saved=self.checkpoint()
        for enabled in (False,True):
            checked=solve(problem,seconds=0,resume=saved,options={'connection_branching':enabled})['checkpoint']
            self.assertEqual(saved['stack'],checked['stack'])

    def test_corrupt_grouped_checkpoints_are_rejected(self):
        problem,saved=self.checkpoint()
        for mutation in ('missing','overlap','representative','empty','child','extra'):
            bad=copy.deepcopy(saved);frame=bad['stack'][0]
            if mutation=='missing':frame['choice_masks'][-1]='0x18'
            elif mutation=='overlap':frame['choice_masks'][-1]='0x3c'
            elif mutation=='representative':frame['choices'][0]=1
            elif mutation=='empty':frame['choice_masks'][0]='0x0'
            elif mutation=='child':bad['stack'][1]['domains'][0]='0x8'
            else:frame['choice_masks'].append('0x1')
            bad['digest']=_checkpoint_digest(bad)
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):solve(problem,seconds=0,resume=bad)

    def test_native_conversion_preserves_every_mask_and_cursor(self):
        frames=[{'domains':[63,63],'cell':0,'choices':[0,3],'choice_masks':[7,56],'next':1},
                {'domains':[7,63],'cell':1,'choices':[4,2,0,3,5,1],'next':2},
                {'domains':[7,4],'cell':-1,'choices':[],'next':0}]
        self.assertEqual([from_native_frame(to_native_frame(f)) for f in frames],frames)

    def test_cancellation_during_selection_does_not_mutate_domains(self):
        problem=SearchProblem(simple_tiles(),double_face());domains=[63,63];before=domains.copy()
        def stop():raise RuntimeError('stop')
        with self.assertRaisesRegex(RuntimeError,'stop'):ConnectionBranching(problem).choose(domains,0,stop,dict.fromkeys(STAT_KEYS,0))
        self.assertEqual(domains,before)

if __name__=='__main__':unittest.main()
