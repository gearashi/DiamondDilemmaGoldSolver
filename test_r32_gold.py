"""Independent finite comparisons for Gold side projection and loss-budget ownership."""
from itertools import combinations, product
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from importlib.util import find_spec

from r27_structure_fixtures import planted, independent
from r31_geographic import physical_bonds
from r32_gold_bonds import endpoints, inventory, loop_groups, bounded_loop_cores
from r32_gold_loss import (Certificates, dropout, first_loss_partition, geographic, master, pose_lower,
                           reference_groups)
from r32_gold_syndrome import OutsideSyndromes, ranked_combinations
from r32_gold_watch import BondWatcher, placed_bonds, verified_placement
from constraint_shared_conflicts import VerifiedConflictStore


def exhaustive_cost(p, reference, groups, required, outside, errors):
    # Direct phase assignments; no dynamic programming, projected rows or search oracle.
    from r29_gold_phase import EndpointCuts, _reciprocal
    from collections import Counter
    oriented=EndpointCuts(p).sides
    best=None
    for x in product((0,1),repeat=p.n):
        labels=[(m,c^x[i//3]) for i,(m,c) in enumerate(oriented)]
        counts=Counter(labels)
        # Each reciprocal pair has equal counts modulo two. A singleton signed
        # class can only be consumed in pairs, hence its own count must be even.
        if any(((n + counts.get(_reciprocal(m,c),0))%2 if _reciprocal(m,c)!=(m,c) else n%2)
               for (m,c),n in counts.items()):continue
        def broken(b):return labels[b[0]]!=_reciprocal(*labels[b[1]])
        if any(broken(b) for b in required):continue
        loss={b for b in reference if broken(b)}
        if len(loss & set(outside))!=errors:continue
        minimum=len(loss)+sum(not g&loss for g in groups)
        best=minimum if best is None else min(best,minimum)
    return best


class GoldTests(unittest.TestCase):
    def test_rank_seek(self):
        for n in range(1,9):
            for k in range(n+1):
                plain=list(combinations(range(n),k))
                for start in [0,len(plain)//2,len(plain)]:
                    self.assertEqual([tuple(row) for _,row in ranked_combinations(n,k,start,len(plain))],
                                     plain[start:])

    def test_physical_mate_pruning_only_removes_nongold(self):
        for seed in (11,17,23):
            for split in (True,False):
                p,ds=planted(1,3,seed,split,True)
                rules=inventory(p)
                for solution in independent(p,ds):
                    actual=set(physical_bonds(p,solution))
                    self.assertTrue(rules.required<=actual<=rules.possible)
                    self.assertFalse(actual & rules.loops)
                self.assertIs(rules,inventory(p))

    def test_sewing_certificate_transport_and_guarded_loss(self):
        p,ds=planted(1,0,23,True,True)
        codes=[3*i for i in range(p.n)]
        bonds=physical_bonds(p,codes)
        cert=Certificates(p)
        loops=loop_groups(cert.arcs,cert.sides,bonds,p.total_segments)
        self.assertTrue(loops)
        group=min(loops,key=len)
        cells=placed_bonds(p,enumerate(codes))
        witness=sorted({pose for b in group for pose in cells[b]})
        self.assertTrue(verified_placement(p,group,witness))
        store=VerifiedConflictStore(p,'gold')
        self.assertTrue(store.admit_physical_conflict(group,witness))
        self.assertFalse(store.admit_physical_conflict(group,witness))
        with self.assertRaises(ValueError):store.admit_physical_conflict(group,[(0,9999)])
        self.assertFalse(VerifiedConflictStore(p,'edges').admit_physical_conflict(group,witness))
        watch=BondWatcher(p,[group])
        domains=list(ds)
        self.assertTrue(watch.prune(domains))
        self.assertIn(0,domains)
        # Free a single tile in a wider context; its forbidden realization has no support.
        domains=list(ds)
        c,v=witness[0]
        domains[c]=(1<<(3*p.n))-1
        self.assertTrue(watch.prune(domains))
        self.assertFalse(domains[c]>>v&1)

    def test_short_loop_probes_respect_positive_controls(self):
        p,ds=planted(1,3,23,False,True)
        required=inventory(p).required
        clauses,checks=bounded_loop_cores(p,required,500)
        self.assertGreater(checks,0)
        for board in independent(p,ds):
            actual=set(physical_bonds(p,board))
            self.assertFalse(any(c<=actual for c in clauses))

    def test_recursive_splits_have_exactly_one_home(self):
        a,b,c,d=(0,3),(1,4),(2,5),(6,9)
        groups=[{a,b,c},{b,d}]
        def leaves(path):
            node=first_loss_partition(groups,path=path)
            if node is None:return []
            kept,lost,options=node
            if not options:return [(set(kept),set(lost))]
            return sum((leaves([*path,i]) for i in range(len(options))),[])
        branches=leaves([])
        for bits in product((0,1),repeat=4):
            preserved={p for p,v in zip((a,b,c,d),bits) if v}
            self.assertEqual(sum(kept<=preserved and not lost&preserved for kept,lost in branches),
                             int(not any(g<=preserved for g in groups)))

    def test_projected_syndromes_against_all_phase_assignments(self):
        for split in [True,False]:
            p,_=planted(1,2,35,split,True)
            centers=[[3*i for i in range(p.n)], [3*((i+3)%p.n) for i in range(p.n)]]
            for board in centers:
                ref,cert,groups=reference_groups(p,board)
                required=inventory(p).required
                q=OutsideSyndromes(p,ref,groups,required)
                for err in range(min(2,len(q.outside))+1):
                    expected=exhaustive_cost(p,ref,q.groups,required,q.outside,err)
                    for cap in (len(q.groups)+err,len(q.groups)+err+1):
                        for method in (q.indexed,q.check):
                            out=method(cap,err,5)
                            self.assertEqual(out['status']=='PHASE_RELAXATION_FEASIBLE',
                                             expected is not None and expected<=cap)

    @unittest.skipUnless(find_spec('ortools'),'Optional OR-Tools')
    def test_loss_relaxations_and_saved_geographic_decisions(self):
        p,ds=planted(1,0,23,False,True)
        board=[3*i for i in range(p.n)]
        ref,c,g=reference_groups(p,board)
        for st in ('xor','integer','euler','mating'):
            out=dropout(p,ref,0,g,3,st,inventory(p).required,(),ds)
            self.assertEqual(out['relaxation_cost'],0)
        self.assertEqual(pose_lower(p,ref,g,board,2)['lower'],0)
        self.assertEqual(master(p,ref,c,0,3,domains=ds)['status'],'FEASIBLE_SEWING_RELAXATION')
        with TemporaryDirectory() as d:
            name=str(Path(d)/'geo.sqlite')
            first=geographic(p,ref,g,c.rows,[],0,0,None,4,name,ds)
            self.assertEqual(first['status'],'GOLD')
            self.assertTrue(p.validate(first['codes'])['valid'])
            self.assertTrue(geographic(p,ref,g,c.rows,[],0,0,None,4,name,ds)['reused'])

    def test_reopened_complete_prefixes(self):
        from r32_gold import run_syndrome
        p,_=planted(1,3,23,True,True)
        ref,c,g=reference_groups(p,[3*i for i in range(p.n)])
        with TemporaryDirectory() as d:
            path=str(Path(d)/'loss.sqlite')
            a=run_syndrome(p,ref,g,0,[0],path,3,page=3)
            b=run_syndrome(p,ref,g,0,[0],path,3,page=4)
            self.assertEqual(a['status'],'INFEASIBLE')
            self.assertEqual(b['new_pages'],0)


if __name__=='__main__':unittest.main()
