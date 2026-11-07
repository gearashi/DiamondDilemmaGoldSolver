import copy,json,random,unittest
from constraint_affine_reasons_round26 import produce_affine_conflict,verify_affine_conflict
from constraint_algebra import AffinePruner
from constraint_deduction_chain_round26 import verify_chain,ChainError,FORMAT
from test_constraint_sat_domains import fixture,admitted
from test_constraint_dfs import brute

class AffineReasonTests(unittest.TestCase):
    def test_random_actual_conflicts_equal_affine_basis_and_preserve_all_completions(self):
        p,_=fixture();solutions=brute(p,'edges');rng=random.Random(260408);found=[]
        for _ in range(120):
            domains=[sum(1<<c for c in row if rng.random()<.35) or 1<<row[0] for row in p.domains]
            expected=AffinePruner(p)._build(domains,lambda:None) is None;certificate=produce_affine_conflict(p,domains)
            self.assertEqual(certificate is not None,expected)
            if certificate is not None:
                self.assertFalse(admitted(solutions,domains));self.assertTrue(verify_affine_conflict(p,domains,json.loads(json.dumps(certificate))))
                found.append((domains,certificate))
        self.assertTrue(found)
        pruner=AffinePruner(p,explain=True);current=found[0][0].copy();self.assertTrue(pruner.prune(current));exported=list(pruner.drain_deduction_chains(target='edges'));self.assertTrue(exported)
        for exported_chain in exported:verify_chain(p,json.loads(json.dumps(exported_chain)),target='edges')
        self.assertGreater(pruner.reason_stats['affine_reason_equations'],0)
        domains,certificate=found[0];chain={'format':FORMAT,'fingerprint':p.fingerprint,'target':'edges',
          'original':[hex(sum(1<<c for c in row)) for row in p.domains],'root':[hex(v) for v in domains],
          'decisions':[],'events':[],'terminal':{'rule':'affine','data':{'certificate':certificate}}}
        self.assertTrue(verify_chain(p,chain,target='edges'))
        bad=copy.deepcopy(chain);bad['terminal']['data']['certificate']['equations'][0][2]^=1
        with self.assertRaises(ChainError):verify_chain(p,bad,target='edges')
        bad=copy.deepcopy(chain);bad['terminal']['data']['certificate']['equations'][0][2]=True
        with self.assertRaises(ChainError):verify_chain(p,bad,target='edges')
        self.assertIsNone(produce_affine_conflict(p,domains,max_equations=0))
        def stop():raise ValueError('caller cancelled')
        with self.assertRaisesRegex(ValueError,'caller cancelled'):verify_affine_conflict(p,domains,certificate,stop)

if __name__=='__main__':unittest.main()
