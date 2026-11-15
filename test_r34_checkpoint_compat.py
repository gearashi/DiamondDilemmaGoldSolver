"""Replay a nonempty checkpoint collected from the v1.14.0 crossover controller."""
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from r27_incumbent_compat import compatible_sources, legacy_sources
from sampling_crossover_round26 import OptimizingCrossover, _digest
from test_gpu_recombination_round26 import complementary_fixture


class HeuristicMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record=json.loads((Path(__file__).parent/'docs/R34_V114_CROSSOVER.json').read_text(encoding='utf-8'))

    def test_published_nonempty_native_and_uniform_layouts_continue_exactly(self):
        recent=json.loads((Path(__file__).parent/'docs/R35_V115_CROSSOVER.json').read_bytes())
        for release,record in (('v114',self.record),('v115',recent)):
            for layout,headers in record['profiles'].items():
                with self.subTest(release=release,layout=layout):
                    p,archive,_=complementary_fixture()
                    saved={**deepcopy(self.record['saved']),**headers}
                    controller=OptimizingCrossover(p,gold_aware=True,max_nodes=1,resume=saved)
                    self.assertEqual(controller._identity(release+'-'+layout),headers['binding'])
                    self.assertEqual(controller.stats['recombination_nodes'],1)
                    self.assertTrue(controller.models)
                    for candidate,solution,nodes in self.record['following']:
                        result=controller.propose(archive)
                        self.assertEqual(result['candidate'],candidate)
                        self.assertEqual(result['solution'],solution)
                        self.assertEqual(controller.stats['recombination_nodes'],nodes)
                    self.assertTrue(p.validate(result['solution'])['valid'])

    def test_migration_rejects_changed_policy_domain_or_source(self):
        p,_,_=complementary_fixture()
        snapshot=self.record['saved']
        with self.assertRaises(ValueError):
            OptimizingCrossover(p,gold_aware=True,max_nodes=2,resume=snapshot)
        fake={**snapshot,'binding':'c'*64}
        fake['digest']=_digest({k:v for k,v in fake.items() if k!='digest'})
        with self.assertRaises(ValueError):
            OptimizingCrossover(p,gold_aware=True,max_nodes=1,resume=fake)
        p.domains[0]=p.domains[0][:-1]
        with self.assertRaises(ValueError):
            OptimizingCrossover(p,gold_aware=True,max_nodes=1,resume=snapshot)
        names=('constraint_work.py','r27_structure_filters.py','r32_gold_bonds.py','search_options.py','r34_coverage_index.py')
        current={name:sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in names}
        old=legacy_sources(current,'v114-git')
        self.assertEqual(set(old),set(names[:-1]))
        self.assertTrue(compatible_sources(old,current))
        for name in names:
            modified={**current,name:'e'*64}
            self.assertIsNone(legacy_sources(modified,'v114-git'))
            self.assertFalse(compatible_sources(old,modified))
        for missing in names[:-1]:
            self.assertFalse(compatible_sources({k:v for k,v in old.items() if k!=missing},current))

    def test_foreign_option_edit_does_not_receive_the_reviewed_alias(self):
        import r27_incumbent_compat as compat
        name='search_options.py'
        reviewed=(Path(__file__).parent/name).read_bytes()
        changed=reviewed.replace(b'lookahead_ms', b'foreign_lookahead_ms')
        self.assertNotEqual(changed,reviewed)
        historical=legacy_sources({name:sha256(reviewed).hexdigest()},'v114-git')
        self.assertIsNotNone(historical)
        with TemporaryDirectory() as folder:
            (Path(folder)/name).write_bytes(changed)
            with patch.object(compat,'__file__',str(Path(folder)/'compat.py')):
                current={name:sha256(changed).hexdigest()}
                self.assertIsNone(legacy_sources(current,'v114-git'))
                self.assertFalse(compatible_sources(historical,current))


if __name__=='__main__':
    unittest.main()
