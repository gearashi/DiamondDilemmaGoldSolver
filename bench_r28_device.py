"""Compare native support work and checkpoint transfers on identical CUDA cubes."""
import argparse
import json
from pathlib import Path
import numpy as np
from constraint_work import WorkScope
from gpu_domain_round24 import DomainCUDA
from gpu_domain_runner import split_cubes
from search_problem import SearchProblem
from io_utils import atomic_json


def compare(problem, lanes=32, dispatches=3):
    cubes = split_cubes([WorkScope.for_problem(problem, "gold").root],lanes)
    records = []
    states = []
    for sharing in (False, True):
        engine = DomainCUDA(problem,cubes,support_metrics=True,cross_lane=sharing)
        try:
            base = engine.checkpoint()
            frozen = engine.incremental_snapshot()
            assert frozen['state_sha256'] == base['state_sha256']
            results = []
            for i in range(dispatches):
                engine.step(1024)
                snap = engine.incremental_snapshot()
                assert snap['state_sha256'] == engine.checkpoint()['state_sha256']
                stats = engine.stats(snapshot=snap['state'])
                state = snap['state'].copy()
                start = engine.offsets['support_counters']
                state[start:start+14] = 0
                import hashlib
                results.append(hashlib.sha256(state.tobytes()).hexdigest())
            mirror = engine._page_mirror
            assert base['state_sha256'] == frozen['state_sha256']
            engine.restore(base)
            assert engine.incremental_snapshot()['state_sha256'] == base['state_sha256']
            records.append({'cross_lane':sharing,'lanes':lanes,'dispatches':dispatches,'decisions':stats['decisions'],
                            'support_checks':stats['support_checks'],'support_words':stats['support_word_scans'],
                            'device_transfer_bytes':mirror.transferred if mirror else None,
                            'full_transfer_bytes':(dispatches+2)*engine.state_bytes})
            states.append(results)
        finally:
            engine.close()
    if states[0] != states[1]:
        raise RuntimeError('Cross-lane search state differs from independent lanes')
    return records


def donation_restore():
    from unittest.mock import patch
    import benchmark_r27_cuda as bench

    class Checked(DomainCUDA):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs, cross_lane=True)
            self.incremental_snapshot()
        def checkpoint(self):
            snap = self.incremental_snapshot()
            control = super().checkpoint()
            if snap['state_sha256'] != control['state_sha256']:
                raise RuntimeError('Donated native state differs from full snapshot')
            return snap

    with patch.object(bench,'DomainCUDA',Checked):
        return [bench.donation(share) for share in (False,True)]


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    problem=SearchProblem(json.loads(a.tiles.read_text(encoding='utf-8-sig')))
    results=[row for lanes in (1,8,32,64) for row in compare(problem,lanes)]
    results = dict(dispatches=results, donation=donation_restore())
    atomic_json(a.output,results)
    print(json.dumps(results))
