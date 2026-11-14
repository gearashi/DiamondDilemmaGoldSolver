"""Reproduce the three complete shortest-route screens and the next-distance controls."""
import argparse
from pathlib import Path
import time

import numpy as np

from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, read_json
from r28_exchange_rotations import RotationOracle
from r29_exchange_clusters import StarClusters
from r29_exchange_route import ROOT, domain_binding, rotation_optimum, run
from search_problem import SearchProblem


def benchmark(problem,database,batch=128):
    record = read_json(ROOT/'docs'/'R29_EXCHANGE_CENTERS.json')
    model = BoardModel(problem)
    if domain_binding(model)!=record['domain']:
        raise ValueError('These published physical centers require the original Gold dataset and domains.')
    centers = [record['centers'][i] for i in (1,0,2)]
    result = dict(triples=[],separated=[],support_samples=[],four_swap=[])
    for row in centers:
        codes = [3*t for t in row['pieces']]
        bound = ExchangeBound(model,codes)
        clusters = StarClusters(bound)
        oracle = RotationOracle(model)
        scores = []
        def exact(seed):
            ans = rotation_optimum(bound,oracle,seed)
            scores.append(ans['score'])
            return ans
        cert = clusters.separated_certificate(208,exact)
        if cert!=row['separated']:
            raise RuntimeError('Recomputed assignment certificate differs from release data.')
        result['separated'].append(dict(sha256=row['sha256'],exact=scores,upper_twice=cert['upper_twice']))
        before = time.monotonic()
        work = run(problem,codes,208,database,batch=batch)
        result['triples'].append(dict(work,wall_seconds=time.monotonic()-before))
    codes = [3*t for t in centers[0]['pieces']]
    bound = ExchangeBound(model,codes)
    clusters = StarClusters(bound)
    rng = np.random.default_rng(16)
    for size in (3,4,6,7,8,10,15):
        closed = [0,0]
        for i in range(128):
            if i<64:
                cells = tuple(rng.choice(problem.n,size,replace=False))
            else:
                cells = [int(rng.integers(problem.n))]
                while len(cells)<size:
                    next_cells = sorted({z for c in cells for z in bound.neighbors[c]}-set(cells)-{problem.n})
                    cells.append(int(rng.choice(next_cells)))
            closed[int(i>=64)] += int(clusters.support_screen(cells,208))
            order = rng.permutation(size)
            if clusters.price(cells,order)//2!=bound.bounds([cells],order)[0]:
                raise RuntimeError('Incident-star correction differs from full local recomputation.')
        result['support_samples'].append(dict(size=size,each_tested=64,random_closed=closed[0],connected_closed=closed[1]))
    for enabled in (False,True):
        # Both controls use ordinary colex order by disabling previous release
        # proofs here, so the exact same twelve supports and 2,520 endpoints compare.
        result['four_swap'].append(run(problem,codes,208,':memory:',size=7,distance=4,backend='cpu',
                                       batch=12,max_pages=1,cluster_screen=enabled,reuse_balls=False))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tiles',default='data/tiles.json')
    p.add_argument('--database',required=True,help='Use a fresh path for cold-work ownership counts.')
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    atomic_json(args.output,benchmark(SearchProblem(read_json(args.tiles)),args.database))
