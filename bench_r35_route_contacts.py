"""Compare exact wildcard support screening against the released colex intervals."""
from collections import Counter
from pathlib import Path
from time import monotonic
import argparse
from math import comb
from io_utils import atomic_json
from r27_incumbent import BoardModel
from r28_exchange import ExchangeBound, read_json
from r29_exchange_clusters import StarClusters
from r30_cycles import certified_contacts
from r30_bounds import budget_upper, defect_packings, defect_cores
from r31_route_geometry import proved_connected
from r32_route_ranges import PrefixRanges
from search_problem import SearchProblem


def run(tiles, path, ranks, heldout=0):
    p = SearchProblem(read_json(tiles))
    model = BoardModel(p)
    archive = read_json('docs/R29_EXCHANGE_CENTERS.json')['centers']
    bs = [ExchangeBound(model, [3*t for t in entry['pieces']]) for entry in archive]
    close = StarClusters(bs[0]).close
    prep = [defect_packings(b) for b in bs]
    cores = [defect_cores(b,208,pack) for b,pack in zip(bs,prep)]
    caps = [budget_upper(b,pack,4) for b,pack in zip(bs,prep)]
    out = dict(puzzle=p.fingerprint, ranks=ranks, trials=[])
    for family,starts in [((2,2,3),(200000000000,301414649175,390001234567)),
                          ((2,2,2,2),(5360481967048,1014012345678,7000012345678))]:
        required = [certified_contacts(b,208,[family]) for b in bs]
        geo = [{family} if proved_connected(b,208,family) else set() for b in bs]
        free = tuple(sorted(model.free))
        if heldout:
            import random
            rng=random.Random(3516+sum(family))
            starts=[rng.randrange(comb(len(free),sum(family))-ranks) for _ in range(heldout)]
        for start in starts:
            trial = dict(family=family,start=start,ranks=ranks,variants={})
            for mode in (False,True,18,32,64,160):
                engine = PrefixRanges(free,close,cores,required,geo,(family,),caps,208,joint=bool(mode),joint_four=mode if type(mode) is int else 0)
                work = Counter()
                t = monotonic()
                intervals = []
                for offset in range(start,start+ranks,256):
                    intervals.extend(engine.spans(sum(family),offset,min(offset+256,start+ranks),work))
                trial['variants'][str(mode)] = dict(work=dict(work),
                    open_ranks=sum(b-a for a,b in intervals),seconds=round(monotonic()-t,3),
                    memo=engine._joint.cache_info()._asdict())
            out['trials'].append(trial)
            print(family,start,trial['variants'],flush=True)
            atomic_json(path,out)


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',default='../data/tiles.json')
    cli.add_argument('--ranks',type=int,default=4096)
    cli.add_argument('--output',type=Path,required=True)
    cli.add_argument('--heldout',type=int,default=0)
    args=cli.parse_args()
    run(args.tiles,args.output,args.ranks,args.heldout)
