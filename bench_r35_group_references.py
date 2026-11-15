"""Check wider certified departure references and the scope for boundary-table follow-ups."""
import argparse
from collections import Counter
from math import inf
from pathlib import Path
from random import Random
import json
import sqlite3
from time import perf_counter, monotonic

from r32_gold_bonds import inventory
from r32_gold_loss import read_json, reference_groups
from r32_gold_syndrome import OutsideSyndromes
from r35_group_images import GroupImages
from bench_r35_group_fallback import legacy_drops
from search_problem import SearchProblem


def sampled_patterns(query, seed, count):
    # Construct outside discriminators from labelled, parity-feasible tile
    # assignments. These are representative direct joint inputs, not claimed
    # arrivals from a sparse near-minimum-error layer.
    rng = Random(seed)
    patterns = []
    phase = 0
    for i in range(count):
        phase = rng.getrandbits(max((v[0].bit_length() for v in query.base.values()),default=0))
        for pivot, (row, rhs) in sorted(query.base.items()):
            if ((row & phase).bit_count() & 1) != rhs:
                phase ^= 1 << pivot
        errors = tuple(j for j,(row,rhs) in enumerate(query.out_eq) if ((row & phase).bit_count() & 1)!=rhs)
        patterns.append(errors)
    return patterns


def compare(problem, cpu, cuda, docs):
    evidence=[]
    required=inventory(problem).required
    records=read_json(docs/'R31_GOLD_RESULTS.json')['departures']
    for kind, path in [('cpu',cpu),('cuda',cuda)]:
        db=sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)
        try:
            for row in records['fresh-'+kind]:
                seed,score=row['seed'],row['score']
                saved=db.execute('select result from r30_starts where seed=? and score=?',(seed,score)).fetchone()
                if not saved:
                    raise ValueError('Required frozen seed/score unavailable')
                codes=json.loads(saved[0])['codes']
                ref,cert,groups=reference_groups(problem,codes,row['witnesses'])
                if any(not any(checked <= frozenset(tuple(sorted(b)) for b in c) for checked in cert.rows)
                       for c in row['witnesses']):
                    raise ValueError('Invalid frozen departure proof')
                q=OutsideSyndromes(problem,ref,groups,required)
                images=GroupImages.compile(q.groups,q.required,q._joint_bonds,len(q.outside),max_rank=14)
                if images is None:
                    raise ValueError('Unexpected high rank')
                tally=Counter()
                result=[]
                for flipped in sampled_patterns(q,211+seed,8):
                    minimum=images.minimum(flipped,inf)
                    if minimum==inf:
                        continue
                    threshold=minimum+len(flipped)
                    for cap in (threshold-1, threshold):
                        expected=cap>=threshold
                        start=perf_counter()
                        try:
                            actual=legacy_drops(q,flipped,cap,monotonic()+.15)
                        except TimeoutError:
                            actual=None
                        tally['drop_seconds']+=perf_counter()-start
                        if actual is not None and actual!=expected:
                            raise AssertionError((kind,seed,cap,expected,actual))
                        tally['drop_unknown' if actual is None else 'drop_completed']+=1
                        result.append(dict(errors=len(flipped),cap=cap,minimum=threshold,drop_result=actual))
                evidence.append(dict(backend=kind,seed=seed,score=score,group_count=len(q.groups),
                                     images=len(images.images), cases=result,work=dict(tally)))
                print(kind,seed,score,len(images.images),dict(tally),flush=True)
        finally:
            db.close()
    return evidence


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tiles',type=Path,default=Path('data/tiles.json'))
    parser.add_argument('--cpu',type=Path,required=True)
    parser.add_argument('--cuda',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=compare(SearchProblem(read_json(args.tiles)),args.cpu,args.cuda,Path(__file__).parent/'docs')
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
