"""Measure shared Gold repair obligations on the two published 207 boards and proper rotations."""
from collections import Counter
from contextlib import nullcontext
import argparse
import json
from pathlib import Path
from random import Random
from types import SimpleNamespace
from unittest.mock import patch

from constraint_repair_round24 import core_guided_repair
from constraint_symmetry import proper_rotations
from r35_repair_seeds import Repairs, active_catalog, archive_rows, certified_distance
from search_problem import SearchProblem
from validator import validate_arrangement


DOCS=Path(__file__).resolve().parent/'docs'


def identity(_problem, record):
    return (tuple(enumerate(record['codes'])),)


def counts(problem, seed):
    rows=set()
    archive_rows(SimpleNamespace(p=problem,codes=seed,check=lambda:None,
                                 add=lambda row,_kind:rows.add(frozenset(row))))
    lower,hint=certified_distance(problem,seed)
    return lower,len(hint) if hint is not None else None,rows


def bounded(data,seed,fixed,old):
    problem=SearchProblem(data,fixed={c:seed[c] for c in fixed},physical_gold_support=True)
    work=Counter()
    def count(method):
        original=getattr(Repairs,method)
        def wrapper(self,*args,**kwargs):
            work[method]+=1
            return original(self,*args,**kwargs)
        return patch.object(Repairs,method,wrapper)
    scope=patch('r35_repair_seeds.catalog_images',identity) if old else nullcontext()
    with scope,count('add'),count('neighbors'),count('projected'):
        result=core_guided_repair(problem,seed,max_queries=1,seconds=15)
    return {'status':result['status'],'reason':result.get('reason'),'lower':result.get('certified_seed_distance'),
            'oracle_queries':result['stats']['queries'],'master_calls':result['stats']['hitting_set_solves'],
            'row_candidates':work['add'],'neighbor_passes':work['neighbors'],'bond_projections':work['projected']}


def measure(data):
    p=SearchProblem(data,physical_gold_support=True)
    ref=json.loads((DOCS/'R29_INCUMBENT.json').read_bytes())['codes']
    alt=json.loads((DOCS/'R29_GOLD_RESULTS.json').read_bytes())['residual_reference']
    rotations=proper_rotations(p.board)
    result={'input':p.fingerprint,'centers':{},'end_to_end':{}}
    for label,seed in (('reference',ref),('alternate',alt)):
        validated=validate_arrangement(data,seed)
        expected,hint,base=counts(p,seed)
        cases=[]
        for rotation in rotations:
            moved=rotation.apply(seed)
            lo,proposed,rows=counts(p,moved)
            with patch('r35_repair_seeds.catalog_images',identity):
                prior,_,old_rows=counts(p,moved)
            expected_rows={frozenset(rotation.cells[c] for c in row) for row in base}
            assert rows==expected_rows and (lo,proposed)==(expected,hint)
            cases.append(dict(old_lower=prior,lower=lo,old_rows=len(old_rows),rows=len(rows)))
        result['centers'][label]=dict(score=validated['matched_edges'],components=validated['line_components'],rotations=cases)
    # The exact capped Hamming neighborhood holds 68 poses and can change at most 92.
    seed=rotations[5].apply(alt)
    lower,hit=certified_distance(p,seed)
    fixed=sorted(set(range(p.n))-hit)+[min(hit)]
    assert (p.n-len(fixed),lower)==(92,93)
    result['end_to_end']={label:bounded(data,seed,fixed,old) for label,old in (('identity',True),('all_rotations',False))}
    # A far-away permutation seldom has even two aligned reference poses per image.
    far=ref[:]
    Random(289).shuffle(far)
    lower,_,rows=counts(p,far)
    result['far']={'seed':289,'lower':lower,'rows':len(rows)}
    catalog=active_catalog(p);source_rows=[*catalog['rows'],*catalog['post_master']]
    result['archive_rows']=len(source_rows)
    for name,seed in (('reference',rotations[5].apply(ref)),('alternate',rotations[5].apply(alt)),('far',far)):
        visits=0
        for r in rotations:
            image=r.apply(ref)
            agree={c for c,d in enumerate(r.cells) if seed[d]==image[d]}
            visits+=sum(row[0] in agree for row in source_rows)
        result.setdefault('indexed_candidate_rows',{})[name]=visits
    return result


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',type=Path,required=True)
    args=cli.parse_args()
    print(json.dumps(measure(json.loads(args.tiles.read_bytes())),indent=2))
