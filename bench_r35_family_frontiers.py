"""Measure repeated local DP states in authenticated and held-out physical loss queries."""
import argparse
from collections import Counter
from contextlib import ExitStack
from functools import lru_cache
import hashlib
import json
from math import inf
from pathlib import Path
from unittest.mock import patch

from bench_r35_geographic_neighbors import four_flip_neighbors
from bench_r35_loss_boundary344 import problem_scope
from io_utils import atomic_json
from r32_gold_loss import read_json
import r35_family_frontiers as tables
from r35_geographic_replay import geographic_departures
import r35_mate_price as pricing


def measure(problem, reference, groups, required, phases, seed, cap, mode):
    counts=Counter()
    caches=(tables._physical_exact,tables._local_exact,tables._original_relaxed,tables._local_relaxed)
    for cache in caches:
        cache.cache_clear()
    cached_exact=tables.exact_frontier
    raw=pricing._matching_prices

    def matching(*args):
        counts['exact_family_searches']+=1
        tracked=[]
        def observed(size):
            def decorate(fn):
                wrapped=lru_cache(size)(fn)
                tracked.append(wrapped)
                return wrapped
            return decorate
        with patch.object(pricing,'lru_cache',observed):
            result=raw(*args)
        counts['visited_matching_subsets']+=sum(f.cache_info().misses for f in tracked)
        return result

    def exact(solver, family, choices, named, covers):
        if mode=='cached':
            result=cached_exact(solver,family,choices,named,covers)
        else:
            result=tuple(solver(family,choices,named,covers).items())
        counts['requested_exact_families']+=1
        counts['neutral_families']+=result==((0,0),)
        return result

    def relaxed(solver,named,covers):
        if mode=='cached':
            before=tables._original_relaxed.cache_info()
            result=tables.relaxed_frontier(solver,named,covers)
            if tables._original_relaxed.cache_info()==before:
                counts['direct_relaxed_searches']+=1
        else:
            result=tuple(solver(named,covers).items())
            counts['direct_relaxed_searches']+=1
        counts['requested_relaxed_families']+=1
        counts['neutral_families']+=result==((0,0),)
        return result

    with ExitStack() as patches:
        patches.enter_context(patch.object(pricing,'_matching_prices',matching))
        patches.enter_context(patch.object(pricing,'exact_frontier',exact))
        patches.enter_context(patch.object(tables,'exact_frontier',exact))
        patches.enter_context(patch.object(pricing,'relaxed_frontier',relaxed))
        model=pricing.ClassPrices(problem,reference,groups,required,limit=8)
        fixed=[model.memo(phase) for phase in phases]
        mask=model.conflict(seed,cap,float('inf'))
    counts['evaluated_relaxed_tables']=counts['direct_relaxed_searches']+tables._local_relaxed.cache_info().misses
    fixed=['no matching' if x==inf else x for x in fixed]
    return dict(counts=dict(counts),cache_sizes=[f.cache_info().currsize for f in caches],
                results=fixed,pin_mask=hex(mask),pin_count=mask.bit_count())


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args()
    root=Path(__file__).resolve().parent
    p,ref,groups,req=problem_scope(args.tiles)
    data=read_json(root/'docs/R35_GEOGRAPHIC_LOSS_CORES.json')
    extra=list(geographic_departures(p,ref,req,data['geographic_cores']))
    phase=int(read_json(root/'docs/R35_NEAR_REFERENCE_TOURS.json')['physical']['phase'],16)
    others={phase}
    for case in data['reports']:
        donor=int(case['phase'],16)
        _,neighbors=four_flip_neighbors(p,req,donor,100000)
        others.update([donor,*neighbors])
    phases=sorted(others)
    report=dict(fingerprint=p.fingerprint,held_out_count=len(phases),phases_sha256=hashlib.sha256(
        json.dumps([hex(v) for v in phases]).encode()).hexdigest(),scopes={})
    for name,claims,cap in (('13',groups[:13],23),('13+2',groups[:13]+extra,24),('344+2',groups+extra,24)):
        old=measure(p,ref,claims,req,phases,phase,cap,'uncached')
        new=measure(p,ref,claims,req,phases,phase,cap,'cached')
        if old['results']!=new['results'] or old['pin_mask']!=new['pin_mask']:
            raise ArithmeticError('Local table reuse changed a phase price or the complete proof minimization.')
        result_hash=hashlib.sha256(json.dumps(old.pop('results')).encode()).hexdigest()
        new.pop('results')
        report['scopes'][name]=dict(uncached=old,cached=new,all_phase_prices_sha256=result_hash)
        atomic_json(args.output,report)
        print(name,len(phases),old['counts']['visited_matching_subsets'],new['counts']['visited_matching_subsets'],flush=True)


if __name__=='__main__':
    main()
