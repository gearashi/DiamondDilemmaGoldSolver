"""Cross-check every retained geometric group and reprice the complete nearby phase family."""
import argparse
import json
from time import perf_counter
from pathlib import Path

from bench_r35_loss_boundary344 import problem_scope
from r32_gold_syndrome import OutsideSyndromes
from r35_geographic_replay import geographic_departures
from r35_loss_witness import physical_price
from r35_overlap_loss import OverlapPrices


def reprice(tiles,claims,donors,output):
    p,ref,groups,req=problem_scope(tiles)
    packed,_,outside=OutsideSyndromes.layout(ref,groups[:13],req)
    given=json.loads(claims.read_bytes())
    full=[frozenset(map(tuple,g)) for g in given['geographic_cores']]
    started=perf_counter()
    projected=geographic_departures(p,ref,req,full)
    elapsed=perf_counter()-started
    if any(not any(c<=g-req for c in projected) for g in full):
        raise ArithmeticError('One supplied core failed replay without a valid dominating subset.')
    allgroups=[*groups,*projected]
    prices=OverlapPrices(p,packed,allgroups,req,outside)
    previous=json.loads(donors.read_bytes())
    rows=[*previous['held_out'],*previous['phase_donors']]
    out=dict(fingerprint=p.fingerprint,claims=len(full),distinct_replayed=len(projected),public_replay_seconds=elapsed,prices=[])
    for row in rows:
        phase=int(row['phase'],16)
        weak,failed=physical_price(p.tiles,ref,groups,req,phase)
        strong=physical_price(p.tiles,ref,allgroups,req,phase)
        assert strong is not None
        fixed=tuple(i for i,bond in enumerate(outside) if bond in failed)
        assert prices.price(phase,fixed)==strong[0]
        out['prices'].append(dict(phase=hex(phase),physical=weak,initial_geo=row.get('geo344'),expanded=strong[0]))
        output.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument('--tiles',required=True)
    cli.add_argument('--claims',required=True,type=Path);cli.add_argument('--phases',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    args=cli.parse_args()
    reprice(args.tiles,args.claims,args.phases,args.output)
