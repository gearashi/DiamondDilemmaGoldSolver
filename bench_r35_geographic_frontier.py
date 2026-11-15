"""Seek a boundary witness with simultaneously embeddable retained pairs for one sign phase."""
import argparse
import json
from pathlib import Path

from bench_r35_geographic_loss_cores import price_and_separate
from bench_r35_loss_boundary344 import problem_scope
from io_utils import atomic_json
from r28_geo_sewing import PosePacking
from r35_geographic_packing_check import check_packing
from r35_loss_witness import physical_price


def search(tiles, claims, output, phase, limit):
    p, ref, groups, req = problem_scope(tiles)
    source = json.loads(claims.read_bytes())
    if source['puzzle'] != p.fingerprint:
        raise ValueError('Different physical puzzle.')
    cores = [frozenset(map(tuple, g)) for g in source['geographic_cores']]
    original = len(cores)
    packing = PosePacking(p)
    entries = []
    def save(rounds):
        atomic_json(output, dict(puzzle=p.fingerprint, phase=hex(phase), initial=original,
                                 boundaries=entries, active_cuts=len(rounds),
                                 new_geographic_cores=[sorted(g) for g in cores[original:]]))
    while len(cores) - original < limit:
        price, _ = physical_price(p.tiles, ref, [*groups, *(c-req for c in cores)], req, phase)
        result = price_and_separate(p, ref, groups, req, phase, price, packing, cores,
                                    disconnected=True, max_rounds=limit-(len(cores)-original), checkpoint=save)
        row = dict(start=price, exact_cost=result['minimum'], status=result['status'], cuts=len(result['rounds']))
        if 'witness_departures' in result:
            departing = set(map(tuple, result['witness_departures']))
            check = check_packing(p, (ref-departing)|req)
            if result['status'] == 'pose packing feasible' and check['feasible'] is not True:
                raise ArithmeticError('A physical retained-pair witness did not independently pack.')
            row['packing']=check
            row['lost']=sorted(departing)
        entries.append(row)
        save([])
        if result['status'] != 'all losses at cap excluded':
            return
    save([])


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True)
    cli.add_argument('--claims',required=True,type=Path)
    cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--phase',required=True,type=lambda v:int(v,0))
    cli.add_argument('--limit',default=48,type=int)
    args=cli.parse_args();search(args.tiles,args.claims,args.output,args.phase,args.limit)
