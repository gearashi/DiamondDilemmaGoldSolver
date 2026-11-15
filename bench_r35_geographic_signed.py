"""Compare conditional geographic departures at fixed public signed SAT conflict budgets."""
import argparse
import json
from pathlib import Path
from random import Random

from bench_r35_geographic_loss_cores import connected_pose_count
from bench_r35_loss_boundary344 import problem_scope
from r31_gold_connectivity import PhaseInterface
from r32_gold_syndrome import OutsideSyndromes
from r35_geographic_replay import geographic_departures
from r35_integer_sat import InventorySAT
from r35_loss_witness import physical_price


def neighbors(p,required,phase,limit=12):
    """Independent eligible two-tile flips by exact integer/parity delta, then raw repricing."""
    interface=PhaseInterface(p)
    delta=[]
    for i,(low,high) in enumerate(interface.rows):
        sign=-1 if phase>>i&1 else 1
        delta.append(tuple(sign*(b-a) for a,b in zip(low[:-1],high[:-1])))
    pairs=[]
    for a in range(1,p.n):
        for b in range(a+1,p.n):
            if any(((x//3 in (a,b))!=(y//3 in (a,b))) for x,y in required): continue
            if all((x+y)%2==0 if mod else x+y==0 for x,y,mod in zip(delta[a],delta[b],interface.mod[:-1])):
                pairs.append((a,b))
    rng=Random(35080)
    rng.shuffle(pairs)
    return [phase^(1<<a)^(1<<b) for a,b in pairs[:limit]]


def replay(check,p,ref,groups,required,cap,pins,budget):
    check.solver.conf_budget(budget)
    verdict=check.solver.solve_limited(assumptions=pins)
    result=dict(verdict=verdict,**check.solver.accum_stats())
    if result['conflicts']>budget+max(16,budget//50):
        raise RuntimeError('Installed SAT solver exceeded its requested conflict limit.')
    if verdict:
        phase=check.phase()
        price=physical_price(p.tiles,ref,groups,required,phase)
        assert price is not None and price[0]<=cap
        result.update(phase=hex(phase),loss=price[0])
    return result


def run(tiles,small_budget,large_budget,output):
    p,ref,groups,required=problem_scope(tiles)
    raw=json.loads((Path(__file__).parent/'docs/R35_GEOGRAPHIC_LOSS_CORES.json').read_bytes())
    bases=tuple(int(row['phase'],16) for row in sorted(raw['reports'],key=lambda x:-x['cap']))
    cores=[frozenset(map(tuple,x)) for x in raw['geographic_cores']]
    assert all(connected_pose_count(p,c)==0 for c in cores)
    projected=geographic_departures(p,ref,required,cores)
    outside_bonds=set.union(*(set(c)-ref for c in cores))
    assert len(outside_bonds)==1 and not geographic_departures(p,ref,required-outside_bonds,cores)
    packed,_,outside=OutsideSyndromes.layout(ref,groups[:13],required)
    fixture=sorted(set(f for w in bases for f in neighbors(p,required,w)))
    prices=[]
    for phase in [*bases,*fixture]:
        row={'phase':hex(phase),'held_out':phase not in bases}
        for label,gs in [('base13',groups[:13]),('geo13',[*groups[:13],*projected]),
                         ('base344',groups),('geo344',[*groups,*projected])]:
            value=physical_price(p.tiles,ref,gs,required,phase)
            assert value is not None
            row[label]=value[0]
        prices.append(row)
    cases=[]
    receipt=dict(puzzle=p.fingerprint,native='cadical153',outside_condition=sorted(outside_bonds),prices=prices,sat=cases)
    output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    def compare(label,gs,cap,pins,budget):
        result=dict(control=label,cap=cap,pins=len(pins),budget=budget)
        for allow in (0,1):
            charged=[*gs,*(projected if allow else ())]
            extra=[g for g in charged if g not in packed]
            check=InventorySAT(p,packed,required,outside,cap,native='cadical153',additional_groups=extra)
            try: result['geo' if allow else 'base']=replay(check,p,ref,charged,required,cap,pins,budget)
            finally: check.close()
        cases.append(result)
        output.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
    for size in (13,344):
        for cap in (21,22,25):
            for budget in (small_budget,large_budget):
                compare('unfixed-'+str(size),groups[:size],cap,[],budget)
    # Matched partial pin scopes include both published phases and genuine held-out signed phases.
    for row in [*prices[:2],*prices[2:8]]:
        phase=int(row['phase'],16)
        ranking=list(range(1,p.n));Random(35).shuffle(ranking)
        for count in (120,152,159):
            pins=[i+1 if phase>>i&1 else -i-1 for i in ranking[:count]]
            compare(('heldout' if row['held_out'] else 'published')+'-'+row['phase'],groups[:13],
                    row['base13'],pins,small_budget)
    return receipt


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',required=True);cli.add_argument('--output',required=True,type=Path)
    cli.add_argument('--small-budget',default=1500,type=int);cli.add_argument('--large-budget',default=15000,type=int)
    opts=cli.parse_args()
    result=run(opts.tiles,opts.small_budget,opts.large_budget,opts.output)
    opts.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
