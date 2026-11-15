"""Minimize fixed-phase geographic conflicts and measure their held-out reuse."""
import argparse
from collections import Counter
import json
from pathlib import Path

from bench_r35_geographic_neighbors import four_flip_neighbors
from io_utils import atomic_json
from r32_gold_bonds import inventory
from r35_phase_anchors import narrow
from search_problem import SearchProblem


def minimize(problem,phase,anchor):
    checks=Counter()
    def rejected(tiles):
        checks['calls']+=1
        pins={t:(phase>>t)&1 for t in tiles}
        result=narrow(problem,None,pins,anchors=[anchor],unique=False,project=False,repeat=False)
        checks['root_probes']+=result['stats'].get('root_probes',0)
        return result['status']=='INFEASIBLE'
    known=list(range(problem.n))
    if not rejected(known):
        raise ValueError('The complete phase is not refuted at the proposed anchor.')
    step=max(1,len(known)//2)
    while step:
        index=0
        while index<len(known):
            trial=known[:index]+known[index+step:]
            if rejected(trial):known=trial
            else:index+=step
        step//=2
    if not rejected(known):
        raise ArithmeticError('Reduced pin set failed its final independent call.')
    return dict(anchor=anchor,pins=[[t,(phase>>t)&1] for t in known],work=dict(checks))


def matches(phase,pins):
    return len({((phase>>tile)&1)^value for tile,value in pins})<=1


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--tiles',type=Path,required=True)
    cli.add_argument('--output',type=Path,required=True)
    args=cli.parse_args();root=Path(__file__).resolve().parent/'docs'
    read=lambda path:json.loads(path.read_bytes())
    p=SearchProblem(read(args.tiles))
    checked=read(root/'R35_PHASE_ANCHORS.json')['cases']
    known={int(q['phase'],16) for q in checked};held=set()
    for donor in read(root/'R35_GEOGRAPHIC_LOSS_CORES.json')['reports']:
        phase=int(donor['phase'],16)
        _,neighbors=four_flip_neighbors(p,inventory(p).required,phase,100000)
        held.update(neighbors);held.add(phase)
    held-=known
    report=dict(fingerprint=p.fingerprint,held_out=len(held),rules=[])
    for q in checked:
        if q['independent']['unresolved_roots']:continue
        zero=next(row for row in q['probes'] if not row['remaining'])
        rule=minimize(p,int(q['phase'],16),zero['anchor'])
        rule.update(source_phase=q['phase'],held_out_matches=[hex(h) for h in sorted(held) if matches(h,rule['pins'])])
        report['rules'].append(rule);atomic_json(args.output,report)
        print(q['phase'],len(rule['pins']),len(rule['held_out_matches']),rule['work'],flush=True)
